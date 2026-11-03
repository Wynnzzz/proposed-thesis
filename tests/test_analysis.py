import copy, json, os, tempfile, unittest
import numpy as np
from adaptive_routing.baselines.registry import PolicySpec, build_specs, run_spec, spec_from_dict, spec_to_dict
from adaptive_routing.config import Config
from adaptive_routing.data.types import Query
from adaptive_routing.llm.recording import RecordingLLM
from adaptive_routing.metrics.budget import direct_test_estimate, paid_cost_per_query, project, reasoning_share
from adaptive_routing.metrics.pareto import pareto_indices
from adaptive_routing.metrics.selection import (SelectionError, feasible, freeze_selection, references, select_scenarios,
                                                verify_selection)
from adaptive_routing.metrics.dev_stats import rank_nodes_by_dev_accuracy
from adaptive_routing.metrics.outcome import make_outcome
from adaptive_routing.routing.collect import collect_full_trace
from adaptive_routing.routing.policy import Theta
from adaptive_routing.routing.runner import SimGroupRunner
from adaptive_routing.routing.sessions import LiveSession
from adaptive_routing.runners.analyze_dev import analyze_dev
from adaptive_routing.runners.replay import replay_dev
from adaptive_routing.tracing.schema import TraceManifest
from adaptive_routing.tracing.store import TraceError, read_trace, validate_trace, write_trace
from tests.helpers import CFG, PB, make_mock, make_queries

INF = float("inf")


def build_traces(tmp, split="dev", cfg_hash=None, n=(20, 10)):
    qs, gold = make_queries(*n)
    mock, clock = make_mock(gold)
    rec = RecordingLLM(mock, PB, "trace", cache_enabled=True)
    runner = SimGroupRunner(clock)
    qs = [Query(q.query_id, q.dataset, split, q.body, q.gold, q.meta) for q in qs]
    for q in qs:
        for S in ("heterogeneous", "homogeneous"):
            collect_full_trace(LiveSession(q, rec, CFG.raw, S, 3, runner), 3, baselines=True, k_sc=5)
    dirs = []
    for ds in ("mmlu", "hotpotqa"):
        qd = [q for q in qs if q.dataset == ds]
        ids = {q.query_id for q in qd}
        man = TraceManifest(f"run-{ds}", "trace_dev", cfg_hash or CFG.hash(), None, "2026-11-01T00:00:00Z", ds, split,
                            len(qd), [q.query_id for q in qd], {}, {}, 1)
        d = os.path.join(tmp, "traces", ds, split)
        write_trace(d, man, qd, [c for c in rec.records if c.query_id in ids])
        dirs.append(d)
    return dirs, qs, gold


class TestAnalysisPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dirs, cls.qs, cls.gold = build_traces(cls.tmp.name)
        cls.out = os.path.join(cls.tmp.name, "results")
        cls.report = analyze_dev(cls.dirs, CFG, cls.out)

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def test_outputs_exist(self):
        for ds in ("mmlu", "hotpotqa"):
            for f in ("outcomes.jsonl", "summary.json", "frontier.json", "hypervolume.json", "frontier.png"):
                p = os.path.join(self.out, "dev_replay", ds, f)
                self.assertTrue(os.path.getsize(p) > 0, p)
        self.assertTrue(os.path.exists(os.path.join(self.out, "selected_policies.json")))

    def test_frozen_selection_verifies_and_detects_tampering(self):
        p = os.path.join(self.out, "selected_policies.json")
        d = verify_selection(p)
        self.assertTrue(d["config_unchanged_since_trace"])
        self.assertEqual(d["latency_kind_for_selection"], "composed_estimated")
        with open(p, encoding="utf-8") as f: raw = f.read()
        bad = os.path.join(self.tmp.name, "tampered.json")
        with open(bad, "w", encoding="utf-8") as f: f.write(raw.replace("cost_cap_50", "cost_cap_99", 1))
        with self.assertRaises(SelectionError): verify_selection(bad)

    def test_selected_are_feasible_pareto_and_spec_roundtrip(self):
        d = verify_selection(os.path.join(self.out, "selected_policies.json"))
        for ds, info in d["per_dataset"].items():
            with open(os.path.join(self.out, "dev_replay", ds, "summary.json"), encoding="utf-8") as f:
                rows = json.load(f)["rows"]
            by = {r["policy_id"]: r for r in rows}
            fam_front = {}
            for k in {r["kind"] for r in rows}:
                fr_rows = [r for r in rows if r["kind"] == k]
                fam_front[k] = {fr_rows[i]["policy_id"] for i in pareto_indices([[r["cost"], r["latency"], r["error"]] for r in fr_rows])}
            sc_by = {s["name"]: s for s in d["scenarios_declared"]}
            refs = {"B2": by["B2"], "B4_ref": by[info["references"]["B4_ref"]["policy_id"]]}
            n_sel = 0
            for sc, byk in info["selected"].items():
                for kind, v in byk.items():
                    if v is None: continue
                    n_sel += 1
                    r = by[v["policy_id"]]
                    self.assertTrue(feasible(r, sc_by[sc], refs))
                    self.assertIn(v["policy_id"], fam_front[kind])             # Pareto-optimal DALAM keluarganya
                    self.assertEqual(spec_from_dict(v["spec"]).policy_id, v["policy_id"])
                    lo, hi = v["dev"]["error_ci"]; self.assertLessEqual(lo, v["dev"]["error"] + 1e-9); self.assertGreaterEqual(hi, v["dev"]["error"] - 1e-9)
            self.assertGreater(n_sel, 0)

    def test_hypervolume_all_dominates_each_family(self):
        for ds in ("mmlu", "hotpotqa"):
            with open(os.path.join(self.out, "dev_replay", ds, "hypervolume.json"), encoding="utf-8") as f:
                hv = json.load(f)
            for k, v in hv["by_family"].items():
                self.assertGreaterEqual(hv["all"] + 1e-12, v, (ds, k))
            self.assertGreater(hv["all"], 0)

    def test_selected_policy_matches_direct_execution_on_dev(self):
        """Pemeriksaan silang end-to-end: spec dari file beku -> eksekusi langsung (mock) == hasil replay tersimpan."""
        d = verify_selection(os.path.join(self.out, "selected_policies.json"))
        ds = "mmlu"
        v = next(x for byk in d["per_dataset"][ds]["selected"].values() for x in byk.values() if x and x["spec"]["kind"] == "A3")
        spec = spec_from_dict(v["spec"])
        tie = d["per_dataset"][ds]["tie_orders"][f"{spec.theta.S}|{spec.theta.N}"]
        replayed = {}
        with open(os.path.join(self.out, "dev_replay", ds, "outcomes.jsonl"), encoding="utf-8") as f:
            for line in f:
                o = json.loads(line)
                if o["policy_id"] == spec.policy_id: replayed[o["query_id"]] = o
        self.assertTrue(replayed)
        for q in [q for q in self.qs if q.dataset == ds]:
            mock, clock = make_mock(self.gold)
            sim = SimGroupRunner(clock)
            r = run_spec(spec, LiveSession(q, RecordingLLM(mock, PB, "d"), CFG.raw, spec.theta.S, 3, sim), tie)
            o = make_outcome(q, r, spec.policy_id, "measured_e2e", measured_latency_s=sim.clock())
            rp = replayed[q.query_id]
            self.assertEqual((o.final_answer, o.em, o.n_calls), (rp["final_answer"], rp["em"], rp["n_calls"]))
            self.assertAlmostEqual(o.cost_usd, rp["cost_usd"], places=12); self.assertAlmostEqual(o.latency_s, rp["latency_s"], places=9)
            self.assertEqual(rp["latency_kind"], "composed_estimated")

    def test_budget_projection_labelled_as_extrapolation(self):
        b = self.report["datasets"]["mmlu"]["budget_direct_test_extrapolated"]
        self.assertGreater(b["upper_total"], b["expected_total"])
        self.assertIn("ekstrapolasi", self.report["datasets"]["mmlu"]["budget_trace_dev_extrapolated"]["note"])


class TestGuards(unittest.TestCase):
    def test_test_split_is_refused_everywhere(self):
        with tempfile.TemporaryDirectory() as tmp:
            dirs, qs, _ = build_traces(tmp, split="test", n=(4, 2))
            man, q, c = read_trace(dirs[0])
            with self.assertRaises(TraceError): validate_trace(man, q, c, "dev")
            with self.assertRaises(TraceError): analyze_dev(dirs, CFG, os.path.join(tmp, "o"))
            with self.assertRaises(ValueError): replay_dev(q, c, CFG.raw, [PolicySpec("B1")], {})

    def test_config_change_since_trace_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            dirs, _, _ = build_traces(tmp, cfg_hash="deadbeef", n=(8, 4))
            rep = analyze_dev(dirs, CFG, os.path.join(tmp, "o"), make_plots=False)
            self.assertFalse(rep["config_unchanged_since_trace"])
            self.assertFalse(verify_selection(os.path.join(tmp, "o", "selected_policies.json"))["config_unchanged_since_trace"])

    def test_mixed_config_hash_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            dirs, _, _ = build_traces(tmp, n=(6, 3))
            mp = os.path.join(dirs[1], "manifest.json")
            with open(mp, encoding="utf-8") as f: m = json.load(f)
            m["config_hash"] = "other"
            with open(mp, "w", encoding="utf-8") as f: json.dump(m, f)
            with self.assertRaises(ValueError): analyze_dev(dirs, CFG, os.path.join(tmp, "o"), make_plots=False)

    def test_impossible_scenario_is_none_not_relaxed(self):
        raw = copy.deepcopy(CFG.raw)
        raw["protocol"]["selection"]["scenarios"] = [{"name": "impossible", "max_cost_ratio_vs_B2": 1e-6, "max_latency_ratio_vs_B4": None}]
        with tempfile.TemporaryDirectory() as tmp:
            dirs, _, _ = build_traces(tmp, cfg_hash=Config(raw).hash(), n=(8, 4))
            rep = analyze_dev(dirs, Config(raw), os.path.join(tmp, "o"), make_plots=False)
            for r in rep["datasets"].values():
                self.assertTrue(all(v is None for v in r["selected"]["impossible"].values()))
            d = verify_selection(os.path.join(tmp, "o", "selected_policies.json"))
            self.assertEqual(len(d["per_dataset"]["mmlu"]["infeasible"]), len(raw["protocol"]["selection"]["families"]))

    def test_config_validation_of_selection(self):
        raw = copy.deepcopy(CFG.raw); raw["protocol"]["selection"]["reference_b4"]["R"] = 9
        with self.assertRaises(Exception): Config(raw).validate(PB)
        raw = copy.deepcopy(CFG.raw); raw["protocol"]["selection"]["scenarios"] = [{"name": "x", "max_cost_ratio_vs_B2": None, "max_latency_ratio_vs_B4": None}]
        with self.assertRaises(Exception): Config(raw).validate(PB)


class TestSelectionUnit(unittest.TestCase):
    def row(self, pid, kind, c, l, e): return {"policy_id": pid, "kind": kind, "spec": {}, "cost": c, "latency": l, "error": e}

    def test_lexicographic_ties_and_feasibility(self):
        rows = [self.row("B2", "B2", 1.0, 1.0, 0.1), self.row("B4|N=3|R=2|S=het", "B4", 2.0, 10.0, 0.2),
                self.row("A3|x", "A3", 0.4, 3.0, 0.15), self.row("A3|y", "A3", 0.3, 3.0, 0.15), self.row("A3|z", "A3", 0.6, 1.0, 0.12),
                self.row("A3|w", "A3", 0.2, 9.0, 0.30)]
        raw = copy.deepcopy(CFG.raw)
        raw["protocol"]["selection"]["scenarios"] = [{"name": "c50", "max_cost_ratio_vs_B2": 0.5, "max_latency_ratio_vs_B4": None},
                                                    {"name": "lat", "max_cost_ratio_vs_B2": None, "max_latency_ratio_vs_B4": 0.2}]
        raw["protocol"]["selection"]["families"] = ["A3"]
        s = select_scenarios(rows, raw)
        self.assertEqual(s["c50"]["A3"]["policy_id"], "A3|y")        # seri error 0.15 -> biaya terendah; |z| melanggar cap
        self.assertEqual(s["lat"]["A3"]["policy_id"], "A3|z")        # latensi <= 0.2*10 = 2

    def test_missing_reference_raises(self):
        with self.assertRaises(SelectionError): references([self.row("B2", "B2", 1, 1, 0)], CFG.raw)

    def test_spec_roundtrip_with_inf(self):
        sp = PolicySpec("A3", Theta(-INF, 0.0, 0.6365, 3, 2, "homogeneous"))
        self.assertEqual(spec_from_dict(json.loads(json.dumps(spec_to_dict(sp)))), sp)
        sp = PolicySpec("B5", tau=INF)
        self.assertEqual(spec_from_dict(json.loads(json.dumps(spec_to_dict(sp)))), sp)

    def test_freeze_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "s.json"); freeze_selection(p, {"a": 1, "b": [1, 2]})
            self.assertEqual(verify_selection(p)["a"], 1)


class TestBudget(unittest.TestCase):
    def test_paid_cost_ignores_cache_hits_and_project(self):
        from adaptive_routing.tracing.schema import CallRecord
        mk = lambda q, c, hit=False, m="a", out=100, rs=40: CallRecord(call_id=f"{q}{c}{hit}", run_id="r", query_id=q, dataset="mmlu", split="dev",
             stage="s2_agent", group_id="s2", node_id="n", model_key=m, cost_usd=c, cache_hit=hit, output_tokens=out, reasoning_tokens=rs)
        calls = [mk("q1", 0.01), mk("q1", 0.01, hit=True), mk("q2", 0.03)]
        self.assertEqual(paid_cost_per_query(calls), {"q1": 0.01, "q2": 0.03})
        p = project([0.01, 0.03], 100, safety=2.0)
        self.assertAlmostEqual(p["expected_total"], 2.0); self.assertGreaterEqual(p["upper_total"], p["expected_total"])
        self.assertAlmostEqual(reasoning_share(calls)["a"], 0.4)
        self.assertAlmostEqual(direct_test_estimate([{"policy_id": "x", "cost": 0.01}], 300, 2)["expected_total"], 6.0)
        with self.assertRaises(ValueError): project([], 10)


if __name__ == "__main__":
    unittest.main()
