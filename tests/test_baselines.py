"""B1-B5, A1-A2 (unit dengan sesi skrip), ambang tau1, peringkat agen, oracle, registri."""
import unittest
from dataclasses import dataclass
from adaptive_routing.baselines.oracle import oracle_outcomes
from adaptive_routing.baselines.policies import run_b1, run_b2, run_b3, run_b4, run_b5
from adaptive_routing.baselines.registry import PolicySpec, build_specs, run_spec
from adaptive_routing.ablations.variants import run_a1, run_a2
from adaptive_routing.data.types import Query
from adaptive_routing.metrics.dev_stats import rank_nodes_by_dev_accuracy
from adaptive_routing.metrics.outcome import make_outcome
from adaptive_routing.routing.policy import PolicyResult, Theta
from adaptive_routing.routing.sessions import RoundResult, Stage1Result
from adaptive_routing.routing.thresholds import tau1_candidates
from adaptive_routing.tracing.schema import CallRecord, PolicyOutcome
from tests.helpers import CFG

INF = float("inf")
NODES = ["a", "b", "c"]


@dataclass
class C:
    group_id: str
    latency_s: float
    cost_usd: float


def rr(ans, g, lat=1.0, cost=0.01):
    return RoundResult(list(ans), NODES, [""] * 3, [C(g, lat, cost) for _ in ans])


class Fake:
    def __init__(self, s1=None, s2=None, revs=(), strong=None, sc=None, slm=None):
        self._s1, self._s2, self._revs, self._strong, self._sc, self._slm = s1, s2, list(revs), strong, sc, slm
        self.asked = []

    def stage1(self): self.asked.append("stage1"); return self._s1
    def slm_answer(self): self.asked.append("slm"); return self._slm
    def strong(self): self.asked.append("strong"); return self._strong
    def sc(self, k): self.asked.append(f"sc{k}"); return self._sc
    def stage2(self): self.asked.append("s2"); return self._s2
    def revision(self, r): self.asked.append(f"r{r}"); return self._revs[r - 1]


def s1(ans, p): return Stage1Result(ans, p, [C("s1a", .2, .001)] + ([C("s1v", .1, .001)] if ans else []))
TH = Theta(0.8, 0.0, 0.0, 3, 2, "heterogeneous")


class TestBaselines(unittest.TestCase):
    def test_b1_only_answer_call(self):
        s = Fake(slm=Stage1Result("A", None, [C("s1a", .2, .001)]))
        r = run_b1(s)
        self.assertEqual((r.final_answer, s.asked, len(r.calls)), ("A", ["slm"], 1))

    def test_b2(self):
        s = Fake(strong=Stage1Result("C", None, [C("b2", 2, .01)]))
        self.assertEqual(run_b2(s).final_answer, "C")

    def test_b3_majority_and_tie_by_index(self):
        mk = lambda ans: RoundResult(ans, [f"slm#{i}" for i in range(len(ans))], [""] * len(ans), [C("b3", 1, .001)] * len(ans))
        self.assertEqual(run_b3(Fake(sc=mk(["A", "B", "A", None, "B"])), 5).final_answer, "A")   # 2-2: sampel indeks terkecil (A)
        self.assertEqual(run_b3(Fake(sc=mk(["B", "A", "A", None, "B"])), 5).final_answer, "B")
        self.assertIsNone(run_b3(Fake(sc=mk([None] * 3)), 3).final_answer)

    def test_b4_runs_all_rounds_without_gates_and_without_stage1(self):
        s = Fake(s2=rr("AAA", "s2"), revs=[rr("AAA", "s3_r1"), rr("AAA", "s3_r2")])
        r = run_b4(TH, s, NODES)
        self.assertEqual((s.asked, r.stopped_at, r.final_answer), (["s2", "r1", "r2"], "s3_end", "A"))
        self.assertNotIn("stage1", s.asked)

    def test_b5(self):
        st = Stage1Result("C", None, [C("b2", 2, .01)])
        s = Fake(s1=s1("A", 0.9), strong=st)
        r = run_b5(0.8, s)
        self.assertEqual((r.final_answer, "strong" in s.asked), ("A", False))
        s = Fake(s1=s1("A", 0.5), strong=st)
        r = run_b5(0.8, s)
        self.assertEqual((r.final_answer, r.stopped_at, r.escalation_reason), ("C", "b5_strong", "low_p_true"))
        self.assertEqual(len(r.calls), 3)
        self.assertEqual(run_b5(0.8, Fake(s1=s1(None, None), strong=st)).escalation_reason, "slm_invalid")
        self.assertEqual(run_b5(-INF, Fake(s1=s1("A", None), strong=st)).escalation_reason, "p_true_undefined")
        self.assertEqual(run_b5(INF, Fake(s1=s1("A", 1.0), strong=st)).final_answer, "C")    # +inf: selalu model kuat

    def test_a1_gate_then_fixed_debate(self):
        s = Fake(s1=s1("A", 0.1), s2=rr("AAA", "s2"), revs=[rr("AAA", "s3_r1"), rr("AAA", "s3_r2")])
        r = run_a1(TH, s, NODES)                       # stage 2 sepakat bulat, tetapi A1 tidak punya gerbang disagreement
        self.assertEqual((s.asked, r.stopped_at), (["stage1", "s2", "r1", "r2"], "s3_end"))
        s = Fake(s1=s1("A", 0.9))
        self.assertEqual(run_a1(TH, s, NODES).stopped_at, "s1")

    def test_a2_no_stage1_but_gates(self):
        s = Fake(s2=rr("AAA", "s2"))
        r = run_a2(TH, s, NODES)
        self.assertEqual((s.asked, r.stopped_at, r.final_answer, r.escalation_reason), (["s2"], "s2", "A", "no_stage1"))
        self.assertFalse(any(c.group_id.startswith("s1") for c in r.calls))
        s = Fake(s2=rr(["A", "A", "B"], "s2"), revs=[rr("AAA", "s3_r1")])
        self.assertEqual(run_a2(Theta(INF, 0.0, 0.0, 3, 2, "heterogeneous"), s, NODES).stopped_at, "s3_r1")


class TestThresholdsRegistry(unittest.TestCase):
    def test_tau1_candidates(self):
        ps = [0.1, 0.2, 0.2, 0.9, None]
        t = tau1_candidates(ps, [0.0, 0.5, 1.0])
        self.assertEqual(t, [float("-inf"), 0.1, 0.2, 0.9, float("inf")])
        self.assertTrue(set(tau1_candidates(ps, [0.0, 0.3, 0.7, 1.0], False)) <= {0.1, 0.2, 0.9})
        self.assertEqual(tau1_candidates([], [0.5]), [float("-inf"), float("inf")])
        self.assertEqual(tau1_candidates([None], [0.5], False), [])

    def test_build_specs_counts_and_unique_ids(self):
        sp = build_specs(CFG.raw, [-INF, 0.5, 0.8, INF])
        from collections import Counter
        c = Counter(s.kind for s in sp)
        self.assertEqual(c, {"B1": 1, "B2": 1, "B3": 1, "B4": 6, "B5": 4, "A1": 24, "A2": 24, "A3": 96})
        self.assertEqual(len({s.policy_id for s in sp}), len(sp))
        self.assertEqual(PolicySpec("A3", Theta(0.5, 0.0, 0.6365, 3, 2, "homogeneous")).policy_id,
                         "A3|t1=0.5|t2=0|t3=0.6365|N=3|R=2|S=hom")
        self.assertEqual(PolicySpec("B5", tau=-INF).policy_id, "B5|t=-inf")

    def test_run_spec_dispatch_and_unknown(self):
        s = Fake(slm=Stage1Result("A", None, [C("s1a", .2, .001)]))
        self.assertEqual(run_spec(PolicySpec("B1"), s, NODES).final_answer, "A")
        with self.assertRaises(ValueError):
            run_spec(PolicySpec("ZZ"), s, NODES)


class TestOutcomeOracleRanking(unittest.TestCase):
    def q(self, g="A"): return Query("q1", "mmlu", "dev", "b", g)

    def test_outcome_labels(self):
        r = PolicyResult("A", "s1", ["s1"], [C("s1a", .2, .001), C("s1v", .1, .002)])
        o = make_outcome(self.q(), r, "A3", "composed_estimated")
        self.assertEqual((o.em, o.correct, o.latency_kind), (1.0, True, "composed_estimated"))
        self.assertAlmostEqual(o.latency_s, 0.3); self.assertAlmostEqual(o.cost_usd, 0.003)
        o = make_outcome(self.q(), r, "A3", "measured_e2e", measured_latency_s=0.9)
        self.assertEqual((o.latency_s, o.latency_kind), (0.9, "measured_e2e"))
        with self.assertRaises(ValueError): make_outcome(self.q(), r, "A3", "measured_e2e")
        with self.assertRaises(ValueError): make_outcome(self.q(), r, "A3", "composed_estimated", measured_latency_s=1.0)
        self.assertEqual(make_outcome(self.q("B"), r, "x", "composed_estimated").correct, False)

    def test_oracle_picks_correct_then_cheapest(self):
        def o(pid, em, cost, lat, f1=None):
            return PolicyOutcome("q1", pid, "A", em, f1, em == 1.0, ["x"], 1, cost, lat, "measured_e2e")
        by = {"B1": [o("B1", 0.0, 0.001, 0.2)], "B2": [o("B2", 1.0, 0.01, 2.0)], "B4": [o("B4", 1.0, 0.005, 5.0)]}
        r = oracle_outcomes(by, "mmlu")[0]
        self.assertEqual((r.policy_id, r.stages_run, r.latency_kind, r.cost_usd), ("B7", ["oracle:B4"], "composed_estimated", 0.005))
        by = {"B1": [o("B1", 0.0, 0.001, 0.2)], "B2": [o("B2", 0.0, 0.01, 2.0)]}      # tidak ada yang benar: termurah
        self.assertEqual(oracle_outcomes(by, "mmlu")[0].stages_run, ["oracle:B1"])
        with self.assertRaises(ValueError):
            oracle_outcomes({"B1": [o("B1", 1, 0, 0)], "B2": []}, "mmlu")

    def test_oracle_hotpot_uses_f1(self):
        def o(pid, em, f1, cost):
            return PolicyOutcome("q1", pid, "x", em, f1, em == 1.0, ["x"], 1, cost, 1.0, "measured_e2e")
        by = {"B1": [o("B1", 0.0, 0.6, 0.001)], "B2": [o("B2", 0.0, 0.3, 0.0001)]}
        self.assertEqual(oracle_outcomes(by, "hotpotqa")[0].stages_run, ["oracle:B1"])

    def test_rank_nodes_dev_only(self):
        qs = [Query(f"q{i}", "mmlu", "dev", "b", "A") for i in range(4)]
        def call(q, node, ans):
            return CallRecord(call_id=f"{q}{node}", run_id="r", query_id=q, dataset="mmlu", split="dev", stage="s2_agent",
                              group_id="s2", node_id=node, parsed_answer=ans, parse_ok=ans is not None)
        calls = []
        for i in range(4):
            calls += [call(f"q{i}", "agent_qwen", "A" if i < 1 else "B"), call(f"q{i}", "agent_llama3_8b_lite", "B"),
                      call(f"q{i}", "agent_glm", "A" if i < 3 else None)]
        self.assertEqual(rank_nodes_by_dev_accuracy(qs, calls, "heterogeneous", 3, CFG.raw),
                         ["agent_glm", "agent_qwen", "agent_llama3_8b_lite"])
        with self.assertRaises(ValueError):
            rank_nodes_by_dev_accuracy([Query("q", "mmlu", "test", "b", "A")], calls, "heterogeneous", 3, CFG.raw)


if __name__ == "__main__":
    unittest.main()
