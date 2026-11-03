"""Tes inti desain D2: replay dari trace lengkap == eksekusi langsung (jawaban, tahap, biaya, latensi),
dan eksekusi langsung hanya memanggil tahap yang dibutuhkan. Backend MOCK; bukan hasil eksperimen."""
import itertools, unittest
from adaptive_routing.llm.mock import SimClock
from adaptive_routing.llm.recording import RecordingLLM
from adaptive_routing.routing.accounting import cost_latency
from adaptive_routing.routing.collect import collect_full_trace
from adaptive_routing.routing.policy import Theta, run_policy
from adaptive_routing.routing.runner import SimGroupRunner
from adaptive_routing.routing.sessions import LiveSession, TraceSession, TraceMissing
from adaptive_routing.baselines.registry import build_specs, run_spec
from adaptive_routing.metrics.dev_stats import rank_nodes_by_dev_accuracy
from adaptive_routing.metrics.outcome import make_outcome
from adaptive_routing.routing.thresholds import tau1_candidates
from tests.helpers import CFG, PB, make_mock, make_queries

INF = float("inf")
TIE = ["agent_glm", "agent_qwen", "agent_llama3_8b_lite", "agent_qwen#0", "agent_qwen#1", "agent_qwen#2"]


def grid():
    for t1, t2, t3, R, S in itertools.product([-INF, 0.5, 0.7, 0.9, INF], [0.0, 0.6365], [0.0, 0.6365], [1, 2, 3],
                                              ["heterogeneous", "homogeneous"]):
        yield Theta(t1, t2, t3, 3, R, S)


class TestReplayEqualsDirect(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qs, gold = make_queries()
        cls.gold = gold
        mock, clock = make_mock(gold)
        cls.collect_rec = RecordingLLM(mock, PB, "trace", cache_enabled=True, allow_estimated=False)
        runner = SimGroupRunner(clock)
        for q in cls.qs:
            for S in ("heterogeneous", "homogeneous"):
                collect_full_trace(LiveSession(q, cls.collect_rec, CFG.raw, S, 3, runner), 3)
        cls.calls = cls.collect_rec.records

    def direct(self, q, theta):
        mock, clock = make_mock(self.gold)
        rec = RecordingLLM(mock, PB, "direct", cache_enabled=False)
        runner = SimGroupRunner(clock)
        t0 = runner.clock()
        res = run_policy(theta, LiveSession(q, rec, CFG.raw, theta.S, 3, runner), TIE)
        return res, runner.clock() - t0, rec

    def test_equivalence_over_grid(self):
        n_checked, fewer = 0, 0
        for q in self.qs:
            for th in grid():
                rep = run_policy(th, TraceSession(q, self.calls, th.S, 3, CFG.raw), TIE)
                dr, e2e, rec = self.direct(q, th)
                self.assertEqual(rep.final_answer, dr.final_answer, (q.query_id, th))
                self.assertEqual(rep.stopped_at, dr.stopped_at)
                self.assertEqual(rep.stages_run, dr.stages_run)
                self.assertEqual(rep.escalation_reason, dr.escalation_reason)
                c_rep, l_rep = cost_latency(rep.calls, 0.0)
                c_dr, l_dr = cost_latency(dr.calls, 0.0)
                self.assertAlmostEqual(c_rep, c_dr, places=12)
                self.assertAlmostEqual(l_rep, l_dr, places=9)
                self.assertAlmostEqual(l_dr, e2e, places=9)                  # composed == terukur (jam simulasi, o=0)
                self.assertAlmostEqual(sum(r.cost_usd for r in rec.records), c_dr, places=12)
                self.assertEqual(len(rec.records), len(dr.calls))             # tidak ada pemanggilan di luar yang dipakai
                full = len([c for c in self.calls if c.query_id == q.query_id and not c.cache_hit])
                fewer += len(rec.records) < full // 2 + 8
                n_checked += 1
        self.assertGreater(n_checked, 3000)
        self.assertGreater(fewer, 0)

    def test_direct_executes_lazily(self):
        q = self.qs[0]
        _, _, rec = self.direct(q, Theta(-INF, 0, 0, 3, 3, "heterogeneous"))
        self.assertLessEqual(len(rec.records), 2)                            # hanya tahap 1

    def test_trace_has_cache_hits_for_shared_stage1(self):
        hits = [c for c in self.calls if c.cache_hit]
        self.assertTrue(hits and all(c.stage.startswith("s1") for c in hits))

    def test_missing_trace_is_explicit_error(self):
        q = self.qs[0]
        with self.assertRaises(TraceMissing):
            run_policy(Theta(INF, 0, 0, 3, 1, "heterogeneous"), TraceSession(q, [], "heterogeneous", 3, CFG.raw), TIE)

    def test_both_logprob_formats_give_same_decisions(self):
        q = self.qs[2]
        out = []
        for fmt in ("openai_content", "together_tokens"):
            mock, clock = make_mock(self.gold, fmt=fmt)
            rec = RecordingLLM(mock, PB, "f")
            s = LiveSession(q, rec, CFG.raw, "heterogeneous", 3, SimGroupRunner(clock))
            out.append(s.stage1().p_true)
        self.assertEqual(out[0], out[1])


class TestEdgeBranchesEquivalent(unittest.TestCase):
    """Cabang ekstrem lewat jalur live: banyak jawaban invalid dan backend tanpa logprobs."""

    def _run(self, fmt, invalid_rate):
        qs, gold = make_queries(16, 8)
        mock, clock = make_mock(gold, fmt=fmt, invalid_rate=invalid_rate, seed=11)
        rec = RecordingLLM(mock, PB, "t", cache_enabled=True)
        runner = SimGroupRunner(clock)
        for q in qs:
            for S in ("heterogeneous", "homogeneous"):
                collect_full_trace(LiveSession(q, rec, CFG.raw, S, 3, runner), 3)
        reasons, finals_none = set(), 0
        for q in qs:
            for th in grid():
                rep = run_policy(th, TraceSession(q, rec.records, th.S, 3, CFG.raw), TIE)
                m2, c2 = make_mock(gold, fmt=fmt, invalid_rate=invalid_rate, seed=11)
                r2 = RecordingLLM(m2, PB, "d")
                sim = SimGroupRunner(c2)
                dr = run_policy(th, LiveSession(q, r2, CFG.raw, th.S, 3, sim), TIE)
                self.assertEqual((rep.final_answer, rep.stopped_at, rep.escalation_reason),
                                 (dr.final_answer, dr.stopped_at, dr.escalation_reason), (q.query_id, th))
                self.assertAlmostEqual(cost_latency(rep.calls)[0], cost_latency(dr.calls)[0], places=12)
                self.assertAlmostEqual(cost_latency(rep.calls)[1], sim.clock(), places=9)
                reasons.add(rep.escalation_reason); finals_none += rep.final_answer is None
        return reasons, finals_none

    def test_invalid_heavy(self):
        reasons, finals_none = self._run("openai_content", 0.5)
        self.assertIn("slm_invalid", reasons)
        self.assertGreater(finals_none, 0)               # cabang 'tidak ada jawaban valid' tercapai

    def test_no_logprobs_backend(self):
        reasons, _ = self._run("none", 0.04)
        self.assertIn("p_true_undefined", reasons)
        self.assertNotIn("low_p_true", reasons)           # tanpa logprobs tidak ada keputusan berbasis p


class TestAllSpecsReplayEqualsDirect(unittest.TestCase):
    """Semua kebijakan (B1-B5, A1-A3) lewat registri: replay dari trace == eksekusi langsung (mock)."""

    @classmethod
    def setUpClass(cls):
        cls.qs, gold = make_queries(20, 10)
        cls.gold = gold
        mock, clock = make_mock(gold)
        cls.rec = RecordingLLM(mock, PB, "trace", cache_enabled=True)
        runner = SimGroupRunner(clock)
        for q in cls.qs:
            for S in ("heterogeneous", "homogeneous"):
                collect_full_trace(LiveSession(q, cls.rec, CFG.raw, S, 3, runner), 3, baselines=True, k_sc=5)
        ps = [c.p_true for c in cls.rec.records if c.stage == "s1_verify"]
        cls.tau1 = tau1_candidates(ps, CFG.raw["policy_space"]["tau1"]["quantile_levels"][::2])
        cls.specs = build_specs(CFG.raw, cls.tau1)
        cls.tie = {S: rank_nodes_by_dev_accuracy(cls.qs, cls.rec.records, S, 3, CFG.raw)
                   for S in ("heterogeneous", "homogeneous")}

    def test_every_spec_equivalent(self):
        n = 0
        kinds = set()
        for spec in self.specs:
            S = spec.theta.S if spec.theta else "heterogeneous"
            for q in self.qs:
                rep = run_spec(spec, TraceSession(q, self.rec.records, S, 3, CFG.raw), self.tie[S])
                mock, clock = make_mock(self.gold)
                rec2 = RecordingLLM(mock, PB, "d")
                sim = SimGroupRunner(clock)
                t0 = sim.clock()
                dr = run_spec(spec, LiveSession(q, rec2, CFG.raw, S, 3, sim), self.tie[S])
                e2e = sim.clock() - t0
                self.assertEqual((rep.final_answer, rep.stopped_at, rep.stages_run),
                                 (dr.final_answer, dr.stopped_at, dr.stages_run), (spec.policy_id, q.query_id))
                o_rep = make_outcome(q, rep, spec.policy_id, "composed_estimated")
                o_dr = make_outcome(q, dr, spec.policy_id, "measured_e2e", measured_latency_s=e2e)
                self.assertEqual((o_rep.em, o_rep.f1, o_rep.n_calls), (o_dr.em, o_dr.f1, o_dr.n_calls))
                self.assertAlmostEqual(o_rep.cost_usd, o_dr.cost_usd, places=12)
                self.assertAlmostEqual(o_rep.latency_s, o_dr.latency_s, places=9)
                self.assertEqual(len(rec2.records), len(dr.calls))
                n += 1; kinds.add(spec.kind)
        self.assertEqual(kinds, {"B1", "B2", "B3", "B4", "B5", "A1", "A2", "A3"})
        self.assertGreater(n, 4000)

    def test_cost_ordering_sanity(self):
        """B1 < B1+verifikasi (B5 dengan tau=-inf) dan A2 tidak membayar tahap 1."""
        q = self.qs[0]
        by = {s.policy_id: s for s in self.specs}
        cost = lambda pid: make_outcome(q, run_spec(by[pid], TraceSession(q, self.rec.records, "heterogeneous", 3, CFG.raw),
                                                    self.tie["heterogeneous"]), pid, "composed_estimated").cost_usd
        self.assertLess(cost("B1"), cost("B5|t=-inf"))
        a2 = [s for s in self.specs if s.kind == "A2" and s.theta.R == 1 and s.theta.S == "heterogeneous"][0]
        res = run_spec(a2, TraceSession(q, self.rec.records, "heterogeneous", 3, CFG.raw), self.tie["heterogeneous"])
        self.assertFalse(any(c.stage.startswith("s1") for c in res.calls))

    def test_oracle_dominates_every_component(self):
        from adaptive_routing.baselines.oracle import oracle_outcomes
        from adaptive_routing.metrics.qa import primary_score
        S = "heterogeneous"
        comps = [s for s in self.specs if s.kind in ("B1", "B2", "B3") or (s.kind == "B4" and s.theta.S == S)]
        for ds in ("mmlu", "hotpotqa"):
            qs = [q for q in self.qs if q.dataset == ds]
            by = {sp.policy_id: [make_outcome(q, run_spec(sp, TraceSession(q, self.rec.records, S, 3, CFG.raw), self.tie[S]),
                                              sp.policy_id, "composed_estimated") for q in qs] for sp in comps}
            orc = {o.query_id: o for o in oracle_outcomes(by, ds)}
            self.assertEqual(len(orc), len(qs))
            for outs in by.values():
                for o in outs:
                    self.assertGreaterEqual(primary_score(ds, orc[o.query_id].em, orc[o.query_id].f1) + 1e-12,
                                            primary_score(ds, o.em, o.f1))
            self.assertTrue(all(o.latency_kind == "composed_estimated" for o in orc.values()))

if __name__ == "__main__":
    unittest.main()
