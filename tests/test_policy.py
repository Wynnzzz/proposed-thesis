import unittest
from dataclasses import dataclass
from adaptive_routing.routing.accounting import cost_latency
from adaptive_routing.routing.policy import Theta, run_policy
from adaptive_routing.routing.sessions import RoundResult, Stage1Result

INF = float("inf")


@dataclass
class C:
    group_id: str
    latency_s: float
    cost_usd: float


class Fake:
    """Sesi skrip: mencatat tahap yang diminta (membuktikan evaluasi lazy)."""
    def __init__(self, s1, s2, revs):
        self.s1, self.s2, self.revs, self.asked = s1, s2, revs, []

    def stage1(self): self.asked.append("s1"); return self.s1
    def stage2(self): self.asked.append("s2"); return self.s2
    def revision(self, r): self.asked.append(f"r{r}"); return self.revs[r - 1]


NODES = ["a", "b", "c"]


def rr(answers, g, lat=1.0, cost=0.01):
    return RoundResult(list(answers), NODES, [""] * 3, [C(g, lat, cost) for _ in answers])


def s1(ans, p, cost=0.001):
    calls = [C("s1a", 0.2, cost)] + ([C("s1v", 0.1, cost)] if ans is not None else [])
    return Stage1Result(ans, p, calls)


T = lambda **k: Theta(**{**dict(tau1=0.8, tau2=0.0, tau3=0.0, N=3, R=2, S="heterogeneous"), **k})


class TestPolicy(unittest.TestCase):
    def run_(self, theta, sess, order=NODES):
        return run_policy(theta, sess, order)

    def test_accept_stage1_is_lazy(self):
        s = Fake(s1("A", 0.9), None, [])
        r = self.run_(T(), s)
        self.assertEqual((r.final_answer, r.stopped_at, s.asked), ("A", "s1", ["s1"]))

    def test_tau1_boundary_inclusive(self):
        self.assertEqual(self.run_(T(tau1=0.8), Fake(s1("A", 0.8), None, [])).stopped_at, "s1")
        s = Fake(s1("A", 0.79), rr("AAA", "s2"), [])
        self.assertEqual(self.run_(T(tau1=0.8), s).stopped_at, "s2")

    def test_sentinels(self):
        self.assertEqual(self.run_(T(tau1=-INF), Fake(s1("A", 0.0), None, [])).stopped_at, "s1")
        s = Fake(s1("A", 1.0), rr("BBB", "s2"), [])
        r = self.run_(T(tau1=INF), s)
        self.assertEqual((r.final_answer, r.stopped_at), ("B", "s2"))

    def test_invalid_slm_and_undefined_p_force_escalation(self):
        r = self.run_(T(tau1=-INF), Fake(s1(None, None), rr("BBB", "s2"), []))
        self.assertEqual((r.stopped_at, r.escalation_reason), ("s2", "slm_invalid"))
        r = self.run_(T(tau1=-INF), Fake(s1("A", None), rr("BBB", "s2"), []))
        self.assertEqual(r.escalation_reason, "p_true_undefined")

    def test_stage2_gate_tau2(self):
        s = Fake(s1("A", 0.1), rr(["A", "A", "B"], "s2"), [rr("AAA", "s3_r1")])
        r = self.run_(T(tau2=0.0), s)                      # tau2=0 menolak 2-1-0, lanjut debat
        self.assertEqual((r.final_answer, r.stopped_at), ("A", "s3_r1"))
        s = Fake(s1("A", 0.1), rr(["A", "A", "B"], "s2"), [])
        r = self.run_(T(tau2=0.6365), s)                   # cukup mayoritas
        self.assertEqual((r.final_answer, r.stopped_at, s.asked), ("A", "s2", ["s1", "s2"]))

    def test_no_unique_majority_never_accepted_at_stage2(self):
        s = Fake(s1("A", 0.1), rr(["A", "B", "C"], "s2"), [rr("BBB", "s3_r1")])
        r = self.run_(T(tau2=9.0), s)
        self.assertEqual(r.stopped_at, "s3_r1")

    def test_debate_stops_early_and_is_lazy(self):
        s = Fake(s1("A", 0.1), rr(["A", "B", "C"], "s2"),
                 [rr(["A", "A", "B"], "s3_r1"), rr("BBB", "s3_r2"), rr("CCC", "s3_r3")])
        r = self.run_(T(R=3, tau3=0.6365), s)
        self.assertEqual((r.final_answer, r.stopped_at, s.asked), ("A", "s3_r1", ["s1", "s2", "r1"]))

    def test_debate_exhausts_rounds_and_uses_tie_break(self):
        s = Fake(s1("A", 0.1), rr(["A", "B", "C"], "s2"),
                 [rr(["A", "B", "C"], "s3_r1"), rr(["A", "B", "C"], "s3_r2")])
        r = self.run_(T(R=2, tau3=0.0), s, order=["c", "a", "b"])
        self.assertEqual((r.final_answer, r.stopped_at), ("C", "s3_end"))
        self.assertEqual(len(r.stages_run), 4)

    def test_all_invalid_at_end_gives_none(self):
        s = Fake(s1(None, None), rr([None] * 3, "s2"), [rr([None] * 3, "s3_r1")])
        r = self.run_(T(R=1), s)
        self.assertIsNone(r.final_answer)

    def test_cost_latency_accounting(self):
        s = Fake(s1("A", 0.1), rr(["A", "B", "C"], "s2", lat=2.0), [rr(["A", "A", "B"], "s3_r1", lat=3.0)])
        r = self.run_(T(R=1, tau3=0.6365), s)
        cost, lat = cost_latency(r.calls, overhead_s=0.5)
        self.assertAlmostEqual(cost, 0.002 + 0.03 + 0.03)
        self.assertAlmostEqual(lat, 0.2 + 0.1 + (2.0 + 0.5) + (3.0 + 0.5))   # stage1 tanpa overhead


if __name__ == "__main__":
    unittest.main()
