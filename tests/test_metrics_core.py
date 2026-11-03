import itertools, unittest
import numpy as np
from scipy import stats as sps
from adaptive_routing.metrics.hypervolume import hypervolume, minmax_bounds, normalize
from adaptive_routing.metrics.pareto import dominates, pareto_indices
from adaptive_routing.metrics.stats import bootstrap_ci, compare_policies, mcnemar_exact, paired_bootstrap_diff
from adaptive_routing.metrics.summary import summarize
from adaptive_routing.tracing.schema import PolicyOutcome


def out(qid, em, cost, lat, kind="composed_estimated", f1=None, ans="A", pid="p"):
    return PolicyOutcome(qid, pid, ans, em, f1, em == 1.0, ["s1"], 2, cost, lat, kind)


class TestPareto(unittest.TestCase):
    def test_dominance(self):
        self.assertTrue(dominates([1, 1, 1], [1, 2, 1])); self.assertFalse(dominates([1, 1, 1], [1, 1, 1]))
        self.assertFalse(dominates([1, 3, 1], [2, 2, 2]))

    def test_front(self):
        P = [[1, 5, 1], [2, 2, 2], [3, 3, 3], [1, 5, 1], [0.5, 9, 9]]
        self.assertEqual(pareto_indices(P), [0, 1, 3, 4])           # duplikat sama-sama tetap; [3,3,3] didominasi [2,2,2]

    def test_front_is_nondominated_and_complete_on_random(self):
        P = np.random.default_rng(1).random((200, 3))
        idx = set(pareto_indices(P))
        for i in range(200):
            dominated = any(dominates(P[j], P[i]) for j in range(200))
            self.assertEqual(i in idx, not dominated)


class TestHV(unittest.TestCase):
    ref = (1.0, 1.0, 1.0)

    def test_single_and_ignored_points(self):
        self.assertAlmostEqual(hypervolume([[0.5, 0.5, 0.5]], self.ref), 0.125)
        self.assertEqual(hypervolume([[1.0, 0.2, 0.2], [1.5, 0, 0]], self.ref), 0.0)
        self.assertEqual(hypervolume(np.empty((0, 3)), self.ref), 0.0)

    def brute(self, P):
        ref = np.array(self.ref); tot = 0.0
        for r in range(1, len(P) + 1):
            for sub in itertools.combinations(range(len(P)), r):
                m = np.max(np.array([P[i] for i in sub]), axis=0)
                tot += (-1) ** (r + 1) * float(np.prod(np.maximum(ref - m, 0)))
        return tot

    def test_matches_inclusion_exclusion(self):
        rng = np.random.default_rng(3)
        for _ in range(25):
            P = rng.random((rng.integers(1, 8), 3))
            self.assertAlmostEqual(hypervolume(P, self.ref), self.brute(P), places=10)

    def test_dominated_points_do_not_change_hv_and_duplicates_ok(self):
        P = np.array([[0.2, 0.3, 0.4], [0.5, 0.5, 0.5], [0.2, 0.3, 0.4]])
        self.assertAlmostEqual(hypervolume(P, self.ref), hypervolume(P[:1], self.ref))

    def test_monotone_and_tied_first_coordinate(self):
        P = np.array([[0.3, 0.6, 0.2], [0.3, 0.2, 0.6], [0.7, 0.1, 0.1]])
        self.assertAlmostEqual(hypervolume(P, self.ref), self.brute(P), places=10)
        self.assertGreater(hypervolume(np.vstack([P, [0.1, 0.9, 0.9]]), self.ref), hypervolume(P, self.ref))

    def test_normalize(self):
        P = np.array([[0, 10, 1], [2, 20, 1]])
        lo, hi = minmax_bounds(P)
        N = normalize(P, lo, hi)
        self.assertEqual(N.tolist(), [[0, 0, 0], [1, 1, 0]])        # span nol -> 0, bukan pembagian nol


class TestStats(unittest.TestCase):
    def test_mcnemar_matches_scipy(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            a, b = rng.random(60) < 0.6, rng.random(60) < 0.5
            r = mcnemar_exact(a, b)
            n = r["b"] + r["c"]
            want = 1.0 if n == 0 else sps.binomtest(min(r["b"], r["c"]), n, 0.5).pvalue
            self.assertAlmostEqual(r["p"], want, places=10)
        self.assertEqual(mcnemar_exact([1, 0], [1, 0])["p"], 1.0)
        self.assertAlmostEqual(mcnemar_exact([1] * 10 + [0] * 2, [0] * 10 + [1] * 2)["p"], 0.03857421875, places=10)

    def test_bootstrap_ci_covers_and_is_seeded(self):
        v = np.random.default_rng(1).normal(5, 1, 300)
        m, lo, hi = bootstrap_ci(v, B=2000, seed=4)
        self.assertLess(lo, m); self.assertLess(m, hi)
        self.assertAlmostEqual(hi - lo, 2 * 1.96 / np.sqrt(300), delta=0.08)
        self.assertEqual(bootstrap_ci(v, B=2000, seed=4), (m, lo, hi))
        self.assertGreater(bootstrap_ci(v, "p95", B=500)[0], bootstrap_ci(v, "median", B=500)[0])

    def test_paired_diff_exact_zero_variance_when_constant_shift(self):
        a = np.random.default_rng(2).random(100); b = a - 0.25
        r = paired_bootstrap_diff(a, b, B=500)
        self.assertAlmostEqual(r["diff"], 0.25); self.assertAlmostEqual(r["lo"], 0.25); self.assertAlmostEqual(r["hi"], 0.25)
        with self.assertRaises(ValueError): paired_bootstrap_diff([1, 2], [1])

    def test_compare_policies_guards(self):
        A = [out(f"q{i}", float(i % 2), 0.01, 1.0) for i in range(30)]
        B = [out(f"q{i}", float(i % 3 == 0), 0.02, 2.0) for i in range(30)]
        r = compare_policies(A, B, "mmlu", B=500)
        self.assertAlmostEqual(r["cost"]["diff"], -0.01); self.assertEqual(r["n"], 30)
        self.assertEqual(r["mcnemar_correct"]["b"] + r["mcnemar_correct"]["c"] > 0, True)
        with self.assertRaises(ValueError):
            compare_policies(A, [out(f"q{i}", 1.0, 0, 1, kind="measured_e2e") for i in range(30)], "mmlu")
        with self.assertRaises(ValueError):
            compare_policies(A, B[:-1], "mmlu")

    def test_summarize(self):
        o = [out("a", 1.0, 0.01, 1.0), out("b", 0.0, 0.03, 3.0, ans=None), out("c", 1.0, 0.02, 2.0), out("d", 1.0, 0.04, 10.0)]
        s = summarize(o, "mmlu")
        self.assertAlmostEqual(s["error"], 0.25); self.assertAlmostEqual(s["cost_mean"], 0.025)
        self.assertAlmostEqual(s["latency_median"], 2.5); self.assertAlmostEqual(s["latency_p95"], np.percentile([1, 3, 2, 10], 95))
        self.assertAlmostEqual(s["none_rate"], 0.25)
        h = summarize([out("a", 0.0, 0.01, 1.0, f1=0.5)], "hotpotqa")
        self.assertAlmostEqual(h["error"], 0.5); self.assertAlmostEqual(h["f1"], 0.5)
        with self.assertRaises(ValueError): summarize([out("a", 1, 0, 1), out("b", 1, 0, 1, kind="measured_e2e")], "mmlu")
        with self.assertRaises(ValueError): summarize([], "mmlu")


if __name__ == "__main__":
    unittest.main()
