import unittest
from adaptive_routing.routing.disagreement import analyze, accepts, aggregate


class TestDisagreement(unittest.TestCase):
    def test_patterns_n3(self):
        d = analyze(["A", "A", "A"]); self.assertEqual(d.counts, (3,)); self.assertAlmostEqual(d.d, 0); self.assertEqual(d.majority, "A")
        d = analyze(["A", "A", "B"]); self.assertEqual(d.counts, (2, 1)); self.assertAlmostEqual(d.d, 0.6365, places=4)
        d = analyze(["A", "B", "C"]); self.assertEqual(d.counts, (1, 1, 1)); self.assertAlmostEqual(d.d, 1.0986, places=4)
        self.assertFalse(d.unique); self.assertIsNone(d.majority)

    def test_invalid_are_distinct_singletons(self):
        d = analyze(["A", None, None])
        self.assertEqual(d.counts, (1, 1, 1)); self.assertEqual(d.n_invalid, 2); self.assertFalse(d.unique)
        d = analyze([None, None, None]); self.assertIsNone(d.majority)
        d = analyze(["A", "A", None]); self.assertEqual(d.majority, "A"); self.assertTrue(d.unique)

    def test_accepts_rule(self):
        self.assertTrue(accepts(analyze(["A", "A", "A"]), 0.0))
        self.assertFalse(accepts(analyze(["A", "A", "B"]), 0.0))
        self.assertTrue(accepts(analyze(["A", "A", "B"]), 0.6365))     # batas inklusif + toleransi
        self.assertFalse(accepts(analyze(["A", "B", "C"]), 1.0986))    # tanpa mayoritas unik tidak pernah diterima

    def test_n5_unique_majority_with_high_entropy(self):
        d = analyze(["A", "A", "B", "C", "D"])    # 2-1-1-1: entropi tinggi, mayoritas unik
        self.assertTrue(d.unique); self.assertFalse(accepts(d, 1.0)); self.assertTrue(accepts(d, 1.4))
        self.assertFalse(analyze(["A", "A", "B", "B", "C"]).unique)

    def test_aggregate_tie_break(self):
        nodes = ["x", "y", "z"]
        self.assertEqual(aggregate(["A", "B", "C"], nodes, ["z", "y", "x"]), "C")
        self.assertEqual(aggregate(["A", "B", "C"], nodes, ["y", "x", "z"]), "B")
        self.assertEqual(aggregate(["A", "A", "B"], nodes, ["z", "y", "x"]), "A")      # mayoritas menang atas peringkat
        self.assertEqual(aggregate([None, "B", None], nodes, nodes), "B")
        self.assertIsNone(aggregate([None, None, None], nodes, nodes))
        self.assertEqual(aggregate(["A", "B", None], nodes, []), "A")                   # cadangan leksikografis


if __name__ == "__main__":
    unittest.main()
