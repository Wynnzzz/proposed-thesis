import unittest
from adaptive_routing.routing.entropy import (achievable_entropies, effective_thresholds, entropy_from_counts,
                                              accepted_patterns, unique_majority)


class TestEntropy(unittest.TestCase):
    def test_n3_values_match_decision_d4(self):
        a = achievable_entropies(3)
        self.assertEqual(len(a), 3)
        for got, want in zip(a, [0.0, 0.6365, 1.0986]):
            self.assertAlmostEqual(got, want, places=4)

    def test_counts(self):
        self.assertAlmostEqual(entropy_from_counts((3,)), 0.0)
        self.assertAlmostEqual(entropy_from_counts((2, 1)), 0.6365, places=4)
        self.assertAlmostEqual(entropy_from_counts((1, 1, 1)), 1.0986, places=4)

    def test_n3_unique_majority_equals_low_entropy(self):
        self.assertTrue(unique_majority((2, 1)))
        self.assertFalse(unique_majority((1, 1, 1)))

    def test_n3_effective_thresholds_collapse(self):
        # tau=1.0986 menerima pola yang sama dengan tau=0.6365 karena syarat mayoritas unik
        eff = effective_thresholds(3)
        self.assertEqual(len(eff), 2)
        self.assertAlmostEqual(eff[0], 0.0)
        self.assertAlmostEqual(eff[1], 0.6365, places=4)
        self.assertEqual(accepted_patterns(3, 1.0986), accepted_patterns(3, 0.6365))

    def test_rounded_config_threshold_still_accepts_exact_level(self):
        # regresi: 0.6365 (dibulatkan, dari D4) < eksak 0.636514; tanpa toleransi pola 2-1-0 akan ditolak
        self.assertGreater(entropy_from_counts((2, 1)), 0.6365)
        self.assertIn((2, 1), accepted_patterns(3, 0.6365))
        self.assertNotIn((2, 1), accepted_patterns(3, 0.0))
        self.assertNotIn((1, 1, 1), accepted_patterns(3, 0.6365))

    def test_boundary_inclusive(self):
        # d <= tau: nilai tepat sama diterima
        self.assertIn((2, 1), accepted_patterns(3, entropy_from_counts((2, 1))))

    def test_n5_nonmonotone_majority(self):
        # (2,1,1,1) entropi lebih tinggi dari (2,2,1) tetapi punya mayoritas unik
        self.assertGreater(entropy_from_counts((2, 1, 1, 1)), entropy_from_counts((2, 2, 1)))
        self.assertTrue(unique_majority((2, 1, 1, 1)))
        self.assertFalse(unique_majority((2, 2, 1)))
        self.assertEqual(len(achievable_entropies(5)), 7)


if __name__ == "__main__":
    unittest.main()
