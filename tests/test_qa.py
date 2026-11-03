import unittest
from adaptive_routing.metrics.qa import f1_tokens, primary_score, score


class TestQA(unittest.TestCase):
    def test_mmlu(self):
        self.assertEqual(score("mmlu", "B", "B"), (1.0, None))
        self.assertEqual(score("mmlu", "B", "C"), (0.0, None))
        self.assertEqual(score("mmlu", None, "C"), (0.0, None))

    def test_hotpot_em_f1(self):
        self.assertEqual(score("hotpotqa", "eiffel tower", "The Eiffel Tower"), (1.0, 1.0))
        em, f1 = score("hotpotqa", "eiffel tower paris", "The Eiffel Tower")
        self.assertEqual(em, 0.0); self.assertAlmostEqual(f1, 2 * (2 / 3) * 1 / (2 / 3 + 1))
        self.assertEqual(score("hotpotqa", None, "x"), (0.0, 0.0))

    def test_yes_no_has_no_partial_credit(self):
        self.assertEqual(f1_tokens("yes", "no"), 0.0)
        self.assertEqual(f1_tokens("yes sir", "yes"), 0.0)     # gold special, pred tidak identik: tanpa kredit parsial
        self.assertEqual(f1_tokens("no", "no way"), 0.0)       # pred special, gold tidak identik
        self.assertEqual(f1_tokens("yes", "yes"), 1.0)
        self.assertEqual(f1_tokens("maybe", "yes"), 0.0)

    def test_token_multiset_overlap(self):
        self.assertAlmostEqual(f1_tokens("a a b", "a b b"), 2 * (2 / 3) * (2 / 3) / (4 / 3))

    def test_primary_score(self):
        self.assertEqual(primary_score("mmlu", 1.0, None), 1.0)
        self.assertEqual(primary_score("hotpotqa", 0.0, 0.4), 0.4)


if __name__ == "__main__":
    unittest.main()
