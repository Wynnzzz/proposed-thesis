import unittest
from adaptive_routing.routing.answers import parse_mmlu, parse_hotpotqa, normalize_text, parse_answer


class TestAnswers(unittest.TestCase):
    def test_mmlu_valid_forms(self):
        for t, want in [("blah\nFinal answer: B", "B"), ("Final answer: (c)", "C"), ("Final Answer: **D**", "D"),
                        ("Final answer: A.", "A"), ("Final answer: B. Paris", "B"), ("x\nfinal answer : a", "A")]:
            self.assertEqual(parse_mmlu(t), want, t)

    def test_mmlu_invalid_forms(self):
        for t in ["", None, "The answer is B", "Final answer: E", "Final answer: A cat sat", "Final answer:"]:
            self.assertIsNone(parse_mmlu(t), t)

    def test_last_final_answer_wins(self):
        self.assertEqual(parse_mmlu("Final answer: A\nwait\nFinal answer: C"), "C")

    def test_hotpot_normalization(self):
        self.assertEqual(parse_hotpotqa("Final answer: The Eiffel Tower."), "eiffel tower")
        self.assertEqual(parse_hotpotqa('Final answer: "Yes"'), "yes")
        self.assertIsNone(parse_hotpotqa("Final answer: the"))
        self.assertIsNone(parse_hotpotqa("no marker"))
        self.assertEqual(normalize_text("An  Apple, a day!"), "apple day")

    def test_dispatch(self):
        self.assertEqual(parse_answer("mmlu", "Final answer: B"), "B")
        with self.assertRaises(ValueError):
            parse_answer("x", "Final answer: B")


if __name__ == "__main__":
    unittest.main()
