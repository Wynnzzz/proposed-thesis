import math, unittest
from adaptive_routing.verification.p_true import first_token_alternatives, p_true_from_alternatives


def oa(yes, no, tok="Yes"):
    alts = [{"token": "Yes", "logprob": math.log(yes)}, {"token": "No", "logprob": math.log(no)}]
    return {"content": [{"token": tok, "logprob": math.log(yes), "top_logprobs": alts}]}


class TestPTrue(unittest.TestCase):
    def test_openai_format(self):
        p, m = p_true_from_alternatives(first_token_alternatives(oa(0.6, 0.2)))
        self.assertAlmostEqual(p, 0.75); self.assertEqual(m["p_true_status"], "ok"); self.assertFalse(m["p_true_partial"])

    def test_together_format(self):
        raw = {"tokens": ["Yes"], "token_logprobs": [math.log(0.7)], "top_logprobs": [{"Yes": math.log(0.7), "No": math.log(0.2)}]}
        p, _ = p_true_from_alternatives(first_token_alternatives(raw))
        self.assertAlmostEqual(p, 0.7 / 0.9)

    def test_token_variants_summed(self):
        alts = {" Yes": math.log(0.3), "yes": math.log(0.2), "▁No": math.log(0.1), "NO": math.log(0.1)}
        p, m = p_true_from_alternatives(alts)
        self.assertAlmostEqual(p, 0.5 / 0.7); self.assertAlmostEqual(m["p_no_raw"], 0.2)

    def test_raw_yes_option(self):
        p, _ = p_true_from_alternatives({"Yes": math.log(0.6), "No": math.log(0.2)}, "raw_yes")
        self.assertAlmostEqual(p, 0.6)

    def test_partial_flag_when_one_side_missing(self):
        p, m = p_true_from_alternatives({"Yes": math.log(0.97), "The": math.log(0.01)})
        self.assertAlmostEqual(p, 1.0); self.assertTrue(m["p_true_partial"])

    def test_undefined_cases_are_none(self):
        self.assertIsNone(p_true_from_alternatives(None)[0])
        p, m = p_true_from_alternatives({"Maybe": -0.1, "Sure": -2.0})
        self.assertIsNone(p); self.assertEqual(m["p_true_status"], "yes_no_absent")
        self.assertIsNone(first_token_alternatives({"weird": 1}))
        self.assertIsNone(first_token_alternatives(None))


if __name__ == "__main__":
    unittest.main()
