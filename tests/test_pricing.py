import datetime as dt, unittest
from adaptive_routing.llm.pricing import PriceBook, PricingError

RAW = {"max_age_days": 30, "models": {
    "a": {"input_per_mtok": 0.06, "output_per_mtok": 0.06, "verified_on": "2026-10-20"},
    "b": {"input_per_mtok": 1.0, "output_per_mtok": 5.0, "verified_on": None},
    "c": {"input_per_mtok": 1.0, "output_per_mtok": 5.0, "verified_on": "2026-01-01"}}}


class TestPricing(unittest.TestCase):
    def setUp(self):
        self.pb = PriceBook(RAW)

    def test_cost_arithmetic(self):
        self.assertAlmostEqual(self.pb.cost_usd("b", 2000, 500), (2000 * 1 + 500 * 5) / 1e6)
        self.assertAlmostEqual(self.pb.cost_usd("a", 1_000_000, 1_000_000), 0.12)

    def test_unknown_model_is_error_not_zero(self):
        with self.assertRaises(PricingError):
            self.pb.cost_usd("zzz", 1, 1)

    def test_verification_rules(self):
        today = dt.date(2026, 11, 1)
        self.assertEqual(self.pb.unverified(["a", "b", "c"], today), ["b", "c"])
        with self.assertRaises(PricingError):
            self.pb.require_verified(["a", "b"], today)
        self.pb.require_verified(["a"], today)


if __name__ == "__main__":
    unittest.main()
