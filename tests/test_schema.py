import os, tempfile, unittest
from adaptive_routing.llm.pricing import PriceBook
from adaptive_routing.tracing.schema import (CallRecord, PolicyOutcome, request_key, append_jsonl, read_jsonl)

PB = PriceBook({"models": {"m": {"input_per_mtok": 0.1, "output_per_mtok": 0.2, "verified_on": "2026-10-30"}}})


def rec(**kw):
    base = dict(call_id="c1", run_id="r", query_id="q1", dataset="mmlu", split="dev", stage="s2_agent",
                group_id="s2", node_id="agent_qwen", model_key="agent_qwen", provider="together", model_id="X/Y",
                t_start_mono=1.0, t_end_mono=2.5, latency_s=1.5, input_tokens=1000, output_tokens=100)
    base.update(kw)
    return CallRecord(**base)


class TestSchema(unittest.TestCase):
    def test_cost_from_api_usage(self):
        r = rec(); r.finalize_cost(PB, "m"); r.validate()
        self.assertAlmostEqual(r.cost_usd, (1000 * 0.1 + 100 * 0.2) / 1e6)
        self.assertTrue(r.pricing_verified)

    def test_missing_usage_never_estimated(self):
        r = rec(input_tokens=None, output_tokens=None)
        with self.assertRaises(ValueError):
            r.finalize_cost(PB, "m")

    def test_estimated_tokens_rejected_outside_mock(self):
        r = rec(token_source="estimated"); r.finalize_cost(PB, "m")
        with self.assertRaises(ValueError):
            r.validate()
        r.validate(allow_estimated=True)

    def test_bad_stage_and_split(self):
        r = rec(stage="nope"); r.finalize_cost(PB, "m")
        with self.assertRaises(ValueError):
            r.validate()
        r = rec(split="val"); r.finalize_cost(PB, "m")
        with self.assertRaises(ValueError):
            r.validate()

    def test_request_key_sensitivity(self):
        m = [{"role": "user", "content": "hi"}]
        k = request_key("X", m, {"t": 0})
        self.assertEqual(k, request_key("X", m, {"t": 0}))
        self.assertNotEqual(k, request_key("X", m, {"t": 0.7}))
        self.assertNotEqual(k, request_key("X", m + [{"role": "assistant", "content": "a"}], {"t": 0}))
        self.assertNotEqual(k, request_key("Z", m, {"t": 0}))

    def test_jsonl_roundtrip(self):
        r = rec(logprobs_first_token={"Yes": -0.2, "No": -1.9}); r.finalize_cost(PB, "m")
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "calls.jsonl"); append_jsonl(p, [r])
            back = CallRecord.from_dict(read_jsonl(p)[0])
        self.assertEqual(back, r)

    def test_latency_kind_label_required(self):
        kw = dict(query_id="q", policy_id="A3", final_answer="B", em=None, f1=None, correct=True,
                  stages_run=["s1"], n_calls=2, cost_usd=0.0, latency_s=1.0)
        PolicyOutcome(latency_kind="composed_estimated", **kw)
        PolicyOutcome(latency_kind="measured_e2e", **kw)
        with self.assertRaises(ValueError):
            PolicyOutcome(latency_kind="estimated", **kw)


if __name__ == "__main__":
    unittest.main()
