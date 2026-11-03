import unittest
from adaptive_routing.llm.base import CallContext, LLMError, LLMResponse
from adaptive_routing.llm.recording import BudgetExceeded, BudgetGuard, RecordingLLM, UsageMissingError
from tests.helpers import CFG, PB

M = CFG.raw["backends"]["agents"][0]
CTX = CallContext("q1", "mmlu", "dev", "s2_agent", "s2", "agent_qwen", "agent_qwen")
MSG = [{"role": "user", "content": "hi"}]
P = {"temperature": 0.0, "max_tokens": 8}


class Client:
    def __init__(self, resp=None, exc=None): self.resp, self.exc, self.n = resp, exc, 0
    def chat(self, *a, **k):
        self.n += 1
        if self.exc: raise self.exc
        return self.resp


def ok(i=1000, o=100): return LLMResponse("Final answer: A", "stop", i, o, latency_s=1.0, t_start_mono=0, t_end_mono=1.0)


class TestRecording(unittest.TestCase):
    def test_cost_from_usage_and_sink(self):
        got = []
        rec = RecordingLLM(Client(ok()), PB, "r", sink=got.append)
        r = rec.call(CTX, M, MSG, P)
        self.assertAlmostEqual(r.cost_usd, (1000 * 0.10 + 100 * 0.15) / 1e6)
        self.assertEqual(got, [r]); self.assertFalse(r.pricing_verified)

    def test_missing_usage_stops_run(self):
        with self.assertRaises(UsageMissingError):
            RecordingLLM(Client(ok(None, None)), PB, "r").call(CTX, M, MSG, P)

    def test_api_error_recorded_not_billed(self):
        r = RecordingLLM(Client(exc=LLMError("HTTP 500")), PB, "r").call(CTX, M, MSG, P)
        self.assertEqual((r.error, r.cost_usd), ("HTTP 500", 0.0))
        r.validate()

    def test_budget_guard(self):
        b = BudgetGuard(cap_usd=0.0005, abort_fraction=0.9)
        rec = RecordingLLM(Client(ok(2000, 2000)), PB, "r", budget=b)   # biaya per panggilan 0.0005
        rec.call(CTX, M, MSG, P)
        with self.assertRaises(BudgetExceeded):
            rec.call(CallContext("q2", "mmlu", "dev", "s2_agent", "s2", "agent_qwen", "agent_qwen"), M, MSG, P)

    def test_cache_hits_only_for_same_logical_call(self):
        c = Client(ok())
        rec = RecordingLLM(c, PB, "r", cache_enabled=True, budget=BudgetGuard(1.0))
        a = rec.call(CTX, M, MSG, P)
        b = rec.call(CTX, M, MSG, P)
        self.assertEqual((c.n, a.cache_hit, b.cache_hit), (1, False, True))
        self.assertAlmostEqual(a.cost_usd, b.cost_usd)                # biaya tetap dicatat untuk akuntansi kebijakan
        self.assertAlmostEqual(rec.budget.spent, a.cost_usd)         # tetapi tidak dibayar dua kali
        other_round = CallContext("q1", "mmlu", "dev", "s3_revision", "s3_r1", "agent_qwen", "agent_qwen", 1)
        rec.call(other_round, M, MSG, P)                              # prompt sama, identitas logis beda: tidak boleh hit
        self.assertEqual(c.n, 2)
        other_sample = CallContext("q1", "mmlu", "dev", "s2_agent", "s2", "agent_qwen", "agent_qwen", None, 1)
        rec.call(other_sample, M, MSG, P)
        self.assertEqual(c.n, 3)

    def test_cache_disabled_by_default(self):
        c = Client(ok()); rec = RecordingLLM(c, PB, "r")
        rec.call(CTX, M, MSG, P); rec.call(CTX, M, MSG, P)
        self.assertEqual(c.n, 2)


if __name__ == "__main__":
    unittest.main()
