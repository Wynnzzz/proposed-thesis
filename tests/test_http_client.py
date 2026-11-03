import json, threading, unittest, http.server
from adaptive_routing.llm.base import CallContext, LLMError
from adaptive_routing.llm.http_client import HTTPClient
from adaptive_routing.llm.recording import RecordingLLM, UsageMissingError
from tests.helpers import CFG, PB

CTX = CallContext("q1", "mmlu", "dev", "s1_verify", "s1v", "slm", "slm_llama32_3b")
MSG = [{"role": "user", "content": "x"}]


class Server:
    def __init__(self, script):
        self.script, self.bodies = list(script), []
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def do_POST(self):
                outer.bodies.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                code, payload = outer.script.pop(0) if outer.script else outer.last
                outer.last = (code, payload)
                d = json.dumps(payload).encode()
                self.send_response(code); self.send_header("Content-Length", str(len(d))); self.end_headers(); self.wfile.write(d)

        self.srv = http.server.HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}/v1"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self): self.srv.shutdown(); self.srv.server_close()


GOOD = {"choices": [{"message": {"content": "Yes"}, "finish_reason": "stop",
                     "logprobs": {"tokens": ["Yes"], "token_logprobs": [-0.2], "top_logprobs": [{"Yes": -0.2, "No": -1.9}]}}],
        "usage": {"prompt_tokens": 30, "completion_tokens": 1,
                  "completion_tokens_details": {"reasoning_tokens": 0}, "prompt_tokens_details": {"cached_tokens": 4}}}


class TestHTTP(unittest.TestCase):
    def cfg(self, url, provider="together"):
        return {**CFG.raw["backends"]["slm"], "base_url": url, "provider": provider}

    def test_parse_usage_and_logprobs_style_together(self):
        s = Server([(200, GOOD)])
        try:
            c = HTTPClient(env={"TOGETHER_API_KEY": "k"}, sleep=lambda x: None)
            r = c.chat(self.cfg(s.url), MSG, {"temperature": 0, "max_tokens": 2,
                       "logprobs": {"style": "together_int", "top_k": 5}}, CTX)
        finally: s.close()
        self.assertEqual((r.input_tokens, r.output_tokens, r.reasoning_tokens, r.cached_input_tokens), (30, 1, 0, 4))
        self.assertEqual(s.bodies[0]["logprobs"], 5); self.assertNotIn("top_logprobs", s.bodies[0])
        self.assertEqual(s.bodies[0]["max_tokens"], 2)

    def test_openai_style_and_param_names(self):
        s = Server([(200, GOOD)])
        try:
            HTTPClient(sleep=lambda x: None).chat(self.cfg(s.url, "openai"), MSG, {"max_tokens": 9, "reasoning_effort": "low",
                   "logprobs": {"style": "openai_bool", "top_k": 3}}, CTX)
        finally: s.close()
        b = s.bodies[0]
        self.assertEqual((b["logprobs"], b["top_logprobs"], b["max_completion_tokens"], b["reasoning_effort"]), (True, 3, 9, "low"))
        self.assertNotIn("max_tokens", b)

    def test_auto_style_rejected(self):
        with self.assertRaises(LLMError):
            HTTPClient.build_body(self.cfg("http://x"), MSG, {"max_tokens": 2, "logprobs": {"style": "auto", "top_k": 5}})

    def test_retry_on_5xx_then_success_and_no_retry_on_400(self):
        s = Server([(500, {"error": "boom"}), (429, {"error": "slow"}), (200, GOOD)])
        try:
            r = HTTPClient(max_retries=3, sleep=lambda x: None).chat(self.cfg(s.url), MSG, {"max_tokens": 2}, CTX)
        finally: s.close()
        self.assertEqual((r.attempts, len(s.bodies)), (3, 3))
        s = Server([(400, {"error": "bad"})])
        try:
            with self.assertRaises(LLMError):
                HTTPClient(max_retries=3, sleep=lambda x: None).chat(self.cfg(s.url), MSG, {"max_tokens": 2}, CTX)
        finally: s.close()
        self.assertEqual(len(s.bodies), 1)

    def test_retries_exhausted_becomes_error_record(self):
        s = Server([(500, {"error": "x"})] * 10)
        try:
            rec = RecordingLLM(HTTPClient(max_retries=1, sleep=lambda x: None), PB, "r")
            r = rec.call(CTX, self.cfg(s.url), MSG, {"max_tokens": 2})
        finally: s.close()
        self.assertIn("HTTP 500", r.error); self.assertEqual(r.cost_usd, 0.0)

    def test_server_without_usage_stops_run(self):
        bad = {"choices": [{"message": {"content": "Yes"}, "finish_reason": "stop"}]}
        s = Server([(200, bad)])
        try:
            with self.assertRaises(UsageMissingError):
                RecordingLLM(HTTPClient(sleep=lambda x: None), PB, "r").call(CTX, self.cfg(s.url), MSG, {"max_tokens": 2})
        finally: s.close()


if __name__ == "__main__":
    unittest.main()
