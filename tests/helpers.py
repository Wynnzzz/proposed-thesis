import os
from adaptive_routing.config import Config
from adaptive_routing.data.types import Query
from adaptive_routing.llm.mock import MockLLM, SimClock
from adaptive_routing.llm.pricing import PriceBook

ROOT = os.path.dirname(os.path.dirname(__file__))
CFG = Config.from_file(os.path.join(ROOT, "configs", "default.yaml"))
PB = PriceBook.from_file(os.path.join(ROOT, "configs", "pricing.yaml"))


def make_queries(n_mmlu=24, n_hotpot=12):
    qs, gold = [], {}
    for i in range(n_mmlu):
        g = "ABCD"[i % 4]
        qs.append(Query(f"mmlu-{i}", "mmlu", "dev", f"Question {i}\nA. a\nB. b\nC. c\nD. d", g)); gold[f"mmlu-{i}"] = g
    for i in range(n_hotpot):
        g = f"answer{i}"
        qs.append(Query(f"hp-{i}", "hotpotqa", "dev", f"Context ... Question {i}", g)); gold[f"hp-{i}"] = g
    return qs, gold


def make_mock(gold, clock=None, fmt="openai_content", seed=7, invalid_rate=0.04):
    clock = clock or SimClock()
    skill = {"slm_llama32_3b": 0.5, "agent_qwen": 0.6, "agent_llama3_8b_lite": 0.5, "agent_glm": 0.7, "strong_gpt": 0.9}
    lat = {"slm_llama32_3b": 0.3, "agent_qwen": 1.0, "agent_llama3_8b_lite": 0.5, "agent_glm": 3.0, "strong_gpt": 2.0}
    return MockLLM(gold, skill, lat, clock=clock, seed=seed, logprob_format=fmt, invalid_rate=invalid_rate), clock
