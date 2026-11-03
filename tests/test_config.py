import copy, os, unittest
from adaptive_routing.config import Config, ConfigError
from adaptive_routing.llm.pricing import PriceBook
import yaml

ROOT = os.path.dirname(os.path.dirname(__file__))
PB = PriceBook.from_file(os.path.join(ROOT, "configs", "pricing.yaml"))


def load():
    return Config.from_file(os.path.join(ROOT, "configs", "default.yaml"))


def mutate(path, value):
    c = load(); cur = c.raw
    keys = path.split(".")
    for k in keys[:-1]:
        cur = cur[int(k)] if isinstance(cur, list) else cur[k]
    last = keys[-1]
    if isinstance(cur, list): cur[int(last)] = value
    else: cur[last] = value
    return c


class TestConfig(unittest.TestCase):
    def test_default_valid(self):
        load().validate(PB)

    def test_hash_stable_and_sensitive(self):
        a, b = load(), load()
        self.assertEqual(a.hash(), b.hash())
        self.assertNotEqual(a.hash(), mutate("experiment.seed", 1).hash())

    def test_d1_rejects_non_together_slm(self):
        with self.assertRaises(ConfigError) as cm:
            mutate("backends.slm.provider", "openai").validate(PB)
        self.assertIn("D1", str(cm.exception))

    def test_d2_guards(self):
        for path, val, tag in [("protocol.direct.cache_enabled", True, "cache"),
                               ("protocol.replay.splits", ["dev", "test"], "replay"),
                               ("protocol.trace.collect_splits", ["dev", "test"], "trace"),
                               ("protocol.replay.latency_kind", "measured_e2e", "composed"),
                               ("protocol.selection.split", "test", "dev")]:
            with self.assertRaises(ConfigError, msg=path) as cm:
                mutate(path, val).validate(PB)
            self.assertIn(tag, str(cm.exception))

    def test_d3_split_rules(self):
        with self.assertRaises(ConfigError):
            mutate("datasets.mmlu.dev_split", "test").validate(PB)
        with self.assertRaises(ConfigError):
            mutate("datasets.sizes.pilot", 500).validate(PB)

    def test_d4_entropy_levels_checked(self):
        with self.assertRaises(ConfigError) as cm:
            mutate("entropy.levels", {3: [0.0, 0.5, 1.0986]}).validate(PB)
        self.assertIn("D4", str(cm.exception))
        with self.assertRaises(ConfigError):
            mutate("entropy.log_base", "2").validate(PB)

    def test_n5_requires_five_agents(self):
        with self.assertRaises(ConfigError):
            mutate("policy_space.N", [3, 5]).validate(PB)

    def test_unknown_pricing_key(self):
        with self.assertRaises(ConfigError):
            mutate("backends.strong.pricing_key", "openai/xxx").validate(PB)

    def test_live_run_blocked_until_probe_and_pricing(self):
        c = mutate("experiment.run_mode", "trace_dev")
        env = {"TOGETHER_API_KEY": "x", "OPENAI_API_KEY": "y"}
        with self.assertRaises(ConfigError) as cm:
            c.validate_live(PB, env)
        msg = str(cm.exception)
        self.assertIn("probe_confirmed", msg)
        self.assertIn("tarif belum terverifikasi", msg)

    def test_live_ok_when_everything_confirmed(self):
        c = mutate("experiment.run_mode", "trace_dev")
        for m in c.all_backends():
            m["probe_confirmed"] = True
        with self.assertRaises(ConfigError) as cm:      # style masih 'auto' ditolak
            c.validate_live(PB, {"TOGETHER_API_KEY": "x", "OPENAI_API_KEY": "y"})
        self.assertIn("style", str(cm.exception))
        c.raw["backends"]["slm"]["logprobs"]["style"] = "together_int"
        with open(os.path.join(ROOT, "configs", "pricing.yaml"), encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        import datetime as dt
        for v in raw["models"].values():
            v["verified_on"] = str(dt.date.today())
        env = {"TOGETHER_API_KEY": "x", "OPENAI_API_KEY": "y"}
        c.validate_live(PriceBook(raw), env)

    def test_missing_api_key_reported(self):
        c = mutate("experiment.run_mode", "trace_dev")
        with self.assertRaises(ConfigError) as cm:
            c.validate_live(PB, {})
        self.assertIn("TOGETHER_API_KEY", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
