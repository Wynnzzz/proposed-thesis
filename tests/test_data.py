import os, tempfile, unittest
from collections import Counter
from adaptive_routing.data.io import read_queries, write_queries
from adaptive_routing.data.prepare import (DataSchemaError, build_hotpot, build_mmlu, convert_hotpot_row,
                                           convert_mmlu_row, make_manifest)
from adaptive_routing.data.render import render_hotpot, render_mmlu
from adaptive_routing.data.sampling import allocate_proportional, disjoint_split, stratified_sample

SIZES = {"pilot": 20, "dev": 100, "test": 300}


def mmlu_rows(n_per_subject, tag):
    rows = []
    for s, n in n_per_subject.items():
        for i in range(n):
            rows.append({"question": f"{tag} {s} q{i}?", "subject": s, "choices": ["a", "b", "c", "d"], "answer": i % 4})
    return rows


def hotpot_rows(n):
    return [{"id": f"id{i:05d}", "question": f"q{i}", "answer": "yes" if i % 3 == 0 else f"ans{i}", "type": "bridge",
             "level": "hard", "supporting_facts": {"title": [], "sent_id": []},
             "context": {"title": [f"T{j}" for j in range(10)], "sentences": [[f"s{j}a.", f"s{j}b."] for j in range(10)]}}
            for i in range(n)]


class TestSampling(unittest.TestCase):
    def test_allocation_sums_and_proportional(self):
        c = {"a": 100, "b": 50, "c": 5, "d": 1}
        a = allocate_proportional(c, 40)
        self.assertEqual(sum(a.values()), 40)
        self.assertGreaterEqual(a["a"], a["b"]); self.assertLessEqual(a["d"], 1)
        self.assertTrue(all(a[k] <= c[k] for k in c))
        self.assertEqual(a, allocate_proportional(c, 40))

    def test_allocation_small_strata_can_be_zero_and_total_exact(self):
        c = {f"big{i}": 200 for i in range(10)} | {f"small{i}": 5 for i in range(47)}   # mirip MMLU: strata sangat timpang
        a = allocate_proportional(c, 100)
        self.assertEqual(sum(a.values()), 100)
        self.assertGreater(sum(1 for v in a.values() if v == 0), 0)             # strata kecil dapat 0
        tot = sum(c.values())
        self.assertTrue(all(abs(a[k] - 100 * c[k] / tot) < 1 for k in c))        # selisih dari jatah proporsional < 1
        with self.assertRaises(ValueError):
            allocate_proportional({"a": 3}, 4)

    def test_stratified_sample_deterministic_and_seed_sensitive(self):
        strata = {"a": list(range(0, 50)), "b": list(range(50, 80)), "c": list(range(80, 90))}
        x, _ = stratified_sample(strata, 20, 1, "t"); y, _ = stratified_sample(strata, 20, 1, "t")
        z, _ = stratified_sample(strata, 20, 2, "t")
        self.assertEqual(x, y); self.assertNotEqual(x, z)
        self.assertEqual(len(set(x)), 20)

    def test_disjoint_split_and_order_independent(self):
        ids = [f"i{k}" for k in range(1000)]
        d1, t1 = disjoint_split(ids, 100, 300, 7, "h")
        d2, t2 = disjoint_split(list(reversed(ids)), 100, 300, 7, "h")
        self.assertEqual((d1, t1), (d2, t2))
        self.assertEqual((len(d1), len(t1), len(set(d1) & set(t1))), (100, 300, 0))
        with self.assertRaises(ValueError):
            disjoint_split(ids[:10], 6, 6, 1, "h")


class TestBuild(unittest.TestCase):
    def test_mmlu_build(self):
        val = mmlu_rows({f"sub{i}": 20 + i for i in range(57)}, "v")
        tst = mmlu_rows({f"sub{i}": 200 + 5 * i for i in range(57)}, "t")
        dev, test, info = build_mmlu(val, tst, SIZES, 42)
        self.assertEqual((len(dev), len(test)), (100, 300))
        self.assertTrue(all(q.split == "dev" and q.meta["source_split"] == "validation" for q in dev))
        self.assertTrue(all(q.split == "test" and q.meta["source_split"] == "test" for q in test))
        self.assertEqual(len({q.query_id for q in dev + test}), 400)
        self.assertEqual(info["validation_test_question_overlap"], 0)
        self.assertEqual(sum(info["alloc_dev"].values()), 100)
        self.assertTrue(all(q.gold in "ABCD" and "A. a" in q.body for q in dev))
        cnt = Counter(q.meta["subject"] for q in dev)
        self.assertEqual(dict(cnt), {k: v for k, v in info["alloc_dev"].items() if v})            # sampel sesuai alokasi
        self.assertGreaterEqual(info["alloc_dev"]["sub56"], info["alloc_dev"]["sub0"])
        dev2, _, _ = build_mmlu(val, tst, SIZES, 42)
        self.assertEqual([q.query_id for q in dev], [q.query_id for q in dev2])

    def test_mmlu_overlap_detected(self):
        val = mmlu_rows({"a": 150}, "same"); tst = mmlu_rows({"a": 400}, "same")
        _, _, info = build_mmlu(val, tst, SIZES, 1)
        self.assertEqual(info["validation_test_question_overlap"], 150)

    def test_hotpot_build_disjoint(self):
        dev, test, info = build_hotpot(hotpot_rows(2000), SIZES, 42)
        self.assertEqual((len(dev), len(test)), (100, 300))
        self.assertFalse({q.query_id for q in dev} & {q.query_id for q in test})
        self.assertTrue(dev[0].body.startswith("Context:\n[T0] s0a. s0b.") and "Question: q" in dev[0].body)
        self.assertNotIn("supporting", dev[0].body.lower())

    def test_schema_errors_are_loud(self):
        with self.assertRaises(DataSchemaError) as cm:
            convert_mmlu_row({"question": "q", "subject": "s", "choices": list("abcd")}, "test", 0, "test")
        self.assertIn("answer", str(cm.exception))
        with self.assertRaises(DataSchemaError):
            convert_mmlu_row({"question": "q", "subject": "s", "choices": list("abcd"), "answer": "B"}, "test", 0, "test")
        with self.assertRaises(DataSchemaError):
            convert_hotpot_row({k: 1 for k in ("id", "question", "answer", "type", "level")} | {"context": {"title": []}}, 0, "dev")
        rows = hotpot_rows(500); rows[3]["id"] = rows[2]["id"]
        with self.assertRaises(DataSchemaError):
            build_hotpot(rows, SIZES, 1)
        with self.assertRaises(ValueError):
            render_mmlu("s", "q", ["a", "b"])

    def test_manifest_and_io_roundtrip(self):
        dev, test, info = build_hotpot(hotpot_rows(1000), SIZES, 3)
        man = make_manifest("hotpotqa", {"hf_id": "x", "hf_config": "distractor"}, SIZES, 3, dev, test,
                            {"validation": 1000}, {"validation": {}}, info, {"validation": 7405})
        self.assertEqual(man["splits"]["pilot"], man["splits"]["dev"][:20])
        self.assertEqual(man["count_warnings"]["validation"], {"expected_unverified": 7405, "found": 1000})
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "dev.jsonl"); write_queries(p, dev)
            self.assertEqual(read_queries(p), dev)

    def test_render(self):
        self.assertIn("about high school physics", render_mmlu("high_school_physics", "Q?", list("wxyz")))
        self.assertEqual(render_hotpot("Q?", ["A"], [["x.", "y."]]), "Context:\n[A] x. y.\n\nQuestion: Q?")


if __name__ == "__main__":
    unittest.main()
