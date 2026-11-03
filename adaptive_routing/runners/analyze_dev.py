"""Analisis dev end-to-end dari trace lengkap: replay semua kebijakan, ringkasan, frontier 3 objektif, hypervolume,
seleksi per skenario, pembekuan selected_policies.json, plot, dan proyeksi biaya. Tidak menyentuh data test."""
from __future__ import annotations
import hashlib, json, os
from typing import Any, Dict, List, Sequence

import numpy as np

from ..baselines.registry import build_specs
from ..config import Config
from ..metrics.budget import direct_test_estimate, paid_cost_per_query, project, reasoning_share
from ..metrics.dev_stats import rank_nodes_by_dev_accuracy
from ..metrics.hypervolume import hypervolume, minmax_bounds, normalize
from ..metrics.plots import plot_frontier
from ..metrics.qa import primary_score
from ..metrics.selection import (frontier, freeze_selection, make_rows, objective_matrix, select_scenarios, references)
from ..metrics.stats import bootstrap_ci
from ..routing.thresholds import tau1_candidates
from ..tracing.schema import append_jsonl
from ..tracing.store import read_trace, validate_trace
from .replay import replay_dev


def _enc(x: float):
    return "inf" if x == float("inf") else "-inf" if x == float("-inf") else x


def analyze_dev(trace_dirs: Sequence[str], cfg: Config, out_dir: str, make_plots: bool = True) -> Dict[str, Any]:
    cfg.validate()
    raw = cfg.raw
    queries, calls, mans = [], [], []
    for d in trace_dirs:
        man, qs, cs = read_trace(d)
        validate_trace(man, qs, cs, "dev")
        mans.append(man); queries += qs; calls += cs
    if len({m["config_hash"] for m in mans}) != 1:
        raise ValueError("trace berasal dari konfigurasi berbeda (config_hash tidak sama)")
    trace_hash = mans[0]["config_hash"]
    ps, pr = raw["policy_space"], raw["protocol"]
    overhead = pr["replay"]["coordination_overhead_s"]
    B, level, seed = pr["stats"]["bootstrap_resamples"], pr["stats"]["ci_level"], pr["stats"]["bootstrap_seed"]
    margin = pr["hypervolume"]["reference_point_margin"]
    os.makedirs(out_dir, exist_ok=True)
    report: Dict[str, Any] = {"datasets": {}}
    payload: Dict[str, Any] = {
        "purpose": "kebijakan terpilih pada DEV; dibekukan sebelum eksekusi langsung pada test",
        "config_hash": cfg.hash(), "trace_config_hash": trace_hash, "config_unchanged_since_trace": cfg.hash() == trace_hash,
        "trace_run_ids": [m["run_id"] for m in mans],
        "dev_query_ids_sha256": hashlib.sha256("\n".join(sorted(q.query_id for q in queries)).encode()).hexdigest(),
        "scenarios_declared": pr["selection"]["scenarios"], "latency_kind_for_selection": "composed_estimated",
        "per_dataset": {}}
    for ds in sorted({q.dataset for q in queries}):
        qs = [q for q in queries if q.dataset == ds]
        ids = {q.query_id for q in qs}
        cs = [c for c in calls if c.query_id in ids]
        ps_true = [c.p_true for c in cs if c.stage == "s1_verify" and c.p_true is not None]
        tau1 = tau1_candidates(ps_true, ps["tau1"]["quantile_levels"], ps["tau1"]["include_sentinels"])
        specs = build_specs(raw, tau1)
        ties = {(S, N): rank_nodes_by_dev_accuracy(qs, cs, S, N, raw) for S in ps["S"] for N in ps["N"]}
        outs = replay_dev(qs, cs, raw, specs, ties, overhead)
        rows = make_rows(specs, outs, ds)
        front = frontier(rows)
        fam = {k: [r for r in rows if r["kind"] == k] for k in sorted({r["kind"] for r in rows})}
        M = np.array(objective_matrix(rows))
        lo, hi = minmax_bounds(M)
        ref = [margin] * 3
        hv = {"normalization": "dev_union_minmax", "reference_point": ref, "lo": lo.tolist(), "hi": hi.tolist(),
              "all": hypervolume(normalize(M[[rows.index(r) for r in front]], lo, hi), ref), "by_family": {}}
        for k, rs in fam.items():
            Mk = np.array(objective_matrix(rs))
            hv["by_family"][k] = hypervolume(normalize(Mk, lo, hi), ref)       # HV dari himpunan keluarga (dominasi diurus HV)
        sel = select_scenarios(rows, raw)
        refs = references(rows, raw)
        sel_json: Dict[str, Any] = {}
        for sc, byk in sel.items():
            sel_json[sc] = {}
            for k, r in byk.items():
                if r is None:
                    sel_json[sc][k] = None
                    continue
                errs = [1.0 - primary_score(ds, o.em or 0.0, o.f1) for o in outs[r["policy_id"]]]
                m, l, h = bootstrap_ci(errs, "mean", B, level, seed)
                sel_json[sc][k] = {"policy_id": r["policy_id"], "spec": r["spec"],
                                   "dev": {"cost": r["cost"], "latency": r["latency"], "error": r["error"],
                                           "error_ci": [l, h]}}
        payload["per_dataset"][ds] = {
            "n_dev": len(qs), "tau1_candidates": [_enc(t) for t in tau1],
            "tie_orders": {f"{S}|{N}": v for (S, N), v in ties.items()},
            "references": {k: {"policy_id": v["policy_id"], "cost": v["cost"], "latency": v["latency"], "error": v["error"]}
                           for k, v in refs.items()},
            "selected": sel_json, "infeasible": [[sc, k] for sc, byk in sel.items() for k, r in byk.items() if r is None]}
        d_out = os.path.join(out_dir, "dev_replay", ds)
        os.makedirs(d_out, exist_ok=True)
        op = os.path.join(d_out, "outcomes.jsonl")
        open(op, "w").close()
        for pid, o in outs.items():
            append_jsonl(op, o)
        with open(os.path.join(d_out, "summary.json"), "w", encoding="utf-8") as f:
            json.dump({"latency_kind": "composed_estimated", "rows": rows}, f, indent=1, ensure_ascii=False, default=str)
        with open(os.path.join(d_out, "frontier.json"), "w", encoding="utf-8") as f:
            json.dump({"objectives": ["cost_mean", "latency_mean(composed)", "error"], "policies": [r["policy_id"] for r in front]},
                      f, indent=1)
        with open(os.path.join(d_out, "hypervolume.json"), "w", encoding="utf-8") as f:
            json.dump(hv, f, indent=1)
        if make_plots:
            chosen = {f"{sc}/{k}": v["policy_id"] for sc, byk in sel_json.items() for k, v in byk.items() if v}
            plot_frontier(rows, {r["policy_id"] for r in front}, os.path.join(d_out, "frontier.png"), f"{ds}: frontier dev", chosen)
        # proyeksi biaya (ekstrapolasi)
        paid = list(paid_cost_per_query(cs).values())
        sizes = raw["datasets"]["sizes"]
        sel_rows = [r for byk in sel.values() for r in byk.values() if r]
        uniq = {r["policy_id"]: r for r in sel_rows}.values()
        report["datasets"][ds] = {
            "n_dev": len(qs), "n_policies": len(specs), "frontier_size": len(front), "hypervolume": hv,
            "selected": {sc: {k: (v["policy_id"] if v else None) for k, v in byk.items()} for sc, byk in sel_json.items()},
            "budget_trace_dev_extrapolated": project(paid, sizes["dev"]),
            "budget_direct_test_extrapolated": direct_test_estimate(list(uniq), sizes["test"],
                                                                    pr["direct"]["latency_repeats"]),
            "reasoning_share_by_model": reasoning_share(cs)}
    digest = freeze_selection(os.path.join(out_dir, "selected_policies.json"), payload)
    report["selected_policies_sha256"] = digest
    report["config_unchanged_since_trace"] = payload["config_unchanged_since_trace"]
    with open(os.path.join(out_dir, "dev_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, ensure_ascii=False, default=str)
    return report
