#!/usr/bin/env python3
"""Validate complete full-vocabulary grids and save auditable paper summaries."""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t

from scripts.summarize_prospective_extensions import reference_inverse_size, LAWS

ROOT = Path("results/full_vocabulary_v1")
RUNS = ("grid", "nonoverlap_pairs", "unequal_sizes", "shuffle", "analytic")


def read_checked(run):
    directory = ROOT / run
    config = json.loads((directory / "config.json").read_text())
    metadata = json.loads((directory / "metadata.json").read_text())
    mapping = json.loads((directory / "class_map.json").read_text())
    if mapping["other_class"] is not None or mapping["token_ids_in_class_order"] != list(range(50257)):
        raise ValueError("not an exact, complete GPT-2 content vocabulary")
    if metadata["classes"] != 50257 or not metadata["full_vocabulary"]:
        raise ValueError("incorrect target metadata")
    with (directory / "pair_results.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    pairs = range(1, config["nonoverlap_pool_pairs"] + 1) if config.get("nonoverlap_pool_pairs") else config["experimental_replicates"]
    sizes = config.get("size_pairs") or [(n, n) for n in config["chunk_counts"]]
    expected = {(c, int(p), int(a), int(b), float(q))
                for c, p, (a, b), q in itertools.product(config["corpora"], pairs, sizes, config["mask_rates"])}
    keys = []
    for row in rows:
        row.update(n_a=int(row["n_a_chunks"]), n_b=int(row["n_b_chunks"]),
                   q=float(row["mask_rate"]), replicate=int(row["replicate"]),
                   observed=float(row["observed_disagreement"]))
        keys.append((row["corpus"], row["replicate"], row["n_a"], row["n_b"], row["q"]))
        assert int(row["classes"]) == 50257
        assert float(row["eval_top_token_coverage"]) == 1
        assert int(row["eval_chunks"]) == 128
        for law in config["laws"]:
            np.testing.assert_allclose(float(row[f"{law}_ratio"]),
                                       float(row[f"{law}_prediction"]) / row["observed"])
        if row["observed"] < 0 or not math.isfinite(float(row["skill"])):
            raise ValueError("invalid measured result")
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError(f"{run}: missing, duplicate, or extra conditions")
    common = ("seed", "corpora", "auxiliary_split", "reference_split", "experimental_replicates",
              "evaluation_split", "evaluation_chunks", "lag_radius", "mask_rates", "chunk_counts",
              "smoothing", "stacking_ridge", "bootstrap_draws", "laws")
    old_name = "analytic_pilot" if run == "analytic" else run
    old_config_path = Path(f"configs/prospective_lag_v1_{old_name}.json")
    old_config = json.loads(old_config_path.read_text())
    for field in common:
        if config[field] != old_config[field]:
            raise ValueError(f"{run}: not protocol-matched on {field}")
    for field in ("size_pairs", "nonoverlap_pool_pairs", "shuffle_within_chunk"):
        if config.get(field) != old_config.get(field):
            raise ValueError(f"{run}: not protocol-matched on {field}")
    return rows, metadata


def summarize(run, rows, metadata):
    has_all_laws = all(f"{law}_prediction" in rows[0] for law in LAWS[:-1])
    if has_all_laws:
        reference_inverse_size(rows)
    summaries = []
    for corpus in sorted({r["corpus"] for r in rows}):
        selected = [r for r in rows if r["corpus"] == corpus]
        mean = lambda key: float(np.mean([float(r[key]) for r in selected]))
        record = dict(run=run, corpus=corpus, cells=len(selected),
                      pairs=len({r["replicate"] for r in selected}),
                      skill=mean("skill"),
                      min_skill=min(float(r["skill"]) for r in selected),
                      observed_disagreement=mean("observed_disagreement"),
                      mean_cluster_ratio=mean("source_cluster_ratio"),
                      unseen_auxiliary_target_fraction=mean("zero_prior_target_fraction"))
        if run == "nonoverlap_pairs":
            half = [r for r in selected if r["q"] == .5]
            logs = np.log([float(r["source_cluster_ratio"]) for r in half])
            width = t.ppf(.975, len(logs) - 1) * np.std(logs, ddof=1) / np.sqrt(len(logs))
            record.update(geometric_ratio_q05=float(np.exp(logs.mean())),
                          ratio_ci95_q05=[float(np.exp(logs.mean() - width)), float(np.exp(logs.mean() + width))],
                          skill_q05=float(np.mean([float(r["skill"]) for r in half])))
        if has_all_laws:
            grouped = defaultdict(list)
            for row in selected:
                grouped[row["replicate"]].append(row)
            for law in LAWS:
                errors = [float(np.mean([r[f"{law}_abs_log_error"] for r in pair]))
                          for pair in grouped.values()]
                record[f"{law}_pair_mean_abs_log_error"] = float(np.mean(errors))
        if run == "analytic":
            record["analytic_ratio"] = mean("analytic_source_influence_ratio")
            record["analytic_same_lag_ratio"] = mean("analytic_same_lag_ratio")
        record["run_duration_seconds"] = metadata["duration_seconds"]
        record["peak_working_set_bytes"] = metadata["hardware"].get("peak_working_set_bytes")
        summaries.append(record)
    return summaries


def main():
    output = ROOT / "summary"
    if output.exists():
        raise FileExistsError(output)
    summaries, sources, sizes = [], {}, []
    for run in RUNS:
        rows, metadata = read_checked(run)
        summaries.extend(summarize(run, rows, metadata))
        for file in ("config.json", "metadata.json", "class_map.json", "pair_results.csv"):
            path = ROOT / run / file
            sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        if run == "grid":
            for corpus, n in itertools.product(sorted({r["corpus"] for r in rows}), (512, 1024, 2048)):
                selected = [r for r in rows if r["corpus"] == corpus and r["n_a"] == n]
                sizes.append(dict(corpus=corpus, n=n,
                                  skill=float(np.mean([float(r["skill"]) for r in selected])),
                                  ratio=float(np.mean([float(r["source_cluster_ratio"]) for r in selected]))))
    with (ROOT / "analytic/input_results.csv").open(newline="") as handle:
        input_rows = list(csv.DictReader(handle))
    context = []
    for corpus in sorted({r["corpus"] for r in input_rows}):
        sequences = defaultdict(list)
        for row in input_rows:
            if row["corpus"] == corpus:
                if int(row["n_chunks"]) != 512 or float(row["mask_rate"]) != .5:
                    raise ValueError("unexpected analytic context condition")
                sequences[int(row["sequence_index"])].append(row)
        if set(sequences) != set(range(128)) or any(len(v) != 2 for v in sequences.values()):
            raise ValueError("incomplete analytic input grid")
        values = []
        for _, pair in sorted(sequences.items()):
            for key in ("visible_context_surprisal", "analytic_source_influence_prediction"):
                np.testing.assert_allclose(float(pair[0][key]), float(pair[1][key]))
            values.append([float(pair[0]["visible_context_surprisal"]),
                           np.mean([float(r["observed_disagreement"]) for r in pair]),
                           float(pair[0]["analytic_source_influence_prediction"])])
        values = np.asarray(values)
        ordered = values[np.argsort(values[:, 0], kind="stable")]
        bins = [ordered[i:i + 32, 1:].mean(axis=0) for i in range(0, 128, 32)]
        context.append(dict(
            corpus=corpus, observed_rare_common=float(bins[-1][0] / bins[0][0]),
            analytic_rare_common=float(bins[-1][1] / bins[0][1]),
            quartile_ratios=[float(b[1] / b[0]) for b in bins],
            quartile_observed=[float(b[0]) for b in bins],
            quartile_predicted=[float(b[1]) for b in bins],
        ))
    input_path = ROOT / "analytic/input_results.csv"
    sources[str(input_path)] = hashlib.sha256(input_path.read_bytes()).hexdigest()
    output.mkdir()
    (output / "summary.json").write_text(json.dumps(summaries, indent=2))
    (output / "size_summary.json").write_text(json.dumps(sizes, indent=2))
    (output / "sources.json").write_text(json.dumps(sources, indent=2))
    (output / "context_summary.json").write_text(json.dumps(context, indent=2))
    for row in summaries:
        print(row["run"], row["corpus"], f"skill={row['skill']:.5f}",
              f"ratio={row['mean_cluster_ratio']:.4f}", flush=True)


if __name__ == "__main__":
    main()
