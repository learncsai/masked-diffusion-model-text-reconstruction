#!/usr/bin/env python3
"""Pair-level validation of prospective categorical lag experiments.

The inverse-size comparator is derived entirely from the independent
reference law at q=0.5 in each experiment. It assumes variance proportional
to (1-q)(1/n_A+1/n_B), with no fit to observed A/B disagreement.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t


ROOT = Path("results/prospective_lag_v1")
RUNS = ("grid", "nonoverlap_pairs", "unequal_sizes")
LAWS = ("source_cluster", "independent_chunks", "same_lag",
        "inverse_size_baseline")


def read_rows(name: str) -> list[dict]:
    with (ROOT / name / "pair_results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty result set {name}")
    for row in rows:
        row["n_a"] = int(row.get("n_a_chunks") or row["n_chunks"])
        row["n_b"] = int(row.get("n_b_chunks") or row["n_chunks"])
        row["q"] = float(row["mask_rate"])
        row["replicate"] = int(row["replicate"])
        row["observed"] = float(row["observed_disagreement"])
    return rows


def reference_inverse_size(rows: list[dict]) -> None:
    by_corpus = defaultdict(list)
    for row in rows:
        by_corpus[row["corpus"]].append(row)
    for corpus, selected in by_corpus.items():
        # The sole anchor is the reference-law prediction at q=0.5 and the
        # first size pair. It is identical across A/B pairs at that condition.
        first_size = (selected[0]["n_a"], selected[0]["n_b"])
        anchors = [row for row in selected if row["q"] == 0.5 and
                   (row["n_a"], row["n_b"]) == first_size]
        if not anchors:
            raise ValueError(f"missing reference anchor for {corpus}")
        anchor = float(anchors[0]["source_cluster_prediction"])
        anchor_scale = 1 / first_size[0] + 1 / first_size[1]
        for row in selected:
            scale = (1 - row["q"]) / 0.5 * (
                (1 / row["n_a"] + 1 / row["n_b"]) / anchor_scale
            )
            row["inverse_size_baseline_prediction"] = anchor * scale
            row["inverse_size_anchor_prediction"] = anchor
            for law in LAWS:
                prediction = float(row[f"{law}_prediction"])
                row[f"{law}_abs_log_error"] = abs(math.log(prediction / row["observed"]))


def interval(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        return float("nan"), float("nan")
    center = float(np.mean(values))
    half = float(t.ppf(0.975, len(values) - 1) *
                 np.std(values, ddof=1) / np.sqrt(len(values)))
    return center - half, center + half


def summarize(name: str, rows: list[dict]) -> tuple[list[dict], list[dict]]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["corpus"], row["replicate"])].append(row)
    pair_rows = []
    for (corpus, replicate), selected in groups.items():
        record = {"run": name, "corpus": corpus, "replicate": replicate,
                  "cells": len(selected)}
        for law in LAWS:
            record[f"{law}_mean_abs_log_error"] = float(np.mean([
                row[f"{law}_abs_log_error"] for row in selected
            ]))
        pair_rows.append(record)
    corpus_rows = []
    for corpus in sorted({row["corpus"] for row in rows}):
        selected = [row for row in rows if row["corpus"] == corpus]
        selected_pairs = [row for row in pair_rows if row["corpus"] == corpus]
        q_half = [row for row in selected if row["q"] == 0.5]
        if name == "nonoverlap_pairs":
            log_ratios = [math.log(float(row["source_cluster_prediction"]) /
                                   row["observed"]) for row in q_half]
            low, high = interval(log_ratios)
            ratio = float(np.exp(np.mean(log_ratios)))
            ci = [float(np.exp(low)), float(np.exp(high))]
        else:
            ratio, ci = None, None
        entry = {
            "run": name, "corpus": corpus,
            "unique_pairs": len(selected_pairs), "correlated_cells": len(selected),
            "mean_named_target_fraction_q05": float(np.mean([
                float(row.get("named_target_fraction") or "nan") for row in q_half
            ])),
            "mean_named_target_skill_q05": float(np.mean([
                float(row.get("named_target_skill") or "nan") for row in q_half
            ])),
            "mean_other_target_skill_q05": float(np.mean([
                float(row.get("other_target_skill") or "nan") for row in q_half
            ])),
            "geometric_cluster_ratio_q05": ratio,
            "conditional_pair_t95_ratio_q05": ci,
            "mean_source_cluster_ratio": float(np.mean([
                float(row["source_cluster_prediction"]) / row["observed"]
                for row in selected
            ])),
        }
        for law in LAWS:
            errors = [row[f"{law}_mean_abs_log_error"] for row in selected_pairs]
            entry[f"{law}_pair_mean_abs_log_error"] = float(np.mean(errors))
        for comparator in ("independent_chunks", "same_lag", "inverse_size_baseline"):
            differences = [
                row["source_cluster_mean_abs_log_error"] -
                row[f"{comparator}_mean_abs_log_error"] for row in selected_pairs
            ]
            entry[f"full_minus_{comparator}_error"] = float(np.mean(differences))
            entry[f"full_better_than_{comparator}_pairs"] = sum(value < 0 for value in differences)
        corpus_rows.append(entry)
    return corpus_rows, pair_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    output = ROOT / "extensions_summary"
    output.mkdir(exist_ok=True)
    all_summary, all_pairs, all_augmented = [], [], []
    for name in RUNS:
        rows = read_rows(name)
        reference_inverse_size(rows)
        summary, pairs = summarize(name, rows)
        all_summary += summary
        all_pairs += pairs
        for row in rows:
            all_augmented.append({"run": name, **row})
    write_csv(output / "pair_law_errors.csv", all_pairs)
    write_csv(output / "augmented_results.csv", all_augmented)
    (output / "summary.json").write_text(json.dumps(all_summary, indent=2), encoding="utf-8")
    for entry in all_summary:
        print(entry["run"], entry["corpus"],
              "pairs", entry["unique_pairs"],
              "named skill", f"{entry['mean_named_target_skill_q05']:.2%}",
              "OTHER skill", f"{entry['mean_other_target_skill_q05']:.2%}",
              "cluster ratio", f"{entry['mean_source_cluster_ratio']:.3f}",
              "full/chunk/same/naive log-error", *[
                  f"{entry[f'{law}_pair_mean_abs_log_error']:.3f}" for law in LAWS
              ], flush=True)


if __name__ == "__main__":
    main()
