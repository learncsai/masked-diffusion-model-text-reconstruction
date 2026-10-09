"""Validate analytic extensions against frozen full-vocabulary experiments.

No fitted correction or model selection is performed. Bootstrap comparisons
come from the previous runs, after checking identical inputs and outcomes.
The output is a new artifact collection; existing results are never replaced.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

from scripts.summarize_full_vocabulary import read_checked
from scripts.summarize_prospective_extensions import reference_inverse_size

ROOT = Path("results/full_vocabulary_v1")
RUNS = ("grid", "nonoverlap_pairs", "unequal_sizes")
ANALYTIC = ("analytic_source_influence", "analytic_same_lag")
LAWS = ANALYTIC + ("source_cluster", "independent_chunks", "same_lag",
                   "inverse_size_baseline", "analytic_inverse_size")
HASHES = ("mask_sha256", "evaluation_sha256", "side_a_sha256",
          "side_b_sha256", "auxiliary_fit_sha256")


def key(row):
    return (row["corpus"], int(row["replicate"]), int(row["n_a_chunks"]),
            int(row["n_b_chunks"]), float(row["mask_rate"]))


def load_matched(run):
    base, base_meta = read_checked(run)
    path = ROOT / f"analytic_{run}"
    config = json.loads((path / "config.json").read_text())
    meta = json.loads((path / "metadata.json").read_text())
    base_config = json.loads((ROOT / run / "config.json").read_text())
    allowed = {"output", "analytic_source_influence", "analytic_only", "save_input_details"}
    for name in set(config) | set(base_config):
        if name not in allowed and config.get(name) != base_config.get(name):
            raise ValueError(f"{run}: changed experimental setting {name}")
    if not (meta["analytic_only"] and meta["analytic_source_influence"]
            and meta["monte_carlo_draws_per_law"] == 0
            and meta["rng_bootstrap_seed_slots_reserved"]):
        raise ValueError("not a completed analytic-only run")
    if meta["manifest_provenance"] != base_meta["manifest_provenance"]:
        raise ValueError("source provenance changed")
    for source, digest in meta["code_sha256"].items():
        snapshot = path / "code" / Path(source).name
        if hashlib.sha256(snapshot.read_bytes()).hexdigest() != digest:
            raise ValueError(f"code snapshot mismatch: {snapshot}")
    mapping = json.loads((path / "class_map.json").read_text())
    if mapping["other_class"] is not None or mapping["token_ids_in_class_order"] != list(range(50257)):
        raise ValueError("not the full identity token map")
    with (path / "pair_results.csv").open(newline="") as handle:
        analytic = list(csv.DictReader(handle))
    indexed = {key(row): row for row in analytic}
    if len(indexed) != len(analytic) or set(indexed) != {key(row) for row in base}:
        raise ValueError(f"{run}: missing, repeated or extra analytic conditions")
    merged = []
    for old in base:
        new = indexed[key(old)]
        for name in HASHES:
            if old[name] != new[name]:
                raise ValueError(f"{run} {key(old)}: {name} differs")
        for name in ("skill", "baseline_loss", "reconstruction_loss", "observed_disagreement",
                     "zero_prior_target_fraction", "masked_tokens", "sources_a", "sources_b"):
            np.testing.assert_allclose(float(new[name]), float(old[name]), rtol=1e-12, atol=1e-14)
        for law in ANALYTIC:
            value = float(new[f"{law}_prediction"])
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"invalid prediction: {run}, {key(old)}, {law}")
            np.testing.assert_allclose(float(new[f"{law}_ratio"]), value / old["observed"])
        merged.append(dict(old, **{k: v for k, v in new.items() if k.startswith("analytic_")}))
    reference_inverse_size(merged)
    for corpus in config["corpora"]:
        selected = [row for row in merged if row["corpus"] == corpus]
        first = (selected[0]["n_a"], selected[0]["n_b"])
        anchors = [row for row in selected if (row["n_a"], row["n_b"]) == first and row["q"] == .5]
        anchor = float(anchors[0]["analytic_source_influence_prediction"])
        for row in selected:
            row["analytic_inverse_size_prediction"] = anchor * (1 - row["q"]) / .5 * (
                (1 / row["n_a"] + 1 / row["n_b"]) / (1 / first[0] + 1 / first[1]))
            for law in LAWS:
                ratio = float(row[f"{law}_prediction"]) / row["observed"]
                row[f"{law}_ratio"] = ratio
                row[f"{law}_abs_log_error"] = abs(float(np.log(ratio)))
        # The prediction targets the expectation across pairs, not individual outcomes.
        for a, b, q in {(r["n_a"], r["n_b"], r["q"]) for r in selected}:
            cell = [r for r in selected if (r["n_a"], r["n_b"], r["q"]) == (a, b, q)]
            for law in LAWS:
                values = [float(r[f"{law}_prediction"]) for r in cell]
                np.testing.assert_allclose(values, values[0], rtol=1e-12)
    return merged, meta


def ratio_interval(values):
    logs = np.log(np.asarray(values, dtype=float))
    width = t.ppf(.975, len(logs) - 1) * logs.std(ddof=1) / np.sqrt(len(logs))
    return dict(geometric_ratio=float(np.exp(logs.mean())),
                lower=float(np.exp(logs.mean() - width)),
                upper=float(np.exp(logs.mean() + width)), pairs=len(logs))


def summarize(run, rows, meta):
    output, conditions, pairs, intervals = [], [], [], []
    for corpus in dict.fromkeys(r["corpus"] for r in rows):
        selected = [r for r in rows if r["corpus"] == corpus]
        pair_ids = sorted({r["replicate"] for r in selected})
        record = dict(run=run, corpus=corpus, cells=len(selected), pairs=len(pair_ids),
                      mean_skill=float(np.mean([float(r["skill"]) for r in selected])),
                      min_skill=min(float(r["skill"]) for r in selected),
                      duration_seconds=meta["duration_seconds"],
                      peak_working_set_bytes=meta["hardware"].get("peak_working_set_bytes"))
        for law in LAWS:
            ratios = [r[f"{law}_ratio"] for r in selected]
            pair_errors = [float(np.mean([r[f"{law}_abs_log_error"] for r in selected
                                        if r["replicate"] == pair])) for pair in pair_ids]
            record[f"{law}_mean_ratio"] = float(np.mean(ratios))
            record[f"{law}_pair_mean_abs_log_error"] = float(np.mean(pair_errors))
            record[f"{law}_min_ratio"] = float(np.min(ratios))
            record[f"{law}_max_ratio"] = float(np.max(ratios))
            for pair, error in zip(pair_ids, pair_errors):
                pairs.append(dict(run=run, corpus=corpus, replicate=pair, law=law,
                                  mean_abs_log_error=error))
            if run == "nonoverlap_pairs":
                for q in (.2, .5, .8):
                    values = [r[f"{law}_ratio"] for r in selected if r["q"] == q]
                    intervals.append(dict(corpus=corpus, mask_rate=q, law=law,
                                          **ratio_interval(values)))
        for a, b, q in sorted({(r["n_a"], r["n_b"], r["q"]) for r in selected}):
            cell = [r for r in selected if (r["n_a"], r["n_b"], r["q"]) == (a, b, q)]
            entry = dict(run=run, corpus=corpus, n_a=a, n_b=b, mask_rate=q,
                         pairs=len(cell), observed=float(np.mean([r["observed"] for r in cell])),
                         skill=float(np.mean([float(r["skill"]) for r in cell])))
            for law in LAWS:
                entry[f"{law}_prediction"] = float(cell[0][f"{law}_prediction"])
                entry[f"{law}_mean_ratio"] = float(np.mean([r[f"{law}_ratio"] for r in cell]))
            conditions.append(entry)
        output.append(record)
    return output, conditions, pairs, intervals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "analytic_summary")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    summaries, conditions, pair_errors, intervals, merged = [], [], [], [], []
    hashes = {}
    for run in RUNS:
        rows, meta = load_matched(run)
        summary, cells, pairs, bounds = summarize(run, rows, meta)
        summaries.extend(summary)
        conditions.extend(cells)
        pair_errors.extend(pairs)
        intervals.extend(bounds)
        merged.extend(dict(row, run=run) for row in rows)
        for directory in (ROOT / run, ROOT / f"analytic_{run}"):
            for name in ("config.json", "metadata.json", "class_map.json", "pair_results.csv"):
                path = directory / name
                hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    args.output.mkdir(parents=True)
    for name, value in (("summary", summaries), ("conditions", conditions),
                        ("pair_errors", pair_errors), ("pair_intervals", intervals)):
        (args.output / f"{name}.json").write_text(json.dumps(value, indent=2), encoding="utf-8")
    with (args.output / "matched_pair_results.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(dict.fromkeys(k for row in merged for k in row))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(merged)
    (args.output / "sources.json").write_text(json.dumps({
        "input_hashes": hashes,
        "matching": "all conditions checked: masks, evaluation, A/B sources, auxiliary fitting and scores",
        "interval": "descriptive t95 on seven pair log ratios, conditional on shared auxiliary/reference/evaluation",
        "pair_errors": "mean absolute log error within pair over conditions, then mean across pairs",
        "unequal_sizes": "leading variance sum only; mean-output difference not estimated",
        "validation_scope": "fixed protocol extended after earlier pilot; not a preregistered study",
        "reused_pool": True,
    }, indent=2), encoding="utf-8")
    for row in summaries:
        print(row["run"], row["corpus"],
              f"analytic={row['analytic_source_influence_mean_ratio']:.4f}",
              f"bootstrap={row['source_cluster_mean_ratio']:.4f}")


if __name__ == "__main__":
    main()
