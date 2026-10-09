"""Compare reference-only disagreement laws on the frozen three-corpus task.

Prepare portable inputs once with --prepare-inputs. Then run without arguments.
This script never loads a neural checkpoint or connects to the training Pod.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import platform
import shutil
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from diffusion_lm_rmt import sparse_categorical_lag as lag
from diffusion_lm_rmt.categorical_lag import offset_list, pattern_weights, per_sequence_squared_error, source_groups
from diffusion_lm_rmt.sparse_categorical_influence import source_influence_disagreement

CORPORA = ("tinystories", "wikitext", "cnn_dailymail")
LABELS = dict(tinystories="TinyStories", wikitext="WikiText-103", cnn_dailymail="CNN/DailyMail")
SIZES = (512, 1024, 2048)
METHODS = ("multinomial", "source_same_lag", "source_full")
CORE = [f"src/diffusion_lm_rmt/{name}.py" for name in
        ("categorical_lag", "sparse_categorical_lag", "sparse_categorical_influence")]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_records(ids, records):
    if len(ids) != len(records):
        raise ValueError("Token/record length mismatch")
    actual = [hashlib.sha256(np.asarray(row, dtype=np.int32).tobytes()).hexdigest() for row in ids]
    if actual != [r["sha256"] for r in records]:
        raise ValueError("Token bytes do not match the recorded chunk hashes")


def prepare_inputs(frozen, inputs):
    """Export a self-contained token package after checking the frozen protocol."""
    inputs.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(frozen) as archive:
        for corpus in CORPORA:
            folder = inputs / corpus
            folder.mkdir(exist_ok=True)
            original = json.loads(archive.read(f"{corpus}/manifest.json"))
            protocol = original["protocol"]
            for rel, expected in protocol["inputs"].items():
                if sha(ROOT / rel) != expected:
                    raise ValueError(f"Frozen input changed: {rel}")
            for rel in CORE:
                if sha(ROOT / rel) != protocol["code"][rel]:
                    raise ValueError(f"Frozen mathematical implementation changed: {rel}")
            prepared = json.loads(archive.read(f"{corpus}/prepared.json"))
            for name in ("context.npz", "lag_pairs.json", "source_audit.json"):
                content = archive.read(f"{corpus}/{name}")
                if hashlib.sha256(content).hexdigest() != prepared["sha256"][name]:
                    raise ValueError(f"Frozen result changed: {corpus}/{name}")
                (folder / name).write_bytes(content)
            (folder / "original_manifest.json").write_bytes(archive.read(f"{corpus}/manifest.json"))
            split_path = next(ROOT / p for p in protocol["inputs"] if p.endswith("confirmatory_split_manifest.json"))
            split_manifest = read_json(split_path)
            records = split_manifest["splits"]["eval"]["chunks"][:128]
            with np.load(folder / "context.npz") as context:
                validate_records(context["evaluation"], records)
            write_json(folder / "evaluation_records.json", records)
            document_hashes = {r["document_id"]: r["sha256"]
                for split in split_manifest["splits"].values() for r in split["documents"]}
            raw = next(ROOT / p for p in protocol["inputs"] if p.endswith("raw_documents.jsonl"))
            with raw.open(encoding="utf-8") as stream:
                for line in stream:
                    doc = json.loads(line)
                    if hashlib.sha256(doc["text"].encode()).hexdigest() != doc["sha256"]:
                        raise ValueError("Raw source hash mismatch")
                    document_hashes[doc["document_id"]] = doc["sha256"]
            write_json(folder / "source_text_hashes.json", document_hashes)
            for name in ("reference_original", *[f"pair{p}_{arm}" for p in (1, 2, 3) for arm in ("a", "b")]):
                for suffix in (".npz", ".json"):
                    path = next(ROOT / p for p in protocol["inputs"] if p.endswith(f"/{name}{suffix}"))
                    shutil.copyfile(path, folder / f"{name}{suffix}")
            print(f"Verified and prepared {corpus}", flush=True)
    paths = sorted(p for p in inputs.rglob("*") if p.is_file() and p.name != "manifest.json")
    write_json(inputs / "manifest.json", dict(prepared_utc=utc(), frozen_bundle_sha256=sha(frozen),
        inputs_sha256={p.relative_to(inputs).as_posix(): sha(p) for p in paths},
        core_sha256={p: sha(ROOT / p) for p in CORE}))


def multinomial_row_trace(counts, prior, smoothing, scale):
    """Twice the fixed-row-count variance trace, with a reference plug-in law.

    The population conditional is estimated once at the REFERENCE size with
    add-alpha smoothing. Only the expected row count is scaled to target n.
    Includes every part of the smoothed collision probability, including the
    prior cross term and prior square. No A/B counts enter this calculation.
    """
    totals = np.asarray(counts.sum(axis=1)).ravel()
    table, beta = lag.conditional_tables({0: counts}, prior, smoothing)[0]
    collision = lag.Prediction(table, beta, prior).norm2()
    if np.any(collision < -1e-12) or np.any(collision > 1 + 1e-12):
        raise ArithmeticError("Invalid collision probability")
    target_counts = scale * totals
    return 2 * target_counts / (target_counts + smoothing)**2 * np.maximum(1 - collision, 0)


def multinomial_forecast(reference, evaluation, mask, prior, offsets, gram, cross, *, n, smoothing, ridge):
    counts = lag.pair_counts(reference, offsets, len(prior))
    trace = {d: multinomial_row_trace(c, prior, smoothing, n / len(reference)) for d, c in counts.items()}
    rows, _, contexts, visible, patterns = lag.context_design(evaluation, mask, offsets)
    point = np.zeros(len(rows))
    for code in np.unique(patterns):
        if not code:
            continue
        selected = np.flatnonzero(patterns == code)
        active = tuple(j for j in range(len(offsets)) if code & (1 << j))
        weights = pattern_weights(gram, cross, active, ridge)
        for weight, j in zip(weights, active):
            point[selected] += weight**2 * trace[offsets[j]][contexts[j][selected]]
    return per_sequence_squared_error(rows, point, len(evaluation))


def correlation(left, right, *, rank=False):
    if np.ptp(left) == 0 or np.ptp(right) == 0:
        return None
    return float(spearmanr(left, right).statistic if rank else np.corrcoef(left, right)[0, 1])


def summarize(corpus, n, forecasts, observed):
    """Keep input correlations within each pair and each corpus-size condition."""
    pair_rows, summary = [], []
    for method, prediction in zip(METHODS, forecasts):
        group = []
        for pair, actual in enumerate(observed, 1):
            row = dict(corpus=corpus, n=n, method=method, pair=pair,
                predicted_mean=float(prediction.mean()), observed_mean=float(actual.mean()),
                prediction_observation_ratio=float(prediction.mean() / actual.mean()),
                absolute_log_error=float(abs(np.log(prediction.mean() / actual.mean()))),
                pearson=correlation(prediction, actual), spearman=correlation(prediction, actual, rank=True))
            pair_rows.append(row)
            group.append(row)
        summary.append(dict(corpus=corpus, n=n, method=method, pairs=len(observed), chunks=observed.shape[1],
            predicted_mean=float(prediction.mean()), observed_mean=float(observed.mean()),
            prediction_observation_ratio=float(prediction.mean() / observed.mean()),
            mean_absolute_log_error=float(np.mean([r["absolute_log_error"] for r in group])),
            mean_pair_pearson=float(np.mean([r["pearson"] for r in group])),
            mean_pair_spearman=float(np.mean([r["spearman"] for r in group])),
            pair_spearman_min=min(r["spearman"] for r in group), pair_spearman_max=max(r["spearman"] for r in group),
            pearson_to_pair_mean=correlation(prediction, observed.mean(axis=0)),
            spearman_to_pair_mean=correlation(prediction, observed.mean(axis=0), rank=True)))
    return pair_rows, summary


def plot_summary(out, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    styles = dict(multinomial=("#B85D24", "s", "Multinomial"),
        source_same_lag=("#62733B", "^", "Source, same lag"), source_full=("#245A81", "o", "Source, all lags"))
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), layout="constrained", sharex=True, sharey="row")
    handles = []
    for col, corpus in enumerate(CORPORA):
        for method, (color, marker, label) in styles.items():
            group = sorted((r for r in summary if r["corpus"] == corpus and r["method"] == method), key=lambda r: r["n"])
            line, = axes[0, col].plot(SIZES, [r["prediction_observation_ratio"] for r in group], color=color, marker=marker, label=label)
            axes[1, col].plot(SIZES, [r["mean_pair_spearman"] for r in group], color=color, marker=marker)
            if col == 0:
                handles.append(line)
        axes[0, col].set_title(LABELS[corpus])
        axes[0, col].axhline(1, color="#555555", linestyle="--", linewidth=1)
        axes[1, col].set_xlabel("Corpus size (64-token chunks)")
        for ax in axes[:, col]:
            ax.set_xscale("log", base=2)
            ax.set_xticks(SIZES)
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.grid(axis="y", color="#dddddd", linewidth=.6)
    axes[0, 0].set_ylabel("Predicted / observed mean\n(target: 1)")
    axes[1, 0].set_ylabel("Mean within-pair Spearman correlation\n(higher is better)")
    axes[0, 0].set_ylim(bottom=0)
    axes[1, 0].set_ylim(-.05, 1)
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    fig.savefig(out / "disagreement_comparison.png", dpi=180)
    fig.savefig(out / "disagreement_comparison.pdf")
    plt.close(fig)


def run(inputs, out):
    out.mkdir(parents=True, exist_ok=True)
    (out / "per_chunk").mkdir(exist_ok=True)
    manifest = read_json(inputs / "manifest.json")
    for rel, expected in manifest["inputs_sha256"].items():
        if sha(inputs / rel) != expected:
            raise ValueError(f"Analysis input changed: {rel}")
    for rel, expected in manifest["core_sha256"].items():
        if sha(ROOT / rel) != expected:
            raise ValueError(f"Mathematical implementation changed: {rel}")
    code_paths = [*CORE, "scripts/evaluate_lag_disagreement.py"]
    code_hashes = {p: sha(ROOT / p) for p in code_paths}
    contract = dict(input_manifest_sha256=sha(inputs / "manifest.json"), code_sha256=code_hashes,
        corpora=list(CORPORA), sizes=list(SIZES), methods=list(METHODS),
        multinomial="Reference-smoothed conditional collision, scaled expected row counts, fixed-row-count iid variance, no cross-lag terms.",
        source_same_lag="Source-level ratio influence, deleting cross-lag products only.",
        source_full="Source-level ratio influence, all cross-lag products retained.",
        metrics="Ratio of mean forecast to mean observed disagreement. Mean absolute pair log error. Pearson/Spearman within each of three pairs, then arithmetic mean.",
        uncertainty="Descriptive conditional results. No confidence intervals or token/chunk independence assumption for inference.",
        status="Exploratory comparison on existing pairs. Outcomes were already available. No method is tuned using A/B outcomes.")
    signature = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    if (out / "analysis_plan.json").exists():
        if read_json(out / "analysis_plan.json")["signature"] != signature:
            raise ValueError("Analysis plan changed. Use a new output directory.")
    else:
        write_json(out / "analysis_plan.json", dict(saved_utc=utc(), signature=signature, **contract))
    all_summary, all_pairs, all_chunks, checks, audits = [], [], [], [], {}
    started = time.perf_counter()
    for corpus in CORPORA:
        folder = inputs / corpus
        original = read_json(folder / "original_manifest.json")
        config = original["protocol"]["config"]
        assert config["radius"] == 2 and config["classes"] == 50257 and config["pairs"] == 3
        assert config["corpus_sizes"] == list(SIZES) and config["mask_rate"] == .5
        with np.load(folder / "context.npz") as c:
            ids, mask, prior, gram, cross = (c[k] for k in ("evaluation", "mask", "prior", "gram", "cross"))
        with np.load(folder / "reference_original.npz") as z:
            reference = z["ids"]
        records = read_json(folder / "reference_original.json")
        validate_records(reference, records)
        evaluation_records = read_json(folder / "evaluation_records.json")
        validate_records(ids, evaluation_records)
        source_ids = np.array([r["document_id"] for r in evaluation_records])
        groups = source_groups(records)
        offsets = offset_list(config["radius"])
        kwargs = dict(smoothing=config["smoothing"], ridge=config["ridge"])
        role_audit = read_json(folder / "source_audit.json")
        assert role_audit["overlaps"] == 0
        text_hashes = read_json(folder / "source_text_hashes.json")
        role_sets, text_sets = [], []
        for role in role_audit["roles"].values():
            current = set(role["source_ids"])
            current_text = {text_hashes[s] for s in current}
            if any(current & previous for previous in role_sets) or any(current_text & previous for previous in text_sets):
                raise ValueError("Sources or source texts cross roles")
            role_sets.append(current)
            text_sets.append(current_text)
        assert set(source_ids) == set(role_audit["roles"]["evaluation"]["source_ids"])
        assert {r["document_id"] for r in records} == set(role_audit["roles"]["reference"]["source_ids"])
        audits[corpus] = dict(reference_chunks=len(reference), reference_sources=len(groups),
            evaluation_chunks=len(ids), evaluation_sources=len(set(source_ids)), overlaps=0,
            original_protocol_sha256=original["protocol_sha256"])
        # Compute and freeze every prediction before this script loads the A/B scores.
        predictions = {}
        for n in SIZES:
            cache = out / "per_chunk" / f"{corpus}_n{n}_forecasts.npz"
            if cache.exists():
                with np.load(cache) as z:
                    assert str(z["signature"]) == signature
                    forecasts = z["forecasts"]
            else:
                common = (reference, ids, mask, prior, offsets, gram, cross)
                multi = multinomial_forecast(*common, n=n, **kwargs)
                print(f"Forecasting {corpus} n={n}: source, same lag", flush=True)
                same = source_influence_disagreement(reference, groups, ids, mask, prior, offsets, gram, cross,
                    n=n, same_lag=True, **kwargs)
                print(f"Forecasting {corpus} n={n}: source, all lags", flush=True)
                full = source_influence_disagreement(reference, groups, ids, mask, prior, offsets, gram, cross,
                    n=n, same_lag=False, **kwargs)
                forecasts = np.array([multi, same, full])
                if forecasts.shape != (3, len(ids)) or not np.all(np.isfinite(forecasts)) or np.any(forecasts < 0):
                    raise ValueError("Invalid forecasts")
                np.savez_compressed(cache, forecasts=forecasts, methods=np.array(METHODS), source_ids=source_ids,
                    signature=np.array(signature), saved_utc=np.array(utc()))
            predictions[n] = forecasts
        frozen = {(r["n"], r["pair"]): r for r in read_json(folder / "lag_pairs.json")}
        pair_data = {}
        chunk_sets = [{r["sha256"] for r in records}, {r["sha256"] for r in evaluation_records}]
        for pair in (1, 2, 3):
            for arm in ("a", "b"):
                name = f"pair{pair}_{arm}"
                with np.load(folder / f"{name}.npz") as z:
                    tokens = z["ids"][:max(SIZES)]
                rec = read_json(folder / f"{name}.json")[:len(tokens)]
                validate_records(tokens, rec)
                assert {r["document_id"] for r in rec} == set(role_audit["roles"][name]["source_ids"])
                hashes = {r["sha256"] for r in rec}
                if any(hashes & previous for previous in chunk_sets):
                    raise ValueError("Exact chunks cross roles")
                chunk_sets.append(hashes)
                pair_data[pair, arm] = tokens
        for n in SIZES:
            observed = []
            for pair in (1, 2, 3):
                outputs, losses = [], []
                for arm in ("a", "b"):
                    tables = lag.conditional_tables(lag.pair_counts(pair_data[pair, arm][:n], offsets, len(prior)), prior, config["smoothing"])
                    rows, targets, output = lag.masked_prediction(ids, mask, tables, prior, offsets, gram, cross, config["ridge"])
                    outputs.append(output)
                    losses.append(per_sequence_squared_error(rows, output.loss(targets), len(ids)))
                actual = per_sequence_squared_error(rows, lag.squared_difference(*outputs), len(ids))
                old = frozen[n, pair]
                np.testing.assert_allclose(actual, old["per_chunk"]["disagreement"], rtol=1e-11, atol=1e-12)
                for arm, loss in zip(("a", "b"), losses):
                    np.testing.assert_allclose(loss, old["per_chunk"][f"loss_{arm}"], rtol=1e-11, atol=1e-12)
                np.testing.assert_allclose(predictions[n][2].mean(), old["lag_prediction"], rtol=1e-11, atol=1e-12)
                # Independently materialize only three rows to check squared distances.
                for j in (0, len(rows)//2, len(rows)-1):
                    vectors = [o.residual[j].toarray().ravel() + o.prior_weight[j]*prior for o in outputs]
                    np.testing.assert_allclose(lag.squared_difference(*outputs)[j], np.sum((vectors[0]-vectors[1])**2), rtol=1e-11, atol=1e-12)
                observed.append(actual)
                checks.append(dict(corpus=corpus, n=n, pair=pair,
                    max_chunk_disagreement_difference=float(np.max(np.abs(actual-old["per_chunk"]["disagreement"]))),
                    full_forecast_mean_difference=float(abs(predictions[n][2].mean()-old["lag_prediction"]))))
                for chunk, value in enumerate(actual):
                    all_chunks.append(dict(corpus=corpus, n=n, pair=pair, chunk=chunk,
                        source_id=str(source_ids[chunk]), observed=float(value),
                        **{method: float(predictions[n][j, chunk]) for j, method in enumerate(METHODS)}))
            observed = np.array(observed)
            np.savez_compressed(out / "per_chunk" / f"{corpus}_n{n}_observed.npz", observed=observed,
                source_ids=source_ids, signature=np.array(signature))
            pairs, summary = summarize(corpus, n, predictions[n], observed)
            all_pairs.extend(pairs)
            all_summary.extend(summary)
            print(f"Scored {corpus} n={n}: ratios " + ", ".join(f"{r['method']}={r['prediction_observation_ratio']:.3f}" for r in summary), flush=True)
    write_csv(out / "summary.csv", all_summary)
    write_json(out / "summary.json", all_summary)
    write_csv(out / "per_pair.csv", all_pairs)
    write_csv(out / "per_chunk.csv", all_chunks)
    table = []
    for corpus in CORPORA:
        for n in SIZES:
            group = [next(r for r in all_summary if (r["corpus"], r["n"], r["method"]) == (corpus, n, m)) for m in METHODS]
            table.append(f"{LABELS[corpus]} & {n:,} & {group[0]['observed_mean']:.5f} & " +
                " & ".join(f"{r['prediction_observation_ratio']:.3f}" for r in group) + " & " +
                " & ".join(f"{r['mean_pair_spearman']:.3f}" for r in group) + r" \\")
    (out / "table_rows.tex").write_text("\n".join(table) + "\n", encoding="utf-8")
    plot_summary(out, all_summary)
    write_json(out / "metadata.json", dict(completed_utc=utc(), signature=signature,
        runtime_seconds=time.perf_counter()-started, python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
        source_audits=audits, checks=checks, paired_conditions=len(checks), summary_rows=len(all_summary),
        independent_dense_distance_checks_per_pair=3,
        outputs_sha256={p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob("*"))
                        if p.is_file() and p.name not in ("metadata.json", "README.md", "validation.json")}))
    print(json.dumps(dict(completed_utc=utc(), output=str(out), seconds=time.perf_counter()-started), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-inputs", action="store_true")
    parser.add_argument("--frozen", type=Path, default=ROOT / "results/lag_topk_matched_v1/frozen_inputs.zip")
    parser.add_argument("--inputs", type=Path, default=ROOT / "results/lag_disagreement_matched_v1/inputs")
    parser.add_argument("--output", type=Path, default=ROOT / "results/lag_disagreement_matched_v1")
    args = parser.parse_args()
    if args.prepare_inputs:
        prepare_inputs(args.frozen, args.inputs)
    else:
        run(args.inputs, args.output)


if __name__ == "__main__":
    main()
