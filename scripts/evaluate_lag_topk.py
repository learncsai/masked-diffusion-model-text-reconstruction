"""Evaluate exact full-vocabulary top-k ranks on the frozen matched MDLM contexts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from diffusion_lm_rmt import sparse_categorical_lag as lag
from diffusion_lm_rmt.categorical_lag import offset_list

CORPORA = ("tinystories", "wikitext", "cnn_dailymail")
LABELS = dict(tinystories="TinyStories", wikitext="WikiText", cnn_dailymail="CNN/DailyMail")
KS = (1, 2, 4, 8, 16, 32, 64)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def target_ranks(scores, targets):
    """Exact ranks, breaking equal scores by smaller token ID; also return tie bounds."""
    scores = np.asarray(scores)
    targets = np.asarray(targets)
    if scores.ndim != 2 or targets.shape != (len(scores),) or not np.issubdtype(targets.dtype, np.integer):
        raise ValueError("Expected score rows and one integer target per row")
    if not np.all(np.isfinite(scores)) or np.any(targets < 0) or np.any(targets >= scores.shape[1]):
        raise ValueError("Invalid scores or targets")
    target_values = scores[np.arange(len(scores)), targets, None]
    best = 1 + np.count_nonzero(scores > target_values, axis=1)
    equal = scores == target_values
    worst = best + np.count_nonzero(equal, axis=1) - 1
    actual = best + np.count_nonzero(equal & (np.arange(scores.shape[1])[None, :] < targets[:, None]), axis=1)
    return actual.astype(np.int32), best.astype(np.int32), worst.astype(np.int32)


def prediction_ranks(prediction, targets, batch_rows=64):
    """Materialize bounded row blocks, ranking against all classes with no truncation."""
    ranks, best, worst = [], [], []
    for start in range(0, len(targets), batch_rows):
        stop = min(len(targets), start + batch_rows)
        scores = prediction.residual[start:stop].toarray()
        scores += prediction.prior_weight[start:stop, None] * prediction.prior[None, :]
        values = target_ranks(scores, targets[start:stop])
        for chunks, value in zip((ranks, best, worst), values):
            chunks.append(value)
        # Independent full sort for a small, deterministic sample of real predictions.
        if start == 0:
            token_ids = np.arange(scores.shape[1])
            for row in range(min(3, len(scores))):
                order = np.lexsort((token_ids, -scores[row]))
                expected = np.flatnonzero(order == targets[start + row])[0] + 1
                if values[0][row] != expected:
                    raise AssertionError("Target rank disagrees with full vocabulary sort")
    return tuple(np.concatenate(v) for v in (ranks, best, worst))


def chunk_mean(values, rows, chunks):
    counts = np.bincount(rows, minlength=chunks)
    if len(counts) != chunks or np.any(counts == 0):
        raise ValueError("Every evaluation chunk must have masked positions")
    return np.bincount(rows, weights=np.asarray(values, dtype=float), minlength=chunks) / counts


def relative_error_reduction(accuracy, baseline_accuracy):
    return (accuracy - baseline_accuracy) / (1 - baseline_accuracy) if baseline_accuracy < 1 else None


def plot_summary(out, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter, ScalarFormatter
    plt.rcParams.update({"font.size": 12.5, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 3, figsize=(10.0, 6.3), layout="constrained", sharex=True, sharey="row")
    styles = {512: ("#245A81", "o"), 1024: ("#B85D24", "s"), 2048: ("#62733B", "^")}
    handles = {}
    for col, corpus in enumerate(CORPORA):
        groups = [r for r in summary if r["corpus"] == corpus]
        for n, (color, marker) in styles.items():
            group = sorted((r for r in groups if r["n"] == n), key=lambda r: r["k"])
            line, = axes[0, col].plot(KS, [r["reconstructor_accuracy"] for r in group], color=color,
                marker=marker, label=f"Reconstructor: {n:,} chunks")
            handles[line.get_label()] = line
            axes[1, col].plot(KS, [r["relative_topk_error_reduction"] for r in group], color=color, marker=marker)
        base = sorted((r for r in groups if r["n"] == 512), key=lambda r: r["k"])
        line, = axes[0, col].plot(KS, [r["unigram_accuracy"] for r in base], color="#555555", linestyle="--", label="Unigram")
        handles[line.get_label()] = line
        axes[0, col].set_title("WikiText-103" if corpus == "wikitext" else LABELS[corpus])
        axes[1, col].axhline(0, color="#888888", linewidth=.7)
        for ax in axes[:, col]:
            ax.set_xscale("log", base=2)
            ax.set_xticks(KS)
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.grid(axis="y", color="#dddddd", linewidth=.6)
        axes[1, col].set_xlabel("Top k")
    axes[0, 0].set_ylim(bottom=0)
    axes[0, 0].set_ylabel("Top-k accuracy\n(higher is better)")
    axes[1, 0].set_ylabel("Relative top-k error reduction\n(higher is better)")
    fig.legend(handles.values(), handles.keys(), loc="outside lower center", ncol=2,
               frameon=False, fontsize=11.5)
    fig.savefig(out / "topk_comparison.png", dpi=170)
    fig.savefig(out / "topk_comparison.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=ROOT / "results/lag_topk_matched_v1/frozen_inputs.zip")
    parser.add_argument("--output", type=Path, default=ROOT / "results/lag_topk_matched_v1")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "per_chunk").mkdir(exist_ok=True)
    import torch
    import scipy
    torch.set_num_threads(1)
    all_rows, per_model_mse, checks, contexts = [], [], [], {}
    with zipfile.ZipFile(args.inputs) as archive:
        for corpus in CORPORA:
            def read(name):
                return json.loads(archive.read(f"{corpus}/{name}"))
            manifest, prepared = read("manifest.json"), read("prepared.json")
            config = manifest["protocol"]["config"]
            if config["radius"] != 2 or config["classes"] != 50257:
                raise ValueError("Unexpected reconstructor protocol")
            if config["pairs"] != 3 or config["corpus_sizes"] != [512, 1024, 2048]:
                raise ValueError("Unexpected comparison sizes")
            for rel, expected in manifest["protocol"]["inputs"].items():
                if sha(ROOT / rel) != expected:
                    raise ValueError(f"Changed input: {rel}")
            for rel in ["src/diffusion_lm_rmt/sparse_categorical_lag.py", "src/diffusion_lm_rmt/categorical_lag.py"]:
                if sha(ROOT / rel) != manifest["protocol"]["code"][rel]:
                    raise ValueError(f"Changed reconstruction code: {rel}")
            for name in ["context.npz", "lag_pairs.json", "source_audit.json"]:
                if hashlib.sha256(archive.read(f"{corpus}/{name}")).hexdigest() != prepared["sha256"][name]:
                    raise ValueError(f"Changed frozen context: {corpus}/{name}")
            with np.load(io.BytesIO(archive.read(f"{corpus}/context.npz"))) as z:
                context = {k: z[k] for k in z.files}
            ids, mask, prior = context["evaluation"], context["mask"], context["prior"]
            rows, positions = np.nonzero(mask)
            targets, chunks = ids[rows, positions], len(ids)
            eval_path = next(ROOT / rel for rel in manifest["protocol"]["inputs"] if rel.endswith("/eval.pt"))
            evaluation = torch.load(eval_path, map_location="cpu", weights_only=True)
            if not np.array_equal(evaluation["input_ids"].numpy()[:chunks], ids):
                raise ValueError("Evaluation records do not match saved context")
            source_ids = np.array([r["document_id"] for r in evaluation["chunk_records"][:chunks]], dtype=str)
            contexts[corpus] = dict(chunks=chunks, masked_positions=len(targets), source_documents=len(set(source_ids)),
                context_sha256=prepared["sha256"]["context.npz"], protocol_sha256=manifest["protocol_sha256"])
            order = np.lexsort((np.arange(len(prior)), -prior))
            prior_rank = np.empty(len(prior), dtype=np.int32)
            prior_rank[order] = np.arange(1, len(prior) + 1)
            unique, inverse, counts = np.unique(prior, return_inverse=True, return_counts=True)
            greater = len(prior) - np.cumsum(counts)
            base_best = greater[inverse[targets]] + 1
            base_worst = base_best + counts[inverse[targets]] - 1
            base_ranks = prior_rank[targets]
            base_mse = float(chunk_mean(1 - 2 * prior[targets] + prior @ prior, rows, chunks).mean())
            baseline = {k: float(chunk_mean(base_ranks <= k, rows, chunks).mean()) for k in KS}
            np.savez_compressed(out / "per_chunk" / f"{corpus}_evaluation.npz", source_ids=source_ids,
                evaluation=ids, mask=mask, rows=rows, positions=positions, targets=targets,
                unigram_target_rank=base_ranks, unigram_best_tie_rank=base_best, unigram_worst_tie_rank=base_worst)
            old = {(r["n"], r["pair"]): r for r in read("lag_pairs.json")}
            offsets = offset_list(config["radius"])
            for n in config["corpus_sizes"]:
                for pair in range(1, config["pairs"] + 1):
                    for arm in ("a", "b"):
                        path = ROOT / "data/diffusion_llm/reproducibility_followup_v1" / corpus / f"pair{pair}_{arm}.npz"
                        with np.load(path) as z:
                            training = z["ids"][:n]
                        tables = lag.conditional_tables(lag.pair_counts(training, offsets, len(prior)), prior, config["smoothing"])
                        actual_rows, actual_targets, prediction = lag.masked_prediction(ids, mask, tables, prior, offsets,
                            context["gram"], context["cross"], config["ridge"])
                        if not np.array_equal(actual_rows, rows) or not np.array_equal(actual_targets, targets):
                            raise ValueError("Scored masked positions changed")
                        mse = float(chunk_mean(prediction.loss(targets), rows, chunks).mean())
                        previous = old[n, pair]
                        if not np.isclose(mse, previous[f"loss_{arm}"], rtol=1e-11, atol=1e-12):
                            raise ValueError("MSE differs from the frozen RunPod reconstructor result")
                        if not np.isclose(base_mse, previous["baseline_loss"], rtol=1e-11, atol=1e-12):
                            raise ValueError("Unigram differs from the frozen RunPod baseline")
                        rank, best, worst = prediction_ranks(prediction, targets)
                        per_model_mse.append(dict(corpus=corpus, n=n, pair=pair, arm=arm,
                            reconstructor_mse=mse, unigram_mse=base_mse, relative_mse_improvement=1-mse/base_mse))
                        chunk_accuracies = []
                        for k in KS:
                            accuracy_by_chunk = chunk_mean(rank <= k, rows, chunks)
                            accuracy = float(accuracy_by_chunk.mean())
                            chunk_accuracies.append(accuracy_by_chunk)
                            all_rows.append(dict(corpus=corpus, n=n, pair=pair, arm=arm, k=k,
                                reconstructor_accuracy=accuracy, unigram_accuracy=baseline[k],
                                accuracy_gain_pp=100*(accuracy-baseline[k]),
                                relative_topk_error_reduction=relative_error_reduction(accuracy, baseline[k]),
                                accuracy_tie_min=float(chunk_mean(worst <= k, rows, chunks).mean()),
                                accuracy_tie_max=float(chunk_mean(best <= k, rows, chunks).mean()),
                                unigram_accuracy_tie_min=float(chunk_mean(base_worst <= k, rows, chunks).mean()),
                                unigram_accuracy_tie_max=float(chunk_mean(base_best <= k, rows, chunks).mean())))
                        np.savez_compressed(out / "per_chunk" / f"{corpus}_pair{pair}_{arm}_n{n}.npz",
                            k=np.array(KS), target_rank=rank, best_tie_rank=best, worst_tie_rank=worst,
                            accuracy_by_chunk=np.array(chunk_accuracies))
                        checks.append(dict(corpus=corpus, n=n, pair=pair, arm=arm,
                            absolute_mse_difference=abs(mse-previous[f"loss_{arm}"])))
                    print(f"Scored {corpus}: {n} chunks, pair {pair}", flush=True)
    summary, paired = [], []
    for corpus in CORPORA:
        for n in [512, 1024, 2048]:
            for k in KS:
                group = [r for r in all_rows if (r["corpus"],r["n"],r["k"]) == (corpus,n,k)]
                if len(group) != 6:
                    raise ValueError("Incomplete A/B comparison")
                pair_means = []
                for pair in [1,2,3]:
                    arms = [r for r in group if r["pair"]==pair]
                    mean = float(np.mean([r["reconstructor_accuracy"] for r in arms]))
                    pair_means.append(mean)
                    paired.append(dict(corpus=corpus,n=n,k=k,pair=pair,reconstructor_accuracy=mean,
                        unigram_accuracy=group[0]["unigram_accuracy"],
                        relative_topk_error_reduction=relative_error_reduction(mean,group[0]["unigram_accuracy"])))
                row = dict(corpus=corpus,n=n,k=k,pairs=3,models=6)
                for field in list(group[0])[5:]:
                    row[field] = float(np.mean([r[field] for r in group]))
                row.update(pair_mean_accuracy_min=min(pair_means),pair_mean_accuracy_max=max(pair_means))
                summary.append(row)
    write_csv(out / "per_model.csv", all_rows)
    write_csv(out / "per_pair.csv", paired)
    write_csv(out / "summary.csv", summary)
    write_json(out / "summary.json", summary)
    write_csv(out / "mse_check.csv", per_model_mse)
    metadata = dict(completed_utc=datetime.now(timezone.utc).isoformat(), source_run="rtx4090_budget_v1",
        k=list(KS), radius=2, classes=50257, contexts=contexts, frozen_inputs_sha256=sha(args.inputs),
        analysis_script_sha256=sha(Path(__file__)), python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
        aggregation="Mean over masked positions within each chunk, then equal mean over chunks. Mean A/B within each pair, then mean over three pairs.",
        tie_rule="Descending raw score, then ascending token ID. Tie min/max bound arbitrary tie ordering, not statistical uncertainty.",
        ranking="All 50,257 content-token classes including EOS. MASK/PAD excluded. Raw affine reconstructor scores are not clipped or renormalized.",
        relative_topk_error_reduction="1 - (1 - accuracy_f)/(1 - accuracy_p). This is distinct from relative MSE improvement.",
        independence="Three source-disjoint A/B pairs per corpus, conditional on shared auxiliary/evaluation data. Sizes are nested. No token-level bootstrap or confidence intervals.",
        checks=dict(model_evaluations=len(checks), mse_matches_frozen_results=checks,
            independent_full_sorts_per_model=3, test_file="tests/test_lag_topk.py"))
    write_json(out / "metadata.json", metadata)
    plot_summary(out, summary)
    print(json.dumps({"output":str(out),"models_scored":len(per_model_mse),"summary_rows":len(summary)},indent=2))


if __name__ == "__main__":
    main()
