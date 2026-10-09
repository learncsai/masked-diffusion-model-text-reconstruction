"""Score trained MDLM checkpoints on the frozen reconstructor top-k task.

This performs evaluation only. A preparation pass verifies contexts and
checkpoint identities without loading a model. Incomplete scopes are marked.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import numpy as np
import torch
from diffusion_lm_rmt.matched_mdlm import digest, load_model, object_digest, read_json, utc, write_json
from evaluate_lag_topk import KS, chunk_mean, relative_error_reduction, write_csv

CORPORA = ("tinystories", "wikitext", "cnn_dailymail")


@torch.inference_mode()
def score_checkpoint(model, ids, mask, config, device):
    """Exact logit ranks and probability-vector MSE on masked positions."""
    if ids.shape != mask.shape or not np.all(mask.sum(1) > 0):
        raise ValueError("Invalid evaluation masks")
    model.eval()
    ranks, best, worst, loss = [], [], [], []
    batch = config["training"]["microbatch_size"]
    classes = config["classes"]
    for start in range(0, len(ids), batch):
        clean = torch.as_tensor(ids[start:start+batch], dtype=torch.long, device=device)
        hidden = torch.as_tensor(mask[start:start+batch], dtype=torch.bool, device=device)
        corrupted = clean.masked_fill(hidden, config["mask_token_id"])
        attention = torch.ones_like(clean, dtype=torch.bool)
        times = torch.full((len(clean),), config["mask_rate"], device=device)
        logits = model(corrupted, attention, times)[..., :classes]
        masked_logits = logits[hidden].float()
        targets = clean[hidden]
        values = masked_logits.gather(1, targets[:, None])
        equal = masked_logits == values
        first = 1 + (masked_logits > values).sum(1)
        last = first + equal.sum(1) - 1
        actual = first + (equal & (torch.arange(classes, device=device)[None, :] < targets[:, None])).sum(1)
        if not torch.isfinite(masked_logits).all():
            raise FloatingPointError("Nonfinite logits")
        if start == 0:
            # Independent complete stable sorts check the tensor rank arithmetic.
            scores = masked_logits[:3].cpu().numpy()
            expected = [np.flatnonzero(np.lexsort((np.arange(classes), -row)) == int(t))[0]+1
                        for row, t in zip(scores, targets[:3].cpu().numpy())]
            np.testing.assert_array_equal(actual[:3].cpu().numpy(), expected)
        probabilities = torch.softmax(logits, -1)
        mse = 1 + probabilities.square().sum(-1) - 2*probabilities.gather(-1, clean[..., None]).squeeze(-1)
        per_chunk_loss = (mse.double()*hidden).sum(1)/hidden.sum(1)
        loss.extend(per_chunk_loss.cpu().tolist())
        for dest, tensor in ((ranks, actual), (best, first), (worst, last)):
            dest.append(tensor.cpu().numpy().astype(np.int32))
    return dict(target_rank=np.concatenate(ranks), best_tie_rank=np.concatenate(best),
                worst_tie_rank=np.concatenate(worst), mse_by_chunk=np.asarray(loss))


def prepare_jobs(run, lag_results, corpora, requested_sizes=None, requested_updates=None):
    lag_meta = read_json(lag_results / "metadata.json")
    jobs, contexts, evidence = [], {}, {}
    for corpus in corpora:
        folder = run / corpus
        manifest = read_json(folder / "manifest.json")
        prepared = read_json(folder / "prepared.json")
        config = manifest["protocol"]["config"]
        context_hash = digest(folder / "context.npz")
        if context_hash != prepared["sha256"]["context.npz"]:
            raise ValueError("MDLM context changed")
        if context_hash != lag_meta["contexts"][corpus]["context_sha256"]:
            raise ValueError("MDLM and reconstructor contexts differ")
        if manifest["protocol_sha256"] != lag_meta["contexts"][corpus]["protocol_sha256"]:
            raise ValueError("MDLM and reconstructor protocols differ")
        model_path = "src/diffusion_lm_rmt/diffusion/model.py"
        recorded = {k.replace("\\", "/"): v for k, v in manifest["protocol"]["code"].items()}
        if digest(ROOT / model_path) != recorded[model_path]:
            raise ValueError("Neural architecture code differs from training")
        with np.load(folder / "context.npz") as z:
            context = {k: z[k] for k in z.files}
        with np.load(lag_results / "per_chunk" / f"{corpus}_evaluation.npz") as z:
            for name in ("evaluation", "mask"):
                np.testing.assert_array_equal(context[name], z[name])
            context["source_ids"] = z["source_ids"].copy()
            context["unigram_rank"] = z["unigram_target_rank"].copy()
        ids, mask, prior = (context[k] for k in ("evaluation", "mask", "prior"))
        rows = np.nonzero(mask)[0]
        targets = ids[mask]
        order = np.lexsort((np.arange(len(prior)), -prior))
        prior_rank = np.empty(len(prior), dtype=np.int32)
        prior_rank[order] = np.arange(1, len(prior)+1)
        np.testing.assert_array_equal(context["unigram_rank"], prior_rank[targets])
        context["baseline_accuracy"] = {k: float(chunk_mean(prior_rank[targets] <= k, rows, len(ids)).mean()) for k in KS}
        context["baseline_mse"] = float(chunk_mean(1-2*prior[targets]+prior@prior, rows, len(ids)).mean())
        contexts[corpus] = (config, context)
        evidence[corpus] = dict(protocol_sha256=manifest["protocol_sha256"], context_sha256=context_hash,
            manifest_sha256=digest(folder / "manifest.json"), evaluation_chunks=len(ids),
            source_count=len(set(context["source_ids"])), mask_rate=config["mask_rate"])
        sizes = config["corpus_sizes"] if requested_sizes is None else requested_sizes
        updates = config["training"]["checkpoints"] if requested_updates is None else requested_updates
        if not set(sizes) <= set(config["corpus_sizes"]) or not set(updates) <= set(config["training"]["checkpoints"]):
            raise ValueError("Requested size or training stage is absent from the training protocol")
        for n in sizes:
            for steps in updates:
                for pair in range(1, config["pairs"]+1):
                    score_path = folder / "scores" / f"pair{pair}_n{n}_step{steps}.json"
                    score = read_json(score_path)
                    if score["identity"]["context_sha256"] != context_hash:
                        raise ValueError("Saved MSE used a different context")
                    if not np.isclose(score["baseline_loss"], context["baseline_mse"], rtol=1e-12, atol=1e-12):
                        raise ValueError("Unigram MSE mismatch")
                    identities = []
                    for arm in ("a", "b"):
                        model_dir = folder / "models" / f"pair{pair}_{arm}_n{n}"
                        complete = read_json(model_dir / "complete.json")
                        checkpoint = model_dir / f"step_{steps}.pt"
                        checkpoint_hash = digest(checkpoint)
                        if checkpoint_hash != complete["sha256"][checkpoint.name] or checkpoint_hash != score["identity"][f"checkpoint_{arm}_sha256"]:
                            raise ValueError("Checkpoint does not match saved scores")
                        identity = complete["identity"]
                        if identity["protocol_sha256"] != manifest["protocol_sha256"]:
                            raise ValueError("Checkpoint protocol mismatch")
                        identities.append(identity)
                        jobs.append(dict(corpus=corpus, n=n, updates=steps, pair=pair, arm=arm,
                            checkpoint=str(checkpoint.resolve()), checkpoint_sha256=checkpoint_hash,
                            context_sha256=context_hash, score_sha256=digest(score_path),
                            expected_mse=score["per_chunk"][f"loss_{arm}"]))
                    if identities[0]["parent_sha256"] != identities[1]["parent_sha256"] or identities[0]["seed_offset"] != identities[1]["seed_offset"]:
                        raise ValueError("A/B initialization or RNG seed differs")
    return jobs, contexts, evidence


def collect_summary(records, contexts, lag_results):
    with (lag_results / "summary.csv").open(newline="") as stream:
        lag = list(csv.DictReader(stream))
    summaries = []
    for corpus, n, updates in sorted({(r["corpus"], r["n"], r["updates"]) for r in records}):
        config, context = contexts[corpus]
        cell = [r for r in records if (r["corpus"], r["n"], r["updates"]) == (corpus, n, updates)]
        for k in KS:
            selected = [r for r in cell if r["k"] == k]
            expected = {(p, a) for p in range(1, config["pairs"]+1) for a in ("a", "b")}
            if {(r["pair"], r["arm"]) for r in selected} != expected:
                continue
            baseline = context["baseline_accuracy"][k]
            accuracy = float(np.mean([r["accuracy"] for r in selected]))
            mse = float(np.mean([r["mse"] for r in selected]))
            comparison = next(r for r in lag if r["corpus"] == corpus and int(r["n"]) == n and int(r["k"]) == k)
            np.testing.assert_allclose(baseline, float(comparison["unigram_accuracy"]), rtol=1e-12, atol=1e-12)
            summaries.append(dict(corpus=corpus, n=n, updates=updates, k=k, pairs=config["pairs"],
                mdlm_accuracy=accuracy, reconstructor_accuracy=float(comparison["reconstructor_accuracy"]),
                unigram_accuracy=baseline, mdlm_relative_topk_error_reduction=relative_error_reduction(accuracy, baseline),
                reconstructor_relative_topk_error_reduction=float(comparison["relative_topk_error_reduction"]),
                mdlm_mse=mse, unigram_mse=context["baseline_mse"],
                mdlm_relative_mse_improvement=1-mse/context["baseline_mse"]))
    return summaries


def plot_summary(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter, ScalarFormatter
    corpora = [c for c in CORPORA if any(r["corpus"] == c for r in summary)]
    sizes = sorted({r["n"] for r in summary})
    stages = sorted({r["updates"] for r in summary})
    labels = dict(tinystories="TinyStories", wikitext="WikiText-103", cnn_dailymail="CNN/DailyMail")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    for metric, ylabel in (("accuracy", "Top-k accuracy"),
                          ("relative_topk_error_reduction", "Relative top-k error reduction")):
        fig, axes = plt.subplots(len(sizes), len(corpora), squeeze=False,
            figsize=(3.75*len(corpora), 2.65*len(sizes)+1.0), sharex=True, sharey=True)
        fig.subplots_adjust(left=.09, right=.98, top=.9, bottom=.09, hspace=.27, wspace=.17)
        handles = []
        for row, n in enumerate(sizes):
            for col, corpus in enumerate(corpora):
                ax = axes[row, col]
                cell = [r for r in summary if r["corpus"] == corpus and r["n"] == n]
                if not cell:
                    ax.set_visible(False)
                    continue
                for stage_index, stage in enumerate(stages):
                    values = sorted((r for r in cell if r["updates"] == stage), key=lambda r: r["k"])
                    if len(values) != len(KS):
                        continue
                    h, = ax.plot(KS, [r[f"mdlm_{metric}"] for r in values], color="#245A81",
                        linestyle="--" if stage_index == 0 and len(stages)>1 else "-",
                        marker="o", markersize=4,
                        markerfacecolor="white" if stage_index == 0 and len(stages)>1 else "#245A81",
                        label=f"MDLM: {stage:,} updates")
                    if row == 0 and col == 0:
                        handles.append(h)
                first = sorted((r for r in cell if r["updates"] == min(r["updates"] for r in cell)), key=lambda r: r["k"])
                if len(first) != len(KS):
                    continue
                h, = ax.plot(KS, [r[f"reconstructor_{metric}"] for r in first],
                    color="#B85D24", marker="s", markersize=4, label="Reconstructor: radius 2")
                if row == 0 and col == 0:
                    handles.append(h)
                base = [r["unigram_accuracy"] for r in first] if metric == "accuracy" else np.zeros(len(KS))
                h, = ax.plot(KS, base, color="#666666", linestyle=":", label="Unigram")
                if row == 0 and col == 0:
                    handles.append(h)
                ax.set_xscale("log", base=2)
                ax.set_xticks(KS)
                ax.xaxis.set_major_formatter(ScalarFormatter())
                ax.yaxis.set_major_formatter(PercentFormatter(1))
                ax.grid(axis="y", color="#E4E4E4", linewidth=.6)
                ax.set_title(f"{labels[corpus]} | {n:,} chunks", fontsize=10)
                if col == 0:
                    ax.set_ylabel(ylabel)
                if row == len(sizes)-1:
                    ax.set_xlabel("Top k")
        if metric == "accuracy":
            axes[0, 0].set_ylim(bottom=0, top=1)
        fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False)
        fig.text(.5, .017, "Same evaluation tokens, masks and unigram. MDLM uses full context. Means over three A/B pairs.",
                 ha="center", fontsize=9)
        for extension in ("png", "pdf"):
            fig.savefig(output / f"topk_{metric}.{extension}", dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "results/mdlm_runpod_main_v1")
    parser.add_argument("--lag-results", type=Path, default=ROOT / "results/lag_topk_matched_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "results/mdlm_topk_matched_v1")
    parser.add_argument("--corpora", nargs="+", choices=CORPORA, default=list(CORPORA))
    parser.add_argument("--sizes", nargs="+", type=int)
    parser.add_argument("--updates", nargs="+", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--max-models", type=int, help="Partial software check, not a complete experiment")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.max_models is not None and args.max_models < 1:
        parser.error("--max-models must be positive")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    jobs, contexts, evidence = prepare_jobs(args.run, args.lag_results, args.corpora, args.sizes, args.updates)
    args.output.mkdir(parents=True, exist_ok=True)
    contract = dict(k=list(KS), jobs=jobs, contexts=evidence, script_sha256=digest(Path(__file__)),
        lag_summary_sha256=digest(args.lag_results / "summary.csv"),
        ranking_script_sha256=digest(ROOT / "scripts/evaluate_lag_topk.py"),
        torch=torch.__version__, numpy=np.__version__, device=args.device, model_limit=args.max_models,
        ranking="Descending content-token logits, ascending token ID for exact ties. EOS included. MASK/PAD excluded.",
        aggregation="Masked-token mean within a chunk, equal chunk means, equal A/B means, equal pair means.",
        scope="MDLM uses full 64-token context. Reconstructor uses radius two. No neural forecast calibration is claimed.")
    signature = object_digest(contract)
    plan_path = args.output / "analysis_plan.json"
    if plan_path.exists() and read_json(plan_path)["signature"] != signature:
        raise ValueError("Evaluation settings changed. Use a new output directory.")
    write_json(plan_path, dict(signature=signature, **contract))
    if args.prepare_only:
        write_json(args.output / "progress.json", dict(status="prepared", verified_models=len(jobs), utc=utc()))
        print(f"Verified {len(jobs)} checkpoints and matched evaluation contexts. No inference run.")
        return
    device = torch.device(args.device)
    records, checked = [], []
    selected_jobs = jobs if args.max_models is None else jobs[:args.max_models]
    for index, job in enumerate(selected_jobs, 1):
        config, context = contexts[job["corpus"]]
        ids, mask = context["evaluation"], context["mask"]
        rows = np.nonzero(mask)[0]
        stem = f"{job['corpus']}_pair{job['pair']}_{job['arm']}_n{job['n']}_step{job['updates']}"
        dest = args.output / "per_model" / f"{stem}.npz"
        dest.parent.mkdir(exist_ok=True)
        start = time.perf_counter()
        write_json(args.output / "progress.json", dict(status="running", model=stem, completed=index-1, planned=len(jobs), utc=utc()))
        if dest.exists():
            with np.load(dest) as z:
                if str(z["signature"]) != signature:
                    raise ValueError("Cached inference signature changed")
                result = {k: z[k] for k in ("target_rank", "best_tie_rank", "worst_tie_rank", "mse_by_chunk")}
        else:
            model = load_model(config, Path(job["checkpoint"]), device)
            result = score_checkpoint(model, ids, mask, config, device)
            del model
            np.testing.assert_allclose(result["mse_by_chunk"], job["expected_mse"], rtol=1e-5, atol=1e-6)
            np.savez_compressed(dest, **result, source_ids=context["source_ids"], rows=rows,
                targets=ids[mask], signature=np.array(signature), checkpoint_sha256=np.array(job["checkpoint_sha256"]))
        np.testing.assert_allclose(result["mse_by_chunk"], job["expected_mse"], rtol=1e-5, atol=1e-6)
        for k in KS:
            accuracy = float(chunk_mean(result["target_rank"] <= k, rows, len(ids)).mean())
            records.append(dict(corpus=job["corpus"], n=job["n"], updates=job["updates"], pair=job["pair"], arm=job["arm"], k=k,
                accuracy=accuracy, unigram_accuracy=context["baseline_accuracy"][k],
                relative_topk_error_reduction=relative_error_reduction(accuracy, context["baseline_accuracy"][k]),
                mse=float(result["mse_by_chunk"].mean()),
                accuracy_tie_min=float(chunk_mean(result["worst_tie_rank"] <= k, rows, len(ids)).mean()),
                accuracy_tie_max=float(chunk_mean(result["best_tie_rank"] <= k, rows, len(ids)).mean())))
        checked.append(dict(model=stem, max_mse_difference=float(np.max(np.abs(result["mse_by_chunk"]-job["expected_mse"]))), seconds=time.perf_counter()-start))
        print(f"Scored {index}/{len(selected_jobs)}: {stem}", flush=True)
    write_csv(args.output / "per_model.csv", records)
    summary = collect_summary(records, contexts, args.lag_results)
    if summary:
        write_csv(args.output / "summary.csv", summary)
        plot_summary(summary, args.output)
    write_json(args.output / "summary.json", summary)
    status = "complete" if len(selected_jobs) == len(jobs) else "partial_software_check"
    write_json(args.output / "validation.json", dict(status=status, signature=signature, models=len(checked),
        checks=checked, per_chunk_mse_tolerance="rtol=1e-5, atol=1e-6 for cross-device numerical differences",
        outputs_sha256={p.relative_to(args.output).as_posix(): digest(p) for p in args.output.rglob("*")
                        if p.is_file() and p.name not in ("validation.json", "progress.json")}))
    write_json(args.output / "progress.json", dict(status=status, completed=len(selected_jobs), planned=len(jobs), utc=utc()))


if __name__ == "__main__":
    main()
