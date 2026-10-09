"""Retrospective parameter-Jacobian diagnostic, not a reference forecast.

Use the saved A/B parameter difference to test the output linearization.
No parameters are optimized. Both endpoints and their midpoint are reported.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
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
from torch.func import functional_call, jvp
from torch.nn.attention import SDPBackend, sdpa_kernel
from diffusion_lm_rmt.matched_mdlm import digest, load_model, utc, write_json, read_json
from evaluate_mdlm_topk import CORPORA, prepare_jobs
from evaluate_lag_topk import write_csv


def vector_statistics(observed, tangent, chunk_rows, chunks):
    """Squared norms and inner products, with equal weight for each chunk."""
    observed, tangent = observed.double(), tangent.double()
    values = dict(observed=observed.square().sum(-1), linearized=tangent.square().sum(-1),
                  residual=(observed-tangent).square().sum(-1), inner=(observed*tangent).sum(-1))
    return {name: np.asarray([float(value[chunk_rows == i].mean()) for i in range(chunks)])
            for name, value in values.items()}


def check_pair(job_a, job_b, config, context, device, chunk_limit, batch_size):
    model = load_model(config, job_a["checkpoint"], device)
    model.eval()
    params_a = {name: value.detach() for name, value in model.named_parameters()}
    state_b = torch.load(job_b["checkpoint"], map_location=device, weights_only=True)["model"]
    params_b = {name: state_b[name].detach() for name in params_a}
    delta = {name: params_b[name]-value for name, value in params_a.items()}
    midpoint = {name: .5*(value+params_b[name]) for name, value in params_a.items()}
    anchors = dict(a=params_a, b=params_b, midpoint=midpoint)
    ids, mask = context["evaluation"][:chunk_limit], context["mask"][:chunk_limit]
    all_scores = {name: defaultdict(list) for name in anchors}
    finite_difference_checks = []
    max_probability_sum_error = 0.
    max_tangent_sum_error = 0.
    started = time.perf_counter()
    # Math attention supports forward derivatives. Disable the inference-only
    # MHA shortcut so the same differentiable computation is used throughout.
    with sdpa_kernel(SDPBackend.MATH):
        for start in range(0, len(ids), batch_size):
            clean = torch.as_tensor(ids[start:start+batch_size], device=device, dtype=torch.long)
            hidden = torch.as_tensor(mask[start:start+batch_size], device=device, dtype=torch.bool)
            corrupted = clean.masked_fill(hidden, config["mask_token_id"])
            attention = torch.ones_like(clean, dtype=torch.bool)
            times = torch.full((len(clean),), config["mask_rate"], device=device)
            rows = hidden.nonzero()[:, 0]

            def probabilities(parameters):
                logits = functional_call(model, parameters, (corrupted, attention, times))
                return logits[..., :config["classes"]][hidden].softmax(-1)

            with torch.no_grad():
                pa, pb = probabilities(params_a), probabilities(params_b)
                observed = pb-pa
                max_probability_sum_error = max(max_probability_sum_error,
                    float((pa.sum(-1)-1).abs().max()), float((pb.sum(-1)-1).abs().max()))
                for name, anchor in anchors.items():
                    _, tangent = jvp(probabilities, (anchor,), (delta,))
                    if not torch.isfinite(tangent).all():
                        raise FloatingPointError("Nonfinite parameter derivative")
                    max_tangent_sum_error = max(max_tangent_sum_error, float(tangent.sum(-1).abs().max()))
                    stats = vector_statistics(observed, tangent, rows, len(clean))
                    for key, value in stats.items():
                        all_scores[name][key].extend(value)
                    if start == 0:
                        for step in (.001, .0005):
                            plus = {key: value+step*delta[key] for key, value in anchor.items()}
                            minus = {key: value-step*delta[key] for key, value in anchor.items()}
                            fd = (probabilities(plus)-probabilities(minus))/(2*step)
                            relative_error = float(torch.linalg.vector_norm((fd-tangent).double()) /
                                                   torch.linalg.vector_norm(tangent.double()).clamp_min(1e-15))
                            finite_difference_checks.append(dict(anchor=name, step=step, relative_error=relative_error))
                            if relative_error > .005:
                                raise AssertionError(f"Directional derivative check failed: {relative_error}")
    scores = {name: {key: np.asarray(value) for key, value in stats.items()}
              for name, stats in all_scores.items()}
    expected = read_json(Path(job_a["checkpoint"]).parents[2] / "scores" /
                         f"pair{job_a['pair']}_n{job_a['n']}_step{job_a['updates']}.json")["per_chunk"]["disagreement"][:len(ids)]
    np.testing.assert_allclose(scores["a"]["observed"], expected, rtol=2e-5, atol=2e-7)
    return scores, dict(finite_difference=finite_difference_checks,
        max_original_disagreement_error=float(np.max(np.abs(scores["a"]["observed"]-expected))),
        max_probability_sum_error=max_probability_sum_error, max_tangent_sum_error=max_tangent_sum_error,
        parameter_count=model.parameter_count, elapsed_seconds=time.perf_counter()-started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "results/mdlm_runpod_main_v1")
    parser.add_argument("--lag-results", type=Path, default=ROOT / "results/lag_topk_matched_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "results/mdlm_linearization_check_v1")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--corpora", nargs="+", choices=CORPORA, default=list(CORPORA))
    parser.add_argument("--sizes", nargs="+", type=int, default=[2048])
    parser.add_argument("--updates", nargs="+", type=int, default=[8000, 12000])
    parser.add_argument("--chunks", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-pairs", type=int)
    args = parser.parse_args()
    if args.chunks < 1 or args.batch_size < 1:
        parser.error("chunks and batch size must be positive")
    device = torch.device(args.device)
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    jobs, contexts, evidence = prepare_jobs(args.run, args.lag_results, args.corpora, args.sizes, args.updates)
    paired = [(a, next(b for b in jobs if all(a[k] == b[k] for k in ("corpus", "n", "updates", "pair")) and b["arm"] == "b"))
              for a in jobs if a["arm"] == "a"]
    selected = paired if args.max_pairs is None else paired[:args.max_pairs]
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "analysis_plan.json").exists():
        raise FileExistsError("Use a new output directory for a new diagnostic")
    write_json(args.output / "analysis_plan.json", dict(created_utc=utc(),
        scope="Retrospective output linearization using actual A/B parameters. Not a source forecast.",
        formula="q_B-q_A approximately J_theta (theta_B-theta_A)", anchors=["a", "b", "midpoint"],
        normalization="Class sum, masked-position mean within chunk, then equal chunk and pair means.",
        chunks=args.chunks, total_pairs=len(paired), selected_pairs=len(selected), contexts=evidence,
        jobs=jobs, script_sha256=digest(__file__), torch=torch.__version__, device=str(device),
        device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        batch_size=args.batch_size, extra_training=False))
    records, checks = [], []
    started = time.perf_counter()
    for a, b in selected:
        config, context = contexts[a["corpus"]]
        if args.chunks > len(context["evaluation"]):
            raise ValueError("Requested more chunks than saved evaluation set")
        scores, validation = check_pair(a, b, config, context, device, args.chunks, args.batch_size)
        identity = {key: a[key] for key in ("corpus", "n", "updates", "pair")}
        key = f"{a['corpus']}_n{a['n']}_step{a['updates']}_pair{a['pair']}"
        np.savez_compressed(args.output / f"{key}.npz", source_ids=context["source_ids"][:args.chunks],
                            **{f"{anchor}_{metric}": values for anchor, score in scores.items() for metric, values in score.items()})
        for anchor, score in scores.items():
            means = {metric: float(values.mean()) for metric, values in score.items()}
            records.append(dict(**identity, anchor=anchor, chunks=args.chunks, **means,
                prediction_observation_ratio=means["linearized"]/means["observed"],
                relative_vector_error=np.sqrt(means["residual"]/means["observed"]),
                cosine=means["inner"]/np.sqrt(means["observed"]*means["linearized"])))
        checks.append(dict(**identity, **validation, npz_sha256=digest(args.output / f"{key}.npz")))
        write_csv(args.output / "per_pair.csv", records)
        write_json(args.output / "validation.json", dict(checks=checks))
        write_json(args.output / "progress.json", dict(status="running", completed_pairs=len(checks),
            selected_pairs=len(selected), **identity, elapsed_seconds=time.perf_counter()-started))
        print(json.dumps(dict(completed_pairs=len(checks), selected_pairs=len(selected), **identity,
                             elapsed_seconds=time.perf_counter()-started)), flush=True)
    summary = []
    for corpus, n, updates, anchor in sorted({(r["corpus"], r["n"], r["updates"], r["anchor"]) for r in records}):
        cell = [r for r in records if (r["corpus"], r["n"], r["updates"], r["anchor"]) == (corpus, n, updates, anchor)]
        means = {key: float(np.mean([r[key] for r in cell])) for key in ("observed", "linearized", "residual", "inner")}
        summary.append(dict(corpus=corpus, n=n, updates=updates, anchor=anchor, pairs=len(cell), **means,
            prediction_observation_ratio=means["linearized"]/means["observed"],
            relative_vector_error=np.sqrt(means["residual"]/means["observed"]),
            cosine=means["inner"]/np.sqrt(means["observed"]*means["linearized"])))
    write_csv(args.output / "summary.csv", summary)
    write_json(args.output / "progress.json", dict(status="complete" if len(selected) == len(paired) else "partial_scope",
        completed_pairs=len(checks), total_pairs=len(paired), elapsed_seconds=time.perf_counter()-started, finished_utc=utc()))


if __name__ == "__main__":
    main()
