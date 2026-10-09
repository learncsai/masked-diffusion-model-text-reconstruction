"""Independently recompute the reported MDLM top-k summaries from saved ranks."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/mdlm_topk_matched_v1")
    args = parser.parse_args()
    out = args.output
    plan = json.loads((out / "analysis_plan.json").read_text())
    report = json.loads((out / "validation.json").read_text())
    assert report["status"] == "complete"
    for name, expected in report["outputs_sha256"].items():
        assert sha(out / name) == expected, name
    with (out / "per_model.csv").open(newline="") as stream:
        models = list(csv.DictReader(stream))
    with (out / "summary.csv").open(newline="") as stream:
        summary = list(csv.DictReader(stream))
    means = {}
    for job in plan["jobs"]:
        corpus, n, updates, pair, arm = (job[k] for k in ("corpus", "n", "updates", "pair", "arm"))
        name = f"{corpus}_pair{pair}_{arm}_n{n}_step{updates}"
        context_path = Path(job["checkpoint"]).parents[2] / "context.npz"
        assert sha(context_path) == job["context_sha256"]
        with np.load(context_path) as z:
            mask, ids, prior = z["mask"], z["evaluation"], z["prior"]
        with np.load(out / "per_model" / f"{name}.npz") as z:
            assert str(z["signature"]) == plan["signature"]
            assert str(z["checkpoint_sha256"]) == job["checkpoint_sha256"]
            rank, best, worst, mse = (z[k] for k in ("target_rank", "best_tie_rank", "worst_tie_rank", "mse_by_chunk"))
            np.testing.assert_array_equal(z["targets"], ids[mask])
            np.testing.assert_array_equal(z["rows"], np.where(mask)[0])
            assert len(z["source_ids"]) == len(ids)
            assert np.all((1 <= best) & (best <= rank) & (rank <= worst) & (worst <= len(prior)))
            np.testing.assert_allclose(mse, job["expected_mse"], rtol=1e-5, atol=1e-6)
            chunk_rows = z["rows"].copy()
        baseline_loss = np.mean([np.mean(1 - 2*prior[ids[c][mask[c]]] + np.dot(prior, prior)) for c in range(len(ids))])
        order = sorted(range(len(prior)), key=lambda i: (-prior[i], i))
        prior_rank = np.empty(len(prior), dtype=int)
        prior_rank[order] = np.arange(1, len(prior)+1)
        previous = -1.
        for k in plan["k"]:
            accuracy = np.mean([np.mean(rank[chunk_rows == c] <= k) for c in range(len(ids))])
            base = np.mean([np.mean(prior_rank[ids[c][mask[c]]] <= k) for c in range(len(ids))])
            assert accuracy >= previous
            previous = accuracy
            key = (corpus, n, updates, pair, arm, k)
            means[key] = dict(accuracy=float(accuracy), baseline=float(base), mse=float(mse.mean()),
                              baseline_mse=float(baseline_loss))
            matches = [r for r in models if r["corpus"] == corpus and int(r["n"]) == n
                       and int(r["updates"]) == updates and int(r["pair"]) == pair
                       and r["arm"] == arm and int(r["k"]) == k]
            assert len(matches) == 1
            row = matches[0]
            np.testing.assert_allclose([float(row["accuracy"]), float(row["unigram_accuracy"]), float(row["mse"]),
                float(row["relative_topk_error_reduction"])], [accuracy, base, mse.mean(), 1-(1-accuracy)/(1-base)], atol=1e-12, rtol=1e-12)
    for row in summary:
        corpus, n, updates, k = row["corpus"], int(row["n"]), int(row["updates"]), int(row["k"])
        group = [v for (c, size, stage, pair, arm, cutoff), v in means.items()
                 if (c, size, stage, cutoff) == (corpus, n, updates, k)]
        assert len(group) == 2*int(row["pairs"])
        accuracy = np.mean([v["accuracy"] for v in group])
        mse = np.mean([v["mse"] for v in group])
        baseline, baseline_mse = group[0]["baseline"], group[0]["baseline_mse"]
        expected = [accuracy, mse, baseline, baseline_mse, 1-(1-accuracy)/(1-baseline), 1-mse/baseline_mse]
        actual = [float(row[k]) for k in ("mdlm_accuracy", "mdlm_mse", "unigram_accuracy", "unigram_mse",
                                         "mdlm_relative_topk_error_reduction", "mdlm_relative_mse_improvement")]
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert len(models) == len(means)
    result = dict(status="passed", model_checkpoints=len(plan["jobs"]), per_model_rows=len(models),
                  summary_rows=len(summary), validator_sha256=sha(Path(__file__)),
                  checks="Output hashes, masks, targets, ranks and ties, equal chunk weighting, MSE reconciliation, top-k monotonicity, summary aggregation")
    (out / "independent_validation.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
