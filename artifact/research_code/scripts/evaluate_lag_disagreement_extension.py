"""Reproduce the 4096/8192-chunk disagreement comparison from portable inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import evaluate_lag_disagreement as base
from scripts.evaluate_lag_disagreement import (CORPORA, CORE, LABELS, METHODS, lag, offset_list,
    per_sequence_squared_error, read_json, sha, source_groups, source_influence_disagreement,
    summarize, utc, validate_records, write_csv, write_json)

SIZES = (4096, 8192)


def load_context(folder):
    with np.load(folder / "context.npz") as c:
        context = tuple(c[k] for k in ("evaluation", "mask", "prior", "gram", "cross"))
    with np.load(folder / "reference_original.npz") as z:
        reference = z["ids"]
    records = read_json(folder / "reference_original.json")
    validate_records(reference, records)
    evaluation_records = read_json(folder / "evaluation_records.json")
    validate_records(context[0], evaluation_records)
    config = read_json(folder / "original_manifest.json")["protocol"]["config"]
    assert config["radius"] == 2 and config["classes"] == 50257 and config["mask_rate"] == .5
    return context, reference, records, evaluation_records, config


def run(inputs, out):
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    (out / "per_chunk").mkdir(exist_ok=True)
    manifest = read_json(inputs / "manifest.json")
    for rel, digest in manifest["inputs_sha256"].items():
        assert sha(inputs / rel) == digest, rel
    for rel, digest in manifest["core_sha256"].items():
        assert sha(ROOT / rel) == digest, rel
    contract = dict(input_manifest_sha256=sha(inputs / "manifest.json"),
        code_sha256={p:sha(ROOT / p) for p in [*CORE, "scripts/evaluate_lag_disagreement.py", "scripts/evaluate_lag_disagreement_extension.py"]},
        corpora=list(CORPORA), sizes=list(SIZES), methods=list(METHODS), pairs=3,
        reference_chunks=2048, evaluation_chunks=128,
        status="Exploratory extension prompted by the 2048-chunk comparison. Existing 4096 prefixes and newly collected 8192 suffixes. No parameter tuning on new outcomes.",
        metrics="Mean prediction / mean observation. Mean absolute pair log error. Mean within-pair Pearson and Spearman across evaluation chunks.",
        uncertainty="Three pairs sharing reference, auxiliary data and evaluation inputs. No independence of chunks is assumed for inference. No confidence intervals.",
        source_pool="New suffixes use a larger 64000-source prefix. Prefix composition can affect size comparisons. WikiText source units are paragraphs, not whole articles.")
    signature = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    plan = out / "analysis_plan.json"
    if plan.exists():
        assert read_json(plan)["signature"] == signature, "Use a new output folder after changing the analysis"
    else:
        write_json(plan, dict(saved_utc=utc(), signature=signature, **contract))
    # Freeze all reference-only predictions before loading any A/B tokens.
    for corpus in CORPORA:
        folder = inputs / corpus
        (ids, mask, prior, gram, cross), reference, rec, eval_rec, config = load_context(folder)
        offsets = offset_list(config["radius"])
        source_ids = np.array([r["document_id"] for r in eval_rec])
        groups = source_groups(rec)
        kw = dict(smoothing=config["smoothing"], ridge=config["ridge"])
        for n in SIZES:
            path = out / "per_chunk" / f"{corpus}_n{n}_forecasts.npz"
            if path.exists():
                with np.load(path) as z:
                    assert str(z["signature"]) == signature
                continue
            print(f"Forecasting {corpus} n={n}: multinomial", flush=True)
            multi = base.multinomial_forecast(reference, ids, mask, prior, offsets, gram, cross, n=n, **kw)
            print(f"Forecasting {corpus} n={n}: source, same lag", flush=True)
            same = source_influence_disagreement(reference, groups, ids, mask, prior, offsets, gram, cross, n=n, same_lag=True, **kw)
            print(f"Forecasting {corpus} n={n}: source, all lags", flush=True)
            full = source_influence_disagreement(reference, groups, ids, mask, prior, offsets, gram, cross, n=n, same_lag=False, **kw)
            forecasts = np.array([multi, same, full])
            assert forecasts.shape == (3, 128) and np.isfinite(forecasts).all() and (forecasts >= 0).all()
            np.savez_compressed(path, forecasts=forecasts, methods=np.array(METHODS), source_ids=source_ids,
                signature=np.array(signature), saved_utc=np.array(utc()))
    all_summary, all_pairs, all_chunks, checks, audits = [], [], [], [], {}
    for corpus in CORPORA:
        folder = inputs / corpus
        (ids, mask, prior, gram, cross), reference, rec, eval_rec, config = load_context(folder)
        offsets = offset_list(config["radius"])
        source_ids = np.array([r["document_id"] for r in eval_rec])
        role_audit = read_json(folder / "source_audit.json")
        text_hashes = read_json(folder / "source_text_hashes.json")
        excluded = read_json(folder / "excluded_sources.json")
        excluded_ids, excluded_texts, excluded_chunks = (set(excluded[k]) for k in ("source_ids", "text_hashes", "chunk_hashes"))
        role_sets, text_sets = [], []
        for role in role_audit["roles"].values():
            current = set(role["source_ids"])
            texts = {text_hashes[s] for s in current}
            assert all(not current & old for old in role_sets)
            assert all(not texts & old for old in text_sets)
            role_sets.append(current)
            text_sets.append(texts)
        assert {r["document_id"] for r in rec} == set(role_audit["roles"]["reference"]["source_ids"])
        assert set(source_ids) == set(role_audit["roles"]["evaluation"]["source_ids"])
        tokens = {}
        chunk_sets = [{r["sha256"] for r in rec}, {r["sha256"] for r in eval_rec}]
        for pair in (1, 2, 3):
            for arm in ("a", "b"):
                name = f"pair{pair}_{arm}"
                with np.load(folder / f"{name}.npz") as z:
                    tokens[pair, arm] = z["ids"]
                records = read_json(folder / f"{name}.json")
                validate_records(tokens[pair, arm], records)
                assert tokens[pair, arm].shape == (8192, 64)
                assert {r["document_id"] for r in records} == set(role_audit["roles"][name]["source_ids"])
                hashes = {r["sha256"] for r in records}
                assert len(hashes) == 8192
                assert all(not hashes & old for old in chunk_sets)
                chunk_sets.append(hashes)
                added = records[4096:]
                assert not {r["document_id"] for r in added} & excluded_ids
                assert not {text_hashes[r["document_id"]] for r in added} & excluded_texts
                assert not {r["sha256"] for r in added} & excluded_chunks
        audits[corpus] = dict(reference_chunks=len(reference), reference_sources=len(source_groups(rec)),
            evaluation_chunks=len(ids), evaluation_sources=len(set(source_ids)), source_text_and_chunk_overlaps=0,
            pairs={name: {k:v for k,v in role.items() if k != "source_ids"} for name,role in role_audit["roles"].items() if name.startswith("pair")})
        for n in SIZES:
            with np.load(out / "per_chunk" / f"{corpus}_n{n}_forecasts.npz") as z:
                forecasts = z["forecasts"]
                assert str(z["signature"]) == signature
            observed = []
            for pair in (1, 2, 3):
                outputs = []
                for arm in ("a", "b"):
                    tables = lag.conditional_tables(lag.pair_counts(tokens[pair, arm][:n], offsets, len(prior)), prior, config["smoothing"])
                    rows, targets, prediction = lag.masked_prediction(ids, mask, tables, prior, offsets, gram, cross, config["ridge"])
                    outputs.append(prediction)
                distance = lag.squared_difference(*outputs)
                actual = per_sequence_squared_error(rows, distance, len(ids))
                dense_errors = []
                for j in (0, len(rows)//2, len(rows)-1):
                    vectors = [o.residual[j].toarray().ravel() + o.prior_weight[j]*prior for o in outputs]
                    direct = np.sum((vectors[0]-vectors[1])**2)
                    np.testing.assert_allclose(distance[j], direct, rtol=1e-11, atol=1e-12)
                    dense_errors.append(abs(float(distance[j]-direct)))
                assert np.isfinite(actual).all() and (actual >= 0).all()
                observed.append(actual)
                checks.append(dict(corpus=corpus, n=n, pair=pair, max_dense_distance_difference=max(dense_errors)))
                for chunk, value in enumerate(actual):
                    all_chunks.append(dict(corpus=corpus, n=n, pair=pair, chunk=chunk, source_id=str(source_ids[chunk]),
                        observed=float(value), **{m:float(forecasts[j,chunk]) for j,m in enumerate(METHODS)}))
            observed = np.array(observed)
            np.savez_compressed(out / "per_chunk" / f"{corpus}_n{n}_observed.npz", observed=observed,
                source_ids=source_ids, signature=np.array(signature), saved_utc=np.array(utc()))
            pairs, summary = summarize(corpus, n, forecasts, observed)
            all_pairs.extend(pairs)
            all_summary.extend(summary)
            print(f"Scored {corpus} n={n}: " + ", ".join(f"{r['method']}={r['prediction_observation_ratio']:.4f}" for r in summary), flush=True)
    write_csv(out / "summary.csv", all_summary)
    write_json(out / "summary.json", all_summary)
    write_csv(out / "per_pair.csv", all_pairs)
    write_csv(out / "per_chunk.csv", all_chunks)
    table = []
    for corpus in CORPORA:
        for n in SIZES:
            group = [next(r for r in all_summary if (r["corpus"],r["n"],r["method"]) == (corpus,n,m)) for m in METHODS]
            table.append(f"{LABELS[corpus]} & {n:,} & {group[0]['observed_mean']:.5f} & " +
                " & ".join(f"{r['prediction_observation_ratio']:.3f}" for r in group) + " & " +
                " & ".join(f"{r['mean_pair_spearman']:.3f}" for r in group) + r" \\")
    (out / "table_rows.tex").write_text("\n".join(table) + "\n", encoding="utf-8")
    # Reuse the established plot style without changing the frozen helper file.
    base.SIZES = SIZES
    base.plot_summary(out, all_summary)
    write_json(out / "metadata.json", dict(completed_utc=utc(), signature=signature, runtime_seconds=time.perf_counter()-started,
        python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, source_audits=audits, checks=checks,
        summary_rows=len(all_summary), paired_conditions=len(checks), dense_distance_checks_per_pair=3,
        outputs_sha256={p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob("*"))
            if p.is_file() and p.name not in ("metadata.json", "README.md", "validation.json")}))
    print(f"Completed in {time.perf_counter()-started:.1f} seconds: {out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=ROOT / "results/lag_disagreement_extension_v1/inputs")
    parser.add_argument("--output", type=Path, default=ROOT / "results/lag_disagreement_extension_v1")
    args = parser.parse_args()
    run(args.inputs, args.output)
