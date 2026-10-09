"""Fresh source-disjoint replication using frozen full-vocabulary contexts."""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
from scipy.stats import t

from diffusion_lm_rmt import sparse_categorical_lag as backend
from diffusion_lm_rmt import sparse_categorical_influence as influence
from diffusion_lm_rmt.calibration_diagnostics import score_weights
from diffusion_lm_rmt.categorical_lag import offset_list, source_groups
from scripts.prepare_calibration_diagnostics import digest, write_json
from scripts.run_prospective_lag_pilot import load_split, check_disjoint


def interval(values, log=False):
    values = np.asarray(values, float)
    z = np.log(values) if log else values
    error = float(z.std(ddof=1) / np.sqrt(len(z)))
    center = float(z.mean())
    half = float(t.ppf(.975, len(z)-1) * error)
    transform = np.exp if log else float
    return dict(mean=float(transform(center)), lower=float(transform(center-half)),
                upper=float(transform(center+half)), standard_error=error,
                count=len(z), scale="log" if log else "linear")


def prepare(config, corpus):
    from transformers import AutoTokenizer
    from diffusion_lm_rmt.neural_data import fetch_documents, _chunk_split

    base = json.loads(Path(config["base_config"]).read_text())
    primary = json.loads(Path(base["base_runs"]["grid"]).read_text())
    destination = Path(config["prepared"]) / corpus
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "complete.json").exists():
        raise FileExistsError(destination)
    manifest_path = Path(primary["corpora"][corpus]) / "confirmatory_split_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    reserved_ids = {d["document_id"] for s in manifest["splits"].values() for d in s["documents"]}
    reserved_text = {d["sha256"] for s in manifest["splits"].values() for d in s["documents"]}
    reserved_chunks = {d["sha256"] for s in manifest["splits"].values() for d in s["chunks"]}
    calibration = Path(base["prepared"]) / corpus
    # Fresh calibration banks were sometimes drawn from assigned but unused
    # documents. Reserve their actual records explicitly as well.
    for name in ("reference_fresh1.json", "reference_fresh2.json"):
        records = json.loads((calibration / name).read_text())
        reserved_ids.update(r["document_id"] for r in records)
        reserved_chunks.update(r["sha256"] for r in records)
    tokenizer = AutoTokenizer.from_pretrained(base["tokenizer"], local_files_only=True)
    import hashlib
    vocab_sha = hashlib.sha256(json.dumps(sorted(tokenizer.get_vocab().items(), key=lambda x: x[1]),
        ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    assert vocab_sha == manifest["tokenizer"]["vocabulary_sha256"]
    raw_path = destination / "raw_documents.jsonl"
    documents = fetch_documents(raw_path, config["raw_document_count"], corpus)
    # Compare canonical content hashes rather than trusting stream positions.
    eligible = [d for d in documents if d["document_id"] not in reserved_ids
                and d["sha256"] not in reserved_text]
    np.random.default_rng(np.random.SeedSequence([config["seed"], list(primary["corpora"]).index(corpus), 1])).shuffle(eligible)
    original_ids, original_chunks = set(reserved_ids), set(reserved_chunks)
    original_text = set(reserved_text)
    records_by_side, pairs = {}, []
    source_lookup = {d["document_id"]: d["sha256"] for d in documents}
    max_size = max(config["fitted_sizes"])
    for pair in range(config["pairs"]):
        for side in ("a", "b"):
            ids, attention, records = _chunk_split(tokenizer, eligible, max_size, 64,
                forbidden=reserved_chunks, max_chunks_per_document=manifest.get("max_chunks_per_document"))
            if len(ids) != max_size:
                raise RuntimeError(f"insufficient unused data for {corpus} pair {pair+1} {side}")
            assert bool((attention == 1).all())
            current_ids = {r["document_id"] for r in records}
            current_text = {source_lookup[key] for key in current_ids}
            current_chunks = {r["sha256"] for r in records}
            assert not current_ids & reserved_ids
            assert not current_text & reserved_text
            assert not current_chunks & reserved_chunks
            name = f"pair{pair+1}_{side}"
            np.savez_compressed(destination / f"{name}.npz", ids=ids.numpy().astype(np.int32))
            write_json(destination / f"{name}.json", records)
            records_by_side[name] = (ids.numpy(), records)
            pairs.append(dict(pair=pair+1, side=side, chunks=len(ids), sources=len(current_ids),
                path=str(destination / f"{name}.npz"),
                sha256=digest(destination / f"{name}.npz"),
                records_path=str(destination / f"{name}.json"),
                records_sha256=digest(destination / f"{name}.json")))
            reserved_ids.update(current_ids)
            reserved_text.update(current_text)
            reserved_chunks.update(current_chunks)
            eligible = [d for d in eligible if d["document_id"] not in reserved_ids
                        and d["sha256"] not in reserved_text]
            print(f"prepared {corpus} {name}: {len(current_ids)} sources", flush=True)
    check_disjoint(records_by_side)
    write_json(destination / "pairs.json", pairs)
    write_json(destination / "config.json", config)
    write_json(destination / "complete.json", dict(corpus=corpus, raw_sources=len(documents),
        pairs=config["pairs"], source_unit="paragraph" if corpus == "wikitext" else "article" if corpus == "cnn_dailymail" else "story",
        excluded_source_ids=len(original_ids), excluded_source_hashes=len(original_text),
        excluded_chunk_hashes=len(original_chunks), new_source_count=len(reserved_ids-original_ids),
        all_source_and_chunk_overlap_checks_passed=True,
        tokenizer_sha256=vocab_sha, raw_sha256=digest(raw_path), manifest_sha256=digest(manifest_path),
        pairs_sha256=digest(destination / "pairs.json")))


def run(config, corpus):
    started = time.perf_counter()
    base = json.loads(Path(config["base_config"]).read_text())
    primary = json.loads(Path(base["base_runs"]["grid"]).read_text())
    prepared = Path(config["prepared"]) / corpus
    assert json.loads((prepared / "config.json").read_text()) == config
    assert json.loads((prepared / "complete.json").read_text())["all_source_and_chunk_overlap_checks_passed"]
    output = Path(config["output"]) / corpus
    output.mkdir(parents=True, exist_ok=False)
    (output / "code").mkdir()
    code = {}
    for path in (Path(__file__), Path(backend.__file__), Path(influence.__file__)):
        (output / "code" / path.name).write_bytes(path.read_bytes())
        code[str(path)] = digest(path)
    write_json(output / "config.json", config)
    context = Path(base["prepared"]) / corpus / f"{config['context_case']}.npz"
    with np.load(context) as saved:
        ids, mask, prior, gram, cross = [saved[k] for k in ("eval_ids", "mask", "prior", "gram", "cross")]
    weights = score_weights(mask)
    offsets = offset_list(primary["lag_radius"])
    alpha, ridge = primary["smoothing"], primary["stacking_ridge"]
    def predict(counts):
        return backend.masked_prediction(ids, mask, backend.conditional_tables(counts, prior, alpha),
            prior, offsets, gram, cross, ridge)
    forecasts = []
    for bank in config["reference_banks"]:
        path = Path(base["prepared"]) / corpus / f"reference_{bank}.npz"
        with np.load(path) as saved:
            reference = saved["ids"]
        records = json.loads(path.with_suffix(".json").read_text())
        groups = source_groups(records)
        for n in config["fitted_sizes"]:
            for law in ("full", "same_lag", "independent_chunks"):
                grouping = [np.asarray([j]) for j in range(len(reference))] if law == "independent_chunks" else groups
                value = influence.source_influence_disagreement(reference, grouping, ids, mask,
                    prior, offsets, gram, cross, n=n, smoothing=alpha, ridge=ridge, same_lag=law == "same_lag")
                forecasts.append(dict(corpus=corpus, bank=bank, reference_chunks=len(reference),
                    n=n, law=law, prediction=float(np.mean(value)), reference_sha256=digest(path)))
                print(f"forecast {corpus} {bank} n={n} {law}: {np.mean(value):.6g}", flush=True)
        anchor = next(r["prediction"] for r in forecasts if r["bank"] == bank and r["n"] == 512 and r["law"] == "full")
        for n in config["fitted_sizes"]:
            forecasts.append(dict(corpus=corpus, bank=bank, reference_chunks=len(reference),
                n=n, law="inverse_size", prediction=anchor * 512 / n, reference_sha256=digest(path)))
    # Freeze every forecast before accessing or scoring any new A/B tensor.
    write_json(output / "forecasts.json", forecasts)
    rows = []
    pairs = json.loads((prepared / "pairs.json").read_text())
    for pair in range(1, config["pairs"]+1):
        for n in config["fitted_sizes"]:
            predictions, skills, source_counts = [], [], []
            for side in ("a", "b"):
                entry = next(r for r in pairs if r["pair"] == pair and r["side"] == side)
                assert digest(entry["path"]) == entry["sha256"]
                assert digest(entry["records_path"]) == entry["records_sha256"]
                with np.load(entry["path"]) as saved:
                    train = saved["ids"][:n]
                records = json.loads(Path(entry["records_path"]).read_text())[:n]
                source_counts.append(len({r["document_id"] for r in records}))
                _, targets, prediction = predict(backend.pair_counts(train, offsets, len(prior)))
                baseline = 1 - 2 * prior[targets] + float(prior @ prior)
                skills.append(float(1 - (weights @ prediction.loss(targets)) / (weights @ baseline)))
                predictions.append(prediction)
            observed = float(weights @ backend.squared_difference(*predictions))
            rows.append(dict(corpus=corpus, pair=pair, n=n, q=config["mask_rate"], observed=observed,
                skill_a=skills[0], skill_b=skills[1], sources_a=source_counts[0], sources_b=source_counts[1]))
            write_json(output / "observations.json", rows)
            print(f"observed {corpus} pair={pair} n={n}: {observed:.6g}, skill={np.mean(skills):.4f}", flush=True)
    summary = []
    for forecast in forecasts:
        selected = [r for r in rows if r["n"] == forecast["n"]]
        ratios = [forecast["prediction"] / r["observed"] for r in selected]
        summary.append(dict(forecast, ratio=interval(ratios, log=True),
            mean_absolute_log_error=float(np.mean(np.abs(np.log(ratios)))),
            observed=interval([r["observed"] for r in selected]),
            skill=interval([.5*(r["skill_a"]+r["skill_b"]) for r in selected])))
    contrasts = [next(r["observed"] for r in rows if r["pair"] == pair and r["n"] == 512)
                 / next(r["observed"] for r in rows if r["pair"] == pair and r["n"] == 2048)
                 for pair in range(1, config["pairs"]+1)]
    write_json(output / "summary.json", summary)
    write_json(output / "size_contrast.json", interval(contrasts, log=True))
    write_json(output / "metadata.json", dict(complete=True, duration_seconds=time.perf_counter()-started,
        conditional_on_shared_reference_auxiliary_evaluation=True, forecast_before_new_ab_scores=True,
        nested_sizes=True, python=platform.python_version(), code_sha256=code,
        context_sha256=digest(context), prepared_manifest_sha256=digest(prepared / "complete.json"),
        outputs_sha256={name: digest(output / name) for name in
            ("forecasts.json", "observations.json", "summary.json", "size_contrast.json")},
        threads={key: os.environ.get(key) for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--config", type=Path, default=Path("configs/reproducibility_followup_v1.json"))
    parser.add_argument("--corpus", choices=("tinystories", "wikitext", "cnn_dailymail"), required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    (prepare if args.action == "prepare" else run)(config, args.corpus)
