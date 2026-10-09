"""Freeze matched contexts and disjoint reference extensions; no network access."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from diffusion_lm_rmt import sparse_categorical_lag as backend
from diffusion_lm_rmt.categorical_lag import offset_list, token_counts
from diffusion_lm_rmt.calibration_diagnostics import score_weights
from scripts.run_prospective_lag_pilot import load_split, check_disjoint, split_auxiliary
from scripts.summarize_full_vocabulary_analytic import load_matched


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


def case_name(run, a, b, q):
    return f"{run}_{a}_{b}_q{q:g}"


def prepare(config_path):
    config = json.loads(config_path.read_text())
    output = Path(config["prepared"])
    if (output / "complete.json").exists():
        raise FileExistsError(f"completed preparation already exists: {output}")
    if (output / "config.json").exists():
        assert json.loads((output / "config.json").read_text()) == config
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    sources = {str(config_path): digest(config_path)}
    base = json.loads(Path(config["base_runs"]["grid"]).read_text())
    names = [base["auxiliary_split"], base["reference_split"], base["evaluation_split"]]
    names += [f"replicate{i}_train_{s}" for i in base["experimental_replicates"] for s in ("a", "b")]
    data = {}
    manifests = {}
    for corpus, root in base["corpora"].items():
        data[corpus] = {n: load_split(Path(root), n) for n in names}
        check_disjoint(data[corpus])
        for n in names:
            path = Path(root) / f"{n}.pt"
            sources[str(path)] = digest(path)
        mp = Path(root) / "confirmatory_split_manifest.json"
        sources[str(mp)] = digest(mp)
        manifests[corpus] = json.loads(mp.read_text())
        (output / corpus).mkdir(exist_ok=True)

    # Replay EVERY original RNG slot, including unselected masks and law seeds.
    matched = []
    contexts = []
    for run, path in config["base_runs"].items():
        original = json.loads(Path(path).read_text())
        sources[path] = digest(path)
        old_rows, _ = load_matched(run)
        for old_path in (Path("results/full_vocabulary_v1") / run,
                         Path("results/full_vocabulary_v1") / f"analytic_{run}"):
            for filename in ("config.json", "metadata.json", "pair_results.csv"):
                sources[str(old_path / filename)] = digest(old_path / filename)
        rng = np.random.default_rng(original["seed"])
        offsets = offset_list(original["lag_radius"])
        for corpus in original["corpora"]:
            splits = data[corpus]
            aux, records = splits[original["auxiliary_split"]]
            fit, validation = split_auxiliary(aux.astype(np.int32), records, rng)
            prior = token_counts(fit, 50257)
            prior /= prior.sum()
            gram, cross = backend.fitting_equations(
                fit, validation, prior, offsets, original["smoothing"], .5, rng)
            ids, eval_records = splits[original["evaluation_split"]]
            ids = ids[:original["evaluation_chunks"]].astype(np.int32)
            size_pairs = original.get("size_pairs") or [[n, n] for n in original["chunk_counts"]]
            for a, b in size_pairs:
                for q in original["mask_rates"]:
                    mask = rng.random(ids.shape) < q
                    mask[~mask.any(axis=1), 0] = True
                    for _ in ([a] if a == b else [a, b]):
                        for _ in original["laws"]:
                            rng.integers(2**32)
                    if [run, a, b, q] not in config["cases"]:
                        continue
                    key = case_name(run, a, b, q)
                    selected = [r for r in old_rows if r["corpus"] == corpus
                                and (r["n_a"], r["n_b"], r["q"]) == (a, b, q)]
                    assert len(selected) == 2
                    for row in selected:
                        assert hashlib.sha256(mask.tobytes()).hexdigest() == row["mask_sha256"]
                        assert hashlib.sha256(fit.tobytes()).hexdigest() == row["auxiliary_fit_sha256"]
                        assert hashlib.sha256(json.dumps([r["sha256"] for r in eval_records[:len(ids)]]).encode()).hexdigest() == row["evaluation_sha256"]
                        predictions = []
                        for side, n in (("a", a), ("b", b)):
                            train, recs = splits[f"replicate{row['replicate']}_train_{side}"]
                            assert hashlib.sha256(json.dumps([r["sha256"] for r in recs[:n]]).encode()).hexdigest() == row[f"side_{side}_sha256"]
                            tables = backend.conditional_tables(backend.pair_counts(
                                train[:n].astype(np.int32), offsets, len(prior)), prior, original["smoothing"])
                            predictions.append(backend.masked_prediction(ids, mask, tables, prior,
                                offsets, gram, cross, original["stacking_ridge"])[2])
                        observed = float(score_weights(mask) @ backend.squared_difference(*predictions))
                        np.testing.assert_allclose(observed, row["observed"], rtol=1e-12, atol=1e-14)
                        matched.append(dict(corpus=corpus, case=key, replicate=row["replicate"],
                            observed=observed, original_analytic=float(row["analytic_source_influence_prediction"])))
                    target = output / corpus / f"{key}.npz"
                    np.savez_compressed(target, eval_ids=ids, mask=mask, prior=prior, gram=gram, cross=cross)
                    contexts.append(dict(corpus=corpus, case=key, run=run, n_a=a, n_b=b, q=q,
                        path=str(target), sha256=digest(target), smoothing=original["smoothing"],
                        radius=original["lag_radius"], ridge=original["stacking_ridge"]))
                    print(f"matched {corpus} {key}: inputs and both observed scores reproduced", flush=True)

    from transformers import AutoTokenizer
    from diffusion_lm_rmt.neural_data import _chunk_split
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"], local_files_only=True)
    vocab_hash = hashlib.sha256(json.dumps(sorted(tokenizer.get_vocab().items(), key=lambda x: x[1]),
        ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    banks = []
    for index, (corpus, raw_path) in enumerate(config["raw_corpora"].items()):
        assert vocab_hash == manifests[corpus]["tokenizer"]["vocabulary_sha256"]
        sources[raw_path] = digest(raw_path)
        documents = [json.loads(line) for line in Path(raw_path).read_text(encoding="utf-8").splitlines()]
        used = {r["document_id"] for _, recs in data[corpus].values() for r in recs}
        raw_by_id = {d["document_id"]: d for d in documents}
        used_hashes = {raw_by_id[s]["sha256"] for s in used}
        forbidden = {r["sha256"] for _, recs in data[corpus].values() for r in recs}
        eligible = [d for d in documents if d["document_id"] not in used and d["sha256"] not in used_hashes]
        np.random.default_rng(config["seed"] + index).shuffle(eligible)
        counts = dict(corpus=corpus, raw_sources=len(documents), retained_primary_sources=len(used),
                      eligible_unused_sources=len(eligible))
        collected = [data[corpus][base["reference_split"]]]
        for bank_i in range(config["fresh_banks"]):
            ids, attention, recs = _chunk_split(tokenizer, eligible, config["fresh_chunks_per_bank"],
                64, forbidden=forbidden, max_chunks_per_document=manifests[corpus].get("max_chunks_per_document"))
            assert len(ids) == config["fresh_chunks_per_bank"] and bool((attention == 1).all())
            new_ids = {r["document_id"] for r in recs}
            new_hashes = {raw_by_id[s]["sha256"] for s in new_ids}
            assert not (new_ids & used or new_hashes & used_hashes)
            assert not ({r["sha256"] for r in recs} & forbidden)
            used.update(new_ids)
            used_hashes.update(new_hashes)
            forbidden.update(r["sha256"] for r in recs)
            eligible = [d for d in eligible if d["document_id"] not in used and d["sha256"] not in used_hashes]
            collected.append((ids.numpy(), recs))
        bank_items = list(zip(["original", "fresh1", "fresh2"], collected))
        bank_items.append(("pooled", (np.concatenate([i for i, _ in collected]),
                                       sum([r for _, r in collected], []))))
        check_disjoint(dict(bank_items[:-1]))
        for label, (ids, records) in bank_items:
            path = output / corpus / f"reference_{label}.npz"
            np.savez_compressed(path, ids=ids.astype(np.int32))
            rp = path.with_suffix(".json")
            write_json(rp, records)
            banks.append(dict(corpus=corpus, bank=label, chunks=len(ids),
                sources=len({r["document_id"] for r in records}), path=str(path), sha256=digest(path),
                records_path=str(rp), records_sha256=digest(rp)))
        write_json(output / corpus / "source_audit.json", counts)
        print(f"reference banks {corpus}: {counts}", flush=True)
    write_json(output / "contexts.json", contexts)
    write_json(output / "banks.json", banks)
    write_json(output / "matched_observations.json", matched)
    for path in (Path(__file__), Path("src/diffusion_lm_rmt/neural_data.py")):
        sources[str(path)] = digest(path)
    write_json(output / "sources.json", sources)
    write_json(output / "complete.json", {"contexts": len(contexts), "banks": len(banks),
        "matched_observations_sha256": digest(output / "matched_observations.json")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/calibration_diagnostics_v1.json"))
    prepare(parser.parse_args().config)
