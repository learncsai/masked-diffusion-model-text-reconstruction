"""Append source-disjoint data to the frozen matched A/B corpora, up to 8192 chunks."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts.evaluate_lag_disagreement import CORPORA, CORE, read_json, sha, utc, validate_records, write_json


def prepare(corpus, out):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer
    from diffusion_lm_rmt.neural_data import CORPORA as DATASETS, _canonical_text, _chunk_split

    inputs = out / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    folder = inputs / corpus
    if (folder / "extension_complete.json").exists():
        print(f"Already prepared {corpus}", flush=True)
        return
    original = ROOT / "results/lag_disagreement_matched_v1/inputs"
    base_manifest = read_json(original / "manifest.json")
    for rel, digest in base_manifest["inputs_sha256"].items():
        if rel.startswith(corpus + "/"):
            assert sha(original / rel) == digest, rel
    for rel, digest in base_manifest["core_sha256"].items():
        assert sha(ROOT / rel) == digest, rel
    shutil.copytree(original / corpus, folder, dirs_exist_ok=True)
    follow = ROOT / "data/diffusion_llm/reproducibility_followup_v1" / corpus
    extended = ROOT / "data/diffusion_llm/full_vocabulary_4096_v1" / corpus
    complete = read_json(extended / "complete.json")
    assert sha(extended / "extensions.json") == complete["extensions_sha256"]
    assert sha(follow / "raw_documents.jsonl") == complete["raw_sha256"]
    extensions = read_json(extended / "extensions.json")
    protocol = read_json(folder / "original_manifest.json")["protocol"]
    split_path = next(ROOT / p for p in protocol["inputs"] if p.endswith("confirmatory_split_manifest.json"))
    assert sha(split_path) == protocol["inputs"][split_path.relative_to(ROOT).as_posix()]
    split = read_json(split_path)
    reserved_ids = {d["document_id"] for s in split["splits"].values() for d in s["documents"]}
    reserved_texts = {d["sha256"] for s in split["splits"].values() for d in s["documents"]}
    reserved_chunks = {d["sha256"] for s in split["splits"].values() for d in s["chunks"]}
    # Reserve every historical pair and extra reference bank, including pairs
    # that are not used in this three-pair comparison.
    record_paths = [*(follow / f"pair{p}_{s}.json" for p in range(1, 8) for s in ("a", "b")),
        *(ROOT / "data/diffusion_llm/calibration_diagnostics_v1" / corpus / f"reference_{s}.json"
          for s in ("original", "fresh1", "fresh2"))]
    for entry in extensions:
        path = ROOT / entry["records_path"]
        assert sha(path) == entry["records_sha256"]
        assert sha(ROOT / entry["path"]) == entry["sha256"]
        record_paths.append(path)
    for path in record_paths:
        rec = read_json(path)
        reserved_ids.update(r["document_id"] for r in rec)
        reserved_chunks.update(r["sha256"] for r in rec)
    old_raw = [json.loads(line) for line in (follow / "raw_documents.jsonl").read_text(encoding="utf-8").splitlines()]
    old_lookup = {d["document_id"]: d["sha256"] for d in old_raw}
    reserved_texts.update(old_lookup[s] for s in reserved_ids if s in old_lookup)
    raw_dir = ROOT / "data/diffusion_llm/lag_disagreement_extension_v1" / corpus
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path, provenance_path = raw_dir / "raw_documents.jsonl", raw_dir / "provenance.json"
    settings = DATASETS[corpus]
    if raw_path.exists() and provenance_path.exists():
        provenance = read_json(provenance_path)
        assert sha(raw_path) == provenance["raw_sha256"]
        raw = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
    else:
        revision = HfApi().dataset_info(settings["path"]).sha
        print(f"Fetching {corpus}: 64000 unique sources at {revision}", flush=True)
        stream = load_dataset(settings["path"], settings["name"], split=settings["split"], revision=revision, streaming=True)
        raw, seen = [], set()
        for index, row in enumerate(stream):
            text = _canonical_text(row[settings["text_field"]])
            if len(text) < settings["minimum_characters"] or (settings["drop_headings"] and text.startswith("=")):
                continue
            digest = hashlib.sha256(text.encode()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            raw.append(dict(document_id=f"{corpus}:{index}:{digest[:16]}", sha256=digest, text=text))
            if len(raw) % 16000 == 0:
                print(f"Fetched {corpus}: {len(raw)} unique sources", flush=True)
            if len(raw) == 64000:
                break
        assert len(raw) == 64000
        raw_path.write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in raw), encoding="utf-8")
        provenance = dict(dataset=settings, revision=revision, fetched_utc=utc(), raw_sources=len(raw), raw_sha256=sha(raw_path))
        write_json(provenance_path, provenance)
    assert raw[:len(old_raw)] == old_raw, "Dataset prefix changed"
    assert all(hashlib.sha256(d["text"].encode()).hexdigest() == d["sha256"] for d in raw)
    lookup = {d["document_id"]: d["sha256"] for d in raw}
    reserved_texts.update(lookup[s] for s in reserved_ids if s in lookup)
    write_json(folder / "excluded_sources.json", dict(source_ids=sorted(reserved_ids), text_hashes=sorted(reserved_texts), chunk_hashes=sorted(reserved_chunks)))
    eligible = [d for d in raw if d["document_id"] not in reserved_ids and d["sha256"] not in reserved_texts]
    initial_eligible = len(eligible)
    seed = 20261001 + CORPORA.index(corpus)
    np.random.default_rng(seed).shuffle(eligible)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "data/diffusion_llm/gpt2_tokenizer", local_files_only=True)
    vocab_sha = hashlib.sha256(json.dumps(sorted(tokenizer.get_vocab().items(), key=lambda x: x[1]), ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    assert vocab_sha == complete["tokenizer_sha256"] == split["tokenizer"]["vocabulary_sha256"]
    audit = read_json(folder / "source_audit.json")
    source_texts = read_json(folder / "source_text_hashes.json")
    source_texts.update(lookup)
    prefix_checks = []
    for pair in (1, 2, 3):
        for side in ("a", "b"):
            name = f"pair{pair}_{side}"
            with np.load(original / corpus / f"{name}.npz") as z:
                base = z["ids"]
            base_rec = read_json(original / corpus / f"{name}.json")
            entry = next(r for r in extensions if (r["pair"], r["side"]) == (pair, side))
            with np.load(ROOT / entry["path"]) as z:
                ext = z["ids"]
            ext_rec = read_json(ROOT / entry["records_path"])
            validate_records(base, base_rec)
            validate_records(ext, ext_rec)
            assert len(base) == len(ext) == 2048
            tokens, attention, rec = _chunk_split(tokenizer, eligible, 4096, 64, forbidden=reserved_chunks,
                max_chunks_per_document=split.get("max_chunks_per_document"))
            assert len(tokens) == 4096 and bool((attention == 1).all())
            ids_set = {r["document_id"] for r in rec}
            text_set = {lookup[s] for s in ids_set}
            chunk_set = {r["sha256"] for r in rec}
            assert not ids_set & reserved_ids and not text_set & reserved_texts and not chunk_set & reserved_chunks
            reserved_ids.update(ids_set)
            reserved_texts.update(text_set)
            reserved_chunks.update(chunk_set)
            eligible = [d for d in eligible if d["document_id"] not in reserved_ids and d["sha256"] not in reserved_texts]
            combined = np.concatenate((base, ext, tokens.numpy().astype(np.int32)))
            records = base_rec + ext_rec + rec
            validate_records(combined, records)
            np.testing.assert_array_equal(combined[:2048], base)
            np.testing.assert_array_equal(combined[2048:4096], ext)
            np.savez_compressed(folder / f"{name}.npz", ids=combined)
            write_json(folder / f"{name}.json", records)
            all_sources = {r["document_id"] for r in records}
            audit["roles"][name] = dict(chunks=8192, sources=len(all_sources), source_ids=sorted(all_sources),
                sources_at_4096=len({r["document_id"] for r in records[:4096]}))
            prefix_checks.append(dict(role=name, original_npz_sha256=sha(original / corpus / f"{name}.npz"),
                original_records_sha256=sha(original / corpus / f"{name}.json"), extension_entry=entry,
                original_prefix_unchanged=True, prefix_4096_unchanged=True, new_sources=len(ids_set)))
            print(f"Prepared {corpus} {name}: 8192 chunks, {len(all_sources)} sources", flush=True)
    role_ids, role_texts = [], []
    for role in audit["roles"].values():
        ids_set = set(role["source_ids"])
        text_set = {source_texts[s] for s in ids_set}
        assert all(not ids_set & old for old in role_ids)
        assert all(not text_set & old for old in role_texts)
        role_ids.append(ids_set)
        role_texts.append(text_set)
    audit["overlaps"] = 0
    write_json(folder / "source_audit.json", audit)
    write_json(folder / "source_text_hashes.json", source_texts)
    write_json(folder / "extension_complete.json", dict(prepared_utc=utc(), sizes=[4096,8192], seed=seed,
        corpus=corpus, original_inputs_manifest_sha256=sha(original / "manifest.json"), prefix_checks=prefix_checks,
        tokenizer_sha256=vocab_sha, max_chunks_per_document=split.get("max_chunks_per_document"),
        source_unit="paragraph" if corpus == "wikitext" else "story" if corpus == "tinystories" else "article",
        source_pool=provenance, original_32000_source_prefix_verified=True,
        initially_eligible_sources=initial_eligible, remaining_eligible_sources=len(eligible),
        source_text_and_chunk_overlaps=0, all_role_source_overlaps=0,
        preparation_code_sha256=sha(__file__), neural_data_sha256=sha(ROOT / "src/diffusion_lm_rmt/neural_data.py")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", choices=CORPORA)
    parser.add_argument("--output", type=Path, default=ROOT / "results/lag_disagreement_extension_v1")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plan = dict(sizes=[4096,8192], pairs=3, reference_chunks=2048, radius=2, seed=20261001,
        raw_source_prefix=64000, status="Exploratory extension prompted by the 2048-chunk results. Sizes and methods fixed before scoring the extension.",
        allocation="Keep the existing 2048 and 4096 prefixes. Append unused sources from a shuffled 64000-source prefix. Exclude all historical source IDs, text hashes, and chunk hashes.")
    if not (args.output / "design.json").exists():
        write_json(args.output / "design.json", dict(saved_utc=utc(), **plan))
    for corpus in (args.corpus,) if args.corpus else CORPORA:
        prepare(corpus, args.output)
    inputs = args.output / "inputs"
    if all((inputs / corpus / "extension_complete.json").exists() for corpus in CORPORA):
        write_json(inputs / "manifest.json", dict(prepared_utc=utc(), design=plan,
            inputs_sha256={p.relative_to(inputs).as_posix():sha(p) for p in sorted(inputs.rglob("*")) if p.is_file() and p.name != "manifest.json"},
            core_sha256={p:sha(ROOT / p) for p in CORE}))


if __name__ == "__main__":
    main()
