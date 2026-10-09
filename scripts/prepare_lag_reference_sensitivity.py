"""Freeze nested reference banks while preserving every matched A/B input."""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.evaluate_lag_disagreement import CORPORA, CORE, read_json, sha, utc, validate_records, write_json

OUT = ROOT / "results/lag_reference_sensitivity_v1"
OLD = ROOT / "results/lag_disagreement_extension_v1"


def prepare():
    from transformers import AutoTokenizer
    from diffusion_lm_rmt.neural_data import _chunk_split

    OUT.mkdir(parents=True, exist_ok=True)
    plan = dict(reference_sizes=[2048,4096,8192], corpus_sizes=[4096,8192], pairs=3,
        mask_rate=.5, radius=2, smoothing=5., ridge=.01, seed=20261004,
        allocation="Nested original + fresh1 + fresh2 + 2048 new chunks. The first 4096 contain original + fresh1. All reference sources are disjoint from fixed A/B and auxiliary/evaluation roles.",
        status="Exploratory reference sensitivity prompted by prior observed method differences. Sizes and methods set before calculating new forecasts. No outcome-based tuning.",
        diagnostic="Also calculate multinomial variance with its conditional collision recomputed after counts are scaled to target n. Original three methods unchanged.")
    if (OUT / "design.json").exists():
        assert {k:v for k,v in read_json(OUT / "design.json").items() if k != "saved_utc"} == plan
    else:
        write_json(OUT / "design.json", dict(saved_utc=utc(), **plan))
    old_manifest = read_json(OLD / "inputs/manifest.json")
    for rel,digest in old_manifest["inputs_sha256"].items():
        assert sha(OLD / "inputs" / rel) == digest, rel
    for rel,digest in old_manifest["core_sha256"].items():
        assert sha(ROOT / rel) == digest, rel
    banks_path = ROOT / "data/diffusion_llm/calibration_diagnostics_v1/banks.json"
    banks = read_json(banks_path)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "data/diffusion_llm/gpt2_tokenizer", local_files_only=True)
    vocab_sha = hashlib.sha256(json.dumps(sorted(tokenizer.get_vocab().items(),key=lambda x:x[1]),ensure_ascii=False,separators=(",",":")).encode()).hexdigest()
    for ci,corpus in enumerate(CORPORA):
        folder = OUT / "inputs" / corpus
        if (folder / "reference_prepared.json").exists():
            print(f"Already prepared {corpus}",flush=True)
            continue
        shutil.copytree(OLD / "inputs" / corpus, folder, dirs_exist_ok=True)
        info = read_json(folder / "extension_complete.json")
        assert vocab_sha == info["tokenizer_sha256"]
        raw_dir = ROOT / "data/diffusion_llm/lag_disagreement_extension_v1" / corpus
        provenance = read_json(raw_dir / "provenance.json")
        assert sha(raw_dir / "raw_documents.jsonl") == provenance["raw_sha256"]
        raw = [json.loads(line) for line in (raw_dir / "raw_documents.jsonl").read_text(encoding="utf-8").splitlines()]
        assert all(hashlib.sha256(d["text"].encode()).hexdigest()==d["sha256"] for d in raw)
        lookup = read_json(folder / "source_text_hashes.json")
        excluded = read_json(folder / "excluded_sources.json")
        reserved_ids,reserved_texts,reserved_chunks = (set(excluded[k]) for k in ("source_ids","text_hashes","chunk_hashes"))
        audit = read_json(folder / "source_audit.json")
        for role in audit["roles"].values():
            reserved_ids.update(role["source_ids"])
            reserved_texts.update(lookup[s] for s in role["source_ids"])
        for pair in (1,2,3):
            for arm in ("a","b"):
                rec = read_json(folder / f"pair{pair}_{arm}.json")
                reserved_chunks.update(r["sha256"] for r in rec)
        token_banks,record_banks,bank_provenance = [],[],[]
        for bank in ("original","fresh1","fresh2"):
            entry = next(r for r in banks if (r["corpus"],r["bank"])==(corpus,bank))
            path,rp = ROOT / entry["path"],ROOT / entry["records_path"]
            assert sha(path)==entry["sha256"] and sha(rp)==entry["records_sha256"]
            with np.load(path) as z:
                tokens=z["ids"]
            records=read_json(rp)
            validate_records(tokens,records)
            assert len(tokens)==2048
            token_banks.append(tokens)
            record_banks.extend(records)
            bank_provenance.append(entry)
            reserved_ids.update(r["document_id"] for r in records)
            reserved_texts.update(lookup[r["document_id"]] for r in records)
            reserved_chunks.update(r["sha256"] for r in records)
        eligible = [d for d in raw if d["document_id"] not in reserved_ids and d["sha256"] not in reserved_texts]
        initial = len(eligible)
        np.random.default_rng(plan["seed"]+ci).shuffle(eligible)
        tokens,attention,records = _chunk_split(tokenizer,eligible,2048,64,forbidden=reserved_chunks,
            max_chunks_per_document=info["max_chunks_per_document"])
        assert len(tokens)==2048 and bool((attention==1).all())
        assert not {r["document_id"] for r in records}&reserved_ids
        assert not {lookup[r["document_id"]] for r in records}&reserved_texts
        assert not {r["sha256"] for r in records}&reserved_chunks
        token_banks.append(tokens.numpy().astype(np.int32))
        record_banks.extend(records)
        reference=np.concatenate(token_banks)
        validate_records(reference,record_banks)
        assert len({r["sha256"] for r in record_banks})==8192
        other_ids=set().union(*(set(v["source_ids"]) for k,v in audit["roles"].items() if k!="reference"))
        other_texts={lookup[s] for s in other_ids}
        ref_ids={r["document_id"] for r in record_banks}
        assert not ref_ids&other_ids and not {lookup[s] for s in ref_ids}&other_texts
        with np.load(folder / "reference_original.npz") as z:
            np.testing.assert_array_equal(reference[:2048],z["ids"])
        assert record_banks[:2048]==read_json(folder / "reference_original.json")
        # Export known observations only for exact reconciliation. The new
        # runner rebuilds every A/B score from the unchanged token inputs.
        for n in plan["corpus_sizes"]:
            source=OLD / "per_chunk" / f"{corpus}_n{n}_observed.npz"
            shutil.copyfile(source,folder / f"baseline_n{n}_observed.npz")
            source=OLD / "per_chunk" / f"{corpus}_n{n}_forecasts.npz"
            shutil.copyfile(source,folder / f"baseline_n{n}_forecasts.npz")
        np.savez_compressed(folder / "reference_max.npz",ids=reference)
        write_json(folder / "reference_max.json",record_banks)
        write_json(folder / "reference_prepared.json",dict(prepared_utc=utc(),corpus=corpus,seed=plan["seed"]+ci,
            nested_sources={str(n):len({r["document_id"] for r in record_banks[:n]}) for n in plan["reference_sizes"]},
            initial_eligible_sources=initial,original_2048_reference_unchanged=True,source_text_chunk_overlap=0,
            existing_banks=bank_provenance,additional_source_ids=sorted({r["document_id"] for r in records}),
            raw_source_provenance=provenance,tokenizer_sha256=vocab_sha,
            preparation_code_sha256=sha(__file__),neural_data_sha256=sha(ROOT / "src/diffusion_lm_rmt/neural_data.py"),
            fixed_files_sha256={p.relative_to(folder).as_posix():sha(p) for p in folder.iterdir() if p.is_file() and
                (p.name.startswith("pair") or p.name in ("context.npz","evaluation_records.json"))}))
        print(f"Prepared {corpus}: nested 2048/4096/8192 references, no source/text/chunk overlap",flush=True)
    inputs=OUT / "inputs"
    write_json(inputs / "manifest.json",dict(prepared_utc=utc(),design=plan,old_input_manifest_sha256=sha(OLD / "inputs/manifest.json"),
        inputs_sha256={p.relative_to(inputs).as_posix():sha(p) for p in sorted(inputs.rglob("*")) if p.is_file() and p.name!="manifest.json"},
        core_sha256={p:sha(ROOT / p) for p in CORE}))


if __name__ == "__main__":
    prepare()
