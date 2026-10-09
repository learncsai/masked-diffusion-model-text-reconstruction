"""Document-level TinyStories preparation and stable split manifests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_text(text: str) -> str:
    return " ".join(text.replace("\r", "\n").split())


def prepare_tokenizer(directory: Path, model_name: str = "openai-community/gpt2") -> tuple[PreTrainedTokenizerBase, dict[str, Any]]:
    """Load GPT-2, append stable MASK/PAD tokens, and save its exact metadata."""
    directory.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    added = tokenizer.add_special_tokens({"mask_token": "[MASK]", "pad_token": "[PAD]"})
    if added not in {0, 2}:
        raise RuntimeError(f"unexpected number of added special tokens: {added}")
    tokenizer.save_pretrained(directory)
    vocabulary = tokenizer.get_vocab()
    ordered = sorted(vocabulary.items(), key=lambda item: item[1])
    digest = _sha256_bytes(json.dumps(ordered, ensure_ascii=False, separators=(",", ":")).encode())
    metadata = {
        "base_model": model_name,
        "vocab_size": len(tokenizer),
        "mask_token": tokenizer.mask_token, "mask_token_id": tokenizer.mask_token_id,
        "pad_token": tokenizer.pad_token, "pad_token_id": tokenizer.pad_token_id,
        "eos_token": tokenizer.eos_token, "eos_token_id": tokenizer.eos_token_id,
        "vocabulary_sha256": digest,
    }
    (directory / "tokenizer_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return tokenizer, metadata


# Corpora the split pipeline can build from. Every entry must yield
# independent, document-shaped units: the split integrity argument is that no
# document appears in two splits, which is meaningless if the units overlap.
# WikiText ships as line-oriented text, so its paragraphs are the unit and
# section headings ("= Title =") and short fragments are filtered out.
CORPORA: dict[str, dict[str, Any]] = {
    "tinystories": {
        "path": "roneneldan/TinyStories", "name": None, "split": "train",
        "text_field": "text", "minimum_characters": 1, "drop_headings": False,
    },
    "wikitext": {
        "path": "Salesforce/wikitext", "name": "wikitext-103-raw-v1", "split": "train",
        "text_field": "text", "minimum_characters": 400, "drop_headings": True,
    },
    "cnn_dailymail": {
        "path": "abisee/cnn_dailymail", "name": "3.0.0", "split": "train",
        "text_field": "article", "minimum_characters": 400, "drop_headings": False,
    },
}


def fetch_documents(raw_path: Path, count: int, corpus: str = "tinystories") -> list[dict[str, Any]]:
    """Cache a bounded streaming prefix of ``corpus`` as canonical JSONL.

    Documents are deduplicated by content hash and emitted in stream order, so
    a later fetch for a larger ``count`` returns the earlier prefix unchanged
    and previously assigned splits stay valid.
    """
    if corpus not in CORPORA:
        raise ValueError(f"unknown corpus {corpus!r}; known: {sorted(CORPORA)}")
    settings = CORPORA[corpus]
    if raw_path.exists():
        documents = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line]
        if len(documents) >= count:
            return documents[:count]
    from datasets import load_dataset
    stream = load_dataset(settings["path"], settings["name"], split=settings["split"], streaming=True)
    documents: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source_index, row in enumerate(stream):
        text = _canonical_text(row[settings["text_field"]])
        if len(text) < int(settings["minimum_characters"]):
            continue
        if settings["drop_headings"] and text.startswith("="):
            continue
        digest = _sha256_bytes(text.encode())
        if digest in seen:
            continue
        seen.add(digest)
        documents.append({"document_id": f"{corpus}:{source_index}:{digest[:16]}",
                          "sha256": digest, "text": text})
        if len(documents) == count:
            break
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text("".join(json.dumps(document, ensure_ascii=False) + "\n" for document in documents), encoding="utf-8")
    return documents


def fetch_tinystories_documents(raw_path: Path, count: int) -> list[dict[str, Any]]:
    """Backwards-compatible alias; existing callers assume TinyStories."""
    return fetch_documents(raw_path, count, corpus="tinystories")


def _chunk_document(tokenizer: PreTrainedTokenizerBase, text: str, max_length: int) -> Iterator[list[int]]:
    token_ids = tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]
    for start in range(0, len(token_ids), max_length):
        chunk = token_ids[start : start + max_length]
        # Full chunks make non-padding tokens and tokens-seen exactly matched
        # across controlled runs; document tails are intentionally discarded.
        if len(chunk) == max_length:
            yield chunk


def prepare_splits(
    *,
    output_dir: Path,
    tokenizer_dir: Path,
    raw_path: Path,
    document_counts: dict[str, int],
    max_chunks: dict[str, int],
    max_length: int,
    seed: int,
) -> dict[str, Any]:
    """Split unique documents before tokenization and save tensors/manifests."""
    tokenizer, tokenizer_metadata = prepare_tokenizer(tokenizer_dir)
    required = sum(document_counts.values())
    documents = fetch_tinystories_documents(raw_path, required)
    if len(documents) < required:
        raise RuntimeError("not enough unique documents")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(documents))
    split_documents: dict[str, list[dict[str, Any]]] = {}
    cursor = 0
    # Eval is deliberately allocated first, but all assignments are from one permutation.
    for split in ["eval", "train_a", "train_b"]:
        size = int(document_counts[split])
        split_documents[split] = [documents[int(index)] for index in order[cursor : cursor + size]]
        cursor += size

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "source": "roneneldan/TinyStories", "data_seed": seed,
        "split_before_tokenization": True, "max_length": max_length,
        "tokenizer": tokenizer_metadata, "splits": {}, "overlap_checks": {},
    }
    chunk_hash_sets: dict[str, set[str]] = {}
    for split, docs in split_documents.items():
        chunks: list[list[int]] = []
        chunk_records: list[dict[str, Any]] = []
        for document in docs:
            for local_index, chunk in enumerate(_chunk_document(tokenizer, document["text"], max_length)):
                packed = np.asarray(chunk, dtype=np.int32).tobytes()
                digest = _sha256_bytes(packed)
                chunks.append(chunk)
                chunk_records.append({"document_id": document["document_id"], "chunk_index": local_index, "sha256": digest, "non_padding_tokens": len(chunk)})
                if len(chunks) >= int(max_chunks[split]):
                    break
            if len(chunks) >= int(max_chunks[split]):
                break
        ids = torch.full((len(chunks), max_length), int(tokenizer.pad_token_id), dtype=torch.long)
        attention = torch.zeros((len(chunks), max_length), dtype=torch.long)
        for row, chunk in enumerate(chunks):
            ids[row, : len(chunk)] = torch.tensor(chunk)
            attention[row, : len(chunk)] = 1
        torch.save({"input_ids": ids, "attention_mask": attention, "chunk_records": chunk_records}, output_dir / f"{split}.pt")
        chunk_hash_sets[split] = {record["sha256"] for record in chunk_records}
        manifest["splits"][split] = {
            "documents": [{"document_id": doc["document_id"], "sha256": doc["sha256"]} for doc in docs],
            "document_count": len(docs), "chunk_count_n": int(ids.shape[0]),
            "non_padding_tokens": int(attention.sum().item()), "chunks": chunk_records,
        }
    for left, right in [("train_a", "train_b"), ("train_a", "eval"), ("train_b", "eval")]:
        doc_left = {doc["sha256"] for doc in split_documents[left]}
        doc_right = {doc["sha256"] for doc in split_documents[right]}
        manifest["overlap_checks"][f"{left}__{right}"] = {
            "document_hash_overlap": sorted(doc_left & doc_right),
            "exact_chunk_hash_overlap": sorted(chunk_hash_sets[left] & chunk_hash_sets[right]),
        }
    manifest_path = output_dir / "split_manifest.json"
    validate_split_manifest(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def prepare_confirmatory_splits(
    *,
    output_dir: Path,
    tokenizer_dir: Path,
    raw_path: Path,
    replicate_count: int,
    eval_document_count: int,
    per_split_document_count: int,
    max_chunks: dict[str, int],
    max_length: int,
    master_seed: int,
    corpus: str = "tinystories",
    max_chunks_per_document: int | None = None,
) -> dict[str, Any]:
    """Build one frozen shared eval split plus ``replicate_count`` mutually
    document-disjoint (train_a, train_b) pairs, for the crossed-replication
    confirmatory experiment.

    Unlike ``prepare_splits``, every replicate's train_a/train_b documents
    are drawn independently (not nested prefixes of one pool) and are
    disjoint from every other replicate's documents and from the single
    shared eval set, so replicate variance reflects genuine document-level
    resampling rather than overlapping data.
    """
    tokenizer, tokenizer_metadata = prepare_tokenizer(tokenizer_dir)
    required = eval_document_count + replicate_count * 2 * per_split_document_count
    documents = fetch_documents(raw_path, required, corpus=corpus)
    if len(documents) < required:
        raise RuntimeError(f"not enough unique documents: need {required}, have {len(documents)}")
    rng = np.random.default_rng(master_seed)
    order = rng.permutation(len(documents))

    split_documents: dict[str, list[dict[str, Any]]] = {}
    cursor = 0
    split_documents["eval"] = [documents[int(index)] for index in order[cursor : cursor + eval_document_count]]
    cursor += eval_document_count
    replicate_splits: list[str] = []
    for replicate in range(1, replicate_count + 1):
        for split in ["train_a", "train_b"]:
            name = f"replicate{replicate}_{split}"
            replicate_splits.append(name)
            split_documents[name] = [documents[int(index)] for index in order[cursor : cursor + per_split_document_count]]
            cursor += per_split_document_count

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "source": CORPORA[corpus]["path"], "corpus": corpus, "master_seed": master_seed,
        "replicate_count": replicate_count, "split_before_tokenization": True,
        "max_chunks_per_document": max_chunks_per_document,
        "max_length": max_length, "tokenizer": tokenizer_metadata,
        "splits": {}, "overlap_checks": {},
    }
    chunk_hash_sets: dict[str, set[str]] = {}
    # Natural corpora can contain byte-identical windows in genuinely
    # different documents.  Claim chunk hashes as splits are materialised so
    # document independence also implies zero exact-window overlap.  The
    # lower-level helper performs the same check within each split.
    claimed_chunks: set[str] = set()
    for split, docs in split_documents.items():
        chunk_budget = int(max_chunks["eval"] if split == "eval" else max_chunks["train"])
        ids, attention, chunk_records = _chunk_split(
            tokenizer, docs, chunk_budget, max_length, forbidden=claimed_chunks,
            max_chunks_per_document=max_chunks_per_document,
        )
        if ids.shape[0] < chunk_budget:
            raise RuntimeError(
                f"{split} yielded only {ids.shape[0]} unique chunks; need {chunk_budget}"
            )
        torch.save({"input_ids": ids, "attention_mask": attention,
                    "chunk_records": chunk_records}, output_dir / f"{split}.pt")
        chunk_hash_sets[split] = {record["sha256"] for record in chunk_records}
        claimed_chunks |= chunk_hash_sets[split]
        manifest["splits"][split] = {
            "documents": [{"document_id": doc["document_id"], "sha256": doc["sha256"]} for doc in docs],
            "document_count": len(docs), "chunk_count_n": int(ids.shape[0]),
            "non_padding_tokens": int(attention.sum().item()), "chunks": chunk_records,
        }

    all_split_names = ["eval"] + replicate_splits
    checked_pairs = 0
    for left_index, left in enumerate(all_split_names):
        for right in all_split_names[left_index + 1 :]:
            doc_left = {doc["sha256"] for doc in split_documents[left]}
            doc_right = {doc["sha256"] for doc in split_documents[right]}
            manifest["overlap_checks"][f"{left}__{right}"] = {
                "document_hash_overlap": sorted(doc_left & doc_right),
                "exact_chunk_hash_overlap": sorted(chunk_hash_sets[left] & chunk_hash_sets[right]),
            }
            checked_pairs += 1
    for pair, checks in manifest["overlap_checks"].items():
        if checks["document_hash_overlap"]:
            raise ValueError(f"document overlap in {pair}")
        if checks["exact_chunk_hash_overlap"]:
            raise ValueError(f"exact duplicate chunks in {pair}")
    manifest["overlap_checks_performed"] = checked_pairs
    manifest_path = output_dir / "confirmatory_split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def validate_split_manifest(manifest: dict[str, Any]) -> None:
    """Reject document or exact-chunk overlap between every saved split."""
    expected_pairs = {"train_a__train_b", "train_a__eval", "train_b__eval"}
    if set(manifest.get("overlap_checks", {})) != expected_pairs:
        raise ValueError("manifest does not contain all split overlap checks")
    for pair, checks in manifest["overlap_checks"].items():
        if checks["document_hash_overlap"]:
            raise ValueError(f"document overlap in {pair}")
        if checks["exact_chunk_hash_overlap"]:
            raise ValueError(f"exact duplicate chunks in {pair}")


class ChunkDataset(Dataset):
    """In-memory fixed-length token chunks generated by ``prepare_splits``."""

    def __init__(self, path: Path, limit: int | None = None):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.input_ids = payload["input_ids"][:limit]
        self.attention_mask = payload["attention_mask"][:limit]
        self.chunk_records = payload["chunk_records"][:limit]

    def __len__(self) -> int:
        return self.input_ids.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        return {"input_ids": self.input_ids[index], "attention_mask": self.attention_mask[index], "index": index}


def _chunk_split(
    tokenizer: PreTrainedTokenizerBase,
    documents: list[dict[str, Any]],
    max_chunks: int,
    max_length: int,
    forbidden: set[str] | None = None,
    max_chunks_per_document: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    """Tokenize documents into full-length chunks, stopping at ``max_chunks``.

    Splitting at the document level does not by itself guarantee disjoint
    chunks: formulaic corpora emit byte-identical 64-token windows from
    genuinely different documents ("Once upon a time there was a little girl
    named Lily"), and at tens of thousands of chunks per split such collisions
    are common.  Chunks whose hash is already claimed -- by another split via
    ``forbidden``, or earlier within this split -- are skipped, so the
    disjointness the experiment relies on holds at the chunk level too.
    """
    claimed = set(forbidden or ())
    chunks: list[list[int]] = []
    records: list[dict[str, Any]] = []
    duplicates = 0
    if max_chunks_per_document is not None and max_chunks_per_document < 1:
        raise ValueError("max_chunks_per_document must be positive")
    for document in documents:
        document_chunks = list(_chunk_document(tokenizer, document["text"], max_length))
        if max_chunks_per_document is None or len(document_chunks) <= max_chunks_per_document:
            chosen_indices = range(len(document_chunks))
        else:
            # Cover the article body without privileging repeated news ledes.
            chosen_indices = [
                min(len(document_chunks) - 1, int((index + 0.5) * len(document_chunks)
                                                 / max_chunks_per_document))
                for index in range(max_chunks_per_document)
            ]
        for local_index in chosen_indices:
            chunk = document_chunks[local_index]
            digest = _sha256_bytes(np.asarray(chunk, dtype=np.int32).tobytes())
            if digest in claimed:
                duplicates += 1
                continue
            claimed.add(digest)
            records.append({
                "document_id": document["document_id"], "chunk_index": local_index,
                "sha256": digest, "non_padding_tokens": len(chunk),
            })
            chunks.append(chunk)
            if len(chunks) >= max_chunks:
                break
        if len(chunks) >= max_chunks:
            break
    if duplicates:
        print(f"    skipped {duplicates} duplicate chunks while building this split")
    ids = torch.full((len(chunks), max_length), int(tokenizer.pad_token_id), dtype=torch.long)
    attention = torch.zeros((len(chunks), max_length), dtype=torch.long)
    for row, chunk in enumerate(chunks):
        ids[row, : len(chunk)] = torch.tensor(chunk)
        attention[row, : len(chunk)] = 1
    return ids, attention, records


def extend_confirmatory_splits(
    *,
    output_dir: Path,
    tokenizer_dir: Path,
    raw_path: Path,
    new_replicates: list[int],
    per_split_document_count: int,
    max_chunks_train: int,
    max_length: int,
    seed: int,
) -> dict[str, Any]:
    """Add replicates to an existing confirmatory split set, leaving it untouched.

    ``prepare_confirmatory_splits`` derives every split from one permutation of
    the fetched document list, so re-running it with a larger
    ``replicate_count`` changes that permutation and silently invalidates the
    existing splits, the frozen evaluation corruptions and every checkpoint
    trained against them.  This instead treats the already-assigned documents
    as reserved, draws the new replicates only from documents no existing split
    uses, and verifies zero document- and chunk-level overlap against every
    prior split before writing anything.  Existing files are never rewritten.
    """
    # No single manifest file is authoritative. Each extension writes a
    # cumulative copy into the replicate directories it creates and overwrites
    # the top-level extended file, so an extension that read a stale base
    # silently drops earlier replicates from the top-level record -- while
    # their tensors remain on disk. Merge every manifest present instead, so a
    # split is reserved if ANY record mentions it.
    candidates = [output_dir / "confirmatory_split_manifest.json",
                  output_dir / "confirmatory_split_manifest_extended.json",
                  *sorted(output_dir.glob("replicate*/split_manifest.json"))]
    present = [path for path in candidates if path.exists()]
    if not present:
        raise FileNotFoundError(f"no confirmatory manifest under {output_dir}")
    base: dict[str, Any] = {}
    merged_splits: dict[str, Any] = {}
    for path in present:
        loaded = json.loads(path.read_text())
        if len(loaded.get("splits", {})) >= len(base.get("splits", {})):
            base = loaded
        merged_splits.update(loaded.get("splits", {}))
    base = json.loads(json.dumps(base))
    base["splits"] = merged_splits
    known = {name for name in merged_splits if name != "eval"}
    on_disk_splits = {f"replicate{path.parent.name.removeprefix('replicate')}_{path.stem}"
                      for path in output_dir.glob("replicate*/train_*.pt")}
    if on_disk_splits - known:
        raise ValueError(
            f"splits on disk with no manifest record: {sorted(on_disk_splits - known)}; "
            "refusing to extend against an incomplete picture")
    if int(base["max_length"]) != int(max_length):
        raise ValueError("max_length differs from the existing split set")

    reserved_documents = {
        document["sha256"] for split in base["splits"].values() for document in split["documents"]
    }
    reserved_chunks = {
        record["sha256"] for split in base["splits"].values() for record in split["chunks"]
    }
    for replicate in new_replicates:
        target = output_dir / f"replicate{replicate}"
        if target.exists():
            raise FileExistsError(f"{target} already exists; refusing to overwrite")

    tokenizer, tokenizer_metadata = prepare_tokenizer(tokenizer_dir)
    if tokenizer_metadata["vocabulary_sha256"] != base["tokenizer"]["vocabulary_sha256"]:
        raise ValueError("tokenizer vocabulary does not match the existing split set")

    needed = len(new_replicates) * 2 * per_split_document_count
    # Fetch with headroom: the stream is deduplicated, so the reserved prefix is
    # not necessarily contiguous once documents are filtered out.
    documents = fetch_tinystories_documents(raw_path, len(reserved_documents) + 2 * needed)
    pool = [document for document in documents if document["sha256"] not in reserved_documents]
    if len(pool) < needed:
        raise RuntimeError(f"not enough unused documents: need {needed}, have {len(pool)}")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pool))
    assignments: dict[str, list[dict[str, Any]]] = {}
    cursor = 0
    for replicate in new_replicates:
        for split in ("train_a", "train_b"):
            assignments[f"replicate{replicate}_{split}"] = [
                pool[int(index)] for index in order[cursor : cursor + per_split_document_count]
            ]
            cursor += per_split_document_count

    prepared: dict[str, Any] = {}
    claimed_chunks = set(reserved_chunks)
    for name, documents_for_split in assignments.items():
        print(f"  building {name}", flush=True)
        ids, attention, records = _chunk_split(tokenizer, documents_for_split, max_chunks_train,
                                               max_length, forbidden=claimed_chunks)
        claimed_chunks |= {record["sha256"] for record in records}
        if ids.shape[0] < max_chunks_train:
            raise RuntimeError(f"{name} yielded only {ids.shape[0]} chunks, need {max_chunks_train}")
        prepared[name] = {
            "ids": ids, "attention": attention, "records": records,
            "documents": documents_for_split,
        }

    # Overlap gate: new vs every existing split, and new vs new. Nothing is
    # written until every pair is clean.
    checks: dict[str, Any] = {}
    names = list(prepared)
    for index, left in enumerate(names):
        left_documents = {document["sha256"] for document in prepared[left]["documents"]}
        left_chunks = {record["sha256"] for record in prepared[left]["records"]}
        checks[f"existing__{left}"] = {
            "document_hash_overlap": sorted(left_documents & reserved_documents),
            "exact_chunk_hash_overlap": sorted(left_chunks & reserved_chunks),
        }
        for right in names[index + 1 :]:
            right_documents = {document["sha256"] for document in prepared[right]["documents"]}
            right_chunks = {record["sha256"] for record in prepared[right]["records"]}
            checks[f"{left}__{right}"] = {
                "document_hash_overlap": sorted(left_documents & right_documents),
                "exact_chunk_hash_overlap": sorted(left_chunks & right_chunks),
            }
    for pair, result in checks.items():
        if result["document_hash_overlap"]:
            raise ValueError(f"document overlap in {pair}")
        if result["exact_chunk_hash_overlap"]:
            raise ValueError(f"exact duplicate chunks in {pair}")

    extended = json.loads(json.dumps(base))
    extended["extension_seed"] = seed
    extended["replicate_count"] = int(base["replicate_count"]) + len(new_replicates)
    extended["extended_replicates"] = list(new_replicates)
    for name, payload in prepared.items():
        extended["splits"][name] = {
            "documents": [{"document_id": d["document_id"], "sha256": d["sha256"]}
                          for d in payload["documents"]],
            "document_count": len(payload["documents"]),
            "chunk_count_n": int(payload["ids"].shape[0]),
            "non_padding_tokens": int(payload["attention"].sum().item()),
            "chunks": payload["records"],
        }
    extended["overlap_checks"].update(checks)
    extended["overlap_checks_performed"] = len(extended["overlap_checks"])

    on_disk: set[str] = set()
    for existing in sorted(output_dir.glob("replicate*/train_*.pt")):
        payload = torch.load(existing, map_location="cpu", weights_only=False)
        on_disk |= {_sha256_bytes(np.asarray(row, dtype=np.int32).tobytes())
                    for row in payload["input_ids"].numpy()}
    for name, payload in prepared.items():
        collision = {record["sha256"] for record in payload["records"]} & on_disk
        if collision:
            raise ValueError(
                f"{name} shares {len(collision)} chunks with a split already on disk; "
                "the manifest under-reported what exists")

    shared_eval = torch.load(output_dir / "eval.pt", map_location="cpu", weights_only=False)
    serialized = json.dumps(extended, indent=2)
    for replicate in new_replicates:
        target = output_dir / f"replicate{replicate}"
        target.mkdir(parents=True)
        for split in ("train_a", "train_b"):
            payload = prepared[f"replicate{replicate}_{split}"]
            torch.save({"input_ids": payload["ids"], "attention_mask": payload["attention"],
                        "chunk_records": payload["records"]}, target / f"{split}.pt")
        torch.save(shared_eval, target / "eval.pt")
        (target / "split_manifest.json").write_text(serialized, encoding="utf-8")
    (output_dir / "confirmatory_split_manifest_extended.json").write_text(serialized, encoding="utf-8")
    return extended
