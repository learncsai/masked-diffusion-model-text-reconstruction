#!/usr/bin/env python3
"""Prepare the crossed-replication confirmatory experiment's document splits:
one frozen shared eval set plus R mutually document-disjoint (train_a, train_b)
pairs, one pair per replicate."""

from __future__ import annotations

import argparse
from pathlib import Path

from diffusion_lm_rmt.neural_data import prepare_confirmatory_splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--raw-path", type=Path, required=True)
    parser.add_argument("--replicate-count", type=int, required=True)
    parser.add_argument("--eval-document-count", type=int, default=300)
    parser.add_argument("--per-split-document-count", type=int, default=2200)
    parser.add_argument("--max-chunks-train", type=int, default=4096)
    parser.add_argument("--max-chunks-eval", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--master-seed", type=int, default=90210)
    parser.add_argument("--corpus", default="tinystories",
                        help="which registered corpus to build from (see neural_data.CORPORA)")
    parser.add_argument("--max-chunks-per-document", type=int,
                        help="optional cap, with evenly spaced full chunks per source document")
    args = parser.parse_args()

    manifest = prepare_confirmatory_splits(
        output_dir=args.output_dir, tokenizer_dir=args.tokenizer_dir, raw_path=args.raw_path,
        replicate_count=args.replicate_count, eval_document_count=args.eval_document_count,
        per_split_document_count=args.per_split_document_count,
        max_chunks={"train": args.max_chunks_train, "eval": args.max_chunks_eval},
        max_length=args.max_length, master_seed=args.master_seed, corpus=args.corpus,
        max_chunks_per_document=args.max_chunks_per_document,
    )
    for split, values in manifest["splits"].items():
        print(split, "documents", values["document_count"], "chunks n", values["chunk_count_n"], "tokens", values["non_padding_tokens"])
    print("overlap checks performed (all pairs, zero overlap):", manifest["overlap_checks_performed"])


if __name__ == "__main__":
    main()
