#!/usr/bin/env python3
"""Add document-disjoint replicates to the existing confirmatory split set.

The crossed data x optimisation design needs more independent data splits than
the original three replicates provide.  Regenerating with a larger
``--replicate-count`` would repermute the whole document list and invalidate
every existing split, the frozen evaluation corruptions and all 36 trained
checkpoints, so this extends instead: new replicates are drawn only from
documents no existing split uses, and are gated on zero document- and
chunk-level overlap with every prior split before anything is written.
"""

import argparse
from pathlib import Path

from diffusion_lm_rmt.neural_data import extend_confirmatory_splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--raw-path", type=Path, required=True)
    parser.add_argument("--replicate", type=int, action="append", required=True,
                        help="new replicate index; repeat for several")
    parser.add_argument("--per-split-document-count", type=int, default=2200)
    parser.add_argument("--max-chunks-train", type=int, default=4096)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=606060)
    args = parser.parse_args()

    manifest = extend_confirmatory_splits(
        output_dir=args.output_dir, tokenizer_dir=args.tokenizer_dir, raw_path=args.raw_path,
        new_replicates=sorted(args.replicate),
        per_split_document_count=args.per_split_document_count,
        max_chunks_train=args.max_chunks_train, max_length=args.max_length, seed=args.seed,
    )
    for replicate in sorted(args.replicate):
        for split in ("train_a", "train_b"):
            values = manifest["splits"][f"replicate{replicate}_{split}"]
            print(f"replicate{replicate}_{split}: {values['document_count']} documents, "
                  f"{values['chunk_count_n']} chunks, {values['non_padding_tokens']} tokens")
    print(f"overlap checks performed (all pairs, zero overlap): {manifest['overlap_checks_performed']}")
    print(f"total splits now: {len(manifest['splits'])}")


if __name__ == "__main__":
    main()
