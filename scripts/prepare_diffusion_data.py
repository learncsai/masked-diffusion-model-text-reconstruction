#!/usr/bin/env python3
"""Prepare bounded document-level TinyStories splits and GPT-2 tokenizer."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from diffusion_lm_rmt.neural_data import prepare_splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    data = config["data"]
    manifest = prepare_splits(
        output_dir=Path(data["processed_dir"]), tokenizer_dir=Path(data["tokenizer_dir"]),
        raw_path=Path(data["raw_path"]), document_counts=data["document_counts"],
        max_chunks=data["max_chunks"], max_length=int(config["model"]["max_length"]), seed=int(data["seed"]),
    )
    for split, values in manifest["splits"].items():
        print(split, "documents", values["document_count"], "chunks n", values["chunk_count_n"], "tokens", values["non_padding_tokens"])
    print("overlap checks", manifest["overlap_checks"])


if __name__ == "__main__":
    main()
