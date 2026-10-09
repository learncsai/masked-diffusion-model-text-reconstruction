"""Machine-readable audit of the controlled neural training pairs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import yaml


def _run_record(run_dir: Path) -> dict[str, Any]:
    checkpoint = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    parent_path = checkpoint.get("parent_checkpoint")
    initial_dir = Path(parent_path).parent if parent_path else run_dir
    initial_checkpoint = torch.load(initial_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    initial_system = json.loads((initial_dir / "system_info.json").read_text())
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    initial_config = yaml.safe_load((initial_dir / "config.yaml").read_text())
    return {
        "run_dir": str(run_dir),
        "train_split": checkpoint["train_split"],
        "model_config": checkpoint["model_config"],
        "steps": checkpoint["step"],
        "stage_tokens_seen": checkpoint["tokens_seen"],
        "initial_stage_steps": initial_checkpoint["step"],
        "initial_stage_tokens_seen": initial_checkpoint["tokens_seen"],
        "cumulative_tokens_seen": initial_checkpoint["tokens_seen"] + (checkpoint["tokens_seen"] if parent_path else 0),
        "seeds": initial_checkpoint["seeds"],
        "initial_state_sha256": initial_system["initial_state_sha256"],
        "optimizer_config": config["optimizer"],
        "initial_optimizer_config": initial_config["optimizer"],
        "training_config": config["training"],
        "initial_training_config": initial_config["training"],
        "data_generator_state": checkpoint["data_generator_state"],
        "corruption_generator_state": checkpoint["corruption_generator_state"],
        "parent_checkpoint": parent_path,
    }


def audit_controls(
    split_a_init0: Path,
    split_b_init0: Path,
    split_a_init1: Path,
    output: Path,
) -> dict[str, Any]:
    """Verify the intended data and initialization interventions only."""
    a = _run_record(split_a_init0)
    b = _run_record(split_b_init0)
    init = _run_record(split_a_init1)

    def same(left: dict[str, Any], right: dict[str, Any], key: str) -> bool:
        return left[key] == right[key]

    cross = {
        "disjoint_training_split_labels": a["train_split"] != b["train_split"],
        "same_initial_state": same(a, b, "initial_state_sha256"),
        "same_initialization_seed": a["seeds"]["initialization_seed"] == b["seeds"]["initialization_seed"],
        "same_data_seed": a["seeds"]["data_seed"] == b["seeds"]["data_seed"],
        "same_training_seed": a["seeds"]["training_seed"] == b["seeds"]["training_seed"],
        "same_corruption_seed": a["seeds"]["corruption_seed"] == b["seeds"]["corruption_seed"],
        "same_model_config": same(a, b, "model_config"),
        "same_optimizer_config": same(a, b, "optimizer_config"),
        "same_initial_optimizer_config": same(a, b, "initial_optimizer_config"),
        "same_training_config": same(a, b, "training_config"),
        "same_initial_training_config": same(a, b, "initial_training_config"),
        "same_final_stage_steps": same(a, b, "steps"),
        "same_final_stage_tokens": same(a, b, "stage_tokens_seen"),
        "same_initial_stage_steps": same(a, b, "initial_stage_steps"),
        "same_initial_stage_tokens": same(a, b, "initial_stage_tokens_seen"),
        "same_cumulative_tokens": same(a, b, "cumulative_tokens_seen"),
        "same_final_data_rng_state": torch.equal(a["data_generator_state"], b["data_generator_state"]),
        "same_final_corruption_rng_state": torch.equal(a["corruption_generator_state"], b["corruption_generator_state"]),
    }
    initialization = {
        "same_training_split": a["train_split"] == init["train_split"],
        "different_initial_state": a["initial_state_sha256"] != init["initial_state_sha256"],
        "different_initialization_seed": a["seeds"]["initialization_seed"] != init["seeds"]["initialization_seed"],
        "same_data_seed": a["seeds"]["data_seed"] == init["seeds"]["data_seed"],
        "same_training_seed": a["seeds"]["training_seed"] == init["seeds"]["training_seed"],
        "same_corruption_seed": a["seeds"]["corruption_seed"] == init["seeds"]["corruption_seed"],
        "same_model_config": same(a, init, "model_config"),
        "same_optimizer_config": same(a, init, "optimizer_config"),
        "same_initial_optimizer_config": same(a, init, "initial_optimizer_config"),
        "same_training_config": same(a, init, "training_config"),
        "same_initial_training_config": same(a, init, "initial_training_config"),
        "same_final_stage_steps": same(a, init, "steps"),
        "same_final_stage_tokens": same(a, init, "stage_tokens_seen"),
        "same_initial_stage_steps": same(a, init, "initial_stage_steps"),
        "same_initial_stage_tokens": same(a, init, "initial_stage_tokens_seen"),
        "same_cumulative_tokens": same(a, init, "cumulative_tokens_seen"),
        "same_final_data_rng_state": torch.equal(a["data_generator_state"], init["data_generator_state"]),
        "same_final_corruption_rng_state": torch.equal(a["corruption_generator_state"], init["corruption_generator_state"]),
    }
    manifest = json.loads((split_a_init0 / "split_manifest.json").read_text())
    no_overlap = all(
        not check["document_hash_overlap"] and not check["exact_chunk_hash_overlap"]
        for check in manifest["overlap_checks"].values()
    )
    payload = {
        "passed": all(cross.values()) and all(initialization.values()) and no_overlap,
        "cross_split_same_initialization": cross,
        "same_data_different_initialization": initialization,
        "manifest_has_no_document_or_exact_chunk_overlap": no_overlap,
        "initial_state_sha256": {
            "split_a_init0": a["initial_state_sha256"],
            "split_b_init0": b["initial_state_sha256"],
            "split_a_init1": init["initial_state_sha256"],
        },
        "matched_budget": {
            "initial_stage_steps": a["initial_stage_steps"],
            "continuation_steps": a["steps"],
            "cumulative_tokens_seen": a["cumulative_tokens_seen"],
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    return payload
