#!/usr/bin/env python3
"""Prospective document-bootstrap check of categorical lag reconstruction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from diffusion_lm_rmt import categorical_lag as dense_backend
from diffusion_lm_rmt import sparse_categorical_lag as sparse_backend
from diffusion_lm_rmt import sparse_categorical_influence as sparse_influence

from diffusion_lm_rmt.categorical_lag import (
    bootstrap_disagreement,
    conditional_tables,
    draw_reference_chunks,
    fitting_equations,
    masked_prediction,
    offset_list,
    pair_counts,
    per_sequence_squared_error,
    source_influence_disagreement,
    source_groups,
    token_counts,
)


def load_split(root: Path, name: str) -> tuple[np.ndarray, list[dict]]:
    payload = torch.load(root / f"{name}.pt", map_location="cpu", weights_only=True)
    ids = payload["input_ids"].numpy()
    records = payload["chunk_records"]
    if len(ids) != len(records) or np.any(payload["attention_mask"].numpy() != 1):
        raise ValueError(f"unexpected padding or record mismatch in {root}/{name}")
    return ids, records


def check_disjoint(splits: dict[str, tuple[np.ndarray, list[dict]]]) -> None:
    document_sets = {
        name: {record["document_id"] for record in records}
        for name, (_, records) in splits.items()
    }
    chunk_sets = {
        name: {record["sha256"] for record in records}
        for name, (_, records) in splits.items()
    }
    names = list(splits)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            if document_sets[left] & document_sets[right]:
                raise ValueError(f"document overlap between {left} and {right}")
            if chunk_sets[left] & chunk_sets[right]:
                raise ValueError(f"exact chunk overlap between {left} and {right}")


def split_auxiliary(
    ids: np.ndarray, records: list[dict], rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    groups = source_groups(records)
    rng.shuffle(groups)
    boundary = len(groups) // 2
    left = ids[np.concatenate(groups[:boundary])]
    right = ids[np.concatenate(groups[boundary:])]
    if min(len(left), len(right)) < 128:
        raise ValueError("auxiliary document halves are too small")
    return left, right


def make_repartitions(
    splits: dict[str, tuple[np.ndarray, list[dict]]],
    count: int, n: int, rng: np.random.Generator,
) -> dict[int, tuple[tuple[np.ndarray, list[dict]], tuple[np.ndarray, list[dict]]]]:
    """Random document-disjoint A/B allocations from one fixed finite pool."""
    names = ["replicate1_train_a", "replicate1_train_b",
             "replicate2_train_a", "replicate2_train_b"]
    ids = np.concatenate([splits[name][0] for name in names])
    records = sum([splits[name][1] for name in names], [])
    groups = source_groups(records)
    result = {}
    for replicate in range(1, count + 1):
        order = rng.permutation(len(groups))
        sides = []
        cursor = 0
        for _ in ("a", "b"):
            picked = []
            size = 0
            while size < n:
                if cursor >= len(order):
                    raise ValueError("finite experimental pool cannot supply two disjoint sides")
                group = groups[int(order[cursor])]
                picked.append(group)
                size += len(group)
                cursor += 1
            selected = np.concatenate(picked)[:n]
            sides.append((ids[selected], [records[int(i)] for i in selected]))
        if {r["document_id"] for r in sides[0][1]} & {
            r["document_id"] for r in sides[1][1]
        }:
            raise AssertionError("repartition sides share a source")
        result[replicate] = (sides[0], sides[1])
    return result


def make_nonoverlap_pairs(
    splits: dict[str, tuple[np.ndarray, list[dict]]],
    count: int, n: int, rng: np.random.Generator,
) -> dict[int, tuple[tuple[np.ndarray, list[dict]], tuple[np.ndarray, list[dict]]]]:
    """Allocate each source to at most one A/B side across all pairs."""
    names = ["replicate1_train_a", "replicate1_train_b",
             "replicate2_train_a", "replicate2_train_b"]
    ids = np.concatenate([splits[name][0] for name in names])
    records = sum([splits[name][1] for name in names], [])
    groups = source_groups(records)
    order = rng.permutation(len(groups))
    cursor = 0
    result = {}
    used_sources: set[str] = set()
    for replicate in range(1, count + 1):
        sides = []
        for _ in ("a", "b"):
            picked = []
            size = 0
            while size < n:
                if cursor >= len(order):
                    raise ValueError(
                        f"experimental pool exhausted after {len(result)} complete pairs; "
                        f"requested {count} pairs of {n} chunks per side"
                    )
                group = groups[int(order[cursor])]
                picked.append(group)
                size += len(group)
                cursor += 1
            selected = np.concatenate(picked)[:n]
            side_records = [records[int(i)] for i in selected]
            source_ids = {record["document_id"] for record in side_records}
            if used_sources & source_ids:
                raise AssertionError("a source was reused between independent pairs")
            used_sources.update(source_ids)
            sides.append((ids[selected], side_records))
        result[replicate] = (sides[0], sides[1])
    return result


def squared_onehot_loss(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    if isinstance(prediction, sparse_backend.Prediction):
        return prediction.loss(target)
    return 1 - 2 * prediction[np.arange(len(target)), target] + np.einsum(
        "ij,ij->i", prediction, prediction,
    )


def combine_independent_side_laws(
    equal_pair_prediction_a: np.ndarray, equal_pair_prediction_b: np.ndarray,
) -> np.ndarray:
    """Unequal-side law: Var(f_A) + Var(f_B), each input is 2 Var(f)."""
    return 0.5 * (equal_pair_prediction_a + equal_pair_prediction_b)


def scored_prediction(
    ids: np.ndarray, mask: np.ndarray, tables: dict[int, np.ndarray],
    prior: np.ndarray, offsets: tuple[int, ...], gram: np.ndarray,
    cross: np.ndarray, ridge: float, backend=dense_backend,
) -> tuple[np.ndarray, np.ndarray]:
    rows, target, prediction = backend.masked_prediction(
        ids, mask, tables, prior, offsets, gram, cross, ridge,
    )
    losses = squared_onehot_loss(prediction, target)
    return per_sequence_squared_error(rows, losses, len(ids)), prediction


def visible_context_surprisal(
    ids: np.ndarray, mask: np.ndarray, offsets: tuple[int, ...], prior: np.ndarray,
) -> np.ndarray:
    """Mean negative log auxiliary frequency of visible local contexts."""
    output = np.zeros(len(ids), dtype=np.float64)
    for row in range(len(ids)):
        values = []
        for position in np.flatnonzero(mask[row]):
            for d in offsets:
                context = position - d
                if 0 <= context < ids.shape[1] and not mask[row, context]:
                    values.append(-np.log(max(prior[ids[row, context]], 1e-12)))
        output[row] = float(np.mean(values)) if values else np.nan
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/prospective_lag_v1_pilot.json"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--law-seed-offset", type=int, default=0)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.output is not None:
        config["output"] = str(args.output)
    config["law_seed_offset"] = args.law_seed_offset
    analytic_only = bool(config.get("analytic_only", False))
    if analytic_only and not config.get("analytic_source_influence", False):
        raise ValueError("analytic_only requires analytic_source_influence")
    output = Path(config["output"])
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing result directory {output}")
    start = time.perf_counter()
    seed = int(config["seed"])
    rng = np.random.default_rng(seed)
    full_vocabulary = bool(config.get("full_vocabulary", False))
    engine = sparse_backend if config.get("backend") == "sparse" else dense_backend
    if full_vocabulary and engine is not sparse_backend:
        raise ValueError("full vocabulary requires the memory-safe sparse backend")
    influence_function = (sparse_influence.source_influence_disagreement
                          if engine is sparse_backend else source_influence_disagreement)
    corpora = config["corpora"]
    split_names = [
        config["auxiliary_split"], config["reference_split"], config["evaluation_split"],
    ]
    for replicate in config["experimental_replicates"]:
        split_names += [f"replicate{replicate}_train_a", f"replicate{replicate}_train_b"]

    data: dict[str, dict[str, tuple[np.ndarray, list[dict]]]] = {}
    manifests = {}
    for corpus_index, (corpus, path) in enumerate(corpora.items()):
        root = Path(path)
        manifest_path = root / "confirmatory_split_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifests[corpus] = {
            "path": str(manifest_path),
            "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "source": manifest["source"],
            "source_unit": "paragraph" if corpus == "wikitext" else "article" if corpus == "cnn_dailymail" else "story",
            "chunk_cap_per_document": manifest.get("max_chunks_per_document"),
            "tokenizer": manifest["tokenizer"],
        }
        data[corpus] = {name: load_split(root, name) for name in split_names}
        check_disjoint(data[corpus])
        if config.get("shuffle_within_chunk", False):
            shuffle_rng = np.random.default_rng(seed + 100000 + corpus_index)
            data[corpus] = {
                name: (
                    np.take_along_axis(
                        ids, np.argsort(shuffle_rng.random(ids.shape), axis=1), axis=1,
                    ),
                    records,
                )
                for name, (ids, records) in data[corpus].items()
            }
        if manifest["max_length"] != 64:
            raise ValueError("this experiment requires length 64")

    vocabulary_size = max(int(ids.max()) for splits in data.values()
                          for ids, _ in splits.values()) + 1
    pooled = np.zeros(vocabulary_size, dtype=np.int64)
    for splits in data.values():
        pooled += token_counts(splits[config["auxiliary_split"]][0], vocabulary_size).astype(np.int64)
    if full_vocabulary:
        tokenizers = [m["tokenizer"] for m in manifests.values()]
        if len({t["vocabulary_sha256"] for t in tokenizers}) != 1:
            raise ValueError("full-vocabulary corpora must share the same tokenizer")
        tokenizer = tokenizers[0]
        excluded = {tokenizer["mask_token_id"], tokenizer["pad_token_id"]}
        content_ids = sorted(set(range(tokenizer["vocab_size"])) - excluded)
        if content_ids != list(range(len(content_ids))):
            raise ValueError("content IDs must be contiguous; explicit remapping is needed")
        if vocabulary_size > len(content_ids):
            raise ValueError("artificial mask/pad IDs appear in unmasked corpus data")
        ranked = np.asarray(content_ids, dtype=np.int32)
        class_map = ranked.copy()
        top_count = -1  # No OTHER target exists.
        classes = len(ranked)
    else:
        top_count = int(config["vocabulary_classes"]) - 1
        ranked = np.argsort(-pooled, kind="stable")[:top_count]
        class_map = np.full(vocabulary_size, top_count, dtype=np.int32)
        class_map[ranked] = np.arange(top_count, dtype=np.int32)
        classes = len(ranked) + 1
    offsets = offset_list(int(config["lag_radius"]))

    output.mkdir(parents=True)
    code_paths = (Path(__file__), Path(dense_backend.__file__),
                  Path(sparse_backend.__file__), Path(sparse_influence.__file__))
    code_sources = {str(path): path.read_bytes() for path in code_paths}
    (output / "code").mkdir()
    for path, content in code_sources.items():
        (output / "code" / Path(path).name).write_bytes(content)
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output / "class_map.json").write_text(json.dumps({
        "token_ids_in_class_order": ranked.tolist(),
        "other_class": None if full_vocabulary else top_count,
        "selection": "all tokenizer content IDs, including EOS; artificial MASK/PAD excluded"
        if full_vocabulary else "pooled auxiliary token frequency, stable token-ID tie break",
    }, indent=2), encoding="utf-8")
    rows = []
    input_rows = []
    for corpus, splits in data.items():
        auxiliary_ids, auxiliary_records = splits[config["auxiliary_split"]]
        reference_ids, reference_records = splits[config["reference_split"]]
        eval_ids, eval_records = splits[config["evaluation_split"]]
        auxiliary_ids = class_map[auxiliary_ids]
        reference_ids = class_map[reference_ids]
        eval_ids = class_map[eval_ids[:int(config["evaluation_chunks"])]]
        eval_records = eval_records[:len(eval_ids)]
        fit_ids, weight_ids = split_auxiliary(auxiliary_ids, auxiliary_records, rng)
        prior_counts = token_counts(fit_ids, classes)
        prior = prior_counts / prior_counts.sum()
        gram, cross = engine.fitting_equations(
            fit_ids, weight_ids, prior, offsets, float(config["smoothing"]),
            0.5, rng,
        )
        reference_groups = source_groups(reference_records)
        large_reference_tables = None
        if config.get("large_reference_ceiling", False):
            pool = np.concatenate([
                splits[name][0] for name in (
                    "replicate1_train_a", "replicate1_train_b",
                    "replicate2_train_a", "replicate2_train_b",
                )
            ])
            large_reference_tables = engine.conditional_tables(
                engine.pair_counts(class_map[pool], offsets, len(prior)),
                prior, float(config["smoothing"]),
            )
        partitions = None
        if config.get("random_repartitions"):
            partitions = make_repartitions(
                splits, int(config["random_repartitions"]),
                max(int(value) for value in config["chunk_counts"]), rng,
            )
        if config.get("nonoverlap_pool_pairs"):
            if partitions is not None or config.get("size_pairs"):
                raise ValueError("nonoverlap_pool_pairs needs equal sizes and no repartitions")
            if len(config["chunk_counts"]) != 1:
                raise ValueError("use one fixed chunk count for nonoverlap pairs")
            partitions = make_nonoverlap_pairs(
                splits, int(config["nonoverlap_pool_pairs"]),
                int(config["chunk_counts"][0]), rng,
            )
        print(f"{corpus}: aux={len(auxiliary_ids)}, reference={len(reference_ids)} "
              f"in {len(reference_groups)} sources, eval={len(eval_ids)}, "
              f"top-token coverage={1 - np.mean(eval_ids == top_count):.3f}", flush=True)

        size_pairs = config.get("size_pairs") or [
            [int(n), int(n)] for n in config["chunk_counts"]
        ]
        for n_a, n_b in size_pairs:
            n_a, n_b = int(n_a), int(n_b)
            for q in config["mask_rates"]:
                mask = rng.random(eval_ids.shape) < q
                empty = np.where(~mask.any(axis=1))[0]
                mask[empty, 0] = True
                target_rows, target_positions = np.nonzero(mask)
                targets = eval_ids[target_rows, target_positions]
                baseline_point = 1 - 2 * prior[targets] + np.dot(prior, prior)
                baseline_sequence = per_sequence_squared_error(
                    target_rows, baseline_point, len(eval_ids),
                )
                context_surprisal = (
                    visible_context_surprisal(eval_ids, mask, offsets, prior)
                    if config.get("save_input_details", False) else None
                )
                large_reference_skill = None
                if large_reference_tables is not None:
                    large_loss, _ = scored_prediction(
                        eval_ids, mask, large_reference_tables,
                        prior, offsets, gram, cross,
                        float(config["stacking_ridge"]),
                        backend=engine,
                    )
                    large_reference_skill = float(
                        1 - np.mean(large_loss) / np.mean(baseline_sequence)
                    )
                side_laws = []
                for sample_size in ([n_a] if n_a == n_b else [n_a, n_b]):
                    law_values = {}
                    if config.get("analytic_source_influence", False):
                        for label, same_lag in (
                            ("analytic_source_influence", False),
                            ("analytic_same_lag", True),
                        ):
                            value = influence_function(
                                reference_ids, reference_groups, eval_ids, mask,
                                prior, offsets, gram, cross, n=sample_size,
                                smoothing=float(config["smoothing"]),
                                ridge=float(config["stacking_ridge"]),
                                same_lag=same_lag,
                            )
                            law_values[label] = value
                    for law in config["laws"]:
                        law_rng = np.random.default_rng(
                            (int(rng.integers(2**32)) + args.law_seed_offset) % (2**32)
                        )
                        # Reserve the same RNG slots as the original bootstrap run.
                        # This keeps subsequent masks, auxiliary splits and pair
                        # allocations identical without repeating Monte Carlo work.
                        if analytic_only:
                            continue
                        value, _ = engine.bootstrap_disagreement(
                            reference_ids, reference_groups, eval_ids, mask,
                            prior, offsets, gram, cross, n=sample_size,
                            smoothing=float(config["smoothing"]),
                            ridge=float(config["stacking_ridge"]),
                            draws=int(config["bootstrap_draws"]),
                            law=law, rng=law_rng,
                        )
                        law_values[law] = value
                    side_laws.append(law_values)
                laws = side_laws[0] if n_a == n_b else {
                    label: combine_independent_side_laws(side_laws[0][label], side_laws[1][label])
                    for label in side_laws[0]
                }
                for replicate in (
                    range(1, int(config.get("random_repartitions") or
                                 config.get("nonoverlap_pool_pairs")) + 1)
                    if partitions is not None else config["experimental_replicates"]
                ):
                    if partitions is None:
                        a_ids, a_records = splits[f"replicate{replicate}_train_a"]
                        b_ids, b_records = splits[f"replicate{replicate}_train_b"]
                    else:
                        (a_ids, a_records), (b_ids, b_records) = partitions[replicate]
                    a_ids, b_ids = class_map[a_ids[:n_a]], class_map[b_ids[:n_b]]
                    if len(a_ids) != n_a or len(b_ids) != n_b:
                        raise ValueError("insufficient training chunks")
                    outputs = []
                    losses = []
                    point_losses = []
                    for ids in (a_ids, b_ids):
                        tables = engine.conditional_tables(
                            engine.pair_counts(ids, offsets, len(prior)),
                            prior, float(config["smoothing"]),
                        )
                        loss, predicted = scored_prediction(
                            eval_ids, mask, tables, prior, offsets, gram, cross,
                            float(config["stacking_ridge"]),
                            backend=engine,
                        )
                        losses.append(loss)
                        outputs.append(predicted)
                        point_losses.append(squared_onehot_loss(predicted, targets))
                    if engine is sparse_backend:
                        observed_point = sparse_backend.squared_difference(outputs[0], outputs[1])
                    else:
                        observed_point = np.einsum("ij,ij->i", outputs[0] - outputs[1], outputs[0] - outputs[1])
                    observed = per_sequence_squared_error(target_rows, observed_point, len(eval_ids))
                    named = targets != top_count
                    average_point_loss = 0.5 * (point_losses[0] + point_losses[1])
                    named_skill = 1 - float(average_point_loss[named].mean() /
                                            baseline_point[named].mean()) if named.any() else float("nan")
                    other_skill = 1 - float(average_point_loss[~named].mean() /
                                            baseline_point[~named].mean()) if (~named).any() else float("nan")
                    row = {
                        "corpus": corpus, "replicate": replicate,
                        "n_chunks": n_a if n_a == n_b else "",
                        "n_a_chunks": n_a, "n_b_chunks": n_b,
                        "mask_rate": q, "radius": config["lag_radius"],
                        "classes": len(prior), "eval_chunks": len(eval_ids),
                        "masked_tokens": len(targets),
                        "eval_top_token_coverage": float(1 - np.mean(eval_ids == top_count)),
                        "baseline_loss": float(np.mean(baseline_sequence)),
                        "reconstruction_loss": float(np.mean(losses)),
                        "skill": float(1 - np.mean(losses) / np.mean(baseline_sequence)),
                        "named_target_fraction": float(named.mean()),
                        "named_target_skill": named_skill,
                        "other_target_skill": other_skill,
                        "named_baseline_loss": float(baseline_point[named].mean()) if named.any() else float("nan"),
                        "other_baseline_loss": float(baseline_point[~named].mean()) if (~named).any() else float("nan"),
                        "pooled_8192_skill": large_reference_skill,
                        "observed_disagreement": float(np.mean(observed)),
                        "observed_over_baseline": float(np.mean(observed) / np.mean(baseline_sequence)),
                        "sources_a": len({record["document_id"] for record in a_records[:n_a]}),
                        "sources_b": len({record["document_id"] for record in b_records[:n_b]}),
                        "reference_sources": len(reference_groups),
                        "runtime_seconds": time.perf_counter() - start,
                        "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
                        "evaluation_sha256": hashlib.sha256(
                            json.dumps([r["sha256"] for r in eval_records]).encode()
                        ).hexdigest(),
                        "side_a_sha256": hashlib.sha256(
                            json.dumps([r["sha256"] for r in a_records[:n_a]]).encode()
                        ).hexdigest(),
                        "side_b_sha256": hashlib.sha256(
                            json.dumps([r["sha256"] for r in b_records[:n_b]]).encode()
                        ).hexdigest(),
                        "auxiliary_fit_sha256": hashlib.sha256(fit_ids.tobytes()).hexdigest(),
                        "zero_prior_target_fraction": float(np.mean(prior[targets] == 0)),
                        "fitting_mask_rate": 0.5,
                    }
                    for law, per_sequence in laws.items():
                        prediction = float(np.mean(per_sequence))
                        row[f"{law}_prediction"] = prediction
                        row[f"{law}_ratio"] = prediction / max(row["observed_disagreement"], 1e-30)
                        row[f"{law}_inputwise_correlation"] = float(np.corrcoef(per_sequence, observed)[0, 1])
                    rows.append(row)
                    if context_surprisal is not None:
                        for sequence_index in range(len(eval_ids)):
                            detail = {
                                "corpus": corpus, "replicate": replicate,
                                "n_chunks": n_a if n_a == n_b else "",
                                "n_a_chunks": n_a, "n_b_chunks": n_b, "mask_rate": q,
                                "sequence_index": sequence_index,
                                "visible_context_surprisal": context_surprisal[sequence_index],
                                "baseline_loss": baseline_sequence[sequence_index],
                                "reconstruction_loss": 0.5 * (
                                    losses[0][sequence_index] + losses[1][sequence_index]
                                ),
                                "observed_disagreement": observed[sequence_index],
                            }
                            for law, per_sequence in laws.items():
                                detail[f"{law}_prediction"] = per_sequence[sequence_index]
                            input_rows.append(detail)
                    if large_reference_skill is not None and large_reference_skill > 0:
                        row["fraction_of_pooled_gain"] = row["skill"] / large_reference_skill
                    else:
                        row["fraction_of_pooled_gain"] = None
                    message = f"  n={n_a}:{n_b} q={q} pair={replicate}: skill={row['skill']:.3%}"
                    if "analytic_source_influence_ratio" in row:
                        message += f", analytic ratio={row['analytic_source_influence_ratio']:.3f}"
                    if "source_cluster_ratio" in row:
                        message += f", cluster ratio={row['source_cluster_ratio']:.3f}"
                    print(message, flush=True)
                    # Checkpoint measured rows. A completed metadata.json is written only at the end.
                    with (output / "pair_results.csv").open("w", newline="", encoding="utf-8") as handle:
                        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                        writer.writeheader()
                        writer.writerows(rows)

    with (output / "pair_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    if input_rows:
        with (output / "input_results.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=input_rows[0].keys())
            writer.writeheader()
            writer.writerows(input_rows)
    hardware = {"processor": platform.processor(), "logical_cpus": os.cpu_count()}
    try:
        import psutil
        memory = psutil.Process().memory_info()
        hardware.update({
            "total_ram_bytes": psutil.virtual_memory().total,
            "peak_working_set_bytes": getattr(memory, "peak_wset", None),
            "ending_resident_bytes": memory.rss,
        })
    except ImportError:
        hardware["memory_note"] = "optional psutil unavailable"
    (output / "metadata.json").write_text(json.dumps({
        "command": " ".join(sys.argv), "python": platform.python_version(),
        "numpy": np.__version__, "torch": torch.__version__,
        "duration_seconds": time.perf_counter() - start,
        "manifest_provenance": manifests,
        "sampling_law": (
            "first-order reference-source influence covariance; full and same-lag"
            if analytic_only else
            "nonlinear nonparametric reference-source bootstrap; plug-in source distribution"
        ),
        "analytic_source_influence": bool(config.get("analytic_source_influence", False)),
        "analytic_only": analytic_only,
        "rng_bootstrap_seed_slots_reserved": analytic_only,
        "pair_independence": (
            "one allocation of source-disjoint nonoverlapping pairs from the fixed "
            "experimental source pool; conditional uncertainty given the shared "
            "auxiliary, reference and evaluation samples"
            if config.get("nonoverlap_pool_pairs") else
            "conditional random repartitions of one fixed experimental document pool; "
            "A/B are source-disjoint within a repartition but repartitions reuse sources"
            if config.get("random_repartitions") else
            "disjoint original A/B pairs per corpus; shared reference, auxiliary and evaluation splits"
        ),
        "wikitext_caveat": "cached source units are paragraphs; articles may span multiple source IDs",
        "monte_carlo_draws_per_law": 0 if analytic_only else config["bootstrap_draws"],
        "shuffle_within_chunk": bool(config.get("shuffle_within_chunk", False)),
        "backend": "sparse_plus_prior_exact" if engine is sparse_backend else "dense",
        "full_vocabulary": full_vocabulary,
        "classes": classes,
        "auxiliary_weight_fitting_mask_rate": 0.5,
        "weights_shared_across_mask_rates": True,
        "hardware": hardware,
        "thread_settings": {key: os.environ.get(key) for key in
                            ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        "code_sha256": {
            path: hashlib.sha256(content).hexdigest()
            for path, content in code_sources.items()
        },
    }, indent=2), encoding="utf-8")
    print(f"results: {output} ({time.perf_counter() - start:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
