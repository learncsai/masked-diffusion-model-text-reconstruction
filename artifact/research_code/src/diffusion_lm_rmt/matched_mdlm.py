"""Matched corpus-variation pilot with a small masked diffusion denoiser.

The analytic reference calculation belongs to the lag reconstructor only.
Neural results are observations conditional on a shared auxiliary checkpoint.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import sparse_categorical_influence as influence
from . import sparse_categorical_lag as lag
from .categorical_lag import offset_list, per_sequence_squared_error, source_groups, token_counts
from .diffusion.model import DenoiserConfig, MaskedDiffusionTransformer


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def object_digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def save_torch(path, obj):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, temporary)
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def audit_roles(roles, classes, document_hashes):
    """Check actual chunk bytes and disjoint source IDs, texts, and chunks."""
    sets, result = {}, {}
    for name, (ids, records) in roles.items():
        if ids.ndim != 2 or len(ids) != len(records) or np.any((ids < 0) | (ids >= classes)):
            raise ValueError(f"invalid token array or records: {name}")
        hashes = [hashlib.sha256(np.asarray(row, dtype=np.int32).tobytes()).hexdigest() for row in ids]
        if hashes != [record["sha256"] for record in records]:
            raise ValueError(f"chunk bytes do not match recorded hashes: {name}")
        sources = {record["document_id"] for record in records}
        text_hashes = {document_hashes[source] for source in sources}
        current = (sources, text_hashes, set(hashes))
        for other, previous in sets.items():
            if any(left & right for left, right in zip(current, previous)):
                raise ValueError(f"source ID, source text, or exact chunk overlap: {name}, {other}")
        sets[name] = current
        result[name] = dict(chunks=len(ids), sources=len(sources), source_ids=sorted(sources))
    return result


def load_inputs(root, config):
    data = root / "data/diffusion_llm"
    original = data / "processed_64_tinystories_matched"
    fresh = data / "reproducibility_followup_v1/tinystories"
    calibration = data / "calibration_diagnostics_v1/tinystories"
    paths = [original / "replicate3_train_a.pt", original / "eval.pt",
             original / "confirmatory_split_manifest.json",
             fresh / "raw_documents.jsonl", data / "gpt2_tokenizer/tokenizer_metadata.json",
             calibration / "grid_512_512_q0.5.npz",
             calibration / "reference_original.npz", calibration / "reference_original.json"]
    auxiliary = torch.load(paths[0], map_location="cpu", weights_only=True)
    evaluation = torch.load(paths[1], map_location="cpu", weights_only=True)
    if not torch.all(auxiliary["attention_mask"] == 1) or not torch.all(evaluation["attention_mask"] == 1):
        raise ValueError("this pilot requires full unpadded chunks")
    aux_ids = auxiliary["input_ids"].numpy()
    aux_records = auxiliary["chunk_records"]
    groups = source_groups(aux_records)
    np.random.default_rng(config["seed"]).shuffle(groups)
    boundary = len(groups) // 2
    left, right = np.concatenate(groups[:boundary]), np.concatenate(groups[boundary:])
    roles = {"auxiliary_train": (aux_ids[left], [aux_records[i] for i in left]),
             "auxiliary_validation": (aux_ids[right], [aux_records[i] for i in right])}
    with np.load(calibration / "grid_512_512_q0.5.npz") as saved:
        ids, mask = saved["eval_ids"][:config["evaluation_chunks"]], saved["mask"][:config["evaluation_chunks"]]
    if not np.array_equal(ids, evaluation["input_ids"].numpy()[:len(ids)]):
        raise ValueError("saved evaluation context does not match evaluation records")
    if len(ids) != config["evaluation_chunks"] or not np.all(mask.sum(1) > 0):
        raise ValueError("invalid evaluation size or empty masked chunk")
    roles["evaluation"] = ids, evaluation["chunk_records"][:len(ids)]
    with np.load(calibration / "reference_original.npz") as saved:
        roles["reference"] = saved["ids"], read_json(calibration / "reference_original.json")
    for pair in range(1, config["pairs"] + 1):
        for arm in ("a", "b"):
            path = fresh / f"pair{pair}_{arm}.npz"
            paths.extend([path, path.with_suffix(".json")])
            with np.load(path) as saved:
                ids = saved["ids"][:max(config["corpus_sizes"])]
            if len(ids) != max(config["corpus_sizes"]):
                raise ValueError("not enough chunks for requested corpus size")
            roles[f"pair{pair}_{arm}"] = ids, read_json(path.with_suffix(".json"))[:len(ids)]
    manifest = read_json(original / "confirmatory_split_manifest.json")
    document_hashes = {doc["document_id"]: doc["sha256"]
                       for split in manifest["splits"].values() for doc in split["documents"]}
    with (fresh / "raw_documents.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            doc = json.loads(line)
            actual = hashlib.sha256(doc["text"].encode()).hexdigest()
            if actual != doc["sha256"]:
                raise ValueError("raw source hash mismatch")
            document_hashes[doc["document_id"]] = actual
    tokenizer = read_json(data / "gpt2_tokenizer/tokenizer_metadata.json")
    if tokenizer["mask_token_id"] != config["mask_token_id"] or tokenizer["eos_token_id"] != config["classes"] - 1:
        raise ValueError("tokenizer mismatch")
    audit = audit_roles(roles, config["classes"], document_hashes)
    return roles, mask, paths, audit


def prepare(root, config):
    out = root / config["output"]
    out.mkdir(parents=True, exist_ok=True)
    roles, mask, paths, audit = load_inputs(root, config)
    code = sorted((root / "src/diffusion_lm_rmt").rglob("*.py")) + [root / "scripts/run_matched_mdlm_pilot.py"]
    protocol = dict(config=config, inputs={str(p.relative_to(root)): digest(p) for p in paths},
                    code={str(p.relative_to(root)): digest(p) for p in code})
    signature = object_digest(protocol)
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["protocol_sha256"] != signature:
            raise ValueError("configuration, code, or inputs changed. Use a new output directory.")
        if (out / "prepared.json").exists():
            for name, sha in read_json(out / "prepared.json")["sha256"].items():
                if digest(out / name) != sha:
                    raise ValueError(f"prepared artifact changed: {name}")
            return roles, signature
    else:
        write_json(manifest_path, dict(protocol_sha256=signature, frozen_utc=utc(), protocol=protocol,
            git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
            design="Exploratory pilot. Lag outcomes on these saved pairs were already known."))
    write_json(out / "source_audit.json", dict(roles=audit, overlaps=0,
        independence="Sources are disjoint across roles and A/B sides. Sizes reuse nested prefixes within a side."))
    ids, _ = roles["auxiliary_train"]
    val, _ = roles["auxiliary_validation"]
    prior = token_counts(ids, config["classes"])
    prior /= prior.sum()
    offsets = offset_list(config["radius"])
    gram, cross = lag.fitting_equations(ids, val, prior, offsets, config["smoothing"],
        config["mask_rate"], np.random.default_rng(config["seed"] + 1))
    evaluation = roles["evaluation"][0]
    validation = val[:config["validation_chunks"]]
    validation_mask = np.random.default_rng(config["seed"] + 2).random(validation.shape) < config["mask_rate"]
    if not np.all(validation_mask.sum(1) > 0):
        raise ValueError("empty validation mask")
    np.savez_compressed(out / "context.npz", evaluation=evaluation, mask=mask, prior=prior,
        gram=gram, cross=cross, validation=validation, validation_mask=validation_mask)
    reference, records = roles["reference"]
    forecasts = []
    for n in config["corpus_sizes"]:
        prediction = float(np.mean(influence.source_influence_disagreement(reference, source_groups(records),
            evaluation, mask, prior, offsets, gram, cross, n=n, smoothing=config["smoothing"], ridge=config["ridge"])))
        forecasts.append(dict(n=n, lag_prediction=prediction))
        print(json.dumps(dict(phase="reference", n=n, prediction=prediction)), flush=True)
    # Save every reference prediction before computing the A/B scores in this run.
    write_json(out / "lag_forecasts.json", dict(saved_utc=utc(), predictions=forecasts))
    rows = np.nonzero(mask)[0]
    targets = evaluation[mask]
    baseline = per_sequence_squared_error(rows, 1 - 2 * prior[targets] + prior @ prior, len(evaluation))
    lag_rows = []
    for n in config["corpus_sizes"]:
        prediction = next(f["lag_prediction"] for f in forecasts if f["n"] == n)
        for pair in range(1, config["pairs"] + 1):
            outputs, losses = [], []
            for arm in ("a", "b"):
                train = roles[f"pair{pair}_{arm}"][0][:n]
                tables = lag.conditional_tables(lag.pair_counts(train, offsets, config["classes"]), prior, config["smoothing"])
                _, actual_targets, output = lag.masked_prediction(evaluation, mask, tables, prior, offsets, gram, cross, config["ridge"])
                if not np.array_equal(targets, actual_targets):
                    raise ValueError("lag target mismatch")
                outputs.append(output)
                losses.append(per_sequence_squared_error(rows, output.loss(targets), len(evaluation)))
            disagreement = per_sequence_squared_error(rows, lag.squared_difference(*outputs), len(evaluation))
            result = score_summary(losses[0], losses[1], disagreement, baseline)
            result.update(n=n, pair=pair, method="lag_radius_2", lag_prediction=prediction,
                          prediction_observation_ratio=prediction / result["disagreement"],
                          per_chunk=dict(loss_a=losses[0].tolist(), loss_b=losses[1].tolist(),
                                         disagreement=disagreement.tolist(), baseline=baseline.tolist()))
            lag_rows.append(result)
    write_json(out / "lag_pairs.json", lag_rows)
    write_json(out / "prepared.json", dict(completed_utc=utc(), protocol_sha256=signature,
        sha256={name: digest(out / name) for name in ("context.npz", "source_audit.json", "lag_forecasts.json", "lag_pairs.json")}))
    return roles, signature


def score_summary(loss_a, loss_b, disagreement, baseline):
    baseline_mean = float(np.mean(baseline))
    return dict(loss_a=float(np.mean(loss_a)), loss_b=float(np.mean(loss_b)),
                baseline_loss=baseline_mean, relative_improvement_a=1 - float(np.mean(loss_a)) / baseline_mean,
                relative_improvement_b=1 - float(np.mean(loss_b)) / baseline_mean,
                disagreement=float(np.mean(disagreement)))


def chunk_metrics(probabilities_a, probabilities_b, targets, mask):
    """Class-summed Brier loss and disagreement, then equal chunk averages."""
    if not torch.all(mask.sum(1) > 0):
        raise ValueError("each scored chunk must contain a hidden token")
    def reduce(values):
        return (values.double() * mask).sum(1) / mask.sum(1)
    def loss(probabilities):
        return reduce(1 + probabilities.square().sum(-1)
                      - 2 * probabilities.gather(-1, targets.unsqueeze(-1)).squeeze(-1))
    return loss(probabilities_a), loss(probabilities_b), reduce((probabilities_a - probabilities_b).square().sum(-1))


def weighted_objective(logits, clean, mask, times):
    # No extra mask is forced when the Bernoulli draw is empty.
    ce = F.cross_entropy(logits.float().transpose(1, 2), clean, reduction="none")
    return (((ce * mask).sum(1) / clean.shape[1]) / times).mean()


def learning_rate(step, total, config, *, auxiliary=False):
    train = config["training"]
    warmup = min(train["warmup_updates"], max(1, total // 2))
    first_stage = total if auxiliary else train["checkpoints"][0]
    if step <= warmup:
        return train["learning_rate"] * step / warmup
    fraction = min(1., max(0., (step - warmup) / max(1, first_stage - warmup)))
    return train["minimum_learning_rate"] + .5 * (train["learning_rate"] - train["minimum_learning_rate"]) * (1 + math.cos(math.pi * fraction))


def load_model(config, checkpoint, device):
    model = MaskedDiffusionTransformer(DenoiserConfig(**config["model"]))
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model"])
    return model.to(device).eval()


@torch.inference_mode()
def evaluate_models(models, ids, mask, prior, config, device):
    results = [[], [], []]
    batch = config["training"]["microbatch_size"]
    for start in range(0, len(ids), batch):
        clean = torch.tensor(ids[start:start + batch], dtype=torch.long, device=device)
        hidden = torch.tensor(mask[start:start + batch], dtype=torch.bool, device=device)
        corrupted = clean.masked_fill(hidden, config["mask_token_id"])
        attention = torch.ones_like(clean, dtype=torch.bool)
        times = torch.full((len(clean),), config["mask_rate"], device=device)
        probabilities = [torch.softmax(model(corrupted, attention, times)[..., :config["classes"]], -1) for model in models]
        if len(probabilities) == 1:
            probabilities.append(probabilities[0])
        for values, collected in zip(chunk_metrics(*probabilities, clean, hidden), results):
            collected.extend(values.cpu().tolist())
    rows = np.nonzero(mask)[0]
    baseline = per_sequence_squared_error(rows, 1 - 2 * prior[ids[mask]] + prior @ prior, len(ids))
    summary = score_summary(*results, baseline)
    summary["per_chunk"] = dict(loss_a=results[0], loss_b=results[1], disagreement=results[2], baseline=baseline.tolist())
    if not all(np.all(np.isfinite(x)) for x in results):
        raise FloatingPointError("nonfinite evaluation score")
    return summary


def train_job(root, config, signature, name, ids, context, device, *, parent=None, seed_offset=0, stop_after=None):
    """Atomic optimizer/RNG checkpoints support resuming at the next update."""
    out = root / config["output"]
    folder = out / "models" / name
    folder.mkdir(parents=True, exist_ok=True)
    train = config["training"]
    auxiliary = parent is None
    total = train["auxiliary_updates"] if auxiliary else max(train["checkpoints"])
    milestones = [total] if auxiliary else train["checkpoints"]
    data_hash = hashlib.sha256(np.asarray(ids, dtype=np.int32).tobytes()).hexdigest()
    identity = dict(protocol_sha256=signature, name=name, data_sha256=data_hash,
                    parent_sha256=digest(parent) if parent else None, seed_offset=seed_offset)
    complete = folder / "complete.json"
    if complete.exists():
        info = read_json(complete)
        if info["identity"] != identity:
            raise ValueError(f"completed job identity mismatch: {name}")
        for filename, sha in info["sha256"].items():
            if digest(folder / filename) != sha:
                raise ValueError(f"checkpoint changed: {name}/{filename}")
        return folder / f"step_{total}.pt"
    torch.manual_seed(config["seed"] + seed_offset)
    model = MaskedDiffusionTransformer(DenoiserConfig(**config["model"])).to(device)
    if parent:
        model.load_state_dict(torch.load(parent, map_location="cpu", weights_only=True)["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=train["learning_rate"],
        betas=tuple(train["betas"]), weight_decay=train["weight_decay"])
    data_rng = torch.Generator().manual_seed(config["seed"] + 101 + seed_offset)
    mask_rng = torch.Generator().manual_seed(config["seed"] + 102 + seed_offset)
    last_step, elapsed_before = 0, 0.
    resume_path = folder / "resume.pt"
    if resume_path.exists():
        saved = torch.load(resume_path, map_location="cpu", weights_only=True)
        if saved["identity"] != identity:
            raise ValueError(f"resume identity mismatch: {name}")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        data_rng.set_state(saved["data_rng"])
        mask_rng.set_state(saved["mask_rng"])
        last_step, elapsed_before = saved["step"], saved["elapsed_seconds"]
        if last_step in milestones and not (folder / f"step_{last_step}.pt").exists():
            save_torch(folder / f"step_{last_step}.pt", dict(identity=identity, model=model.state_dict(), step=last_step))
    tokens = torch.tensor(ids, dtype=torch.long)
    log_path = folder / "training.jsonl"
    # Remove any log rows written after the last durable checkpoint.
    if log_path.exists():
        retained = []
        for line in log_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row["step"] <= last_step:
                retained.append(row)
        log_path.write_text("".join(json.dumps(row) + "\n" for row in retained), encoding="utf-8")
    started = time.perf_counter()
    print(json.dumps(dict(phase="train", job=name, start_step=last_step, total_steps=total, parameters=model.parameter_count)), flush=True)
    stop = min(total, stop_after) if stop_after is not None else total
    for step in range(last_step + 1, stop + 1):
        model.train()
        lr = learning_rate(step, total, config, auxiliary=auxiliary)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.
        for _ in range(train["gradient_accumulation"]):
            clean = tokens[torch.randint(len(tokens), (train["microbatch_size"],), generator=data_rng)]
            times = train["time_min"] + (train["time_max"] - train["time_min"]) * torch.rand(len(clean), generator=mask_rng)
            mask = torch.rand(clean.shape, generator=mask_rng) < times[:, None]
            corrupted = clean.masked_fill(mask, config["mask_token_id"]).to(device)
            logits = model(corrupted, torch.ones_like(corrupted, dtype=torch.bool), times.to(device))
            loss = weighted_objective(logits[..., :config["classes"]], clean.to(device), mask.to(device), times.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite training loss: {name} step {step}")
            (loss / train["gradient_accumulation"]).backward()
            loss_value += float(loss.detach()) / train["gradient_accumulation"]
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(norm):
            raise FloatingPointError(f"nonfinite gradient: {name} step {step}")
        optimizer.step()
        elapsed = elapsed_before + time.perf_counter() - started
        if step == 1 or step % train["log_every"] == 0 or step in milestones or step == stop:
            row = dict(utc=utc(), job=name, step=step, total_steps=total, training_loss=loss_value,
                       gradient_norm=float(norm), learning_rate=lr, elapsed_seconds=elapsed,
                       seconds_per_update=elapsed / step)
            if step % train["validation_every"] == 0 or step in milestones:
                model.eval()
                score = evaluate_models([model], context["validation"], context["validation_mask"],
                                        context["prior"], config, device)
                row.update(validation_loss=score["loss_a"], validation_relative_improvement=score["relative_improvement_a"])
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            write_json(out / "progress.json", dict(status="running", **row))
            print(json.dumps(row), flush=True)
        if step % train["save_every"] == 0 or step in milestones or step == stop:
            saved = dict(identity=identity, model=model.state_dict(), optimizer=optimizer.state_dict(),
                         data_rng=data_rng.get_state(), mask_rng=mask_rng.get_state(), step=step, elapsed_seconds=elapsed)
            save_torch(resume_path, saved)
            if step in milestones:
                save_torch(folder / f"step_{step}.pt", dict(identity=identity, model=model.state_dict(), step=step))
    if stop == total:
        write_json(complete, dict(identity=identity, completed_utc=utc(), steps=total,
                   sha256={f"step_{step}.pt": digest(folder / f"step_{step}.pt") for step in milestones}))
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return folder / f"step_{total}.pt"


def evaluate_pair(root, config, name, checkpoint_a, checkpoint_b, context, device, metadata):
    out = root / config["output"]
    path = out / "scores" / f"{name}.json"
    identity = dict(checkpoint_a_sha256=digest(checkpoint_a), checkpoint_b_sha256=digest(checkpoint_b),
                    context_sha256=digest(out / "context.npz"))
    if path.exists():
        if read_json(path)["identity"] != identity:
            raise ValueError(f"saved score identity mismatch: {name}")
        return
    models = [load_model(config, checkpoint, device) for checkpoint in (checkpoint_a, checkpoint_b)]
    result = evaluate_models(models, context["evaluation"], context["mask"], context["prior"], config, device)
    result.update(identity=identity, completed_utc=utc(), **metadata)
    write_json(path, result)
    del models
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(json.dumps(dict(phase="score", name=name, disagreement=result["disagreement"],
                         relative_improvement_a=result["relative_improvement_a"],
                         relative_improvement_b=result["relative_improvement_b"])), flush=True)


def summarize(root, config):
    out = root / config["output"]
    rows = read_json(out / "lag_pairs.json")
    rows += [read_json(path) for path in sorted((out / "scores").glob("*.json"))]
    summary = []
    cells = sorted({(row["method"], row.get("updates", 0), row["n"]) for row in rows})
    for method, updates, n in cells:
        cell = [row for row in rows if (row["method"], row.get("updates", 0), row["n"]) == (method, updates, n)]
        summary.append(dict(method=method, updates=updates, n=n, pairs=len(cell),
            mean_disagreement=float(np.mean([row["disagreement"] for row in cell])),
            mean_mse=float(np.mean([(row["loss_a"] + row["loss_b"]) / 2 for row in cell])),
            mean_relative_improvement=float(np.mean([(row["relative_improvement_a"] + row["relative_improvement_b"]) / 2 for row in cell])),
            min_pair_disagreement=min(row["disagreement"] for row in cell),
            max_pair_disagreement=max(row["disagreement"] for row in cell)))
    write_json(out / "summary.json", summary)
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    # Compare nested sizes within each pair. Do not treat chunks as independent replicates.
    scaling = []
    base_n = min(config["corpus_sizes"])
    for row in rows:
        if row["method"] == "mdlm_same_corpus_seed_control" or row["n"] == base_n:
            continue
        baseline = next((r for r in rows if r["method"] == row["method"] and r.get("updates", 0) == row.get("updates", 0)
                         and r["n"] == base_n and r["pair"] == row["pair"]), None)
        if baseline and baseline["disagreement"] > 0 and row["disagreement"] > 0:
            scaling.append(dict(method=row["method"], updates=row.get("updates", 0), pair=row["pair"],
                n=row["n"], base_n=base_n, observed_ratio=row["disagreement"] / baseline["disagreement"],
                inverse_n_ratio=base_n / row["n"],
                effective_power=-math.log(row["disagreement"] / baseline["disagreement"]) / math.log(row["n"] / base_n)))
    write_json(out / "scaling.json", scaling)
    lines = ["# TinyStories MDLM pilot", "", "Exploratory results. Each model uses a shared auxiliary initialization.", "",
             "Lower MSE and higher relative MSE improvement indicate better reconstruction.",
             "Lower disagreement indicates greater stability, which need not imply better reconstruction.",
             "The analytic prediction applies to the lag reconstructor. No MDLM theory calibration is claimed.", "",
             "The neural model uses the full visible 64-token chunk. The lag reconstructor uses radius 2.",
             "Training uses a fixed update budget, not a fixed number of passes through each corpus.",
             f"{config['pairs']} source-disjoint corpus pairs are planned. Sizes within each pair are nested.",
             "Seed controls reuse one corpus and do not add corpus replicates. No token-level confidence intervals are used.", "",
             "| Method | Updates | Chunks | Completed pairs | MSE (lower) | Relative MSE improvement (higher) | Disagreement |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['method']} | {row['updates']} | {row['n']} | {row['pairs']} | {row['mean_mse']:.6f} | {row['mean_relative_improvement']:.6f} | {row['mean_disagreement']:.6g} |")
    lines += ["", "## Files", "", "- manifest.json freezes the configuration, input hashes, and code hashes.",
              "- source_audit.json lists the source IDs in each independent role.",
              "- lag_forecasts.json and lag_pairs.json contain reference predictions and lag observations.",
              "- scores/ contains MDLM pair scores and the per-chunk values behind every mean.",
              "- models/ contains checkpoints, optimizer/RNG resume files, and training logs.",
              "- summary.csv and summary.json contain the table above. scaling.json compares nested sizes with 1/n.",
              "- progress.json gives the latest training status. run.json gives the process status.",
              "- context.npz contains the common evaluation text, masks, unigram, and lag weights.",
              "", "Run or resume from the repository root:", "",
              "    .\\.venv-cuda\\Scripts\\python.exe scripts/run_matched_mdlm_pilot.py", "",
              "The cached TinyStories input files listed in manifest.json must be available.",
              "See docs/matched_mdlm_tinystories_pilot.md for the data provenance and interpretation limits."]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(root, config, device, *, prepare_only=False):
    roles, signature = prepare(root, config)
    summarize(root, config)
    if prepare_only:
        return
    out = root / config["output"]
    with np.load(out / "context.npz") as saved:
        context = {key: saved[key] for key in saved.files}
    parent = train_job(root, config, signature, "auxiliary", roles["auxiliary_train"][0], context, device, seed_offset=1000)
    # Only auxiliary validation is used for monitoring. Final test scores cannot select a checkpoint.
    for pair in range(1, config["pairs"] + 1):
        for n in config["corpus_sizes"]:
            for arm in ("a", "b"):
                name = f"pair{pair}_{arm}_n{n}"
                train_job(root, config, signature, name, roles[f"pair{pair}_{arm}"][0][:n], context, device, parent=parent)
            for updates in config["training"]["checkpoints"]:
                checkpoints = [out / "models" / f"pair{pair}_{arm}_n{n}" / f"step_{updates}.pt" for arm in ("a", "b")]
                evaluate_pair(root, config, f"pair{pair}_n{n}_step{updates}", *checkpoints, context, device,
                              dict(method="mdlm_full_context", pair=pair, n=n, updates=updates))
            summarize(root, config)
    if config["training"]["same_corpus_seed_controls"]:
        for n in config["corpus_sizes"]:
            name = f"seed_control_pair1_a_n{n}"
            train_job(root, config, signature, name, roles["pair1_a"][0][:n], context, device, parent=parent, seed_offset=2000)
            for updates in config["training"]["checkpoints"]:
                first = out / "models" / f"pair1_a_n{n}" / f"step_{updates}.pt"
                second = out / "models" / name / f"step_{updates}.pt"
                evaluate_pair(root, config, f"seed_control_n{n}_step{updates}", first, second, context, device,
                              dict(method="mdlm_same_corpus_seed_control", pair=1, n=n, updates=updates))
            summarize(root, config)
    write_json(out / "progress.json", dict(status="complete", completed_utc=utc()))
