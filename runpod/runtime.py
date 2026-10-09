"""Shared budget, configuration, and export helpers for the portable run."""
from __future__ import annotations

import csv
import json
import math
import os
import time
from pathlib import Path

CORPORA = ("tinystories", "wikitext", "cnn_dailymail")
LABELS = dict(tinystories="TinyStories", wikitext="WikiText", cnn_dailymail="CNN/DailyMail")


class BudgetStop(RuntimeError):
    pass


def atomic_replace(source, destination):
    """Allow transient Windows indexer/antivirus locks without weakening atomic writes."""
    for attempt in range(100):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if os.name != "nt" or attempt == 99:
                raise
            time.sleep(.05)


def budget_requested():
    deadline = os.environ.get("MDLM_TRAIN_DEADLINE")
    return deadline is not None and time.time() >= float(deadline)


def require_time():
    if budget_requested():
        raise BudgetStop("Training deadline reached. Resume from durable checkpoints with a new budget.")


def validate_budget(credit, hourly_rate, reserve):
    if not all(math.isfinite(x) for x in (credit, hourly_rate, reserve)):
        raise ValueError("Budget values must be finite")
    if hourly_rate <= 0 or reserve < 0 or credit <= reserve:
        raise ValueError("Require hourly_rate > 0 and credit > reserve >= 0")
    seconds = (credit - reserve) / hourly_rate * 3600
    if seconds <= 300:
        raise ValueError("Budget must leave at least five minutes after the reserve")
    return seconds


def configurations(root, label, mode):
    from pilot_support import read_json, write_json
    if not label or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in label):
        raise ValueError("Run label must contain only letters, digits, underscores, or hyphens")
    if mode not in ("full", "smoke"):
        raise ValueError("Mode must be full or smoke")
    if mode == "smoke" and "smoke" not in label.lower():
        raise ValueError("Smoke runs must have 'smoke' in their label")
    configs = []
    for corpus in CORPORA:
        config = read_json(root / "configs/matched_mdlm_tinystories_pilot_v1.json")
        config.update(name=f"{label}_{corpus}", corpus=corpus, output=f"results/{label}/{corpus}")
        if mode == "smoke":
            config.update(pairs=1, corpus_sizes=[32, 64], evaluation_chunks=4, validation_chunks=4)
            config["training"].update(auxiliary_updates=4, checkpoints=[4, 8], warmup_updates=2,
                microbatch_size=2, gradient_accumulation=1, log_every=2, save_every=4, validation_every=4)
        path = root / "runtime_configs" / f"{label}_{corpus}.json"
        if path.exists() and read_json(path) != config:
            raise ValueError("Run settings changed. Choose a new run label.")
        write_json(path, config)
        configs.append(config)
    return configs


def job_plan(config):
    train = config["training"]
    final = max(train["checkpoints"])
    jobs = [("auxiliary", train["auxiliary_updates"])]
    jobs += [(f"pair{pair}_{arm}_n{n}", final) for pair in range(1, config["pairs"] + 1)
             for n in config["corpus_sizes"] for arm in ("a", "b")]
    if train["same_corpus_seed_controls"]:
        jobs += [(f"seed_control_pair1_a_n{n}", final) for n in config["corpus_sizes"]]
    return jobs


def remaining_updates(root, configs):
    import torch
    result = 0
    for config in configs:
        for name, total in job_plan(config):
            folder = root / config["output"] / "models" / name
            if (folder / "complete.json").exists():
                continue  # Core runner verifies identity and snapshot hashes before reusing it.
            saved = folder / "resume.pt"
            step = torch.load(saved, map_location="cpu", weights_only=True)["step"] if saved.exists() else 0
            result += max(0, total - step)
    return result


def render_results(root, label, mode):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import PercentFormatter
    from pilot_support import export_small_results, read_json, write_json

    out = root / "results" / label
    rows = []
    configs = configurations(root, label, mode)
    for config in configs:
        path = root / config["output"] / "summary.json"
        if path.exists():
            rows += [dict(corpus=config["corpus"], **row) for row in read_json(path)]
    write_json(out / "summary.json", rows)
    if rows:
        with (out / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
        fig, axes = plt.subplots(3, 3, figsize=(13, 10), layout="constrained")
        handles = {}
        for index, config in enumerate(configs):
            corpus = config["corpus"]
            sizes = np.array(config["corpus_sizes"])
            final = max(config["training"]["checkpoints"])
            for method, title, color, marker in (
                ("lag_radius_2", "Lag reconstructor, radius 2", "#245A81", "o"),
                ("mdlm_full_context", f"MDLM, {final:,} updates", "#B85D24", "s")):
                group = sorted([r for r in rows if r["corpus"] == corpus and r["method"] == method
                    and r["pairs"] == config["pairs"] and r["updates"] == (0 if method == "lag_radius_2" else final)],
                    key=lambda r: r["n"])
                if not group:
                    continue
                x = [r["n"] for r in group]
                line, = axes[index, 0].plot(x, [r["mean_mse"] for r in group], label=title, color=color, marker=marker)
                handles[title] = line
                axes[index, 1].plot(x, [r["mean_relative_improvement"] for r in group], color=color, marker=marker)
                base = next((r["mean_disagreement"] for r in group if r["n"] == min(sizes)), 0)
                if base > 0:
                    axes[index, 2].plot(x, [r["mean_disagreement"] / base for r in group], color=color, marker=marker)
            line, = axes[index, 2].plot(sizes, sizes.min() / sizes, color="#444444", linestyle=":")
            handles["1/n scaling"] = line
            axes[index, 0].set_ylabel("MSE (lower is better)")
            axes[index, 1].set_ylabel("Relative MSE improvement\n(higher is better)")
            axes[index, 1].yaxis.set_major_formatter(PercentFormatter(1))
            axes[index, 1].axhline(0, color="#aaaaaa", linestyle="-", linewidth=.8)
            axes[index, 2].set_ylabel("Disagreement /\nsmallest-size value")
            for axis in axes[index]:
                axis.set_title(LABELS[corpus])
                axis.set_xlabel("Training corpus size (64-token chunks)")
                axis.set_xticks(sizes)
                axis.grid(axis="y", color="#dddddd", linewidth=.6)
        title = "SOFTWARE SMOKE CHECK: not scientific results" if mode == "smoke" else "Matched evaluation: completed pair sets only"
        fig.suptitle(title, fontsize=14)
        fig.legend(list(handles.values()), list(handles), loc="outside lower center", ncol=3, frameon=False)
        fig.savefig(out / "comparison.png", dpi=150)
        fig.savefig(out / "comparison.pdf")
        plt.close(fig)
    # Atomic ZIP replacement means an interrupted export leaves the previous ZIP intact.
    destination = out / f"{label}_results.zip"
    temporary = destination.with_suffix(".zip.tmp")
    export_small_results(out, temporary)
    atomic_replace(temporary, destination)
    return destination
