"""Prepare, benchmark, run, and export the matched experiment on three corpora."""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "colab"), str(ROOT / "runpod")]
from pilot_support import read_json, write_json
from runtime import BudgetStop, configurations, remaining_updates, render_results, require_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--benchmark-steps", type=int, required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--allow-budget-interruption", action="store_true")
    args = parser.parse_args()
    out = ROOT / "results" / args.label
    budget = read_json(out / "budget.json")
    if budget["session"] != args.session:
        raise ValueError("Worker session changed")
    state = dict(session=args.session, mode=args.mode, status="preparing")
    try:
        import numpy as np
        import torch
        from diffusion_lm_rmt import matched_mdlm as core
        torch.set_num_threads(min(6, os.cpu_count() or 1))
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        configs = configurations(ROOT, args.label, args.mode)
        device = torch.device(args.device)
        for config in configs:
            require_time()
            state.update(status="preparing", corpus=config["corpus"])
            write_json(out / "suite_status.json", state)
            core.run(ROOT, config, device, prepare_only=True)
        require_time()
        state.update(status="benchmarking")
        write_json(out / "suite_status.json", state)
        timing = copy.deepcopy(configs[0])
        timing["output"] = f"results/{args.label}/timing/{args.session}"
        timing["training"].update(log_every=1, save_every=1000000, validation_every=1000000)
        roles, _, _, _ = core.load_inputs(ROOT, timing)
        with np.load(ROOT / configs[0]["output"] / "context.npz") as saved:
            context = {key: saved[key] for key in saved.files}
        # Uses the identical float32 optimizer, objective, RNG, microbatches, and clipping path.
        steps = min(args.benchmark_steps, timing["training"]["auxiliary_updates"])
        core.train_job(ROOT, timing, "TIMING_ONLY", "benchmark", roles["auxiliary_train"][0], context,
                       device, seed_offset=1000, stop_after=steps)
        log = ROOT / timing["output"] / "models/benchmark/training.jsonl"
        records = [json.loads(line) for line in log.read_text().splitlines()]
        first = records[max(0, len(records) // 4 - 1)]
        last = records[-1]
        seconds = (last["elapsed_seconds"] - first["elapsed_seconds"]) / (last["step"] - first["step"])
        remaining = remaining_updates(ROOT, configs)
        estimate = remaining * seconds * 1.25 + (1800 if args.mode == "full" else 30)
        available = budget["training_deadline"] - time.time()
        report = dict(gpu=torch.cuda.get_device_name() if args.device == "cuda" else "CPU SMOKE CHECK",
            measured_updates=steps, seconds_per_update=seconds, remaining_training_updates=remaining,
            estimated_remaining_hours=estimate / 3600, estimated_remaining_cost=estimate / 3600 * budget["hourly_rate"],
            available_training_hours=max(0, available) / 3600, estimated_to_fit=estimate <= available,
            scope="Timing estimate with 25% margin and 30 minutes overhead for full runs. No completion guarantee.")
        write_json(out / "benchmark.json", report)
        print(json.dumps(report, indent=2), flush=True)
        if estimate > available and not args.allow_budget_interruption:
            raise BudgetStop("Timing estimate exceeds available budget. No full training started in this session.")
        if estimate > available:
            if not budget.get("allow_budget_interruption"):
                raise ValueError("Budget-interruption preference does not match the launch record")
            state["budget_note"] = "User chose to attempt the unchanged protocol within the hard budget, saving unfinished work."
            print(state["budget_note"], flush=True)
        for config in configs:
            require_time()
            state.update(status="training", corpus=config["corpus"])
            write_json(out / "suite_status.json", state)
            write_json(ROOT / config["output"] / "run.json", dict(status="running", session=args.session))
            core.run(ROOT, config, device)
            write_json(ROOT / config["output"] / "run.json", dict(status="complete", session=args.session))
            render_results(ROOT, args.label, args.mode)
        state.update(status="complete", finished_epoch=time.time())
    except BudgetStop as error:
        state.update(status="budget_stopped", message=str(error), finished_epoch=time.time())
    except BaseException as error:
        state.update(status="failed", message=str(error), traceback=traceback.format_exc(), finished_epoch=time.time())
        traceback.print_exc()
    finally:
        write_json(out / "suite_status.json", state)
        if state["status"] in ("failed", "budget_stopped") and state.get("corpus"):
            write_json(out / state["corpus"] / "run.json", state)
        try:
            destination = render_results(ROOT, args.label, args.mode)
            print("RESULTS ZIP:", destination, flush=True)
        except Exception:
            traceback.print_exc()
            state["export_error"] = traceback.format_exc()
            write_json(out / "suite_status.json", state)
        write_json(out / "export_complete.json", dict(session=args.session, success="export_error" not in state))
        print(json.dumps(state, indent=2), flush=True)


if __name__ == "__main__":
    main()
