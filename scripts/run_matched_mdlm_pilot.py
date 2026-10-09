"""Run or resume the matched TinyStories pilot on cached source-disjoint data."""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import psutil
import torch
from diffusion_lm_rmt.matched_mdlm import read_json, run, utc, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/matched_mdlm_tinystories_pilot_v1.json")
    parser.add_argument("--device", choices=["cuda", "cpu", "mps"], default="cuda")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="short integration check in tmp, not a scientific result")
    args = parser.parse_args()
    config = read_json(args.config)
    if args.smoke:
        config.update(name="mdlm_smoke", output="tmp/mdlm_pilot_smoke", pairs=1, corpus_sizes=[32, 64],
                      evaluation_chunks=4, validation_chunks=4)
        config["training"].update(auxiliary_updates=4, checkpoints=[4, 8], warmup_updates=2,
                                 microbatch_size=2, gradient_accumulation=1, log_every=2,
                                 save_every=4, validation_every=4)
    out = ROOT / config["output"]
    out.mkdir(parents=True, exist_ok=True)
    lock = out / "run.lock"
    if lock.exists():
        previous = read_json(lock)
        if psutil.pid_exists(previous["pid"]):
            process = psutil.Process(previous["pid"])
            if abs(process.create_time() - previous["create_time"]) < 1:
                raise RuntimeError(f"pilot already running with PID {previous['pid']}")
        lock.unlink()
    with lock.open("x", encoding="utf-8") as stream:
        json.dump(dict(pid=os.getpid(), create_time=psutil.Process().create_time()), stream)
    try:
        if args.device == "cuda" and not torch.cuda.is_available() and not args.prepare_only:
            raise RuntimeError("CUDA unavailable. Use the .venv-cuda Python interpreter.")
        if config["model"]["dropout"] != 0:
            raise ValueError("resume and paired RNG protocol require dropout=0")
        if sorted(set(config["training"]["checkpoints"])) != config["training"]["checkpoints"]:
            raise ValueError("checkpoints must be unique and increasing")
        torch.set_num_threads(6)
        torch.set_num_interop_threads(1)
        # Reject nondeterministic kernels rather than silently adding an unmeasured source of noise.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        write_json(out / "run.json", dict(status="preparing" if args.prepare_only else "running", pid=os.getpid(),
                   started_utc=utc(), device=args.device, torch=str(torch.__version__)))
        run(ROOT, config, torch.device(args.device), prepare_only=args.prepare_only)
        write_json(out / "run.json", dict(status="prepared" if args.prepare_only else "complete", pid=os.getpid(), completed_utc=utc()))
    except BaseException as error:
        write_json(out / "run.json", dict(status="failed", pid=os.getpid(), failed_utc=utc(),
                   error=str(error), traceback=traceback.format_exc()))
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
