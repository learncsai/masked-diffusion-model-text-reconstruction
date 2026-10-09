"""Portable notebook helpers. Training and scientific scores remain in src/."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import threading
import time
import zipfile
from runtime import atomic_replace
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    atomic_replace(temporary, path)


def verify_bundle(root):
    manifest = read_json(root / "PACKAGE_MANIFEST.json")
    for name, expected in manifest["files"].items():
        if sha256(root / name) != expected["sha256"]:
            raise ValueError(f"Package hash mismatch: {name}. Extract a fresh copy of the ZIP.")
    return manifest


def make_config(root, run_label, mode):
    if not run_label or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in run_label):
        raise ValueError("RUN_LABEL must contain only letters, numbers, underscores, or hyphens")
    if mode not in ("full", "smoke"):
        raise ValueError("RUN_MODE must be full or smoke")
    config = read_json(root / "configs/matched_mdlm_tinystories_pilot_v1.json")
    config.update(name=run_label, output=f"results/{run_label}")
    if mode == "smoke":
        config.update(pairs=1, corpus_sizes=[32, 64], evaluation_chunks=4, validation_chunks=4)
        config["training"].update(auxiliary_updates=4, checkpoints=[4, 8], warmup_updates=2,
            microbatch_size=2, gradient_accumulation=1, log_every=2, save_every=4, validation_every=4)
    path = root / "runtime_configs" / f"{run_label}.json"
    if path.exists() and read_json(path) != config:
        raise ValueError("This RUN_LABEL already has a different configuration. Choose a new label.")
    write_json(path, config)
    return path, config


def copy_atomic(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    before = source.stat()
    temporary = destination.with_suffix(destination.suffix + ".copying")
    shutil.copy2(source, temporary)
    after = source.stat()
    if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
        temporary.unlink(missing_ok=True)
        return False
    atomic_replace(temporary, destination)
    return True


def mirror_results(source, destination, cache=None):
    """Copy immutable checkpoints before publishing their completion markers."""
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("backup and working directories must differ")
    destination.mkdir(parents=True, exist_ok=True)
    cache = cache if cache is not None else {}
    files = [p for p in source.rglob("*") if p.is_file() and
             p.name not in ("run.lock", "backup_status.json") and not p.name.endswith((".tmp", ".copying"))]
    files.sort(key=lambda p: (p.name in ("complete.json", "prepared.json"), str(p)))
    copied = 0
    for path in files:
        relative = path.relative_to(source)
        target = destination / relative
        stamp = (path.stat().st_mtime_ns, path.stat().st_size)
        if cache.get(str(relative)) == stamp and target.exists():
            continue
        if path.name in ("complete.json", "prepared.json"):
            marker = read_json(path)
            if any(not (target.parent / name).exists() or sha256(target.parent / name) != expected
                   for name, expected in marker["sha256"].items()):
                continue
        if copy_atomic(path, target):
            cache[str(relative)] = stamp
            copied += 1
    write_json(destination / "backup_status.json", dict(last_success_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), copied=copied))
    return copied


def establish_run(root, config, backup):
    """Restore an existing Colab run into an empty workspace, then check identity."""
    output = root / config["output"]
    identity = dict(package_sha256=sha256(root / "PACKAGE_MANIFEST.json"), config=config)
    for folder in (output, backup):
        if folder and (folder / "handoff_identity.json").exists() and read_json(folder / "handoff_identity.json") != identity:
            raise ValueError(f"Different package or settings already use {folder}. Choose a new RUN_LABEL.")
    if not (output / "manifest.json").exists() and backup and (backup / "manifest.json").exists():
        mirror_results(backup, output)
        print("Restored saved checkpoints and scores from:", backup)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "handoff_identity.json", identity)
    return output


def check_environment(output, device):
    import numpy
    import scipy
    import torch
    versions = dict(python=".".join(platform.python_version_tuple()[:2]), torch=str(torch.__version__),
                    numpy=numpy.__version__, scipy=scipy.__version__, device=device,
                    gpu=torch.cuda.get_device_name() if device == "cuda" else None)
    path = output / "runtime_environment.json"
    if path.exists() and read_json(path)["versions"] != versions:
        raise ValueError("The saved run used a different GPU or Python/PyTorch/NumPy/SciPy version. "
                         "Restore that environment to resume, or choose a new RUN_LABEL for a separate run.")
    if not path.exists():
        write_json(path, dict(versions=versions, platform=platform.platform(), cuda_runtime=torch.version.cuda))
        packages = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        (output / "packages.txt").write_text(packages, encoding="utf-8")
    return versions


def assert_not_running(output):
    import psutil
    lock = output / "run.lock"
    if lock.exists():
        previous = read_json(lock)
        if psutil.pid_exists(previous["pid"]):
            process = psutil.Process(previous["pid"])
            if abs(process.create_time() - previous["create_time"]) < 1:
                raise RuntimeError(f"Training is already running (PID {previous['pid']}). Use the status cell.")


def run_training(root, config_path, config, output, backup, device, on_progress=None, backup_interval=30):
    assert_not_running(output)
    stop = threading.Event()
    cache, errors = {}, []
    def sync():
        if backup:
            mirror_results(output, backup, cache)
    def worker():
        while not stop.is_set():
            try:
                sync()
                errors.clear()
            except Exception as error:
                message = str(error)
                if not errors or errors[-1] != message:
                    print("Checkpoint backup will retry:", message, flush=True)
                errors[:] = [message]
            stop.wait(backup_interval)
    thread = threading.Thread(target=worker, daemon=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    environment["OMP_NUM_THREADS"] = str(min(6, os.cpu_count() or 1))
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    command = [sys.executable, "-u", str(root / "scripts/run_matched_mdlm_pilot.py"),
               "--config", str(config_path), "--device", device]
    process = subprocess.Popen(command, cwd=root, env=environment, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1, creationflags=flags)
    thread.start()
    try:
        with (output / "notebook_run.log").open("a", encoding="utf-8") as stream:
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                if on_progress:
                    on_progress(line.rstrip())
            code = process.wait()
        if code:
            tail = (output / "notebook_run.log").read_text(encoding="utf-8").splitlines()[-20:]
            raise RuntimeError(f"Training stopped with exit code {code}:\n" + "\n".join(tail))
    except KeyboardInterrupt:
        if process.poll() is None:
            if os.name == "nt":
                process.terminate()
            else:
                process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        print("Stopped. Rerun this cell to resume from the last saved checkpoint.")
        raise
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=20)
        stop.set()
        thread.join(timeout=120)
        if thread.is_alive():
            raise RuntimeError("Drive backup is still blocked. Check Drive before restarting the cell.")
        sync()
    return read_json(output / "run.json")


def export_small_results(output, destination):
    """Scores, logs, figures, and provenance. Large checkpoints stay in Drive."""
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file() and path.suffix.lower() in (".json", ".jsonl", ".csv", ".txt", ".md", ".png", ".pdf", ".log", ".npz"):
                archive.write(path, path.relative_to(output).as_posix())
    return destination
