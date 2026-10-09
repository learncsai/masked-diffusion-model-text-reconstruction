"""Added bootstrap comparison, with settings recorded before new forecasts."""
from __future__ import annotations
import argparse
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '1')
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts.run_correction_followup import context, halves, OFFSETS
from scripts.evaluate_lag_disagreement import read_json, write_json, sha, utc, source_groups
from diffusion_lm_rmt import sparse_categorical_lag as lag

CONFIG = 'configs/bootstrap_confirmation_v1.json'
PROTOCOL = 'docs/bootstrap_confirmation_protocol.txt'
CODE = ['scripts/run_bootstrap_confirmation.py', 'scripts/run_correction_followup.py',
        'src/diffusion_lm_rmt/sparse_categorical_lag.py',
        'src/diffusion_lm_rmt/categorical_lag.py',
        'scripts/evaluate_lag_disagreement.py']


def initialize():
    cfg = read_json(ROOT / CONFIG)
    out = ROOT / cfg['output']
    out.mkdir(parents=True, exist_ok=True)
    paths = [CONFIG, PROTOCOL, *CODE]
    for corpus in cfg['corpora']:
        base = Path(cfg['inputs']) / corpus
        paths += [str(base / 'context.npz'), str(base / 'source_audit.json')]
        paths += [str(base / f'reference{bank}.{ext}')
                  for bank in range(cfg['reference_banks']) for ext in ('npz', 'json')]
    contract = {name.replace('\\', '/'): sha(ROOT / name) for name in paths}
    path = out / 'method_saved.json'
    if path.exists():
        assert read_json(path)['files_sha256'] == contract, 'Saved method or inputs changed'
    else:
        assert not list(out.glob('*/bank*.npz'))
        write_json(path, dict(saved_utc=utc(), status=cfg['status'],
                             external_preregistration=False, files_sha256=contract))
    return cfg, out


def run(corpus):
    cfg, out = initialize()
    ci = cfg['corpora'].index(corpus)
    folder = ROOT / cfg['inputs'] / corpus
    dest = out / corpus
    dest.mkdir(exist_ok=True)
    e, mask, p, gram, cross = context(folder)
    m, n = cfg['m'], cfg['n']
    started = time.perf_counter()
    for bank in range(cfg['reference_banks']):
        path = dest / f'bank{bank}.npz'
        if path.exists():
            with np.load(path) as z:
                assert str(z['method_sha256']) == sha(out / 'method_saved.json')
            continue
        with np.load(folder / f'reference{bank}.npz') as z:
            tokens = z['ids'][:m]
        records = read_json(folder / f'reference{bank}.json')[:m]
        groups = source_groups(records)
        parts = halves(tokens, groups, cfg['partition_seed'] + 10000*ci + 100*bank + m)
        batches = np.zeros((3, cfg['bootstrap_batches']))
        for component, (sample, local, _) in enumerate([(tokens, groups, 1.), *parts]):
            for batch in range(cfg['bootstrap_batches']):
                rng = np.random.default_rng(np.random.SeedSequence(
                    [cfg['seed'], ci, bank, m, n, component, batch]))
                value, sizes = lag.bootstrap_disagreement(
                    sample, local, e, mask, p, OFFSETS, gram, cross,
                    n=n, smoothing=cfg['smoothing'], ridge=cfg['ridge'],
                    draws=cfg['draws_per_batch'], law='source_cluster', rng=rng)
                assert np.all(sizes == n)
                batches[component, batch] = value.mean()
        fractions = np.array([part[2] for part in parts])
        corrected = 2*batches[0] - fractions @ batches[1:]
        assert np.isfinite(batches).all() and (batches > 0).all()
        np.savez_compressed(path, batches=batches, jackknife_batches=corrected,
            source_fractions=fractions, source_counts=[len(groups), *[len(x[1]) for x in parts]],
            chunk_counts=[len(tokens), *[len(x[0]) for x in parts]],
            raw_jackknife=corrected.mean(),
            forecasts=[batches[0].mean(), max(0., corrected.mean())],
            m=m, n=n, saved_utc=utc(), method_sha256=sha(out / 'method_saved.json'))
        print(f'{corpus} bank {bank+1}/{cfg["reference_banks"]}: '
              f'bootstrap={batches[0].mean():.8f}, jackknife={corrected.mean():.8f}, '
              f'{time.perf_counter()-started:.1f}s', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['initialize', 'run'])
    ap.add_argument('--corpus', choices=['tinystories', 'wikitext', 'cnn_dailymail'])
    args = ap.parse_args()
    if args.action == 'initialize': initialize()
    elif args.corpus: run(args.corpus)
    else:
        for corpus in read_json(ROOT / CONFIG)['corpora']: run(corpus)
