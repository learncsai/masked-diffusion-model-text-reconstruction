"""Replay the added bootstrap comparison from scalar bank and pair records."""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_multiplier_transfer import logs, train_factor, pack

CORE = ['source_full', 'source_jackknife', 'multinomial', 'multinomial_jackknife']
METHODS = CORE + ['multinomial_times_1.25', 'multinomial_calibrated_original',
                  'multinomial_calibrated_other_corpora', 'source_bootstrap', 'bootstrap_jackknife']
LABELS = ['Full source', 'Source jackknife', 'Multinomial', 'Multinomial jackknife',
          r'Multinomial $\times1.25$', 'Multinomial, original-grid factor',
          'Multinomial, other-corpus factors', 'Source bootstrap', 'Bootstrap jackknife']


def read(path): return json.loads(path.read_text(encoding='utf-8'))
def write(path, value): path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')


def switching_forecast(forecasts, methods, ms, ns):
    """Select between the two corrections, including equality in multinomial."""
    multi = forecasts[..., methods.index('multinomial_jackknife')]
    boot = forecasts[..., methods.index('bootstrap_jackknife')]
    gate = np.asarray(ms)[:, None] > np.asarray(ns)[None, :]
    return np.where(gate, boot, multi)


def original_policy(root, cfg):
    rng = np.random.default_rng(cfg['interval_seed'])
    estimates, distributions = [], []
    for corpus in cfg['corpora']:
        with np.load(root/f'results/correction_followup_v1/{corpus}_followup_arrays.npz') as z:
            f, o, names = z['forecasts'], z['observed'], z['methods'].tolist()
        switch = switching_forecast(f, names, [1024, 2048, 4096], [512, 1024, 2048, 4096])
        f = np.stack([f[..., names.index('multinomial_jackknife')],
                      f[..., names.index('bootstrap_jackknife')], switch], axis=-1)
        bi = rng.integers(len(f), size=(cfg['interval_draws'], len(f)))
        pi = rng.integers(len(o), size=(cfg['interval_draws'], len(o)))
        estimates.append(np.abs(np.log(f/o.mean(axis=0)[None, None, :, None])).mean(axis=(0, 1, 2)))
        distributions.append(np.abs(np.log(f[bi]/o[pi].mean(axis=1)[:, None, None, :, None])).mean(axis=(1, 2, 3)))
    estimate, distribution = np.mean(estimates, axis=0), np.mean(distributions, axis=0)
    return dict(status='Retrospective original-grid policy selected on these outcomes.',
                methods=['multinomial_jackknife', 'bootstrap_jackknife', 'switch_m_gt_n'],
                **pack(estimate, distribution), by_corpus=dict(zip(cfg['corpora'], np.asarray(estimates).tolist())),
                switch_minus_multinomial=pack(estimate[2:3]-estimate[0], distribution[:, 2:3]-distribution[:, 0:1]),
                switch_minus_bootstrap=pack(estimate[2:3]-estimate[1], distribution[:, 2:3]-distribution[:, 1:2]))


def summarize(root):
    cfg = read(root/'configs/bootstrap_confirmation_v1.json')
    out = root/cfg['output']
    original = []
    for corpus in cfg['corpora']:
        with np.load(root/f'results/correction_followup_v1/{corpus}_followup_arrays.npz') as z:
            names = z['methods'].tolist()
            original.append((z['forecasts'][..., [names.index(m) for m in CORE]], z['observed']))
    rng = np.random.default_rng(cfg['interval_seed'])
    op, ob = logs(original, rng, cfg['interval_draws'])
    cells = [(i, j) for i in range(3) for j in range(4)]
    factor, bfactor = train_factor(op, cells), train_factor(ob, cells)
    corpus_factor = [train_factor(op, cells, [j for j in range(3) if j != i]) for i in range(3)]
    corpus_bfactor = [train_factor(ob, cells, [j for j in range(3) if j != i]) for i in range(3)]
    points, boots, records, mc = [], [], [], []
    for ci, corpus in enumerate(cfg['corpora']):
        prefix = root/f'results/correction_followup_v1/confirmation/{corpus}'
        with np.load(prefix/'forecasts.npz') as z: f = z['forecasts'][:, 1, 0, :]
        with np.load(prefix/'observed.npz') as z: o = z['observed'][:, 0]
        assert f.shape == (3, 4) and o.shape == (6,)
        extra = []
        for bank in range(cfg['reference_banks']):
            with np.load(out/corpus/f'bank{bank}.npz') as z:
                assert (int(z['m']), int(z['n'])) == (cfg['m'], cfg['n'])
                assert float(z['raw_jackknife']) > 0, 'Report nonpositive correction explicitly'
                np.testing.assert_allclose(z['jackknife_batches'],
                    2*z['batches'][0]-z['source_fractions']@z['batches'][1:], rtol=1e-13)
                np.testing.assert_allclose(z['source_fractions'],
                    z['source_counts'][1:]/z['source_counts'][0], rtol=1e-13)
                np.testing.assert_allclose(z['forecasts'],
                    [z['batches'][0].mean(), z['jackknife_batches'].mean()], rtol=1e-13)
                extra.append(z['forecasts'])
                for method, batch in zip(METHODS[-2:], [z['batches'][0], z['jackknife_batches']]):
                    mc.append(dict(corpus=corpus, bank=bank, method=method,
                        relative_mc_se=float(batch.std(ddof=1)/np.sqrt(len(batch))/batch.mean())))
        f = np.column_stack([f, 1.25*f[:, 2], np.exp(factor)*f[:, 2],
                             np.exp(corpus_factor[ci])*f[:, 2], np.asarray(extra)])
        assert f.shape == (3, 9) and (f > 0).all()
        point = np.log(f/o.mean())
        bi = rng.integers(len(f), size=(cfg['interval_draws'], len(f)))
        pi = rng.integers(len(o), size=(cfg['interval_draws'], len(o)))
        boot = np.log(f[bi]/o[pi].mean(axis=1)[:, None, None])
        boot[:, :, 5] += (bfactor-factor)[:, None]
        boot[:, :, 6] += (corpus_bfactor[ci]-corpus_factor[ci])[:, None]
        points.append(point)
        boots.append(boot)
        for bank in range(3):
            for j, method in enumerate(METHODS):
                records.append(dict(corpus=corpus, bank=bank, m=cfg['m'], n=cfg['n'],
                    method=method, forecast=float(f[bank, j]), observed_mean=float(o.mean()),
                    ratio=float(f[bank, j]/o.mean()), absolute_log_error=float(abs(point[bank, j]))))
    points, boots = np.asarray(points), np.stack(boots, axis=1)
    estimate, distribution = np.abs(points).mean(axis=(0, 1)), np.abs(boots).mean(axis=(1, 2))
    result = dict(status=cfg['status'], methods=METHODS, m=cfg['m'], n=cfg['n'],
                  **pack(estimate, distribution),
                  bootstrap_jackknife_minus_each=pack(estimate[-1]-estimate, distribution[:, -1, None]-distribution),
                  by_corpus={c:dict(error=np.abs(points[i]).mean(axis=0).tolist(),
                                   ratio=np.exp(points[i]).mean(axis=0).tolist()) for i,c in enumerate(cfg['corpora'])},
                  mc={method:dict(median_relative_se=float(np.median([x['relative_mc_se'] for x in mc if x['method']==method])),
                                  max_relative_se=max(x['relative_mc_se'] for x in mc if x['method']==method)) for method in METHODS[-2:]},
                  clipped_corrections=0, uncertainty=cfg['uncertainty'],
                  original_policy=original_policy(root, cfg))
    write(out/'summary.json', result)
    for name, rows in [('bank_records.csv', records), ('monte_carlo.csv', mc)]:
        with (out/name).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    old = read(root/'results/correction_followup_v1/confirmation_summary.json')
    multi = read(root/'results/oracle_multiplier_v1/multiplier_summary.json')
    def value(report, j):
        lo, hi = report['interval'][j]
        return f"{report['error'][j]:.3f} [{lo:.3f}, {hi:.3f}]"
    rows = []
    for j, label in enumerate(LABELS):
        previous = old if j < 4 else multi
        vals = [value(previous[key], j) for key in ['primary', 'full_grid']] if j < 7 else ['Not run', 'Not run']
        rows.append(label+' & '+' & '.join([*vals, value(result, j)]))
    write(out/'tables.json', {'tab:confirmation': rows})
    (out/'tab_confirmation.tex').write_text(' \\\\\n'.join(rows)+' \\\\\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ['methods', 'error', 'interval', 'mc', 'original_policy']}, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=ROOT)
    summarize(ap.parse_args().root)
