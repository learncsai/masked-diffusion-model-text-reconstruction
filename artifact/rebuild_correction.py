"""Rebuild correction-study inference from bank forecasts and A/B pair scores.

These scalar records are exact reductions of the full study's position arrays.
This compact replay does not recreate source counts or unseen-row assignments.
The companion study archive retains those earlier stages and their inputs.
"""
from pathlib import Path
import argparse
import ast
import csv
import json
import numpy as np


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def load_function(path, name):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[node], type_ignores=[])
    context = {'np': np}
    exec(compile(module, str(path), 'exec'), context)
    return context[name]

def save_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def verify_rows(path, actual):
    with path.open(newline='', encoding='utf-8') as stream:
        saved=list(csv.DictReader(stream))
    assert len(saved)==len(actual),path
    keys=('corpus','n','m','method')
    old={tuple(r[k] for k in keys):r for r in saved}
    for r in actual:
        previous=old[tuple(str(r[k]) for k in keys)]
        for key,value in r.items():
            if isinstance(value,(int,float)):
                np.testing.assert_allclose(float(previous[key]),value,rtol=1e-12,atol=1e-14)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    args=parser.parse_args(); folder=args.folder
    config=read(folder/'configs/rare_context_prospective_v1.json')
    result=folder/'results/rare_context_prospective_v1'
    out=folder/'rebuilt';out.mkdir(exist_ok=True)
    crossed=load_function(folder/'code/reference_correction.py','crossed_interval')
    rows=[];extra=[];arrays=[];decisions=[]
    methods=config['methods']+['multinomial_jackknife']
    for ci,corpus in enumerate(config['corpora']):
        with np.load(result/corpus/'analysis_arrays.npz') as z:
            forecasts=z['forecasts'];observed=z['observed'];large=z['large']
        corrected=np.load(result/corpus/'multinomial_jackknife_exploratory.npy')
        assert forecasts.shape==(6,3,4,3) and observed.shape==(20,4) and corrected.shape==(6,3,4)
        arrays.append((forecasts,observed,corrected))
        target=read(result/corpus/'target_saved.json')['target']
        np.testing.assert_allclose(target,.8*np.median(large[:,0]),rtol=1e-13)
        for ni,n in enumerate(config['sizes']):
            for mi,m in enumerate(config['reference_sizes']):
                inferred=crossed(forecasts[:,mi,ni],observed[:,ni],seed=config['seed']+ci*100000+mi*100+ni)
                for j,method in enumerate(config['methods']):
                    rows.append(dict(corpus=corpus,n=n,m=m,method=method,banks=6,pairs=20,
                        forecast=float(forecasts[:,mi,ni,j].mean()),observed=float(observed[:,ni].mean()),
                        ratio=float(inferred['ratio'][j]),ratio_low=float(inferred['ratio_ci'][j,0]),ratio_high=float(inferred['ratio_ci'][j,1]),
                        mean_bank_absolute_log_error=float(inferred['mean_bank_absolute_log_error'][j])))
                inf=crossed(np.column_stack([forecasts[:,mi,ni,0],corrected[:,mi,ni]]),observed[:,ni],
                    seed=config['seed']+44444+ci*100+mi*10+ni)
                extra.append(dict(corpus=corpus,n=n,m=m,method='multinomial_jackknife',
                    ratio=float(inf['ratio'][1]),ratio_low=float(inf['ratio_ci'][1,0]),ratio_high=float(inf['ratio_ci'][1,1]),
                    mean_bank_absolute_log_error=float(inf['mean_bank_absolute_log_error'][1]),status='exploratory'))
            inf=crossed(large[:,ni,None],observed[:,ni],seed=config['seed']+ci*10000+ni)
            rows.append(dict(corpus=corpus,n=n,m=8192,method='source_large_reference',banks=6,pairs=20,
                forecast=float(large[:,ni].mean()),observed=float(observed[:,ni].mean()),ratio=float(inf['ratio'][0]),
                ratio_low=float(inf['ratio_ci'][0,0]),ratio_high=float(inf['ratio_ci'][0,1]),
                mean_bank_absolute_log_error=float(inf['mean_bank_absolute_log_error'][0])))
        rng=np.random.default_rng(config['seed']+90000+ci)
        samples=rng.integers(20,size=(5000,20))
        intervals=np.quantile(observed[samples].mean(axis=1),[.025,.975],axis=0).T
        all_f=np.concatenate([forecasts,corrected[:,:,:,None]],axis=3)
        for bank in range(6):
            for mi,m in enumerate(config['reference_sizes']):
                for j,method in enumerate(methods):
                    eligible=np.flatnonzero(all_f[bank,mi,:,j]<=target)
                    chosen=int(eligible[0]) if len(eligible) else None
                    outcome='abstain' if chosen is None else ('upper_interval_below_target' if intervals[chosen,1]<=target else
                        ('mean_below_target' if observed[:,chosen].mean()<=target else 'mean_above_target'))
                    decisions.append(dict(corpus=corpus,bank=bank,m=m,method=method,
                        selected_n='' if chosen is None else config['sizes'][chosen],outcome=outcome))
    verify_rows(result/'summary.csv',rows)
    verify_rows(result/'corrected_baseline_calibration.csv',extra)
    # Primary and exploratory comparisons retain their original independent RNG streams.
    estimates={};distributions={}
    for label,seed,count in [('primary',888888,3),('exploratory',771177,4)]:
        rng=np.random.default_rng(config['seed']+seed)
        distributions[label]=[];estimates[label]=[]
        for forecasts,observed,corrected in arrays:
            if count==4: forecasts=np.concatenate([forecasts,corrected[:,:,:,None]],axis=3)
            banks=rng.integers(6,size=(5000,6));pairs=rng.integers(20,size=(5000,20))
            denominator=observed[pairs].mean(axis=1)
            draws=[];values=[]
            for m,n in config['primary_cells']:
                mi,ni=config['reference_sizes'].index(m),config['sizes'].index(n)
                draws.append(np.abs(np.log(forecasts[banks,mi,ni]/denominator[:,None,ni,None])).mean(axis=1))
                values.append(np.abs(np.log(forecasts[:,mi,ni]/observed[:,ni].mean())).mean(axis=0))
            distributions[label].append(np.mean(draws,axis=0));estimates[label].append(np.mean(values,axis=0))
        distributions[label]=np.mean(distributions[label],axis=0);estimates[label]=np.mean(estimates[label],axis=0)
    primary=read(result/'primary_result.json');exploratory=read(result/'multinomial_jackknife_exploratory.json')
    for j,r in enumerate(primary['methods']):
        np.testing.assert_allclose(r['mean_absolute_log_error'],estimates['primary'][j],rtol=1e-14)
        np.testing.assert_allclose(r['interval'],np.quantile(distributions['primary'][:,j],[.025,.975]),rtol=1e-14)
    np.testing.assert_allclose(exploratory['mean_absolute_log_error'],estimates['exploratory'],rtol=1e-14)
    np.testing.assert_allclose(exploratory['intervals'],np.quantile(distributions['exploratory'],[.025,.975],axis=0).T,rtol=1e-14)
    for name in ['decisions.csv','corrected_baseline_decisions.csv']:
        with (result/name).open(newline='') as f:
            old=list(csv.DictReader(f))
        for r in old:
            match=next(x for x in decisions if (x['corpus'],x['bank'],x['m'],x['method'])==(r['corpus'],int(r['bank']),int(r['m']),r['method']))
            assert str(match['selected_n'])==r['selected_n'] and match['outcome']==r['outcome']
    labels=['Full source','Source jackknife','Multinomial','Multinomial jackknife']
    table=[];decision_table=[]
    for j,(name,method) in enumerate(zip(labels,methods)):
        study='primary' if j<3 else 'exploratory'
        value=estimates[study][j];lo,hi=np.quantile(distributions[study][:,j],[.025,.975])
        table.append(f'{name} & {value:.3f} [{lo:.3f}, {hi:.3f}]')
        totals=[sum(r['method']==method and r['outcome']==k for r in decisions) for k in ['upper_interval_below_target','mean_above_target','abstain']]
        decision_table.append(name+' & '+' & '.join(str(n) for n in totals))
    (out/'primary_rows.tex').write_text('\n'.join(table)+'\n')
    (out/'decision_rows.tex').write_text('\n'.join(decision_table)+'\n')
    # Rebuilt ratios and intervals drive the paper figures.
    save_csv(result/'summary.csv',rows);save_csv(result/'corrected_baseline_calibration.csv',extra)
    (out/'REPORT.json').write_text(json.dumps(dict(status='passed',calibration_rows=156,decision_rows=216,
        primary_and_exploratory_estimates_and_intervals_recomputed=True,
        scope='Saved scalar bank forecasts and pair scores. Synthetic decomposition plots use saved aggregate components.'),indent=2)+'\n')
    print('PASS correction study: 156 calibration rows, 216 decisions, primary and exploratory inference.')

if __name__=='__main__': main()
