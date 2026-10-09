"""Exploratory correction-matched baseline, added after primary outcomes.

The partition and jackknife coefficient are identical to the saved source
correction. This script does not alter the prospective primary comparison.
"""
from pathlib import Path
import hashlib
import json
import sys
from datetime import datetime,timezone
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.evaluate_lag_disagreement import multinomial_forecast,source_groups
from scripts.evaluate_lag_disagreement_extension import load_context


def summarize_only(root,config):
    rng=np.random.default_rng(config['seed']+771177)
    distributions=[]
    actual=[]
    for corpus in config['corpora']:
        with np.load(root/corpus/'analysis_arrays.npz') as z:
            source=z['forecasts']
            observed=z['observed']
        correction=np.load(root/corpus/'multinomial_jackknife_exploratory.npy')
        assert (correction>0).all()
        forecasts=np.concatenate([source,correction[:,:,:,None]],axis=3)
        banks=rng.integers(config['reference_banks'],size=(5000,config['reference_banks']))
        pairs=rng.integers(config['pairs'],size=(5000,config['pairs']))
        denominator=observed[pairs].mean(axis=1)
        rows=[];estimates=[]
        for m,n in config['primary_cells']:
            mi,ni=config['reference_sizes'].index(m),config['sizes'].index(n)
            rows.append(np.abs(np.log(forecasts[banks,mi,ni]/denominator[:,None,ni,None])).mean(axis=1))
            estimates.append(np.abs(np.log(forecasts[:,mi,ni]/observed[:,ni].mean())).mean(axis=0))
        distributions.append(np.mean(rows,axis=0))
        actual.append(np.mean(estimates,axis=0))
    distribution=np.mean(distributions,axis=0)
    estimate=np.mean(actual,axis=0)
    result=dict(status='exploratory baseline added after primary outcomes',
        methods=['source_full','source_jackknife','multinomial','multinomial_jackknife'],
        mean_absolute_log_error=estimate.tolist(),
        intervals=np.quantile(distribution,[.025,.975],axis=0).T.tolist(),
        source_jackknife_minus_multinomial_jackknife=float(estimate[1]-estimate[3]),
        difference_interval=np.quantile(distribution[:,1]-distribution[:,3],[.025,.975]).tolist(),
        no_new_outcomes=True,no_parameter_tuning=True,primary_analysis_unchanged=True)
    (root/'multinomial_jackknife_exploratory.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


def main(summary_only=False):
    config=json.loads((ROOT/'configs/rare_context_prospective_v1.json').read_text())
    out=ROOT/config['output']
    if summary_only:
        return summarize_only(out,config)
    plan=out/'exploratory_baseline_plan.json'
    if not plan.exists():
        plan.write_text(json.dumps(dict(saved_utc=datetime.now(timezone.utc).isoformat(),
            status='Exploratory follow-up. Primary outcomes already scored.',
            method='Apply the same source partition and coefficient 2 to the multinomial forecast.',
            code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
    for ci,corpus in enumerate(config['corpora']):
        destination=out/corpus/'multinomial_jackknife_exploratory.npy'
        if destination.exists():
            continue
        folder=ROOT/config['inputs']/corpus
        (evaluation,mask,prior,gram,cross),_,_,_,_=load_context(folder)
        kw=dict(smoothing=config['smoothing'],ridge=config['ridge'])
        output=np.zeros((config['reference_banks'],len(config['reference_sizes']),len(config['sizes'])))
        for bank in range(config['reference_banks']):
            with np.load(folder/f'reference{bank}.npz') as z:
                reference=z['ids']
            records=json.loads((folder/f'reference{bank}.json').read_text())
            for mi,m in enumerate(config['reference_sizes']):
                groups=source_groups(records[:m])
                order=np.random.default_rng(config['seed']+10000*ci+100*bank+m).permutation(len(groups))
                halves=np.array_split(order,2)
                with np.load(out/corpus/f'forecast_bank{bank}_m{m}.npz') as z:
                    full=[float(z[f'n{n}_point_forecasts'][0]@z[f'n{n}_point_weight']) for n in config['sizes']]
                corrected=2*np.asarray(full)
                for half in halves:
                    ids=np.concatenate([groups[i] for i in half])
                    for ni,n in enumerate(config['sizes']):
                        mean=multinomial_forecast(reference[ids],evaluation,mask,prior,(-2,-1,1,2),gram,cross,n=n,**kw).mean()
                        corrected[ni]-=len(half)/len(groups)*mean
                output[bank,mi]=np.maximum(corrected,0)
            print(f'{corpus}: exploratory multinomial jackknife bank {bank+1}/6',flush=True)
        np.save(destination,output)
    summarize_only(out,config)


if __name__=='__main__':
    main('--summary-only' in sys.argv)
