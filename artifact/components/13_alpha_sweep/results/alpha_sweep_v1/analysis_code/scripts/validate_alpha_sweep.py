"""Audit saved forecast chronology and exact alpha=5 reproduction locally."""
from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    cfg=json.loads((ROOT/'configs/alpha_sweep_v1.json').read_text());out=ROOT/cfg['output']
    saved=json.loads((out/'forecasts_saved.json').read_text());method=json.loads((out/'method_saved.json').read_text())
    assert saved['signature']==method['signature'];assert method['saved_utc']<=saved['saved_utc']
    for name,digest in saved['files'].items():assert sha(out/name)==digest,name
    analysis=json.loads((out/'analysis_saved.json').read_text())
    assert analysis['saved_utc']<=saved['saved_utc']
    for name,digest in analysis['code_sha256'].items():assert sha(ROOT/name)==digest,name
    maxima=dict(analytic=0.,bootstrap=0.,observed=0.,mse=0.,weight_arm_alpha5=0.)
    for corpus in cfg['corpora']:
        folder=out/corpus
        for bank in range(6):
            for m in cfg['reference_sizes']:
                with np.load(folder/f'analytic_bank{bank}_m{m}.npz') as z:
                    assert str(z['saved_utc'])<=saved['saved_utc']
                    values=z['values'];np.testing.assert_allclose(values[:,0,3],values[:,1,3],rtol=1e-11,atol=1e-12)
                    maxima['weight_arm_alpha5']=max(maxima['weight_arm_alpha5'],float(np.abs(values[:,0,3]-values[:,1,3]).max()))
                    with np.load(ROOT/'results/rare_context_prospective_v1'/corpus/f'forecast_bank{bank}_m{m}.npz') as old:
                        for ni,n in enumerate(cfg['sizes']):
                            expected=old[f'n{n}_point_forecasts']@old[f'n{n}_point_weight']
                            maxima['analytic']=max(maxima['analytic'],float(np.abs(values[0,0,3,ni]-expected).max()))
                for n in cfg['sizes']:
                    old=ROOT/'results/correction_followup_v1/baselines'/corpus/f'bootstrap_bank{bank}_m{m}_n{n}.npz'
                    if old.exists():
                        with np.load(folder/old.name) as z,np.load(old) as prev:
                            maxima['bootstrap']=max(maxima['bootstrap'],float(np.abs(z['batches'][:,:,0,1]-prev['batches']).max()))
        for pair in range(20):
            with np.load(folder/f'observed_pair{pair}.npz') as z:
                assert str(z['saved_utc'])>=saved['saved_utc'];assert str(z['forecast_sha256'])==sha(out/'forecasts_saved.json')
                np.testing.assert_allclose(z['disagreement'][0,3],z['disagreement'][1,3],rtol=1e-11,atol=1e-12)
                for ni,n in enumerate(cfg['sizes']):
                    with np.load(ROOT/'results/rare_context_prospective_v1'/corpus/f'observed_n{n}.npz') as prev:
                        maxima['observed']=max(maxima['observed'],float(abs(z['disagreement'][0,3,ni]-prev['observed'][pair].mean())))
                        maxima['mse']=max(maxima['mse'],float(np.abs(z['mse'][0,3,ni]-prev['mse'][pair]).max()))
    assert max(maxima.values())<1e-9,maxima
    report=dict(status='passed',forecast_files=len(saved['files']),observed_pairs=60,
        all_new_alpha_forecasts_precede_new_alpha_scores=True,analysis_saved_before_scoring=True,
        maximum_absolute_alpha5_discrepancies=maxima,
        scope='Local audit against original arrays. No fresh corpus allocations or neural training.')
    (out/'local_validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':main()
