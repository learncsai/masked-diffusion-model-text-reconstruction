"""Check statistical aggregation with an exactly known paired size response."""
from pathlib import Path
import hashlib,json,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.summarize_alpha_sweep import slope,weight_draws,summarize


def test_slope_and_paired_resampling_preserve_exact_power_law():
    sizes=np.array([512,1024,2048,4096]);beta=np.array([-.9,-.7,-.2])
    y=(sizes[None,:]/512)**beta[:,None]
    np.testing.assert_allclose(slope(y,sizes),beta,atol=1e-14)
    pairs=np.arange(1,21)[:,None,None]*y[None,:,:]
    w=weight_draws(np.random.default_rng(5),20,5000)
    actual=slope(np.tensordot(w,pairs,axes=(1,0)),sizes)
    np.testing.assert_allclose(actual,np.broadcast_to(beta,actual.shape),atol=1e-14)


def test_complete_summary_on_known_synthetic_records(tmp_path):
    cfg=json.loads((ROOT/'configs/alpha_sweep_v1.json').read_text());cfg['inference_draws']=100
    config=tmp_path/'config.json';config.write_text(json.dumps(cfg))
    source=tmp_path/'input';source.mkdir()
    (source/'method_saved.json').write_text(json.dumps(dict(saved_utc='2026-01-01T00:00:00')))
    frozen=source/'forecasts_saved.json';frozen.write_text(json.dumps(dict(saved_utc='2026-01-02T00:00:00')))
    digest=hashlib.sha256(frozen.read_bytes()).hexdigest()
    n=np.array(cfg['sizes']);a=np.array(cfg['alphas']);beta=-.8+.08*np.log2(a)
    base=.03*(n[None,:]/512)**beta[:,None]
    for corpus in cfg['corpora']:
        folder=source/corpus;folder.mkdir()
        for pair in range(20):
            d=np.broadcast_to(base*(1+.01*pair),(2,7,4)).copy()
            bins=np.stack([.4*d,.5*d],axis=-1)
            np.savez_compressed(folder/f'observed_pair{pair}.npz',disagreement=d,
                mse=np.full((2,7,4,2),.8),accuracy=np.full((2,7,4,2),.2),
                same_lag_bins=bins,coverage=bins,cross_lag=.1*d,unigram_mse=1.,unigram_accuracy=.1,
                saved_utc='2026-01-03T00:00:00',forecast_sha256=digest)
        for bank in range(6):
            for m in cfg['reference_sizes']:
                d=np.broadcast_to((base*(1+.02*bank))[None,:,:,None],(2,7,4,3)).copy()
                np.savez_compressed(folder/f'analytic_bank{bank}_m{m}.npz',values=np.stack([d,d,d]),corrected=d,diagnostics=np.full((7,4,2),.2))
                for ni,size in enumerate(n):
                    values=np.broadcast_to(d[0,[1,3,5],ni,0],(3,2,3)).copy()
                    np.savez_compressed(folder/f'bootstrap_bank{bank}_m{m}_n{size}.npz',batches=np.stack([values,values,values]),corrected_batches=values)
        np.savez_compressed(folder/'weights.npz',weight_norm=np.ones((2,7)))
        np.savez_compressed(folder/'row_design.npz',bin_rate=[.001,.1],bin_mass=[.5,.5],poisson_curves=np.broadcast_to(base[None,:,:,None],(2,7,4,2)))
    output=tmp_path/'summary';summarize(source,config,output)
    result=json.loads((output/'validation.json').read_text())
    assert result['slope_rows']==42 and result['quality_rows']==168
    for r in result['hypotheses']:
        if r['test']=='beta20_minus_beta1':
            np.testing.assert_allclose(r['estimate'],.08*np.log2(20),atol=1e-12)
        else:assert r['high']<0
