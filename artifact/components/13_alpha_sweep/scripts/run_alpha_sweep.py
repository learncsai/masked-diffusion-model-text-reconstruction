"""Paired real-text smoothing sweep. Forecast all settings before new scoring."""
from pathlib import Path
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[_key]='1'
import argparse,hashlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.evaluate_lag_disagreement import read_json,write_json,sha,utc,source_groups,validate_records
from diffusion_lm_rmt import alpha_sweep as sweep,sparse_categorical_lag as lag
from diffusion_lm_rmt.categorical_lag import per_sequence_squared_error

CONFIG=ROOT/'configs/alpha_sweep_v1.json'
PROTOCOL=ROOT/'docs/alpha_sweep_protocol.txt'
OFFSETS=(-2,-1,1,2)
CORE=['scripts/run_alpha_sweep.py','src/diffusion_lm_rmt/alpha_sweep.py',
      'src/diffusion_lm_rmt/sparse_categorical_lag.py','src/diffusion_lm_rmt/source_moment_forecast.py',
      'src/diffusion_lm_rmt/categorical_lag.py','tests/test_alpha_sweep.py']

def paths():
    cfg=read_json(CONFIG);out=ROOT/cfg['output'];out.mkdir(parents=True,exist_ok=True)
    return cfg,out,ROOT/cfg['inputs']

def freeze_method():
    cfg,out,inputs=paths()
    contract=dict(config_sha256=sha(CONFIG),protocol_sha256=sha(PROTOCOL),
        code_sha256={name:sha(ROOT/name) for name in CORE})
    signature=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    path=out/'method_saved.json'
    if path.exists():assert read_json(path)['signature']==signature,'Saved design/code changed'
    else:
        assert not list(out.glob('*/observed_pair*.npz'))
        write_json(path,dict(saved_utc=utc(),signature=signature,**contract,
            status='Existing allocations, known alpha=5 outcomes. No new-alpha A/B scores yet.',external_preregistration=False))
        for name in [*CORE,str(CONFIG.relative_to(ROOT)),str(PROTOCOL.relative_to(ROOT))]:
            target=out/'code'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
    return signature

def context(corpus):
    cfg,out,inputs=paths()
    with np.load(inputs/corpus/'context.npz') as z:e,mask,p,g,b=[z[k] for k in ['evaluation','mask','prior','gram','cross']]
    return e,mask,p,g,b

def prepare(corpus):
    import torch
    from scipy.stats import poisson
    signature=freeze_method();cfg,out,inputs=paths();folder=inputs/corpus;dest=out/corpus;dest.mkdir(exist_ok=True)
    if (dest/'prepared.json').exists():
        old=read_json(dest/'prepared.json');assert old['signature']==signature
        for name,digest in old['files'].items():assert sha(dest/name)==digest
        return
    done=read_json(folder/'complete.json');hashes={}
    for name,digest in done['files_sha256'].items():
        if name.endswith(('.npz','.json')):
            assert sha(folder/name)==digest,name;hashes[name]=digest
    manifest=read_json(folder/'original_manifest.json');old=manifest['protocol']['config']
    aux_path=next(ROOT/name for name in manifest['protocol']['inputs'] if name.endswith('replicate3_train_a.pt'))
    assert sha(aux_path)==manifest['protocol']['inputs'][aux_path.relative_to(ROOT).as_posix()]
    aux=torch.load(aux_path,map_location='cpu',weights_only=True);ids=aux['input_ids'].numpy()
    groups=source_groups(aux['chunk_records']);np.random.default_rng(old['seed']).shuffle(groups)
    split=len(groups)//2;train=ids[np.concatenate(groups[:split])];val=ids[np.concatenate(groups[split:])]
    e,mask,p,g,b=context(corpus)
    expected=np.bincount(train.ravel(),minlength=len(p)).astype(float);expected/=expected.sum()
    np.testing.assert_allclose(p,expected,rtol=0,atol=1e-15)
    design=sweep.geometry(e,mask,p,OFFSETS)
    grams=np.empty((2,len(cfg['alphas']),4,4));cross=np.empty((2,len(cfg['alphas']),4))
    weights=[];metrics=[];norms=[]
    for ai,alpha in enumerate(cfg['alphas']):
        gg,bb=lag.fitting_equations(train,val,p,OFFSETS,alpha,.5,np.random.default_rng(old['seed']+1))
        if alpha==5:
            np.testing.assert_allclose(gg,g,rtol=1e-12,atol=1e-10);np.testing.assert_allclose(bb,b,rtol=1e-12,atol=1e-10)
        grams[:,ai]=[g,gg];cross[:,ai]=[b,bb]
    for arm in range(2):
        aw=[];am=[];an=[]
        for ai in range(len(cfg['alphas'])):
            w,metric=sweep.weights_and_metric(design,grams[arm,ai],cross[arm,ai])
            aw.append(w);am.append(metric);an.append(float(design['pw']@(w*w).sum(axis=1)))
        weights.append(aw);metrics.append(am);norms.append(an)
    weights=np.asarray(weights);metrics=np.asarray(metrics)
    np.savez_compressed(dest/'weights.npz',grams=grams,cross=cross,weights=weights,metrics=metrics,weight_norm=norms)
    C=sweep.queried_counts(lag.pair_counts(train,OFFSETS,len(p)),design)
    N=np.asarray(C.sum(axis=1)).ravel();rates=N/len(train);bins=np.where(N==0,0,np.digitize(rates,cfg['row_rate_edges']))
    nb=len(cfg['row_rate_edges'])+1;feature_weight=np.zeros(design['F'])
    for j in range(4):
        sel=design['feature'][:,j]>=0;f=design['feature'][sel,j]
        feature_weight+=np.bincount(f,weights=design['pw'][sel],minlength=design['F'])
    bin_mass=np.bincount(bins,weights=feature_weight,minlength=nb)
    bin_rate=np.full(nb,np.nan)
    for bi in range(1,nb):
        sel=(bins==bi)&(rates>0)
        if sel.any():bin_rate[bi]=np.exp(np.average(np.log(rates[sel]),weights=feature_weight[sel]))
    collision=np.divide(np.asarray(C.multiply(C).sum(axis=1)).ravel(),N*N,out=np.zeros_like(N),where=N>0)
    cp=np.divide(C@p,N,out=np.zeros_like(N),where=N>0);gap=collision-2*cp+p@p
    curves=np.full((2,len(cfg['alphas']),len(cfg['sizes']),nb),np.nan)
    # Illustrative conditional-multinomial curves, averaged after row calculation.
    for ni,n in enumerate(cfg['sizes']):
        first=np.zeros((len(cfg['alphas']),design['F']));second=first.copy()
        for rate in np.unique(rates[rates>0]):
            lam=n*rate;lo=int(poisson.ppf(1e-13,lam));hi=int(poisson.ppf(1-1e-13,lam))
            counts=np.arange(lo,hi+1);prob=poisson.pmf(counts,lam)
            for ai,alpha in enumerate(cfg['alphas']):
                z=counts/(counts+alpha);mean=prob@z
                first[ai,rates==rate]=prob@(counts/(counts+alpha)**2)
                second[ai,rates==rate]=prob@((z-mean)**2)
        for arm in range(2):
            for ai in range(len(cfg['alphas'])):
                d=2*((1-collision)*first[ai]+gap*second[ai]);totals=np.zeros(nb)
                for j in range(4):
                    sel=design['feature'][:,j]>=0;f=design['feature'][sel,j]
                    totals+=np.bincount(bins[f],weights=design['pw'][sel]*weights[arm,ai,sel,j]**2*d[f],minlength=nb)
                totals[bin_mass==0]=np.nan;totals[0]=np.nan;curves[arm,ai,ni]=totals
    np.savez_compressed(dest/'row_design.npz',rates=rates,bins=bins,bin_mass=bin_mass,bin_rate=bin_rate,poisson_curves=curves)
    write_json(dest/'prepared.json',dict(signature=signature,saved_utc=utc(),input_hashes=hashes,
        auxiliary_file=aux_path.relative_to(ROOT).as_posix(),auxiliary_sha256=sha(aux_path),
        auxiliary_seed=old['seed'],auxiliary_train_chunks=len(train),auxiliary_validation_chunks=len(val),
        alpha5_auxiliary_equivalence=True,files={name:sha(dest/name) for name in ['weights.npz','row_design.npz']}))
    print(corpus,'auxiliary preparation verified',flush=True)

def partitions(tokens,records,ci,bank,m):
    groups=source_groups(records[:m]);tokens=tokens[:m]
    order=np.random.default_rng(20261003+10000*ci+100*bank+m).permutation(len(groups))
    result=[(tokens,groups,1.)]
    for selected in np.array_split(order,2):
        indices=np.concatenate([groups[i] for i in selected]);ends=np.cumsum([0,*[len(groups[i]) for i in selected]])
        result.append((tokens[indices],[np.arange(a,b) for a,b in zip(ends[:-1],ends[1:])],len(selected)/len(groups)))
    return result

def forecast(corpus,bank_only=None,bootstrap_only=False):
    prepare(corpus);signature=freeze_method();cfg,out,inputs=paths();ci=cfg['corpora'].index(corpus)
    dest=out/corpus;e,mask,p,g,b=context(corpus);design=sweep.geometry(e,mask,p,OFFSETS)
    with np.load(dest/'weights.npz') as z:grams,cross,metrics=[z[k] for k in ['grams','cross','metrics']]
    core=[cfg['alphas'].index(a) for a in cfg['bootstrap_alphas']];start=time.perf_counter()
    for bank in range(cfg['banks']) if bank_only is None else [bank_only]:
        with np.load(inputs/corpus/f'reference{bank}.npz') as z:tokens=z['ids']
        records=read_json(inputs/corpus/f'reference{bank}.json');validate_records(tokens,records)
        for m in cfg['reference_sizes']:
            parts=partitions(tokens,records,ci,bank,m);fraction=np.array([x[2] for x in parts[1:]])
            path=dest/f'analytic_bank{bank}_m{m}.npz'
            if not path.exists() and not bootstrap_only:
                values=np.empty((3,2,len(cfg['alphas']),len(cfg['sizes']),3));diagnostics=np.empty((len(cfg['alphas']),len(cfg['sizes']),2))
                for component,(reference,groups,_) in enumerate(parts):
                    state=sweep.prepare_moments(reference,groups,e,mask,p,OFFSETS)
                    for arm in range(2):
                        for ai,alpha in enumerate(cfg['alphas']):
                            forecasts=sweep.evaluate_moments(state,grams[arm,ai],cross[arm,ai],cfg['sizes'],smoothing=alpha,ridge=.01)
                            for ni,n in enumerate(cfg['sizes']):
                                values[component,arm,ai,ni]=forecasts[n]['forecasts'].mean(axis=1)
                    if component==0:
                        total_mass=0.;counts=[];ww=[]
                        for j in range(4):
                            sel=state['feature'][:,j]>=0
                            counts.extend(state['totals'][state['feature'][sel,j]]);ww.extend(state['point_weight'][sel])
                        counts=np.asarray(counts);ww=np.asarray(ww);ww/=ww.sum()
                        for ai,alpha in enumerate(cfg['alphas']):
                            for ni,n in enumerate(cfg['sizes']):
                                scaled=counts*n/m;diagnostics[ai,ni]=[ww@(scaled<alpha),ww@(alpha/(scaled+alpha))]
                corrected=2*values[0]-np.tensordot(fraction,values[1:],axes=(0,0))
                # Alpha=5, fixed weights must reproduce all historical raw laws.
                old=ROOT/'results/rare_context_prospective_v1'/corpus/f'forecast_bank{bank}_m{m}.npz'
                with np.load(old) as z:
                    for ni,n in enumerate(cfg['sizes']):
                        expected=z[f'n{n}_point_forecasts']@z[f'n{n}_point_weight']
                        np.testing.assert_allclose(values[0,0,3,ni],expected,rtol=2e-10,atol=1e-12)
                        if f'n{n}_jackknife_mean_raw' in z:np.testing.assert_allclose(corrected[0,3,ni,2],float(z[f'n{n}_jackknife_mean_raw']),rtol=2e-10,atol=1e-12)
                np.savez_compressed(path,values=values,corrected=corrected,source_fractions=fraction,
                    source_counts=[len(x[1]) for x in parts],diagnostics=diagnostics,signature=signature,saved_utc=utc())
            for ni,n in enumerate(cfg['sizes']):
                path=dest/f'bootstrap_bank{bank}_m{m}_n{n}.npz'
                if path.exists():continue
                batches=np.zeros((3,cfg['bootstrap_batches'],2,len(core)))
                for component,(reference,groups,_) in enumerate(parts):
                    for batch in range(cfg['bootstrap_batches']):
                        rng=np.random.default_rng(np.random.SeedSequence([20261017,ci,bank,m,n,component,batch]))
                        batches[component,batch]=sweep.bootstrap_sweep(reference,groups,design,cfg['bootstrap_alphas'],metrics[:,core],
                            n=n,draws=cfg['bootstrap_draws'],rng=rng)
                corrected=2*batches[0]-np.tensordot(fraction,batches[1:],axes=(0,0))
                old=ROOT/'results/correction_followup_v1/baselines'/corpus/f'bootstrap_bank{bank}_m{m}_n{n}.npz'
                if old.exists():
                    with np.load(old) as z:np.testing.assert_allclose(batches[:,:,0,1],z['batches'],rtol=5e-10,atol=2e-12)
                np.savez_compressed(path,batches=batches,corrected_batches=corrected,source_fractions=fraction,
                    signature=signature,saved_utc=utc())
                elapsed=time.perf_counter()-start
                write_json(dest/'progress.json',dict(stage='forecasts',bank=bank,m=m,n=n,elapsed_seconds=elapsed,saved_utc=utc()))
                print(f'{corpus} bank {bank+1}/6 m={m} n={n}: {elapsed:.1f}s',flush=True)

def freeze_forecasts():
    signature=freeze_method();cfg,out,inputs=paths();files={}
    for corpus in cfg['corpora']:
        for bank in range(cfg['banks']):
            for m in cfg['reference_sizes']:
                for name in [f'analytic_bank{bank}_m{m}.npz',*[f'bootstrap_bank{bank}_m{m}_n{n}.npz' for n in cfg['sizes']]]:
                    path=out/corpus/name
                    with np.load(path) as z:assert str(z['signature'])==signature
                    files[f'{corpus}/{name}']=sha(path)
    path=out/'forecasts_saved.json'
    if path.exists():assert read_json(path)['files']==files
    else:
        assert not list(out.glob('*/observed_pair*.npz'))
        write_json(path,dict(signature=signature,saved_utc=utc(),files=files,
            scope='All new-alpha forecasts saved before new-alpha scoring on existing corpus allocations.'))
    return sha(path)

def score(corpus):
    frozen=freeze_forecasts();cfg,out,inputs=paths();dest=out/corpus
    e,mask,p,g,b=context(corpus);design=sweep.geometry(e,mask,p,OFFSETS)
    with np.load(dest/'weights.npz') as z:grams,cross,weights=[z[k] for k in ['grams','cross','weights']]
    with np.load(dest/'row_design.npz') as z:bins=z['bins'];nb=len(z['bin_mass'])
    prior_mse=float(design['pw']@(1-2*p[design['targets']]+p@p));prior_accuracy=float(design['pw']@(design['targets']==np.argmax(p)))
    started=time.perf_counter()
    for pair in range(cfg['pairs']):
        path=dest/f'observed_pair{pair}.npz'
        if path.exists():continue
        tokens=[]
        for arm in ['a','b']:
            with np.load(inputs/corpus/f'pair{pair}_{arm}.npz') as z:tokens.append(z['ids'])
            validate_records(tokens[-1],read_json(inputs/corpus/f'pair{pair}_{arm}.json'))
        shape=(2,len(cfg['alphas']),len(cfg['sizes']))
        distance=np.zeros(shape);mse=np.zeros((*shape,2));accuracy=np.zeros_like(mse)
        energies=np.zeros((*shape,nb));coverage=np.zeros((*shape,2));cross_energy=np.zeros(shape)
        for ni,n in enumerate(cfg['sizes']):
            counts=[lag.pair_counts(t[:n],OFFSETS,len(p)) for t in tokens]
            for ai,alpha in enumerate(cfg['alphas']):
                tables=[lag.conditional_tables(c,p,alpha) for c in counts]
                for arm in range(2):
                    predictions=[]
                    for side in range(2):
                        rows,targets,pred=lag.masked_prediction(e,mask,tables[side],p,OFFSETS,grams[arm,ai],cross[arm,ai],.01)
                        predictions.append(pred);mse[arm,ai,ni,side]=design['pw']@pred.loss(targets)
                        accuracy[arm,ai,ni,side]=design['pw']@(sweep.exact_top1(pred)==targets)
                    value=lag.squared_difference(*predictions);distance[arm,ai,ni]=design['pw']@value
                    energy,cov=sweep.row_decomposition(*counts,design,alpha,weights[arm,ai],bins,nb)
                    energies[arm,ai,ni]=energy;coverage[arm,ai,ni]=cov
                    cross_energy[arm,ai,ni]=distance[arm,ai,ni]-energy.sum()
                    if ai==3 and arm==0:
                        with np.load(ROOT/'results/rare_context_prospective_v1'/corpus/f'observed_n{n}.npz') as z:
                            np.testing.assert_allclose(distance[arm,ai,ni],z['observed'][pair].mean(),rtol=2e-10,atol=1e-12)
                            np.testing.assert_allclose(mse[arm,ai,ni],z['mse'][pair],rtol=2e-10,atol=1e-12)
        np.testing.assert_allclose(energies.sum(axis=-1)+cross_energy,distance,rtol=1e-12,atol=1e-13)
        np.savez_compressed(path,disagreement=distance,mse=mse,accuracy=accuracy,same_lag_bins=energies,
            coverage=coverage,cross_lag=cross_energy,unigram_mse=prior_mse,unigram_accuracy=prior_accuracy,
            forecast_sha256=frozen,saved_utc=utc())
        elapsed=time.perf_counter()-started
        write_json(dest/'progress.json',dict(stage='scoring',pairs_complete=pair+1,elapsed_seconds=elapsed,saved_utc=utc()))
        print(f'{corpus} pair {pair+1}/20: {elapsed:.1f}s',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=['prepare','forecast','freeze','score'])
    ap.add_argument('--corpus',choices=['tinystories','wikitext','cnn_dailymail']);ap.add_argument('--bank',type=int)
    args=ap.parse_args()
    if args.stage=='freeze':print(freeze_forecasts())
    elif args.stage=='prepare':prepare(args.corpus)
    elif args.stage=='forecast':forecast(args.corpus,args.bank)
    elif args.stage=='score':score(args.corpus)
