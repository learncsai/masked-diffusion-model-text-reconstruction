"""Recompute the complete alpha-sweep inference from compact numeric records."""
from pathlib import Path
import argparse,csv,hashlib,json,sys,warnings
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
NAMES={'tinystories':'TinyStories','wikitext':'WikiText-103','cnn_dailymail':'CNN/DailyMail'}
METHODS=['multinomial','source_same_lag','source_full','multinomial_jackknife','source_jackknife','bootstrap','bootstrap_jackknife']

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def csv_write(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def interval(values):return [float(x) for x in np.nanquantile(values,[.025,.975])]
def slope(values,x):
    centered=np.log(x)-np.log(x).mean()
    logs=np.full_like(values,np.nan,dtype=float);np.log(values,out=logs,where=values>0)
    return logs@centered/(centered@centered)
def weight_draws(rng,count,draws):
    indices=rng.integers(count,size=(draws,count))
    return np.eye(count)[indices].mean(axis=1)

def summarize(source,config,output):
    cfg=read(config);output.mkdir(parents=True,exist_ok=True)
    alphas=cfg['alphas'];sizes=np.array(cfg['sizes']);ms=cfg['reference_sizes'];draws=cfg['inference_draws']
    rows=[];cal=[];hyp=[];quality=[];rowbins=[];cover=[];mc=[];diag=[];tables={}
    all_arrays={};regime_groups={};timing=read(source/'forecasts_saved.json');latest=timing['saved_utc']
    weights=read(source/'method_saved.json');assert weights['saved_utc']<=latest
    for ci,corpus in enumerate(cfg['corpora']):
        folder=source/corpus
        files=[folder/f'observed_pair{j}.npz' for j in range(cfg['pairs'])]
        observed=[];mse=[];accuracy=[];same=[];coverage=[];cross=[]
        for path in files:
            with np.load(path) as z:
                assert str(z['saved_utc'])>=latest
                assert str(z['forecast_sha256'])==hashlib.sha256((source/'forecasts_saved.json').read_bytes()).hexdigest()
                for target,key in [(observed,'disagreement'),(mse,'mse'),(accuracy,'accuracy'),(same,'same_lag_bins'),(coverage,'coverage'),(cross,'cross_lag')]:target.append(z[key])
                prior_mse=float(z['unigram_mse']);prior_accuracy=float(z['unigram_accuracy'])
        observed=np.array(observed);mse=np.array(mse);accuracy=np.array(accuracy)
        same=np.array(same);coverage=np.array(coverage);cross=np.array(cross)
        np.testing.assert_allclose(same.sum(axis=-1)+cross,observed,rtol=1e-11,atol=1e-12)
        np.testing.assert_allclose(same.sum(axis=-1),coverage.sum(axis=-1),rtol=1e-11,atol=1e-12)
        F=np.full((cfg['banks'],2,len(alphas),len(ms),len(sizes),len(METHODS)),np.nan)
        for bank in range(cfg['banks']):
            for mi,m in enumerate(ms):
                with np.load(folder/f'analytic_bank{bank}_m{m}.npz') as z:
                    F[bank,:,:,mi,:,:3]=z['values'][0]
                    F[bank,:,:,mi,:,3]=z['corrected'][:,:,:,0]
                    F[bank,:,:,mi,:,4]=z['corrected'][:,:,:,2]
                    for ai,alpha in enumerate(alphas):
                        for ni,n in enumerate(sizes):
                            diag.append(dict(corpus=corpus,bank=bank,m=m,n=int(n),alpha=alpha,
                                below_alpha=float(z['diagnostics'][ai,ni,0]),mean_unigram_share=float(z['diagnostics'][ai,ni,1])))
                for ni,n in enumerate(sizes):
                    with np.load(folder/f'bootstrap_bank{bank}_m{m}_n{n}.npz') as z:
                        for cai,alpha in enumerate(cfg['bootstrap_alphas']):
                            ai=alphas.index(alpha)
                            for method,values in [(5,z['batches'][0]),(6,z['corrected_batches'])]:
                                F[bank,:,ai,mi,ni,method]=values[:,:,cai].mean(axis=0)
                                for arm in range(2):mc.append(dict(corpus=corpus,bank=bank,arm=cfg['weight_arms'][arm],m=m,n=int(n),alpha=alpha,method=METHODS[method],
                                    estimate=float(values[:,arm,cai].mean()),mc_se=float(values[:,arm,cai].std(ddof=1)/np.sqrt(cfg['bootstrap_batches']))))
        rng=np.random.default_rng(cfg['seed']+ci)
        pw=weight_draws(rng,cfg['pairs'],draws);bw=weight_draws(rng,cfg['banks'],draws)
        obs=observed.mean(axis=0);obs_draw=np.tensordot(pw,observed,axes=(1,0))
        pred=F.mean(axis=0);pred_draw=np.tensordot(bw,F,axes=(1,0))
        beta=slope(obs,sizes);beta_draw=slope(obs_draw,sizes)
        beta_pred=slope(np.moveaxis(pred,-2,-1),sizes);beta_pred_draw=slope(np.moveaxis(pred_draw,-2,-1),sizes)
        with np.load(folder/'weights.npz') as z:weight_norm=z['weight_norm']
        with np.load(folder/'row_design.npz') as z:bin_rate=z['bin_rate'];bin_mass=z['bin_mass'];poisson=z['poisson_curves']
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',message='All-NaN slice encountered')
            prediction_intervals=np.nanquantile(beta_pred_draw,[.025,.975],axis=0)
        all_arrays[corpus]=dict(observed=obs,beta=beta,beta_ci=np.quantile(beta_draw,[.025,.975],axis=0),
            predicted=pred,beta_pred=beta_pred,beta_pred_ci=prediction_intervals,
            same=same.mean(axis=0),coverage=coverage.mean(axis=0),cross=cross.mean(axis=0),
            bin_rate=bin_rate,bin_mass=bin_mass,poisson=poisson,weight_norm=weight_norm,
            relative_mse=1-mse.mean(axis=(0,-1))/prior_mse,accuracy=accuracy.mean(axis=(0,-1)))
        for arm,arm_name in enumerate(cfg['weight_arms']):
            delta=float(beta[arm,alphas.index(20)]-beta[arm,alphas.index(1)])
            lo,hi=interval(beta_draw[:,arm,alphas.index(20)]-beta_draw[:,arm,alphas.index(1)])
            hyp.append(dict(corpus=corpus,arm=arm_name,test='beta20_minus_beta1',method='observed',estimate=delta,low=lo,high=hi))
            for method in [0,2]:
                mi=ms.index(cfg['primary_reference'])
                err=np.abs(beta_pred[arm,:,mi,method]-beta[arm]).mean()-np.abs(-1-beta[arm]).mean()
                simulated=np.abs(beta_pred_draw[:,arm,:,mi,method]-beta_draw[:,arm]).mean(axis=1)-np.abs(-1-beta_draw[:,arm]).mean(axis=1)
                lo,hi=interval(simulated)
                hyp.append(dict(corpus=corpus,arm=arm_name,test='slope_error_minus_inverse_size',method=METHODS[method],estimate=float(err),low=lo,high=hi))
            for ai,alpha in enumerate(alphas):
                lo,hi=interval(beta_draw[:,arm,ai]);decline=100*(1-obs[arm,ai,2]/obs[arm,ai,0])
                r=dict(corpus=corpus,arm=arm_name,alpha=alpha,beta=float(beta[arm,ai]),beta_low=lo,beta_high=hi,decline_512_2048=float(decline))
                for mi,methods in [(ms.index(cfg['primary_reference']),[0,2]),(ms.index(cfg['secondary_reference']),[3,4])]:
                    for method in methods:
                        key=METHODS[method];r[key+'_beta']=float(beta_pred[arm,ai,mi,method]);bl,bh=interval(beta_pred_draw[:,arm,ai,mi,method])
                        r[key+'_beta_low']=bl;r[key+'_beta_high']=bh
                        r[key+'_decline']=float(100*(1-pred[arm,ai,mi,2,method]/pred[arm,ai,mi,0,method]))
                rows.append(r)
                for ni,n in enumerate(sizes):
                    model_mse=mse[:,arm,ai,ni].mean(axis=1);model_acc=accuracy[:,arm,ai,ni].mean(axis=1)
                    llo,lhi=interval(1-pw@model_mse/prior_mse);alo,ahi=interval(pw@model_acc)
                    quality.append(dict(corpus=corpus,arm=arm_name,alpha=alpha,n=int(n),mse=float(model_mse.mean()),
                        relative_mse_improvement=float(1-model_mse.mean()/prior_mse),relative_mse_low=llo,relative_mse_high=lhi,
                        top1_accuracy=float(model_acc.mean()),top1_low=alo,top1_high=ahi,
                        unigram_mse=prior_mse,unigram_accuracy=prior_accuracy,weight_squared_norm=float(weight_norm[arm,ai])))
                    cov=coverage[:,arm,ai,ni].mean(axis=0);total_same=float(cov.sum())
                    cover.append(dict(corpus=corpus,arm=arm_name,alpha=alpha,n=int(n),observed=float(obs[arm,ai,ni]),
                        both_seen=float(cov[0]),one_sided=float(cov[1]),one_sided_share=float(cov[1]/total_same),
                        cross_lag=float(cross[:,arm,ai,ni].mean()),same_lag=total_same))
                    for mi,m in enumerate(ms):
                        for method,name in enumerate(METHODS):
                            v=F[:,arm,ai,mi,ni,method]
                            if np.isnan(v).all():continue
                            good=bool((v>0).all());ratio=float(v.mean()/obs[arm,ai,ni])
                            rlo,rhi=interval(pred_draw[:,arm,ai,mi,ni,method]/obs_draw[:,arm,ai,ni])
                            if good:
                                error=float(np.abs(np.log(v/obs[arm,ai,ni])).mean())
                                e=np.sum(bw*np.abs(np.log(v[None,:]/obs_draw[:,arm,ai,ni,None])),axis=1);elo,ehi=interval(e)
                            else:error=elo=ehi=None
                            for regime in ['full','m<n' if m<n else 'm>n' if m>n else 'm=n']:
                                key=(corpus,arm_name,alpha,name,regime)
                                if key not in regime_groups:regime_groups[key]=dict(total=0.,draws=np.zeros(draws),cells=0,failures=0)
                                group=regime_groups[key];group['cells']+=1;group['failures']+=int((v<=0).sum())
                                if good:group['total']+=error;group['draws']+=e
                            cal.append(dict(corpus=corpus,arm=arm_name,alpha=alpha,m=m,n=int(n),method=name,
                                ratio=ratio,ratio_low=rlo,ratio_high=rhi,mean_bank_abs_log_error=error,error_low=elo,error_high=ehi,
                                nonpositive_banks=int((v<=0).sum())))
                means=same[:,arm,ai].mean(axis=0)
                simulated=np.tensordot(pw,same[:,arm,ai],axes=(1,0))
                for bi,rate in enumerate(bin_rate):
                    if bin_mass[bi]==0:continue
                    for ni in range(len(sizes)-1):
                        d1,d2=means[ni,bi],means[ni+1,bi]
                        if d1>0 and d2>0:
                            elasticity=float(np.log(d2/d1)/np.log(sizes[ni+1]/sizes[ni]))
                            with np.errstate(divide='ignore',invalid='ignore'):
                                sample=np.log(simulated[:,ni+1,bi]/simulated[:,ni,bi])/np.log(sizes[ni+1]/sizes[ni])
                            sample[~np.isfinite(sample)]=np.nan;lo,hi=interval(sample)
                        else:elasticity=lo=hi=None
                        pp=poisson[arm,ai,ni:ni+2,bi]
                        theory=float(np.log(pp[1]/pp[0])/np.log(sizes[ni+1]/sizes[ni])) if np.isfinite(pp).all() and (pp>0).all() else None
                        rowbins.append(dict(corpus=corpus,arm=arm_name,alpha=alpha,bin=bi,n1=int(sizes[ni]),n2=int(sizes[ni+1]),
                            rate=None if not np.isfinite(rate) else float(rate),lambda_over_alpha=None if not np.isfinite(rate) else float(np.sqrt(sizes[ni]*sizes[ni+1])*rate/alpha),
                            energy1=float(d1),energy2=float(d2),elasticity=elasticity,low=lo,high=hi,poisson_elasticity=theory))
    regimes=[]
    for (corpus,arm,alpha,method,regime),g in regime_groups.items():
        if g['failures']:value=lo=hi=None
        else:value=g['total']/g['cells'];lo,hi=interval(g['draws']/g['cells'])
        regimes.append(dict(corpus=corpus,arm=arm,alpha=alpha,method=method,regime=regime,
            mean_bank_abs_log_error=value,low=lo,high=hi,cells=g['cells'],nonpositive_bank_cells=g['failures']))
    paired=[]
    for (corpus,arm,alpha,method,regime),g in regime_groups.items():
        if method=='multinomial' or g['failures']:continue
        base=regime_groups[(corpus,arm,alpha,'multinomial',regime)]
        if base['failures']:continue
        lo,hi=interval(g['draws']/g['cells']-base['draws']/base['cells'])
        paired.append(dict(corpus=corpus,arm=arm,alpha=alpha,method=method,comparator='multinomial',regime=regime,
            error_difference=g['total']/g['cells']-base['total']/base['cells'],low=lo,high=hi))
    for name,data in [('slopes',rows),('hypotheses',hyp),('quality',quality),('calibration',cal),('calibration_by_regime',regimes),
                      ('paired_calibration',paired),('row_bins',rowbins),('coverage',cover),('monte_carlo',mc),('reference_diagnostics',diag)]:csv_write(output/(name+'.csv'),data)
    for corpus,data in all_arrays.items():np.savez_compressed(output/f'{corpus}_plot_data.npz',**data)
    # Full tables are in CSV. Compact paper tables retain all 21 fixed-arm cells.
    tables['tab:alpha-slopes']=[]
    for r in rows:
        if r['arm']!='fixed':continue
        q=next(x for x in quality if (x['corpus'],x['arm'],x['alpha'],x['n'])==(r['corpus'],'fixed',r['alpha'],2048))
        tables['tab:alpha-slopes'].append(f"{NAMES[r['corpus']]} & {r['alpha']:g} & {r['beta']:.3f} [{r['beta_low']:.3f}, {r['beta_high']:.3f}] & {r['source_full_beta']:.3f} & {r['multinomial_beta']:.3f} & {r['decline_512_2048']:.1f} & {100*q['relative_mse_improvement']:.2f}")
    tables['tab:alpha-contrasts']=[]
    for corpus in cfg['corpora']:
        for arm in cfg['weight_arms']:
            entries=[next(r for r in hyp if (r['corpus'],r['arm'],r['test'],r['method'])==(corpus,arm,test,method)) for test,method in [('beta20_minus_beta1','observed'),('slope_error_minus_inverse_size','source_full'),('slope_error_minus_inverse_size','multinomial')]]
            cells=[f"{r['estimate']:+.3f} [{r['low']:+.3f}, {r['high']:+.3f}]" for r in entries]
            tables['tab:alpha-contrasts'].append(NAMES[corpus]+' & '+('Fixed' if arm=='fixed' else 'Refit')+' & '+' & '.join(cells))
    tables['tab:alpha-calibration']=[]
    for corpus in cfg['corpora']:
        for arm in cfg['weight_arms']:
            for alpha in cfg['bootstrap_alphas']:
                cells=[]
                for method in ['source_full','source_jackknife','multinomial','multinomial_jackknife','bootstrap','bootstrap_jackknife']:
                    parts=[]
                    for regime in ['m<n','m>n']:
                        r=next(x for x in regimes if (x['corpus'],x['arm'],x['alpha'],x['method'],x['regime'])==(corpus,arm,alpha,method,regime))
                        v=r['mean_bank_abs_log_error'];parts.append('NA' if v is None else f'{v:.3f}')
                    cells.append(' / '.join(parts))
                tables['tab:alpha-calibration'].append(f"{NAMES[corpus]} & {'Fixed' if arm=='fixed' else 'Refit'} & {alpha:g} & "+' & '.join(cells))
    write(output/'tables.json',tables)
    for label,values in tables.items():(output/(label.replace(':','_')+'.tex')).write_text('\n'.join(x+' \\\\' for x in values)+'\n',encoding='utf-8')
    write(output/'validation.json',dict(status='passed',corpora=3,pairs_per_corpus=20,banks_per_corpus=6,
        slope_rows=len(rows),quality_rows=len(quality),calibration_rows=len(cal),hypotheses=hyp,
        row_partition_identities=True,forecast_before_new_alpha_scoring=True,external_preregistration=False))
    print(json.dumps(dict(status='passed',slope_rows=len(rows),quality_rows=len(quality),calibration_rows=len(cal))))

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--source',type=Path,default=ROOT/'results/alpha_sweep_v1')
    ap.add_argument('--config',type=Path,default=ROOT/'configs/alpha_sweep_v1.json');ap.add_argument('--output',type=Path)
    a=ap.parse_args();summarize(a.source,a.config,a.output or a.source/'summary')
