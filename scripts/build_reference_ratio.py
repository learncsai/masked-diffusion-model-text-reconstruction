"""Rebuild the reference-ratio tables and figures from saved measurements."""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator,FuncFormatter

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.evaluate_lag_disagreement import CORPORA,LABELS,METHODS,read_json,sha,write_csv,write_json,utc
from scripts.run_reference_ratio import OUT,expectation

N_COLORS={512:'#A34C25',1024:'#B18C19',2048:'#527644',4096:'#216C9C',8192:'#775994'}
STAGES={'earlier_matched':('o',False,'Earlier matched'),
        'earlier_prospective':('D',False,'Earlier prospective'),
        'new_known':('s',True,'New forecast cells'),
        'fresh_nested':('^',True,'Fresh A/B'),
        'fresh_separate':('v',True,'Fresh A/B, new reference')}


def csv_rows(path):
    return list(csv.DictReader(path.open(encoding='utf-8',newline='')))


def point(corpus,n,m,method,predicted,observed,stage,bank,origin,intervals):
    obs=np.asarray(observed,dtype=float)
    mean=float(obs.mean())
    margin=float(t.ppf(.975,len(obs)-1)*obs.std(ddof=1)/np.sqrt(len(obs)))
    return dict(corpus=corpus,n=n,m=m,reference_target_ratio=m/n,method=method,
        stage=stage,bank=bank,pairs=len(obs),predicted_mean=float(predicted),observed_mean=mean,
        ratio=float(predicted/mean),lower=float(predicted/(mean+margin)) if intervals else None,
        upper=float(predicted/(mean-margin)) if intervals and mean>margin else None,origin=origin)


def collect(include_new=True):
    rows=[]; sources={}
    def track(path):
        sources[path.relative_to(ROOT).as_posix()]=sha(path)
        return path
    audit=read_json(track(OUT/'context_audit.json'))
    for entry in audit['corpora']:
        assert all(r['identical'] for r in entry['historical_pair_reuse'])
        assert entry['comparison']['eval_ids']['identical']
        assert entry['comparison']['mask']['identical']
        assert not entry['comparison']['prior']['identical']
    for experiment,ns,ms in [('lag_disagreement_matched_v1',(512,1024,2048),(2048,)),
                             ('lag_reference_sensitivity_v1',(4096,8192),(2048,4096,8192))]:
        folder=ROOT/'results'/experiment
        for corpus in CORPORA:
            for n in ns:
                op=(folder/'per_chunk'/f'{corpus}_n{n}_observed.npz' if experiment=='lag_disagreement_matched_v1'
                    else folder/corpus/f'n{n}_observed.npz')
                with np.load(track(op)) as z: obs=z['observed'].mean(axis=1)
                for m in ms:
                    fp=(folder/'per_chunk'/f'{corpus}_n{n}_forecasts.npz' if experiment=='lag_disagreement_matched_v1'
                        else folder/corpus/f'ref{m}_n{n}_forecasts.npz')
                    with np.load(track(fp)) as z: preds=z['forecasts'][:3].mean(axis=1)
                    for method,pred in zip(METHODS,preds):
                        rows.append(point(corpus,n,m,method,pred,obs,'earlier_matched','nested',experiment,False))
    for experiment in ['reproducibility_followup_v1','size_4096_v1']:
        for corpus in CORPORA:
            folder=ROOT/'results/full_vocabulary_v1'/experiment/corpus
            observed=read_json(track(folder/'observations.json'))
            forecasts=read_json(track(folder/'forecasts.json'))
            for f in forecasts:
                if f.get('law',f.get('method')) not in ('full','analytic'): continue
                n=f.get('n',4096)
                obs=[o['observed'] for o in observed if o['n']==n]
                rows.append(point(corpus,n,f['reference_chunks'],'source_full',f['prediction'],obs,
                    'earlier_prospective',f['bank'],experiment,True))
    if not include_new:
        return rows,sources
    for corpus in CORPORA:
        path=track(OUT/corpus/'per_pair.csv')
        pairs=csv_rows(path)
        for r in csv_rows(track(OUT/corpus/'summary.csv')):
            n,m=int(r['n']),int(r['m'])
            selected=[p for p in pairs if (p['cohort'],p['bank'],int(p['n']),int(p['m']),p['method'])
                      ==(r['cohort'],r['bank'],n,m,r['method'])]
            stage='new_known' if r['cohort']=='known' else 'fresh_'+r['bank']
            entry=point(corpus,n,m,r['method'],float(r['predicted_mean']),[float(p['observed']) for p in selected],
                        stage,r['bank'],'reference_ratio_v1',r['cohort']=='fresh')
            np.testing.assert_allclose(entry['ratio'],float(r['ratio']),rtol=1e-12)
            rows.append(entry)
    return rows,sources


def plot(rows,destination):
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(2,3,figsize=(7.0,4.6),sharex=True,sharey=True)
    # Small symmetric offsets reveal coincident conditions. Ticks and raw
    # coordinates are retained in all_points.csv. They do not imply new ratios.
    xshift={'earlier_matched':-.027,'earlier_prospective':-.014,'new_known':0,
            'fresh_nested':.014,'fresh_separate':.027}
    plotted=[r for r in rows if r['method'] in ('source_full','multinomial')]
    all_values=[v for r in plotted for v in (r['ratio'],r['lower'],r['upper'])]
    all_values=[v for v in all_values if v is not None]
    limits=(min(.42,min(all_values)-.03),max(1.38,max(all_values)+.03))
    for row,method in enumerate(('source_full','multinomial')):
        for col,corpus in enumerate(CORPORA):
            ax=axes[row,col]
            group=[r for r in rows if r['corpus']==corpus and r['method']==method]
            for stage,(marker,filled,_) in STAGES.items():
                for r in (p for p in group if p['stage']==stage):
                    color=N_COLORS[r['n']]
                    x=r['reference_target_ratio']*2**xshift[stage]
                    if r['lower'] is not None and r['upper'] is not None:
                        ax.vlines(x,r['lower'],r['upper'],color=color,alpha=.55,lw=.65,zorder=2)
                    ax.plot(x,r['ratio'],marker=marker,ms=4.5 if stage.startswith('fresh') else 3.6,
                            mfc=color if filled else 'white',mec=color,mew=.85,ls='none',zorder=3)
            ax.axhline(1,color='#333333',lw=.8,ls=':',zorder=1)
            ax.set_xscale('log',base=2)
            ax.set_xlim(.21,18.5)
            ax.set_ylim(*limits)
            ax.set_yticks([.5,.75,1,1.25])
            ax.grid(axis='y',color='#dddddd',lw=.5)
            ax.xaxis.set_major_locator(FixedLocator([.25,.5,1,2,3,4,8,16]))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:g}'))
            if row==0: ax.set_title(LABELS[corpus],fontsize=9)
            if col==0:
                ax.set_ylabel(('Full source' if row==0 else 'Multinomial')+'\nForecast / observed')
    nhandles=[Line2D([],[],color=c,lw=2,label=f'n = {n:,}') for n,c in N_COLORS.items()]
    stages=[Line2D([],[],ls='none',marker=mk,ms=4,mfc='#444444' if fill else 'white',mec='#444444',label=label)
            for mk,fill,label in STAGES.values()]
    fig.legend(handles=nhandles,loc='upper center',bbox_to_anchor=(.52,1),ncol=5,frameon=False,fontsize=8.5)
    fig.legend(handles=stages,loc='lower center',bbox_to_anchor=(.52,0),ncol=3,frameon=False,fontsize=8.5,
               columnspacing=1.2,handletextpad=.4)
    fig.text(.54,.12,'Reference size / target size, m/n',ha='center',fontsize=9)
    fig.subplots_adjust(top=.875,bottom=.20,left=.095,right=.985,hspace=.16,wspace=.10)
    fig.savefig(destination.with_suffix('.pdf'),bbox_inches='tight',
                metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(destination.with_suffix('.png'),dpi=220,bbox_inches='tight')
    plt.close(fig)


def tables_and_checks(rows):
    results={}
    for stage in ['new_known','fresh_nested','fresh_separate']:
        selected=[r for r in rows if r['stage']==stage and r['method']=='source_full'
                  and r['reference_target_ratio'] in (1,2,3)]
        results[stage]=dict(conditions=len(selected),within_range=sum(
            expectation(r['n'],r['m'])[0]<=r['ratio']<=expectation(r['n'],r['m'])[1] for r in selected),
            misses=[dict(corpus=r['corpus'],n=r['n'],m=r['m'],ratio=r['ratio'],
                         expected=expectation(r['n'],r['m'])) for r in selected
                    if not expectation(r['n'],r['m'])[0]<=r['ratio']<=expectation(r['n'],r['m'])[1]])
        distances=[max(expectation(r['n'],r['m'])[0]-r['ratio'],0,
                       r['ratio']-expectation(r['n'],r['m'])[1]) for r in selected]
        results[stage]['maximum_distance_outside_range']=max(distances,default=0)
        results[stage]['mean_distance_outside_range']=float(np.mean(distances)) if distances else None
    comparisons=[]
    for r in rows:
        if r['stage']!='fresh_nested' or r['n']!=4096 or r['method']!='source_full':continue
        multi=next(x for x in rows if (x['corpus'],x['stage'],x['n'],x['m'],x['method'])
                   ==(r['corpus'],r['stage'],r['n'],r['m'],'multinomial'))
        comparisons.append(dict(corpus=r['corpus'],n=r['n'],m=r['m'],source_ratio=r['ratio'],
            multinomial_ratio=multi['ratio'],source_closer_abs_ratio=abs(r['ratio']-1)<abs(multi['ratio']-1),
            source_closer_abs_log=bool(abs(np.log(r['ratio']))<abs(np.log(multi['ratio'])))))
    results['fresh_n4096_comparisons']=comparisons
    table=[]
    for corpus in CORPORA:
        for n in (2048,4096):
            cells=[('nested',n),('nested',2*n),('nested',3*n)]
            if n==2048:
                cells.append(('nested',8192))
            cells.append(('separate',4096))
            for bank,m in cells:
                rs=[r for r in rows if (r['corpus'],r['n'],r['m'],r['stage'])==(corpus,n,m,'fresh_'+bank)]
                values={r['method']:r for r in rs}
                f=values['source_full'];s=values['source_same_lag'];multi=values['multinomial']
                table.append(f"{LABELS[corpus]} & {n:,} & {m:,} & {'New' if bank=='separate' else 'Nested'} & "
                    f"{f['ratio']:.3f} [{f['lower']:.3f}, {f['upper']:.3f}] & {s['ratio']:.3f} & {multi['ratio']:.3f} \\\\")
    (OUT/'fresh_table_rows.tex').write_text('\n'.join(table)+'\n',encoding='utf-8')
    known=[]
    for corpus in CORPORA:
        for ratio in (1,2,3,4,8,16):
            rs=sorted([r for r in rows if r['stage']=='new_known' and r['corpus']==corpus and
                       r['reference_target_ratio']==ratio and r['method']=='source_full'],key=lambda r:r['n'])
            for r in rs:
                error=100*(r['ratio']-1)
                known.append(f"{LABELS[corpus]} & {r['n']:,} & {r['m']:,} & {ratio:g} & {r['ratio']:.3f} & ${error:+.1f}$ \\\\")
    (OUT/'new_cells_rows.tex').write_text('\n'.join(known)+'\n',encoding='utf-8')
    coverage=[]
    for corpus in CORPORA:
        data=csv_rows(OUT/corpus/'coverage_strata.csv')
        for m in (4096,8192,12288):
            rs=[r for r in data if (r['cohort'],r['bank'],int(r['n']),int(r['m']),r['method'])
                ==('fresh','nested',4096,m,'source_full')]
            total=sum(float(r['observed_contribution']) for r in rs)
            low=[r for r in rs if r['stratum'] in ('0','1-4')]
            high=[r for r in rs if r['stratum'] not in ('0','1-4')]
            query=100*sum(float(r['position_weight']) for r in low)
            energy=100*sum(float(r['observed_contribution']) for r in low)/total
            low_error=100*sum(float(r['signed_error_contribution']) for r in low)/total
            high_error=100*sum(float(r['signed_error_contribution']) for r in high)/total
            coverage.append(dict(corpus=corpus,n=4096,m=m,low_count_position_percent=query,
                low_count_observed_energy_percent=energy,low_count_error_percent=low_error,
                other_error_percent=high_error,total_error_percent=low_error+high_error))
    write_csv(OUT/'coverage_summary.csv',coverage)
    (OUT/'coverage_rows.tex').write_text('\n'.join(
        f"{LABELS[r['corpus']]} & {r['m']:,} & {r['low_count_position_percent']:.1f} & "
        f"{r['low_count_observed_energy_percent']:.1f} & {r['low_count_error_percent']:.1f} & "
        f"{r['other_error_percent']:.1f} \\\\" for r in coverage)+'\n',encoding='utf-8')
    synthetic=read_json(ROOT/'results/full_vocabulary_v1/reproducibility_followup_v1/synthetic/summary.json')
    results['synthetic_total_ratios']=[dict(n=r['n'],m=r['reference_chunks'],ratio=r['total_ratio']) for r in synthetic]
    write_json(OUT/'hypothesis_checks.json',results)
    print(json.dumps(results,indent=2))


def main():
    rows,sources=collect()
    write_csv(OUT/'all_points.csv',rows)
    plot(rows,OUT/'reference_ratio_comparison')
    tables_and_checks(rows)
    write_json(OUT/'figure_sources.json',dict(built_utc=utc(),source_sha256=sources,
        definition='Forecast divided by arithmetic mean observed disagreement',
        intervals='Conditional t interval for mean observed disagreement, inverted against fixed forecast',
        x_offsets='Small deterministic display offsets at coincident reference/target ratios',
        historical_contexts='Earlier prospective and later matched studies share evaluation text, masks, and three A/B pairs but differ in auxiliary estimates. New follow-up cells fix the matched auxiliary estimates. Historical points are not independent replications.',
        code_sha256=sha(__file__),output_sha256={name:sha(OUT/name) for name in
            ('all_points.csv','reference_ratio_comparison.pdf','reference_ratio_comparison.png',
             'fresh_table_rows.tex','new_cells_rows.tex','coverage_summary.csv','coverage_rows.tex','hypothesis_checks.json')}))


if __name__=='__main__':main()
