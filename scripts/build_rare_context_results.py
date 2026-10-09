"""Plots and readable tables from the saved correction-study measurements."""
from pathlib import Path
import csv
import json
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/rare_context_prospective_v1'
DEV=ROOT/'results/rare_context_development_v1'
LABELS={'tinystories':'TinyStories','wikitext':'WikiText-103','cnn_dailymail':'CNN/DailyMail'}
METHODS={'source_full':('Full source','#436b95','o'),
         'source_jackknife':('Source jackknife','#c2633d','s'),
         'multinomial':('Multinomial','#488265','^')}


def read_csv(path):
    with path.open(newline='',encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def save(fig,name):
    fig.savefig(OUT/f'{name}.pdf',metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(OUT/f'{name}.png',dpi=190)
    plt.close(fig)


def calibration(rows,sizes,name):
    fig,axes=plt.subplots(len(sizes),3,figsize=(8.7,2.45*len(sizes)+.9),squeeze=False,sharey=True)
    for ri,n in enumerate(sizes):
        for ci,(corpus,label) in enumerate(LABELS.items()):
            ax=axes[ri,ci]
            ax.axhline(1,color='.3',ls=':',lw=1,zorder=2.2)
            for method,(_,color,marker) in METHODS.items():
                data=sorted([r for r in rows if r['corpus']==corpus and int(r['n'])==n and r['method']==method],key=lambda r:int(r['m']))
                x=np.array([int(r['m'])/n for r in data])
                y=np.array([float(r['ratio']) for r in data])
                lo=np.array([float(r['ratio_low']) for r in data])
                hi=np.array([float(r['ratio_high']) for r in data])
                ax.errorbar(x,y,yerr=[y-lo,hi-y],color=color,marker=marker,ms=4,lw=1,capsize=2)
            r=next(r for r in rows if r['corpus']==corpus and int(r['n'])==n and r['method']=='source_large_reference')
            y=float(r['ratio'])
            ax.errorbar([8192/n],[y],yerr=[[y-float(r['ratio_low'])],[float(r['ratio_high'])-y]],
                color='.25',marker='*',ms=8,ls='none',capsize=2)
            ax.set_xscale('log',base=2)
            ticks=np.array([1024,2048,4096,8192])/n
            ax.set_xticks(ticks,[f'{v:g}' for v in ticks])
            ax.set_ylim(.35,1.4)
            ax.grid(axis='y',color='.9',lw=.6)
            ax.spines[['top','right']].set_visible(False)
            if ri==0:
                ax.set_title(label,fontsize=10)
            if ci==0:
                ax.set_ylabel(f'n = {n:,} chunks\nForecast / observed')
            if ri==len(sizes)-1:
                ax.set_xlabel('Reference / target size, m/n')
    handles=[Line2D([0],[0],color=color,marker=marker,lw=1,label=label) for label,color,marker in METHODS.values()]
    handles.append(Line2D([0],[0],color='.25',marker='*',lw=0,label='Source, larger reference'))
    exploratory=len(handles)>4
    fig.legend(handles=handles,loc='upper center',ncol=3 if exploratory else 4,frameon=False,fontsize=8.3,bbox_to_anchor=(.53,.995))
    fig.subplots_adjust(left=.105,right=.99,bottom=.20 if exploratory else (.18 if len(sizes)==2 else .115),
        top=.83 if exploratory else (.88 if len(sizes)==2 else .93),hspace=.24,wspace=.14)
    note='Six source-disjoint reference banks and 20 A/B pairs per corpus. Bars: 95% crossed bootstrap intervals.\nAuxiliary and evaluation data are fixed. Mask rate q = 0.5. The dotted line marks exact calibration.'
    if exploratory:
        note+='\nThe purple curve is an exploratory comparison added after the primary outcomes.'
    fig.text(.105,.025,note,fontsize=8)
    save(fig,name)


def mechanism():
    rows=json.loads((DEV/'missing_row_decomposition/summary.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(8.7,3.55),sharey=True)
    parts=[('coverage_gap','Unseen rows + cross terms','#c2633d'),
           ('seen_reference_error','Observed-row reference error','#436b95'),
           ('propagation_error','Linearization error','#488265')]
    for ax,case,label in zip(axes,['iid_sparse','markov_sparse','clustered_sparse'],
                            ['Independent tokens','Dependent tokens','Four chunks per source']):
        low,high=np.zeros(3),np.zeros(3)
        for part,_,color in parts:
            y=np.array([100*next(r['fraction'] for r in rows if (r['case'],r['n'],r['m'],r['component'])==(case,1024,m,part)) for m in [128,512,2048]])
            bottom=np.where(y<0,low,high)
            ax.bar(np.arange(3),y,bottom=bottom,width=.56,color=color)
            low+=np.minimum(y,0); high+=np.maximum(y,0)
        source=json.loads((DEV/case/'summary.json').read_text())
        values=[next(r for r in source if (r['n'],r['m'],r['method'])==(1024,m,'source_full')) for m in [128,512,2048]]
        y=np.array([100*(r['ratio']-1) for r in values])
        lo=np.array([100*(r['ratio_ci'][0]-1) for r in values])
        hi=np.array([100*(r['ratio_ci'][1]-1) for r in values])
        ax.errorbar(np.arange(3),y,yerr=[y-lo,hi-y],fmt='D',color='.15',ms=4,capsize=2,label='Net error')
        ax.axhline(0,color='.3',lw=.8)
        ax.set_xticks(np.arange(3),['0.125','0.5','2'])
        ax.set_xlabel('Reference / target size, m/n')
        ax.set_title(label,fontsize=10)
        ax.spines[['top','right']].set_visible(False)
        ax.set_ylim(-85,18)
        ax.grid(axis='y',color='.92',lw=.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('Calibration error (percentage points)')
    handles=[Line2D([0],[0],color=color,lw=5,label=label) for _,label,color in parts]
    handles.append(Line2D([0],[0],color='.15',marker='D',lw=0,label='Net error'))
    fig.legend(handles=handles,loc='upper center',ncol=2,frameon=False,fontsize=8.3)
    fig.subplots_adjust(left=.10,right=.99,bottom=.22,top=.75,wspace=.12)
    fig.text(.10,.04,'Known population: K = 512, L = 8, alpha = 5, n = 1,024. Four independent references and 80 A/B pairs.\nBars sum to net error. Unseen-row contributions include covariance with observed-row contributions.',fontsize=8)
    save(fig,'known_population_decomposition')


def main():
    plt.rcParams.update({'font.size':9,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,
                         'pdf.fonttype':42,'ps.fonttype':42,'font.family':'DejaVu Sans'})
    rows=read_csv(OUT/'summary.csv')
    calibration(rows,[2048,4096],'calibration_large_targets')
    calibration(rows,[512,1024,2048,4096],'calibration_all_targets')
    mechanism()
    primary=json.loads((OUT/'primary_result.json').read_text())
    decisions=read_csv(OUT/'decisions.csv')
    decision_summary=[]
    for method in METHODS:
        data=[r for r in decisions if r['method']==method]
        counts={name:sum(r['outcome']==name for r in data) for name in
                ['upper_interval_below_target','mean_below_target','mean_above_target','abstain']}
        decision_summary.append(dict(method=method,**counts,total=len(data)))
    (OUT/'decision_summary.json').write_text(json.dumps(decision_summary,indent=2)+'\n')
    lines=[]
    for r in primary['methods']:
        name=METHODS[r['method']][0]
        lo,hi=r['interval']
        dlo,dhi=r['difference_vs_source_interval']
        lines.append(f"{name} & {r['mean_absolute_log_error']:.3f} [{lo:.3f}, {hi:.3f}] & {r['difference_vs_source']:+.3f} [{dlo:+.3f}, {dhi:+.3f}] " + chr(92)*2)
    (OUT/'primary_table_rows.tex').write_text('\n'.join(lines)+'\n')
    if (OUT/'corrected_baseline_calibration.csv').exists():
        METHODS['multinomial_jackknife']=('Multinomial jackknife (exploratory)','#8756a4','v')
        calibration([*rows,*read_csv(OUT/'corrected_baseline_calibration.csv')],
                    [2048,4096],'calibration_corrected_methods')
    print('Built calibration figures, exact mechanism decomposition, primary table and decision summary.')


if __name__=='__main__':
    main()
