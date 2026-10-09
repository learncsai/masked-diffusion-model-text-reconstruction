"""Manuscript figures from the saved finite-reference correction measurements."""
from pathlib import Path
import argparse
import csv
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
NAMES = {'tinystories':'TinyStories', 'wikitext':'WikiText-103', 'cnn_dailymail':'CNN/DailyMail'}
METHODS = {
    'source_full': ('Full source', '#436b95', 'o'),
    'source_jackknife': ('Source jackknife', '#c2633d', 's'),
    'multinomial': ('Multinomial', '#488265', '^'),
    'multinomial_jackknife': ('Multinomial jackknife', '#8756a4', 'v'),
}

def read_csv(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))

def save(fig, output, name):
    fig.savefig(output / (name+'.pdf'), metadata={'CreationDate':None, 'ModDate':None})
    fig.savefig(output / (name+'.png'), dpi=170)
    plt.close(fig)

def calibration(rows, sizes, output, name):
    nr = len(sizes)
    fig, axes = plt.subplots(nr, 3, figsize=(7.0, 2.18 if nr==1 else 7.6), sharey=True, squeeze=False)
    for ri, n in enumerate(sizes):
        for ci, (corpus, label) in enumerate(NAMES.items()):
            ax = axes[ri,ci]
            ax.axhline(1, color='.35', ls=':', lw=.85)
            for method, (_, color, marker) in METHODS.items():
                selected=sorted((r for r in rows if (r['corpus'],int(r['n']),r['method'])==(corpus,n,method)), key=lambda r:int(r['m']))
                y=np.array([float(r['ratio']) for r in selected])
                low=np.array([float(r['ratio_low']) for r in selected])
                high=np.array([float(r['ratio_high']) for r in selected])
                ax.errorbar([int(r['m'])/n for r in selected],y,yerr=[y-low,high-y],
                    color=color,marker=marker,ms=3.2,lw=.9,capsize=2)
            r=next(r for r in rows if (r['corpus'],int(r['n']),r['method'])==(corpus,n,'source_large_reference'))
            y=float(r['ratio'])
            ax.errorbar([8192/n],[y],yerr=[[y-float(r['ratio_low'])],[float(r['ratio_high'])-y]],
                color='.25',marker='*',ms=6,ls='none',capsize=2)
            ax.set_xscale('log',base=2)
            x=np.array([1024,2048,4096,8192])/n
            ax.set_xticks(x,[f'{v:g}' for v in x])
            ax.set_ylim(.35,1.4 if nr>1 else 1.12)
            ax.set_yticks([.4,.6,.8,1.0,1.2,1.4] if nr>1 else [.4,.6,.8,1.0])
            ax.grid(axis='y',color='.9',lw=.5)
            ax.spines[['top','right']].set_visible(False)
            if ri==0: ax.set_title(label,fontsize=9,pad=4)
            if ci==0: ax.set_ylabel('Forecast / observed' if nr==1 else f'n = {n:,}\nForecast / observed')
            if ri==nr-1: ax.set_xlabel('Reference / target size, m/n',labelpad=3)
    handles=[Line2D([0],[0],color=color,marker=marker,ms=3.2,lw=.9,label=label) for label,color,marker in METHODS.values()]
    handles.append(Line2D([0],[0],color='.25',marker='*',ms=6,lw=0,label='Source, 8,192 reference'))
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.53,1.0),ncol=3,frameon=False,
               fontsize=7.8,handlelength=1.8,columnspacing=1.2)
    fig.subplots_adjust(left=.082,right=.99,top=.68 if nr==1 else .91,bottom=.20 if nr==1 else .057,
                        hspace=.34,wspace=.14)
    save(fig,output,name)

def mechanism(source,output):
    rows=json.loads((source/'rare_context_development_v1/missing_row_decomposition/summary.json').read_text())
    parts=[('coverage_gap','Unseen rows and cross terms','#c2633d'),
           ('seen_reference_error','Observed-row reference error','#436b95'),
           ('propagation_error','Linearization error','#488265')]
    fig,axes=plt.subplots(1,3,figsize=(7.0,2.8),sharey=True)
    for ax,case,label in zip(axes,['iid_sparse','markov_sparse','clustered_sparse'],
                           ['Independent tokens','Dependent tokens','Four chunks per source']):
        low,high=np.zeros(3),np.zeros(3)
        for part,_,color in parts:
            y=np.array([100*next(r['fraction'] for r in rows if (r['case'],r['n'],r['m'],r['component'])==(case,1024,m,part)) for m in [128,512,2048]])
            ax.bar(np.arange(3),y,bottom=np.where(y<0,low,high),width=.56,color=color)
            low+=np.minimum(y,0); high+=np.maximum(y,0)
        summary=json.loads((source/'rare_context_development_v1'/case/'summary.json').read_text())
        selected=[next(r for r in summary if (r['n'],r['m'],r['method'])==(1024,m,'source_full')) for m in [128,512,2048]]
        y=np.array([100*(r['ratio']-1) for r in selected])
        lo=np.array([100*(r['ratio_ci'][0]-1) for r in selected])
        hi=np.array([100*(r['ratio_ci'][1]-1) for r in selected])
        ax.errorbar(np.arange(3),y,yerr=[y-lo,hi-y],fmt='D',color='.15',ms=3,capsize=2)
        np.testing.assert_allclose(low+high,y,atol=1e-8)
        ax.axhline(0,color='.35',ls=':',lw=.85)
        ax.set_xticks(np.arange(3),['0.125','0.5','2'])
        ax.set_xlabel('Reference / target size, m/n')
        ax.set_title(label,fontsize=9,pad=5)
        ax.spines[['top','right']].set_visible(False)
        ax.set_ylim(-85,18)
        ax.grid(axis='y',color='.92',lw=.5); ax.set_axisbelow(True)
    axes[0].set_ylabel('Calibration error\n(percentage points)')
    handles=[Line2D([0],[0],color=color,lw=4,label=label) for _,label,color in parts]
    handles.append(Line2D([0],[0],color='.15',marker='D',ms=3,lw=0,label='Net error'))
    fig.legend(handles=handles,loc='upper center',ncol=2,frameon=False,fontsize=8)
    fig.subplots_adjust(left=.10,right=.99,bottom=.17,top=.71,wspace=.13)
    save(fig,output,'fig_missing_context_decomposition')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'results')
    parser.add_argument('--output',type=Path,default=ROOT/'paper/figures')
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':8,'axes.labelsize':8,'xtick.labelsize':7.5,'ytick.labelsize':7.5,
                         'pdf.fonttype':42,'ps.fonttype':42,'font.family':'DejaVu Sans'})
    paths=[args.source/'rare_context_prospective_v1'/name for name in ['summary.csv','corrected_baseline_calibration.csv']]
    rows=sum((read_csv(p) for p in paths),[])
    calibration(rows,[4096],args.output,'fig_reference_correction')
    calibration(rows,[512,1024,2048,4096],args.output,'fig_reference_correction_all')
    mechanism(args.source,args.output)
    inputs=paths+list((args.source/'rare_context_development_v1').glob('*/summary.json'))
    (args.output/'correction_figure_sources.json').write_text(json.dumps({str(p.relative_to(args.source)):
        hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},indent=2)+'\n')

if __name__=='__main__': main()
