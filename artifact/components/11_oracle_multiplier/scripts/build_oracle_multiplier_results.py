"""Render the multiplier diagnostic and numerical rows for the paper."""
from pathlib import Path
import argparse
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[1]
NAMES={'tinystories':'TinyStories','wikitext':'WikiText-103','cnn_dailymail':'CNN/DailyMail'}
CASES={'iid_sparse':'Independent tokens', 'markov_sparse':'Dependent tokens',
       'clustered_sparse':r'Four chunks, $\alpha=5$',
       'clustered_alpha1_matched':r'Four chunks, $\alpha=1$',
       'clustered_alpha20_matched':r'Four chunks, $\alpha=20$'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=ROOT);a=ap.parse_args()
    root=a.root;out=root/'results/oracle_multiplier_v1'
    m=json.loads((out/'multiplier_summary.json').read_text())
    oracle=json.loads((out/'oracle_summary.json').read_text())
    rows=[]
    labels=[r'Multinomial $\times1.25$', 'Multinomial, calibrated on original grid',
            'Multinomial, calibrated on other corpora']
    for j,label in enumerate(labels,4):
        vals=[]
        for key in ['primary','full_grid']:
            lo,hi=m[key]['interval'][j]
            vals.append(f"{m[key]['error'][j]:.3f} [{lo:.3f}, {hi:.3f}]")
        rows.append(label+' & '+' & '.join(vals))
    tables={'tab:confirmation':rows}
    rows=[]
    for r in oracle:
        vals=[]
        for j,key in enumerate(['source_ratio','multinomial_ratio']):
            lo,hi=r['ratio_ci'][j];vals.append(f"{r[key]:.3f} [{lo:.3f}, {hi:.3f}]")
        rows.append(f"{CASES[r['case']]} & {r['n']:,} & "+' & '.join(vals))
    tables['tab:population-oracle']=rows
    (out/'tables.json').write_text(json.dumps(tables,indent=2)+'\n')
    for label,rows in tables.items():
        (out/(label.replace(':','_')+'.tex')).write_text(' \\\\\n'.join(rows)+' \\\\\n')
    cells=list(csv.DictReader((out/'multiplier_cells.csv').open(newline='')))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(7.0,2.75),sharey=True)
    styles=[(512,'o','#436b95'),(1024,'s','#c2633d'),(2048,'^','#488265'),(4096,'D','#8756a4')]
    for ax,(c,name) in zip(axes,NAMES.items()):
        for n,marker,color in styles:
            selected=sorted([r for r in cells if r['corpus']==c and int(r['n'])==n],key=lambda r:float(r['ratio']))
            y=np.array([float(r['factor']) for r in selected]);x=[float(r['ratio']) for r in selected]
            low=np.array([float(r['low']) for r in selected]);high=np.array([float(r['high']) for r in selected])
            ax.errorbar(x,y,yerr=[y-low,high-y],color=color,marker=marker,ms=3.4,lw=.8,capsize=2)
        ax.axhline(1.25,color='.2',ls='--',lw=.8)
        ax.axhline(m['global_factor'],color='.45',ls=':',lw=.9)
        ax.set_xscale('log',base=2);ax.set_xticks([.25,.5,1,2,4,8],['.25','.5','1','2','4','8'])
        ax.set_title(name,fontsize=9,pad=5);ax.set_xlabel('Reference / target size, m/n')
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',color='.9',lw=.5)
        ax.set_ylim(.96,1.46)
    axes[0].set_ylabel('Best multiplier on original outcomes')
    handles=[Line2D([0],[0],color=color,marker=marker,lw=.8,ms=3.4,label=f'n = {n:,}') for n,marker,color in styles]
    handles += [Line2D([0],[0],color='.2',ls='--',lw=.8,label='Fixed 1.25'),
                Line2D([0],[0],color='.45',ls=':',lw=.9,label=f'Global {m["global_factor"]:.3f}')]
    fig.legend(handles=handles,ncol=3,loc='upper center',bbox_to_anchor=(.54,1),frameon=False,
               fontsize=8,columnspacing=1.6,handlelength=2)
    fig.subplots_adjust(left=.09,right=.985,bottom=.18,top=.72,wspace=.14)
    fig.savefig(out/'fig_multiplier_transfer.pdf',metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(out/'fig_multiplier_transfer.png',dpi=170)
    plt.close(fig)
    print('Wrote one multiplier figure and two sets of manuscript rows.')


if __name__=='__main__':main()
