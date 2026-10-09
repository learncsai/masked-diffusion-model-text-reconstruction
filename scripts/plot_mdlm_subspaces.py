"""Render the fixed subspace comparison from saved, source-grouped results."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, ScalarFormatter
ROOT=Path(__file__).resolve().parents[1]
CORPORA=('tinystories','wikitext','cnn_dailymail')
LABELS=('TinyStories','WikiText-103','CNN/DailyMail')
STYLES=(('reconstructor','Observed reconstructor','#555555','D','-'),
        ('source_full','Full source forecast','#7651A8','^','-'),
        ('source_same_lag','Same-lag source forecast','#64893D','v','--'),
        ('multinomial','Multinomial forecast','#C36B25','s','--'),
        ('affine','Full-context affine','#B54F66','P','-'),
        ('unigram','Unigram covariance control','#24718C','o',':'))
KS=np.array([1,2,4,8,16])


def read(path):
    with path.open(newline='',encoding='utf-8') as f:
        return [{k:v if k in ('corpus','method','mode') else float(v) for k,v in r.items()} for r in csv.DictReader(f)]


def one(rows,**keys):
    matched=[r for r in rows if all(r[k]==v for k,v in keys.items())]
    assert len(matched)==1, keys
    return matched[0]


def build(out,copy=False):
    rows=read(out/'summary.csv')
    assert len(rows)==2160 and all(r['pairs']==3 for r in rows)
    plt.rcParams.update({'font.size':11,'axes.titlesize':12,'legend.fontsize':10,
        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    for name,dimensions in [('mdlm_subspaces',[64]),('mdlm_subspaces_dimension_check',[64,128])]:
        fig,axes=plt.subplots(len(dimensions),3,figsize=(11.7,3.6*len(dimensions)+1),squeeze=False,sharex=True,sharey=True)
        fig.subplots_adjust(left=.08,right=.98,top=1-.95/(3.6*len(dimensions)+1),bottom=.15 if len(dimensions)==1 else .09,
                            wspace=.12,hspace=.24)
        handles=[]
        for ri,dimension in enumerate(dimensions):
            for ci,corpus in enumerate(CORPORA):
                ax=axes[ri,ci]
                for method,label,color,marker,ls in STYLES:
                    values=[one(rows,corpus=corpus,n=2048,updates=12000,dimension=dimension,
                                mode='source_crossfit',method=method,k=k) for k in KS]
                    line,=ax.plot(KS,[r['capture'] for r in values],color=color,marker=marker,ls=ls,lw=1.4,ms=4,label=label)
                    if ri==0 and ci==0:handles.append(line)
                line,=ax.plot(KS,KS/dimension,color='black',ls=':',lw=1.2,label='Random-subspace expectation k/d')
                if ri==0 and ci==0:handles.append(line)
                ax.set_xscale('log',base=2);ax.set_xticks(KS);ax.xaxis.set_major_formatter(ScalarFormatter())
                ax.set_ylim(0,1);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(axis='y',color='#dddddd',lw=.5)
                if ri==0:ax.set_title(LABELS[ci])
                if ci==0:ax.set_ylabel(f'Captured MDLM energy\n(d = {dimension}, higher is better)')
                if ri==len(dimensions)-1:ax.set_xlabel('Number of directions k')
        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,1.01),ncol=3,frameon=False)
        for ext in ('pdf','png'):
            kw=dict(metadata={'CreationDate':None,'ModDate':None}) if ext=='pdf' else dict(dpi=180)
            fig.savefig(out/f'{name}.{ext}',bbox_inches='tight',**kw)
        plt.close(fig)
    lines=[];ranges=[]
    for method,label,*_ in STYLES:
        vals=[one(rows,corpus=c,n=2048,updates=12000,dimension=64,mode='source_crossfit',method=method,k=4) for c in CORPORA]
        lines.append(label+' & '+' & '.join(f"{100*r['capture']:.1f}" for r in vals)+r' \\')
        ranges.append(label+' & '+' & '.join(f"{100*r['capture']:.1f} [{100*r['minimum']:.1f}, {100*r['maximum']:.1f}]" for r in vals)+r' \\')
    lines.append(r'Random expectation & 6.25 & 6.25 & 6.25 \\')
    ranges.append(r'Random expectation & 6.25 & 6.25 & 6.25 \\')
    (out/'table_rows.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (out/'range_table_rows.tex').write_text('\n'.join(ranges)+'\n',encoding='utf-8')
    if copy:
        name='mdlm_subspaces';target=ROOT/'paper/figures/fig_mdlm_subspaces.pdf'
        shutil.copyfile(out/f'{name}.pdf',target)
        sources=[out/'summary.csv',out/'per_pair.csv',ROOT/'docs/mdlm_subspace_protocol.txt',Path(__file__)]
        manifest=dict(figure=target.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
            sources={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
        (ROOT/'paper/figures/mdlm_subspace_figure_sources.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'results/mdlm_subspaces_v1')
    p.add_argument('--copy-figure',action='store_true')
    a=p.parse_args();build(a.output,a.copy_figure)
