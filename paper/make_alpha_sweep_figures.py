"""Publication plots for the separate real-text smoothing appendix."""
from pathlib import Path
import argparse,csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter,PercentFormatter,FuncFormatter,NullFormatter

CORPORA=['tinystories','wikitext','cnn_dailymail']
LABELS=['TinyStories','WikiText-103','CNN/DailyMail']
COLORS={1:'#245A81',5:'#B85D24',20:'#62733B',50:'#8C4E80'}

def load_csv(path):
    with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))

def main(source,output,config):
    cfg=json.loads(config.read_text());output.mkdir(parents=True,exist_ok=True)
    alphas=cfg['alphas'];n=np.array(cfg['sizes']);mi=cfg['reference_sizes'].index(cfg['primary_reference'])
    data={}
    for c in CORPORA:
        with np.load(source/f'{c}_plot_data.npz') as z:data[c]={k:z[k] for k in z.files}
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    # Two rows separate the laws, allowing observed and predicted curves to be read.
    for arm,name in [(0,'fixed'),(1,'refit')]:
        fig,axes=plt.subplots(2,3,figsize=(10.6,6.6),layout='constrained',sharex=True,sharey=True)
        for col,c in enumerate(CORPORA):
            d=data[c]
            for row,method in enumerate((2,0)):
                ax=axes[row,col]
                for alpha,color in COLORS.items():
                    ai=alphas.index(alpha);obs=d['observed'][arm,ai];pred=d['predicted'][arm,ai,mi,:,method]
                    ax.plot(n,obs/obs[0],'-o',color=color,markersize=4)
                    ax.plot(n,pred/pred[0],'--',color=color,linewidth=1.7)
                ax.plot(n,n[0]/n,':',color='#333333',linewidth=1.5)
                ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_xticks(n);ax.xaxis.set_major_formatter(ScalarFormatter())
                ax.set_yticks([.125,.25,.5,1]);ax.yaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:g}'));ax.yaxis.set_minor_formatter(NullFormatter())
                ax.grid(axis='y',color='#dedede',linewidth=.6)
                if row==0:ax.set_title(LABELS[col])
                else:ax.set_xlabel('Corpus size n\n(64-token chunks)')
            axes[0,0].set_ylabel('Disagreement / own n=512 value\nSource forecast')
            axes[1,0].set_ylabel('Disagreement / own n=512 value\nMultinomial forecast')
        handles=[Line2D([],[],color=c,label=f'α = {a}') for a,c in COLORS.items()]
        handles += [Line2D([],[],color='#333333',marker='o',label='Observed'),Line2D([],[],color='#333333',linestyle='--',label='Forecast'),Line2D([],[],color='#333333',linestyle=':',label='1/n benchmark')]
        fig.legend(handles=handles,loc='outside lower center',ncol=4,frameon=False)
        fig.savefig(output/f'fig_alpha_size_{name}.pdf');plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(10.6,6.4),layout='constrained',sharex=True,sharey=True)
    for col,c in enumerate(CORPORA):
        d=data[c]
        for arm in range(2):
            ax=axes[arm,col]
            ax.plot(alphas,d['beta'][arm],'-o',color='#333333',label='Observed',markersize=4)
            ax.fill_between(alphas,d['beta_ci'][0,arm],d['beta_ci'][1,arm],color='#777777',alpha=.16)
            for method,color,label,marker in [(2,'#245A81','Source forecast','s'),(0,'#B85D24','Multinomial forecast','^')]:
                ax.plot(alphas,d['beta_pred'][arm,:,mi,method],'--'+marker,color=color,label=label,markersize=4)
            ax.axhline(-1,color='#777777',linestyle=':',label='1/n benchmark')
            ax.set_xscale('log',base=2);ax.set_xticks(alphas);ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:g}'));ax.grid(axis='y',color='#dedede',linewidth=.6)
            if arm==0:ax.set_title(LABELS[col])
            else:ax.set_xlabel('Smoothing α')
    axes[0,0].set_ylabel('Size slope β\nFixed weights');axes[1,0].set_ylabel('Size slope β\nRefit auxiliary weights')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4,frameon=False)
    fig.savefig(output/'fig_alpha_slopes.pdf');plt.close(fig)
    # Per-bin comparison uses one stated adjacent-size contrast to keep axes readable.
    records=load_csv(source/'row_bins.csv')
    fig,axes=plt.subplots(1,3,figsize=(10.6,3.8),layout='constrained',sharey=True)
    for col,c in enumerate(CORPORA):
        for alpha,color in COLORS.items():
            rr=[r for r in records if r['corpus']==c and r['arm']=='fixed' and float(r['alpha'])==alpha and int(r['n1'])==1024 and r['lambda_over_alpha'] and r['elasticity']]
            rr.sort(key=lambda r:float(r['lambda_over_alpha']))
            x=[float(r['lambda_over_alpha']) for r in rr];y=[float(r['elasticity']) for r in rr]
            axes[col].plot(x,y,'-o',color=color,markersize=4)
            pp=[r for r in rr if r['poisson_elasticity']]
            axes[col].plot([float(r['lambda_over_alpha']) for r in pp],[float(r['poisson_elasticity']) for r in pp],'--',color=color,linewidth=1)
        axes[col].axhline(0,color='#777777',linewidth=.8);axes[col].axhline(-1,color='#777777',linestyle=':',linewidth=.8)
        axes[col].axvline(1,color='#999999',linewidth=.7,linestyle=':');axes[col].set_xscale('log')
        axes[col].set_title(LABELS[col]);axes[col].set_xlabel('Auxiliary rate × √(1024 × 2048) / α')
    axes[0].set_ylabel('Same-lag bin elasticity\n1024 to 2048 chunks')
    handles=[Line2D([],[],color=color,label=f'α = {a}') for a,color in COLORS.items()]
    handles += [Line2D([],[],color='#333333',marker='o',label='Observed'),Line2D([],[],color='#333333',linestyle='--',label='Poisson reference curve')]
    fig.legend(handles=handles,loc='outside lower center',ncol=3,frameon=False)
    fig.savefig(output/'fig_alpha_rows.pdf');plt.close(fig)
    # Accuracy/stability tradeoff and one-sided same-lag energy use all alphas.
    q=load_csv(source/'quality.csv');cov=load_csv(source/'coverage.csv')
    fig,axes=plt.subplots(2,3,figsize=(10.6,6.2),layout='constrained',sharex=True)
    for col,c in enumerate(CORPORA):
        for arm,color,label in [('fixed','#245A81','Fixed weights'),('refit','#B85D24','Refit auxiliary weights')]:
            qq=sorted([r for r in q if r['corpus']==c and r['arm']==arm and int(r['n'])==2048],key=lambda r:float(r['alpha']))
            cc=sorted([r for r in cov if r['corpus']==c and r['arm']==arm and int(r['n'])==2048],key=lambda r:float(r['alpha']))
            axes[0,col].plot(alphas,[float(r['relative_mse_improvement']) for r in qq],'-o',color=color,label=label,markersize=4)
            axes[1,col].plot(alphas,[float(r['one_sided_share']) for r in cc],'-o',color=color,label=label,markersize=4)
        for row in range(2):
            axes[row,col].set_xscale('log',base=2);axes[row,col].set_xticks(alphas);axes[row,col].xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:g}'))
            axes[row,col].yaxis.set_major_formatter(PercentFormatter(1));axes[row,col].grid(axis='y',color='#dedede',linewidth=.6)
        axes[0,col].set_title(LABELS[col]);axes[1,col].set_xlabel('Smoothing α')
    axes[0,0].set_ylabel('Relative MSE improvement\n(higher is better)');axes[1,0].set_ylabel('One-sided context share\nof same-lag energy')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False)
    fig.savefig(output/'fig_alpha_quality_coverage.pdf');plt.close(fig)
    print('Saved five smoothing-sweep figures')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=Path('results/alpha_sweep_v1/summary'))
    ap.add_argument('--output',type=Path,default=Path('paper/figures'));ap.add_argument('--config',type=Path,default=Path('configs/alpha_sweep_v1.json'))
    a=ap.parse_args();main(a.source,a.output,a.config)
