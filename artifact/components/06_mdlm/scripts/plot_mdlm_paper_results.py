"""Publication figures for the matched neural and count-based experiments."""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter, PercentFormatter

CORPORA=("tinystories","wikitext","cnn_dailymail")
LABELS=("TinyStories","WikiText-103","CNN/DailyMail")
SIZES=np.array([512,1024,2048])
COLORS=dict(mdlm="#245A81",lag="#555555",source_full="#73538C",multinomial="#B85D24",source_same_lag="#62733B")
STYLES=[("mdlm",8000,"MDLM, 8k updates",COLORS["mdlm"],"o","--"),
        ("mdlm",12000,"MDLM, 12k updates",COLORS["mdlm"],"o","-"),
        ("lag",0,"Reconstructor observed",COLORS["lag"],"D","-"),
        ("source_full",0,"Source forecast (lag)",COLORS["source_full"],"^","-"),
        ("multinomial",0,"Multinomial forecast (lag)",COLORS["multinomial"],"s","-")]


def selected(rows,**keys):
    return sorted([r for r in rows if all(r[k]==v for k,v in keys.items())],key=lambda r:r.get("n",r.get("k",0)))


def decorate(ax, xlabel=True):
    ax.set_xscale("log",base=2)
    ax.set_xticks(SIZES)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlim(460,2250)
    ax.grid(axis="y",color="#dddddd",linewidth=.5)
    if xlabel: ax.set_xlabel("Corpus size n\n(64-token chunks)")


def save(fig,out,name):
    fig.savefig(out/f"{name}.pdf",bbox_inches="tight",metadata={"CreationDate":None,"ModDate":None})
    fig.savefig(out/f"{name}.png",dpi=180,bbox_inches="tight")
    plt.close(fig)


def plot_all(data,out):
    plt.rcParams.update({"font.size":12,"axes.titlesize":13,"axes.labelsize":12,"legend.fontsize":10,
        "axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42,"font.family":"DejaVu Sans"})
    fig,axs=plt.subplots(2,3,figsize=(11.6,6.3),sharex=True,sharey="row")
    fig.subplots_adjust(left=.09,right=.985,bottom=.12,top=.80,hspace=.25,wspace=.13)
    handles=[]
    for col,c in enumerate(CORPORA):
        for m,t,label,color,marker,ls in STYLES:
            records=data["forecasts"] if m in ("source_full","multinomial") else data["summary"]
            keys=dict(corpus=c,method=m)
            if records is data["summary"]: keys["updates"]=t
            rows=selected(records,**keys)
            y=np.array([r["disagreement"] for r in rows])
            line,=axs[0,col].plot(SIZES,y,color=color,marker=marker,linestyle=ls,lw=1.6,ms=4.5,
                                    markerfacecolor="white" if t==8000 else color,label=label)
            if col==0: handles.append(line)
            if records is data["summary"]:
                low=np.array([r["minimum_disagreement"] for r in rows]);high=np.array([r["maximum_disagreement"] for r in rows])
                axs[0,col].errorbar(SIZES,y,yerr=[y-low,high-y],fmt="none",ecolor=color,elinewidth=.7,capsize=2)
            axs[1,col].plot(SIZES,y/y[0],color=color,marker=marker,linestyle=ls,lw=1.6,ms=4.5,
                            markerfacecolor="white" if t==8000 else color)
        axs[0,col].set_title(LABELS[col])
        axs[0,col].set_yscale("log")
        axs[0,col].set_ylim(.006,.24)
        axs[0,col].set_yticks([.01,.03,.1,.2],labels=["0.01","0.03","0.10","0.20"])
        axs[1,col].plot(SIZES,512/SIZES,":",color="black",lw=1.4)
        axs[1,col].set_ylim(.1,1.1)
        for row in (0,1): decorate(axs[row,col],xlabel=row==1)
    axs[0,0].set_ylabel("Mean squared disagreement\n(log scale)")
    axs[1,0].set_ylabel("Disagreement / own n=512 value")
    handles.append(Line2D([],[],color="black",ls=":",label="1/n benchmark (bottom row)"))
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.52,1),ncol=3,frameon=False)
    save(fig,out,"mdlm_disagreement")

    fig,axs=plt.subplots(2,3,figsize=(11.6,6.8))
    fig.subplots_adjust(left=.085,right=.985,bottom=.09,top=.82,hspace=.85,wspace=.2)
    handles=[]
    for col,c in enumerate(CORPORA):
        for m,t,label,color,marker,ls in STYLES[:3]:
            rows=selected(data["summary"],corpus=c,method=m,updates=t)
            line,=axs[0,col].plot(SIZES,[r["relative_mse_improvement"] for r in rows],color=color,marker=marker,ls=ls,lw=1.7,
                                 markerfacecolor="white" if t==8000 else color,label=label)
            if col==0: handles.append(line)
            top=sorted(selected(data["topk"],corpus=c,n=2048,method=m,updates=t),key=lambda r:r["k"])
            axs[1,col].plot([r["k"] for r in top],[r["accuracy"] for r in top],color=color,marker=marker,ls=ls,lw=1.7,ms=4,
                            markerfacecolor="white" if t==8000 else color)
        axs[0,col].axhline(0,color="black",ls=":",lw=1)
        axs[0,col].set_ylim(-.085,.225)
        axs[0,col].yaxis.set_major_formatter(PercentFormatter(1))
        decorate(axs[0,col])
        axs[0,col].set_title(LABELS[col])
        axs[1,col].plot([r["k"] for r in top],[r["unigram_accuracy"] for r in top],color="black",ls=":",lw=1.5)
        axs[1,col].set_xscale("log",base=2)
        axs[1,col].set_xticks([1,2,4,8,16,32,64])
        axs[1,col].xaxis.set_major_formatter(ScalarFormatter())
        axs[1,col].set_xlabel("k (n=2,048 chunks)")
        axs[1,col].set_ylim(0,.86)
        axs[1,col].yaxis.set_major_formatter(PercentFormatter(1))
        axs[1,col].grid(axis="y",color="#dddddd",lw=.5)
    axs[0,0].set_ylabel("Relative MSE improvement\n(higher is better)")
    axs[1,0].set_ylabel("Top-k accuracy\n(higher is better)")
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.52,1),ncol=3,frameon=False)
    fig.text(.52,.9,"Top row: dotted line = zero improvement over unigram",ha="center",fontsize=10)
    fig.text(.52,.44,"Bottom row: dotted line = unigram top-k accuracy",ha="center",fontsize=10)
    save(fig,out,"mdlm_quality")

    fig,axs=plt.subplots(2,3,figsize=(11.6,6.1),sharex=True,sharey=True)
    fig.subplots_adjust(left=.1,right=.985,bottom=.12,top=.83,hspace=.25,wspace=.13)
    handles=[]
    rankstyles=[("multinomial","Multinomial forecast (lag)",COLORS["multinomial"],"s","-"),
                ("source_same_lag","Source, same lag",COLORS["source_same_lag"],"v","--"),
                ("source_full","Source, full covariance",COLORS["source_full"],"^","-"),
                ("lag_observed","Observed reconstructor (same pair)",COLORS["lag"],"D",":")]
    for col,c in enumerate(CORPORA):
        for row,t in enumerate((8000,12000)):
            for m,label,color,marker,ls in rankstyles:
                vals=selected(data["rank_summary"],corpus=c,updates=t,comparator=m)
                y=np.array([r["mean_spearman"] for r in vals])
                line,=axs[row,col].plot(SIZES,y,color=color,marker=marker,ls=ls,lw=1.6,ms=4.5,label=label)
                if col==0 and row==0: handles.append(line)
                # Pair ranges are available for all comparators and remain visible.
                lo=np.array([r["minimum_spearman"] for r in vals]);hi=np.array([r["maximum_spearman"] for r in vals])
                axs[row,col].errorbar(SIZES,y,yerr=[y-lo,hi-y],fmt="none",ecolor=color,elinewidth=.65,capsize=1.5,alpha=.65)
            decorate(axs[row,col],xlabel=row==1)
            axs[row,col].set_ylim(0,.68)
        axs[0,col].set_title(LABELS[col])
    axs[0,0].set_ylabel("Mean Spearman correlation\n(MDLM, 8k updates)")
    axs[1,0].set_ylabel("Mean Spearman correlation\n(MDLM, 12k updates)")
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.53,1),ncol=2,frameon=False)
    save(fig,out,"mdlm_input_ranking")
    return ["mdlm_disagreement","mdlm_quality","mdlm_input_ranking"]
