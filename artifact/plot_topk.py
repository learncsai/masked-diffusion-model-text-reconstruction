"""Publication styling for the saved reconstructor top-k measurements."""
CORPORA = ("tinystories", "wikitext", "cnn_dailymail")
LABELS = dict(zip(CORPORA, ("TinyStories", "WikiText-103", "CNN/DailyMail")))
KS = [1, 2, 4, 8, 16, 32, 64]

def plot_summary(out, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter, ScalarFormatter
    plt.rcParams.update({"font.size": 12.5, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 3, figsize=(10.0, 6.3), layout="constrained", sharex=True, sharey="row")
    styles = {512: ("#245A81", "o"), 1024: ("#B85D24", "s"), 2048: ("#62733B", "^")}
    handles = {}
    for col, corpus in enumerate(CORPORA):
        groups = [r for r in summary if r["corpus"] == corpus]
        for n, (color, marker) in styles.items():
            group = sorted((r for r in groups if r["n"] == n), key=lambda r: r["k"])
            line, = axes[0, col].plot(KS, [r["reconstructor_accuracy"] for r in group], color=color,
                marker=marker, label=f"Reconstructor: {n:,} chunks")
            handles[line.get_label()] = line
            axes[1, col].plot(KS, [r["relative_topk_error_reduction"] for r in group], color=color, marker=marker)
        base = sorted((r for r in groups if r["n"] == 512), key=lambda r: r["k"])
        line, = axes[0, col].plot(KS, [r["unigram_accuracy"] for r in base], color="#555555", linestyle="--", label="Unigram")
        handles[line.get_label()] = line
        axes[0, col].set_title("WikiText-103" if corpus == "wikitext" else LABELS[corpus])
        axes[1, col].axhline(0, color="#888888", linewidth=.7)
        for ax in axes[:, col]:
            ax.set_xscale("log", base=2)
            ax.set_xticks(KS)
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.grid(axis="y", color="#dddddd", linewidth=.6)
        axes[1, col].set_xlabel("Top k")
    axes[0, 0].set_ylim(bottom=0)
    axes[0, 0].set_ylabel("Top-k accuracy\n(higher is better)")
    axes[1, 0].set_ylabel("Relative top-k error reduction\n(higher is better)")
    fig.legend(handles.values(), handles.keys(), loc="outside lower center", ncol=2,
               frameon=False, fontsize=11.5)
    fig.savefig(out / "topk_comparison.png", dpi=170)
    fig.savefig(out / "topk_comparison.pdf")
    plt.close(fig)
