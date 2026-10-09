"""Rebuild the main-text MDLM comparison from saved scores, without training."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
CORPORA = ("tinystories", "wikitext", "cnn_dailymail")
LABELS = dict(tinystories="TinyStories", wikitext="WikiText-103", cnn_dailymail="CNN/DailyMail")
SIZES = (512, 1024, 2048)
STAGES = (8000, 12000)
KS = (1, 2, 4, 8, 16, 32, 64)
SOURCES = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tracked(path):
    SOURCES[Path(path).relative_to(ROOT).as_posix()] = sha(path)
    return Path(path)


def read_json(path):
    return json.loads(tracked(path).read_text(encoding="utf-8"))


def read_csv(path):
    with tracked(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def matched(rows, **keys):
    return [r for r in rows if all(str(r[k]) == str(v) for k, v in keys.items())]


def one(rows, **keys):
    selected = matched(rows, **keys)
    if len(selected) != 1:
        raise ValueError(f"Expected one row, got {len(selected)}: {keys}")
    return selected[0]


def pair_metrics(score):
    pc = score["per_chunk"]
    for name in ("loss_a", "loss_b", "disagreement"):
        if len(pc[name]) != 128 or not np.isfinite(pc[name]).all():
            raise ValueError("Invalid per-chunk score")
        np.testing.assert_allclose(np.mean(pc[name]), score[name], rtol=1e-12, atol=1e-12)
    baseline = float(score["baseline_loss"])
    np.testing.assert_allclose(np.mean(pc["baseline"]), baseline, rtol=1e-12)
    mse = .5*(score["loss_a"]+score["loss_b"])
    for side in ("a", "b"):
        np.testing.assert_allclose(score[f"relative_improvement_{side}"], 1-score[f"loss_{side}"]/baseline, rtol=1e-12, atol=1e-12)
    return dict(disagreement=float(score["disagreement"]), mse=mse,
                relative_mse_improvement=1-mse/baseline, baseline_mse=baseline)


def collect():
    run = ROOT / "results/mdlm_runpod_main_v1"
    lag = ROOT / "results/lag_disagreement_matched_v1"
    lag_meta = read_json(lag / "metadata.json")
    lag_summary = read_csv(lag / "summary.csv")
    original_summary = read_csv(run / "summary.csv")
    topk = read_csv(ROOT / "results/mdlm_topk_matched_v1/per_model.csv")
    lag_topk = read_csv(ROOT / "results/lag_topk_matched_v1/per_model.csv")
    topk_meta = read_json(ROOT / "results/lag_topk_matched_v1/metadata.json")
    assert read_json(ROOT / "results/mdlm_topk_matched_v1/independent_validation.json")["status"] == "passed"
    pair_rows, ranking_rows, forecast_rows, protocol = [], [], [], []
    chunk_rows = []
    for corpus in CORPORA:
        folder = run / corpus
        manifest = read_json(folder / "manifest.json")
        config = manifest["protocol"]["config"]
        prepared = read_json(folder / "prepared.json")
        audit = read_json(folder / "source_audit.json")
        assert manifest["protocol_sha256"] == lag_meta["source_audits"][corpus]["original_protocol_sha256"]
        assert manifest["protocol_sha256"] == topk_meta["contexts"][corpus]["protocol_sha256"]
        context_sha = prepared["sha256"]["context.npz"]
        assert context_sha == topk_meta["contexts"][corpus]["context_sha256"]
        assert config["mask_rate"] == .5 and config["classes"] == 50257 and config["pairs"] == 3
        assert config["corpus_sizes"] == list(SIZES) and config["training"]["checkpoints"] == list(STAGES)
        protocol.append(dict(corpus=corpus, config=config, source_counts={k:dict(chunks=v["chunks"], sources=v["sources"]) for k,v in audit["roles"].items()},
                             context_sha256=context_sha, protocol_sha256=manifest["protocol_sha256"]))
        lag_pairs = read_json(folder / "lag_pairs.json")
        assert sha(folder / "lag_pairs.json") == prepared["sha256"]["lag_pairs.json"]
        for n in SIZES:
            files = {kind: lag / "per_chunk" / f"{corpus}_n{n}_{kind}.npz" for kind in ("forecasts", "observed")}
            for path in files.values():
                assert sha(tracked(path)) == lag_meta["outputs_sha256"][path.relative_to(lag).as_posix()]
            with np.load(files["forecasts"]) as z:
                forecasts = dict(zip(z["methods"].tolist(), z["forecasts"].copy()))
                source_ids = z["source_ids"].copy()
            with np.load(files["observed"]) as z:
                np.testing.assert_array_equal(source_ids, z["source_ids"])
                observed_lag = z["observed"].copy()
            record_path = lag / "inputs" / corpus / "evaluation_records.json"
            assert sha(record_path) == lag_meta["outputs_sha256"][record_path.relative_to(lag).as_posix()]
            eval_records = read_json(record_path)
            np.testing.assert_array_equal(source_ids, [r["document_id"] for r in eval_records])
            assert set(source_ids) == set(audit["roles"]["evaluation"]["source_ids"])
            for name, values in forecasts.items():
                saved = one(lag_summary, corpus=corpus, n=n, method=name)
                np.testing.assert_allclose(values.mean(), float(saved["predicted_mean"]), rtol=1e-12)
                forecast_rows.append(dict(corpus=corpus, n=n, method=name, disagreement=float(values.mean()), reference_chunks=2048))
            for pair in (1, 2, 3):
                lag_score = one(lag_pairs, n=n, pair=pair)
                np.testing.assert_allclose(lag_score["per_chunk"]["disagreement"], observed_lag[pair-1], rtol=1e-12, atol=1e-14)
                pair_rows.append(dict(corpus=corpus, n=n, pair=pair, method="lag", updates=0, **pair_metrics(lag_score)))
                for steps in STAGES:
                    score = read_json(folder / "scores" / f"pair{pair}_n{n}_step{steps}.json")
                    assert score["identity"]["context_sha256"] == context_sha
                    complete = [read_json(folder / "models" / f"pair{pair}_{side}_n{n}" / "complete.json") for side in ("a", "b")]
                    assert complete[0]["identity"]["parent_sha256"] == complete[1]["identity"]["parent_sha256"]
                    assert complete[0]["identity"]["seed_offset"] == complete[1]["identity"]["seed_offset"]
                    for side, record in zip(("a", "b"), complete):
                        assert score["identity"][f"checkpoint_{side}_sha256"] == record["sha256"][f"step_{steps}.pt"]
                    pair_rows.append(dict(corpus=corpus, n=n, pair=pair, method="mdlm", updates=steps, **pair_metrics(score)))
                    actual = np.asarray(score["per_chunk"]["disagreement"])
                    for name, values in {**forecasts, "lag_observed":observed_lag[pair-1]}.items():
                        rho = float(spearmanr(values, actual).statistic)
                        ranking_rows.append(dict(corpus=corpus, n=n, pair=pair, updates=steps, comparator=name, spearman=rho,
                            comparator_mean=float(values.mean()), mdlm_mean=float(actual.mean()), unscaled_mean_ratio=float(values.mean()/actual.mean())))
                    for e, value in enumerate(actual):
                        chunk_rows.append(dict(corpus=corpus, n=n, pair=pair, updates=steps, chunk=e, source_id=str(source_ids[e]),
                            mdlm_disagreement=float(value), lag_disagreement=float(observed_lag[pair-1,e]),
                            **{name:float(values[e]) for name,values in forecasts.items()}))
            for steps in STAGES:
                score = read_json(folder / "scores" / f"seed_control_n{n}_step{steps}.json")
                assert score["identity"]["context_sha256"] == context_sha
                original = read_json(folder / "models" / f"pair1_a_n{n}" / "complete.json")
                control = read_json(folder / "models" / f"seed_control_pair1_a_n{n}" / "complete.json")
                for key in ("data_sha256", "parent_sha256", "protocol_sha256"):
                    assert original["identity"][key] == control["identity"][key]
                assert original["identity"]["seed_offset"] != control["identity"]["seed_offset"]
                assert score["identity"]["checkpoint_a_sha256"] == original["sha256"][f"step_{steps}.pt"]
                assert score["identity"]["checkpoint_b_sha256"] == control["sha256"][f"step_{steps}.pt"]
                pair_rows.append(dict(corpus=corpus, n=n, pair=1, method="seed_control", updates=steps, **pair_metrics(score)))
    summary = []
    names = dict(lag="lag_radius_2", mdlm="mdlm_full_context", seed_control="mdlm_same_corpus_seed_control")
    for c,n,m,t in sorted({(r["corpus"],r["n"],r["method"],r["updates"]) for r in pair_rows}):
        cell = matched(pair_rows,corpus=c,n=n,method=m,updates=t)
        assert len(cell) == (1 if m == "seed_control" else 3)
        means = {key:float(np.mean([r[key] for r in cell])) for key in ("disagreement", "mse", "relative_mse_improvement", "baseline_mse")}
        prior = one(original_summary,corpus=c,n=n,method=names[m],updates=t)
        for col, old in (("disagreement","mean_disagreement"),("mse","mean_mse"),("relative_mse_improvement","mean_relative_improvement")):
            np.testing.assert_allclose(means[col],float(prior[old]),rtol=1e-12,atol=1e-12)
        summary.append(dict(corpus=c,n=n,method=m,updates=t,pairs=len(cell),**means,
                            minimum_disagreement=min(r["disagreement"] for r in cell),maximum_disagreement=max(r["disagreement"] for r in cell)))
    ranks=[]
    for c,n,t,m in sorted({(r["corpus"],r["n"],r["updates"],r["comparator"]) for r in ranking_rows}):
        cell=matched(ranking_rows,corpus=c,n=n,updates=t,comparator=m)
        ranks.append(dict(corpus=c,n=n,updates=t,comparator=m,pairs=3,mean_spearman=float(np.mean([r["spearman"] for r in cell])),
            minimum_spearman=min(r["spearman"] for r in cell),maximum_spearman=max(r["spearman"] for r in cell)))
    accuracy=[]
    for c in CORPORA:
        for n in SIZES:
            for m,t,data,col in (("lag",0,lag_topk,"reconstructor_accuracy"),("mdlm",8000,topk,"accuracy"),("mdlm",12000,topk,"accuracy")):
                for k in KS:
                    keys=dict(corpus=c,n=n,k=k)
                    if t: keys["updates"]=t
                    cell=matched(data,**keys)
                    assert {(int(r["pair"]),r["arm"]) for r in cell} == {(p,a) for p in (1,2,3) for a in ("a","b")} and len(cell)==6
                    a=float(np.mean([float(r[col]) for r in cell]))
                    base=float(np.mean([float(r["unigram_accuracy"]) for r in cell]))
                    if t:
                        np.testing.assert_allclose(np.mean([float(r["mse"]) for r in cell]),one(summary,corpus=c,n=n,method=m,updates=t)["mse"],rtol=1e-5,atol=1e-6)
                    accuracy.append(dict(corpus=c,n=n,method=m,updates=t,k=k,accuracy=a,unigram_accuracy=base,relative_topk_error_reduction=1-(1-a)/(1-base)))
    scaling=[]
    for c in CORPORA:
        for m,t in (("lag",0),("mdlm",8000),("mdlm",12000),("multinomial",0),("source_full",0),("source_same_lag",0)):
            data=forecast_rows if m in ("multinomial","source_full","source_same_lag") else summary
            keys=dict(corpus=c,method=m)
            if data is summary: keys["updates"]=t
            lo,hi=(one(data,n=n,**keys)["disagreement"] for n in (512,2048))
            scaling.append(dict(corpus=c,method=m,updates=t,ratio_2048_to_512=hi/lo,decrease_percent=100*(1-hi/lo)))
    return dict(summary=summary,per_pair=pair_rows,forecasts=forecast_rows,rank_summary=ranks,rank_per_pair=ranking_rows,
                per_chunk=chunk_rows,topk=accuracy,scaling=scaling), protocol


def table_rows(data, out):
    d, f, a = data["summary"],data["forecasts"],data["topk"]
    lines=[]
    for c in CORPORA:
        for n in SIZES:
            vals=[one(d,corpus=c,n=n,method="lag",updates=0)["disagreement"]]
            vals += [one(f,corpus=c,n=n,method=m)["disagreement"] for m in ("source_full","multinomial")]
            vals += [one(d,corpus=c,n=n,method=m,updates=t)["disagreement"] for m in ("mdlm","seed_control") for t in STAGES]
            lines.append(f"{LABELS[c]} & {n:,} & "+" & ".join(f"{v:.4f}" for v in vals)+r" \\")
    (out/"disagreement_rows.tex").write_text("\n".join(lines)+"\n")
    lines=[]
    for c in CORPORA:
        for m,t,label in (("lag",0,"Reconstructor"),("mdlm",8000,"MDLM, 8k"),("mdlm",12000,"MDLM, 12k")):
            row=one(d,corpus=c,n=2048,method=m,updates=t)
            acc=[one(a,corpus=c,n=2048,method=m,updates=t,k=k)["accuracy"] for k in (1,64)]
            lines.append(f"{LABELS[c]} & {label} & {row['mse']:.4f} & {100*row['relative_mse_improvement']:.2f} & {100*acc[0]:.2f} & {100*acc[1]:.2f}"+r" \\")
    (out/"quality_rows.tex").write_text("\n".join(lines)+"\n")
    lines=[]
    for c in CORPORA:
        for t in STAGES:
            vals=[one(data["rank_summary"],corpus=c,n=2048,updates=t,comparator=m)["mean_spearman"] for m in ("multinomial","source_same_lag","source_full","lag_observed")]
            lines.append(f"{LABELS[c]} & {t:,} & "+" & ".join(f"{v:.3f}" for v in vals)+r" \\")
    (out/"ranking_rows.tex").write_text("\n".join(lines)+"\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"results/mdlm_paper_comparison_v1")
    parser.add_argument("--copy-figures",action="store_true")
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    data,protocol=collect()
    for name,rows in data.items(): write_csv(args.output/f"{name}.csv",rows)
    table_rows(data,args.output)
    from plot_mdlm_paper_results import plot_all
    names=plot_all(data,args.output)
    provenance=dict(sources=dict(sorted(SOURCES.items())),code={f"scripts/{name}":sha(ROOT/"scripts"/name) for name in ("build_mdlm_paper_results.py","plot_mdlm_paper_results.py")},
        protocol=protocol,scope="Exploratory matched denoising comparison. Source and multinomial predictions target the lag reconstructor. Neural comparisons are observations and descriptive transfer checks.",
        aggregation="Equal masked-token means within chunks, equal chunk means within pairs, equal pair means. Rank correlations are computed separately within each pair.",
        uncertainty="Error bars span the three observed pair means. They are not confidence intervals. Shared inputs and training randomness are held fixed.",
        checks=["Frozen context and protocol hashes agree across methods","A/B initialization and seed identities match","Original per-chunk means and derived MSE improvement reconcile","All population-forecast input arrays match saved hashes","All MDLM and lag summaries reconcile with original RunPod export","Top-k means reconcile with six model/side rows","Three pairs per corpus-size-stage, one same-corpus seed control"],
        outputs_sha256={p.name:sha(p) for p in args.output.iterdir() if p.is_file() and p.suffix in (".csv",".tex",".pdf",".png")})
    write_json(args.output/"provenance.json",provenance)
    if args.copy_figures:
        for name in names: shutil.copyfile(args.output/f"{name}.pdf",ROOT/"paper/figures"/f"fig_{name}.pdf")
        write_json(ROOT/"paper/figures/mdlm_comparison_figure_sources.json",dict(sources=provenance["sources"]))
    print(json.dumps(dict(status="complete",rows={k:len(v) for k,v in data.items()},figures=names)))


if __name__ == "__main__": main()
