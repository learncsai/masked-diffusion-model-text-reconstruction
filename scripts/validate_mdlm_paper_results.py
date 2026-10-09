"""Independently check the neural comparison from saved records."""
from __future__ import annotations
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/mdlm_paper_comparison_v1"


def read_csv(path):
    with path.open(newline="") as stream: return list(csv.DictReader(stream))


def match(rows,**keys):
    return [r for r in rows if all(str(r[k])==str(v) for k,v in keys.items())]


def mean(values):
    values=list(values)
    return math.fsum(values)/len(values)


def ranks(values):
    """Average ranks, independently of scipy.stats.spearmanr."""
    order=sorted(range(len(values)),key=lambda i:values[i])
    result=np.empty(len(values)); i=0
    while i<len(order):
        j=i+1
        while j<len(order) and values[order[j]]==values[order[i]]: j+=1
        for position in range(i,j): result[order[position]]=(i+j-1)/2+1
        i=j
    return result


def correlation(left,right):
    x,y=ranks(left),ranks(right)
    x=x-mean(x);y=y-mean(y)
    return math.fsum(x*y)/math.sqrt(math.fsum(x*x)*math.fsum(y*y))


def main():
    provenance=json.loads((OUT/"provenance.json").read_text())
    for category in ("sources","code"):
        for path,digest in provenance[category].items():
            assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    for path,digest in provenance["outputs_sha256"].items():
        assert hashlib.sha256((OUT/path).read_bytes()).hexdigest()==digest,path
    chunks=read_csv(OUT/"per_chunk.csv")
    pair_rows=read_csv(OUT/"per_pair.csv")
    ranked=read_csv(OUT/"rank_per_pair.csv")
    summary=read_csv(OUT/"summary.csv")
    assert len(chunks)==6912 and len(pair_rows)==99 and len(ranked)==216 and len(summary)==45
    for row in pair_rows:
        c,n,t,p,m=(row[k] for k in ("corpus","n","updates","pair","method"))
        root=ROOT/"results/mdlm_runpod_main_v1"/c
        if m=="lag":
            scores=json.loads((root/"lag_pairs.json").read_text())
            score=next(s for s in scores if s["n"]==int(n) and s["pair"]==int(p))
        else:
            name=f"seed_control_n{n}_step{t}.json" if m=="seed_control" else f"pair{p}_n{n}_step{t}.json"
            score=json.loads((root/"scores"/name).read_text())
        d=mean(score["per_chunk"]["disagreement"])
        loss=mean(score["per_chunk"]["loss_a"]+score["per_chunk"]["loss_b"])
        base=mean(score["per_chunk"]["baseline"])
        np.testing.assert_allclose([float(row[k]) for k in ("disagreement","mse","baseline_mse","relative_mse_improvement")],
                                   [d,loss,base,1-loss/base],rtol=1e-12,atol=1e-12)
        if m=="mdlm":
            points=sorted(match(chunks,corpus=c,n=n,updates=t,pair=p),key=lambda r:int(r["chunk"]))
            assert len(points)==128
            np.testing.assert_array_equal([float(x["mdlm_disagreement"]) for x in points],score["per_chunk"]["disagreement"])
    maximum_rank_error=0.
    for row in ranked:
        keys={k:row[k] for k in ("corpus","n","updates","pair")}
        points=sorted(match(chunks,**keys),key=lambda r:int(r["chunk"]))
        col="lag_disagreement" if row["comparator"]=="lag_observed" else row["comparator"]
        x=[float(p[col]) for p in points];y=[float(p["mdlm_disagreement"]) for p in points]
        error=abs(correlation(x,y)-float(row["spearman"]))
        maximum_rank_error=max(maximum_rank_error,error)
        assert error<1e-12
    for row in summary:
        cell=match(pair_rows,**{k:row[k] for k in ("corpus","n","updates","method")})
        assert len(cell)==int(row["pairs"])
        for key in ("disagreement","mse","baseline_mse","relative_mse_improvement"):
            np.testing.assert_allclose(float(row[key]),mean(float(x[key]) for x in cell),rtol=1e-12,atol=1e-12)
    for row in read_csv(OUT/"rank_summary.csv"):
        cell=match(ranked,**{k:row[k] for k in ("corpus","n","updates","comparator")})
        assert len(cell)==3
        np.testing.assert_allclose(float(row["mean_spearman"]),mean(float(x["spearman"]) for x in cell),rtol=1e-12)
    for row in read_csv(OUT/"forecasts.csv"):
        with np.load(ROOT/"results/lag_disagreement_matched_v1/per_chunk"/f"{row['corpus']}_n{row['n']}_forecasts.npz") as z:
            vector=z["forecasts"][z["methods"].tolist().index(row["method"])]
            np.testing.assert_allclose(float(row["disagreement"]),mean(vector),rtol=1e-12)
    neural=read_csv(ROOT/"results/mdlm_topk_matched_v1/per_model.csv")
    lag=read_csv(ROOT/"results/lag_topk_matched_v1/per_model.csv")
    for row in read_csv(OUT/"topk.csv"):
        keys={k:row[k] for k in ("corpus","n","k")}
        if row["method"]=="mdlm": keys["updates"]=row["updates"]
        cells=match(neural if row["method"]=="mdlm" else lag,**keys)
        col="accuracy" if row["method"]=="mdlm" else "reconstructor_accuracy"
        assert len(cells)==6
        accuracy=mean(float(x[col]) for x in cells);baseline=mean(float(x["unigram_accuracy"]) for x in cells)
        np.testing.assert_allclose([float(row[k]) for k in ("accuracy","unigram_accuracy","relative_topk_error_reduction")],
                                   [accuracy,baseline,1-(1-accuracy)/(1-baseline)],atol=1e-12,rtol=1e-12)
    paper=ROOT/"paper/aistatsready.tex"
    if paper.exists():
        manuscript=paper.read_text(encoding="utf-8")
        for name in ("disagreement","quality","ranking"):
            for line in (OUT/f"{name}_rows.tex").read_text().splitlines(): assert line in manuscript,line
        for row in read_csv(OUT/"scaling.csv"):
            if row["method"] in ("lag","mdlm"):
                assert f"{float(row['decrease_percent']):.1f}\\%" in manuscript
    result=dict(status="passed",per_chunk_rows=6912,pair_rows=99,within_pair_correlations=216,
        summary_rows=45,ranking_summaries=72,topk_rows=189,maximum_independent_rank_error=maximum_rank_error,
        limitations="Recomputes saved scalar summaries and rank correlations. Does not rerun training or full-vocabulary inference. Prior top-k validation checks the underlying target ranks.")
    (OUT/"independent_validation.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))


if __name__=="__main__": main()
