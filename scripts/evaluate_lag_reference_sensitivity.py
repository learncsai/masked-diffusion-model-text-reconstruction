"""Vary only the reference bank in the matched disagreement comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import evaluate_lag_disagreement as base
from scripts.evaluate_lag_disagreement import (CORPORA,CORE,LABELS,METHODS,lag,offset_list,
    pattern_weights,per_sequence_squared_error,read_json,sha,source_groups,
    source_influence_disagreement,utc,validate_records,write_csv,write_json)
from scripts.evaluate_lag_disagreement_extension import load_context

SIZES=(4096,8192)
REFERENCES=(2048,4096,8192)
DIAGNOSTIC="multinomial_conditional_at_n"
ALL_METHODS=(*METHODS,DIAGNOSTIC)
DEFAULT=ROOT / "results/lag_reference_sensitivity_v1"


def target_conditional_multinomial(reference,ids,mask,prior,offsets,gram,cross,*,n,smoothing,ridge):
    counts=lag.pair_counts(reference,offsets,len(prior))
    trace={d:base.multinomial_row_trace(c*(n/len(reference)),prior,smoothing,1.) for d,c in counts.items()}
    rows,_,contexts,visible,patterns=lag.context_design(ids,mask,offsets)
    point=np.zeros(len(rows))
    for code in np.unique(patterns):
        if not code:
            continue
        selected=np.flatnonzero(patterns==code)
        active=tuple(j for j in range(len(offsets)) if code&(1<<j))
        weights=pattern_weights(gram,cross,active,ridge)
        for weight,j in zip(weights,active):
            point[selected]+=weight**2*trace[offsets[j]][contexts[j][selected]]
    return per_sequence_squared_error(rows,point,len(ids))


def summarize(corpus,n,m,forecasts,observed):
    pairs,summary=base.summarize(corpus,n,forecasts[:3],observed)
    extra_pairs,extra_summary=base.summarize(corpus,n,np.repeat(forecasts[3:4],3,axis=0),observed)
    # The base helper is intentionally left unchanged. Its first method's
    # metrics are also valid for the separately labelled diagnostic vector.
    for row in extra_pairs[:3]:
        pairs.append(dict(row,method=DIAGNOSTIC))
    summary.append(dict(extra_summary[0],method=DIAGNOSTIC))
    return [dict(r,reference_chunks=m) for r in pairs],[dict(r,reference_chunks=m) for r in summary]


def validate_inputs(inputs,corpus=None):
    manifest=read_json(inputs / "manifest.json")
    for rel,digest in manifest["inputs_sha256"].items():
        if corpus is None or rel.startswith(corpus+"/"):
            assert sha(inputs / rel)==digest,rel
    for rel,digest in manifest["core_sha256"].items():
        assert sha(ROOT / rel)==digest,rel
    return manifest


def initialize(inputs,out):
    validate_inputs(inputs)
    out.mkdir(parents=True,exist_ok=True)
    code=[*CORE,"scripts/evaluate_lag_disagreement.py","scripts/evaluate_lag_disagreement_extension.py",
          "scripts/evaluate_lag_reference_sensitivity.py"]
    contract=dict(input_manifest_sha256=sha(inputs / "manifest.json"),code_sha256={p:sha(ROOT / p) for p in code},
        corpora=list(CORPORA),corpus_sizes=list(SIZES),reference_sizes=list(REFERENCES),methods=list(ALL_METHODS),
        pairs=3,evaluation_chunks=128,mask_rate=.5,
        baseline="Reuse verified 2048-reference source forecasts from the prior experiment. Recompute its multinomial forecast and all A/B observations to check identity.",
        diagnostic="The additional multinomial conditional-at-n variant changes only the conditional collision estimate. The expected row counts and variance prefactor stay fixed.",
        status="Exploratory sensitivity on already observed fixed A/B corpora. Nested reference banks give a conditional comparison, not independent reference replicates.",
        limits="Changing reference size also changes its source composition, coverage, covariance estimate and effective smoothing. Three A/B pairs. WikiText sources are paragraphs. No confidence intervals or significance tests.")
    signature=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    path=out / "analysis_plan.json"
    if path.exists():
        assert read_json(path)["signature"]==signature,"Use a new output folder if code or inputs change"
    else:
        write_json(path,dict(saved_utc=utc(),signature=signature,**contract))
    return signature


def run_corpus(inputs,out,corpus):
    started=time.perf_counter()
    validate_inputs(inputs,corpus)
    plan=read_json(out / "analysis_plan.json")
    assert sha(inputs / "manifest.json")==plan["input_manifest_sha256"]
    for rel,digest in plan["code_sha256"].items():
        assert sha(ROOT / rel)==digest,rel
    signature=plan["signature"]
    folder=inputs / corpus
    dest=out / corpus
    dest.mkdir(parents=True,exist_ok=True)
    (ids,mask,prior,gram,cross),original,original_rec,eval_rec,config=load_context(folder)
    offsets=offset_list(config["radius"])
    kw=dict(smoothing=config["smoothing"],ridge=config["ridge"])
    source_ids=np.array([r["document_id"] for r in eval_rec])
    with np.load(folder / "reference_max.npz") as z:
        reference=z["ids"]
    records=read_json(folder / "reference_max.json")
    validate_records(reference,records)
    assert reference.shape==(8192,64)
    np.testing.assert_array_equal(reference[:2048],original)
    assert records[:2048]==original_rec
    audit=read_json(folder / "source_audit.json")
    lookup=read_json(folder / "source_text_hashes.json")
    ref_ids={r["document_id"] for r in records}
    ref_texts={lookup[s] for s in ref_ids}
    ref_chunks={r["sha256"] for r in records}
    assert len(ref_chunks)==8192
    role_ids=[ref_ids]
    role_texts=[ref_texts]
    for name,role in audit["roles"].items():
        if name=="reference":
            continue
        current=set(role["source_ids"])
        texts={lookup[s] for s in current}
        assert all(not current&old for old in role_ids),name
        assert all(not texts&old for old in role_texts),name
        role_ids.append(current)
        role_texts.append(texts)
    assert not {r["sha256"] for r in eval_rec}&ref_chunks
    assert set(source_ids)==set(audit["roles"]["evaluation"]["source_ids"])
    checks=[]
    for m in REFERENCES:
        ref,rec=reference[:m],records[:m]
        groups=source_groups(rec)
        for n in SIZES:
            path=dest / f"ref{m}_n{n}_forecasts.npz"
            if path.exists():
                with np.load(path) as z:
                    assert str(z["signature"])==signature
                continue
            write_json(dest / "progress.json",dict(status="forecasting",reference_chunks=m,n=n,started_utc=utc()))
            print(f"{corpus}: reference={m}, A/B={n}, multinomial",flush=True)
            multi=base.multinomial_forecast(ref,ids,mask,prior,offsets,gram,cross,n=n,**kw)
            diagnostic=target_conditional_multinomial(ref,ids,mask,prior,offsets,gram,cross,n=n,**kw)
            if m==2048:
                with np.load(folder / f"baseline_n{n}_forecasts.npz") as z:
                    baseline=z["forecasts"]
                    np.testing.assert_array_equal(z["source_ids"],source_ids)
                np.testing.assert_allclose(multi,baseline[0],rtol=1e-12,atol=1e-14)
                same,full=baseline[1:]
            else:
                print(f"{corpus}: reference={m}, A/B={n}, source same lag ({len(groups)} sources)",flush=True)
                same=source_influence_disagreement(ref,groups,ids,mask,prior,offsets,gram,cross,n=n,same_lag=True,**kw)
                print(f"{corpus}: reference={m}, A/B={n}, source all lags",flush=True)
                full=source_influence_disagreement(ref,groups,ids,mask,prior,offsets,gram,cross,n=n,same_lag=False,**kw)
            if m==n:
                np.testing.assert_allclose(multi,diagnostic,rtol=0,atol=1e-14)
            forecasts=np.array([multi,same,full,diagnostic])
            assert forecasts.shape==(4,128) and np.isfinite(forecasts).all() and (forecasts>=0).all()
            np.savez_compressed(path,forecasts=forecasts,methods=np.array(ALL_METHODS),source_ids=source_ids,
                signature=np.array(signature),saved_utc=np.array(utc()),baseline_source_reused=np.array(m==2048))
    # Rebuild each A/B observation once. The exact same vectors are compared
    # with every reference bank, so reference size cannot change the target.
    token_pairs={}
    chunk_sets=[ref_chunks,{r["sha256"] for r in eval_rec}]
    prep=read_json(folder / "reference_prepared.json")
    for rel,digest in prep["fixed_files_sha256"].items():
        assert sha(folder / rel)==digest,rel
    for pair in (1,2,3):
        for arm in ("a","b"):
            name=f"pair{pair}_{arm}"
            with np.load(folder / f"{name}.npz") as z:
                tokens=z["ids"]
            rec=read_json(folder / f"{name}.json")
            validate_records(tokens,rec)
            assert tokens.shape==(8192,64)
            assert {r["document_id"] for r in rec}==set(audit["roles"][name]["source_ids"])
            chunks={r["sha256"] for r in rec}
            assert len(chunks)==8192 and all(not chunks&old for old in chunk_sets)
            chunk_sets.append(chunks)
            token_pairs[pair,arm]=tokens
    all_summary,all_pairs,all_chunks=[],[],[]
    for n in SIZES:
        observed=[]
        with np.load(folder / f"baseline_n{n}_observed.npz") as z:
            baseline=z["observed"]
            np.testing.assert_array_equal(z["source_ids"],source_ids)
        for pair in (1,2,3):
            outputs=[]
            for arm in ("a","b"):
                counts=lag.pair_counts(token_pairs[pair,arm][:n],offsets,len(prior))
                tables=lag.conditional_tables(counts,prior,config["smoothing"])
                rows,_,prediction=lag.masked_prediction(ids,mask,tables,prior,offsets,gram,cross,config["ridge"])
                outputs.append(prediction)
            distances=lag.squared_difference(*outputs)
            actual=per_sequence_squared_error(rows,distances,len(ids))
            np.testing.assert_allclose(actual,baseline[pair-1],rtol=1e-12,atol=1e-14)
            dense_errors=[]
            for j in (0,len(rows)//2,len(rows)-1):
                vectors=[o.residual[j].toarray().ravel()+o.prior_weight[j]*prior for o in outputs]
                direct=np.sum((vectors[0]-vectors[1])**2)
                np.testing.assert_allclose(distances[j],direct,rtol=1e-11,atol=1e-12)
                dense_errors.append(abs(float(distances[j]-direct)))
            observed.append(actual)
            checks.append(dict(n=n,pair=pair,max_observation_difference=float(np.max(np.abs(actual-baseline[pair-1]))),
                max_dense_distance_difference=max(dense_errors)))
        observed=np.array(observed)
        np.savez_compressed(dest / f"n{n}_observed.npz",observed=observed,source_ids=source_ids,signature=np.array(signature))
        for m in REFERENCES:
            with np.load(dest / f"ref{m}_n{n}_forecasts.npz") as z:
                forecasts=z["forecasts"]
                assert str(z["signature"])==signature
            pairs,summary=summarize(corpus,n,m,forecasts,observed)
            all_pairs.extend(pairs)
            all_summary.extend(summary)
            for pair,actual in enumerate(observed,1):
                for chunk,value in enumerate(actual):
                    all_chunks.append(dict(corpus=corpus,n=n,reference_chunks=m,pair=pair,chunk=chunk,
                        source_id=str(source_ids[chunk]),observed=float(value),
                        **{method:float(forecasts[j,chunk]) for j,method in enumerate(ALL_METHODS)}))
            print(f"{corpus}: reference={m}, A/B={n}: "+", ".join(f"{r['method']}={r['prediction_observation_ratio']:.4f}" for r in summary),flush=True)
    write_json(dest / "summary.json",all_summary)
    write_csv(dest / "per_pair.csv",all_pairs)
    write_csv(dest / "per_chunk.csv",all_chunks)
    write_json(dest / "complete.json",dict(corpus=corpus,completed_utc=utc(),signature=signature,
        runtime_seconds=time.perf_counter()-started,checks=checks,source_text_chunk_overlap=0,
        reference_sources=prep["nested_sources"],outputs_sha256={p.name:sha(p) for p in dest.iterdir()
            if p.is_file() and p.name not in ("complete.json","progress.json")}))
    write_json(dest / "progress.json",dict(status="complete",completed_utc=utc()))


def collect(out):
    import csv
    summary,pairs,chunks=[],[],[]
    plan=read_json(out / "analysis_plan.json")
    completed={}
    for corpus in CORPORA:
        folder=out / corpus
        meta=read_json(folder / "complete.json")
        assert meta["signature"]==plan["signature"]
        for rel,digest in meta["outputs_sha256"].items():
            assert sha(folder / rel)==digest,rel
        completed[corpus]=meta
        summary.extend(read_json(folder / "summary.json"))
        for name,values in (("per_pair.csv",pairs),("per_chunk.csv",chunks)):
            with (folder / name).open(newline="") as f:
                values.extend(csv.DictReader(f))
    write_json(out / "summary.json",summary)
    write_csv(out / "summary.csv",summary)
    write_csv(out / "per_pair.csv",pairs)
    write_csv(out / "per_chunk.csv",chunks)
    table=[]
    for corpus in CORPORA:
        for m in REFERENCES:
            vals=[next(r["prediction_observation_ratio"] for r in summary if
                (r["corpus"],r["n"],r["reference_chunks"],r["method"])==(corpus,n,m,method)) for n in SIZES for method in METHODS]
            table.append(f"{LABELS[corpus]} & {m:,} & "+" & ".join(f"{v:.3f}" for v in vals)+r" \\")
    (out / "table_rows.tex").write_text("\n".join(table)+"\n",encoding="utf-8")
    plot(out,summary)
    write_json(out / "metadata.json",dict(completed_utc=utc(),signature=plan["signature"],corpora=completed,
        python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,summary_rows=len(summary),
        outputs_sha256={p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob("*"))
            if p.is_file() and p.name not in ("metadata.json","README.md","validation.json")}))
    print(f"Collected {len(summary)} summaries in {out}",flush=True)


def plot(out,summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    styles=dict(multinomial=("#B85D24","s","Multinomial"),source_same_lag=("#62733B","^","Source, same lag"),
        source_full=("#245A81","o","Source, all lags"))
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(2,3,figsize=(11,6.5),layout="constrained",sharex=True,sharey=True)
    handles=[]
    for col,corpus in enumerate(CORPORA):
        axes[0,col].set_title(LABELS[corpus])
        for row,n in enumerate(SIZES):
            ax=axes[row,col]
            for method,(color,marker,label) in styles.items():
                vals=[next(r["prediction_observation_ratio"] for r in summary if
                    (r["corpus"],r["n"],r["reference_chunks"],r["method"])==(corpus,n,m,method)) for m in REFERENCES]
                line,=ax.plot(REFERENCES,vals,color=color,marker=marker,label=label)
                if col==row==0:
                    handles.append(line)
            ax.axhline(1,color="#555555",linestyle="--",linewidth=1)
            ax.set_xscale("log",base=2)
            ax.set_xticks(REFERENCES)
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.grid(axis="y",color="#dddddd",linewidth=.6)
            ax.set_ylim(bottom=0)
        axes[1,col].set_xlabel("Reference size (64-token chunks)")
    for row,n in enumerate(SIZES):
        axes[row,0].set_ylabel(f"A/B size: {n:,} chunks\nPredicted / observed mean (target: 1)")
    fig.legend(handles=handles,loc="outside lower center",ncol=3,frameon=False)
    fig.savefig(out / "reference_sensitivity.png",dpi=180)
    fig.savefig(out / "reference_sensitivity.pdf")
    plt.close(fig)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs",type=Path,default=DEFAULT / "inputs")
    parser.add_argument("--output",type=Path,default=DEFAULT)
    parser.add_argument("--corpus",choices=CORPORA)
    parser.add_argument("--initialize-only",action="store_true")
    parser.add_argument("--collect-only",action="store_true")
    args=parser.parse_args()
    if args.collect_only:
        collect(args.output)
    elif args.corpus:
        run_corpus(args.inputs,args.output,args.corpus)
    else:
        initialize(args.inputs,args.output)
        if not args.initialize_only:
            for corpus in CORPORA:
                run_corpus(args.inputs,args.output,corpus)
            collect(args.output)
