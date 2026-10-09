"""Check saved position-level measurements, then rebuild follow-up outputs.

Works in the compact review archive without text corpora or model weights.
This reruns aggregation and plotting, not source collection or prediction.
"""
from __future__ import annotations
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
from scipy.stats import t

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.evaluate_lag_disagreement import CORPORA,read_json,sha,write_json
from scripts.run_reference_ratio import OUT,OLD,expectation,BIN_NAMES


def check_allocation_records(corpus):
    """Recheck recorded source/text/chunk separation without corpus text."""
    old=OLD/'inputs'/corpus
    folder=OUT/'inputs'/corpus
    prepared=read_json(folder/'prepared.json')
    lookup=read_json(old/'source_text_hashes.json')
    excluded=read_json(old/'excluded_sources.json')
    ids=set(excluded['source_ids']);texts=set(excluded['text_hashes']);chunks=set(excluded['chunk_hashes'])
    for role in read_json(old/'source_audit.json')['roles'].values():ids.update(role['source_ids'])
    names=['reference_max','evaluation_records',*[f'pair{p}_{a}' for p in (1,2,3) for a in ('a','b')]]
    for name in names:
        records=read_json(old/(name+'.json'))
        ids.update(r['document_id'] for r in records)
        chunks.update(r['sha256'] for r in records)
    texts.update(lookup[s] for s in ids if s in lookup)
    new_names=['reference_extension','reference_separate',*[f'fresh{p}_{a}' for p in (1,2,3) for a in ('a','b')]]
    for name in new_names:
        path=folder/(name+'.json')
        assert sha(path)==prepared['files_sha256'][path.name]
        records=read_json(path)
        current_ids={r['document_id'] for r in records}
        current_texts={lookup[s] for s in current_ids}
        current_chunks={r['sha256'] for r in records}
        assert len(records)==len(current_chunks)==4096
        assert not ids&current_ids and not texts&current_texts and not chunks&current_chunks
        ids.update(current_ids);texts.update(current_texts);chunks.update(current_chunks)
    nested=read_json(folder/'reference_nested.json')
    assert nested==read_json(old/'reference_max.json')+read_json(folder/'reference_extension.json')


def main():
    figure_sources=read_json(OUT/'figure_sources.json')
    for rel,digest in figure_sources['source_sha256'].items():
        assert sha(ROOT/rel)==digest,rel
    for rel,digest in read_json(OUT/'analysis_plan.json')['code_sha256'].items():
        assert sha(ROOT/rel)==digest,rel
    freeze=read_json(OUT/'forecasts_frozen.json')
    freeze_time=datetime.fromisoformat(freeze['saved_utc'])
    for rel,digest in freeze['files_sha256'].items():
        assert sha(OUT/rel)==digest,rel
        with np.load(OUT/rel) as z:assert datetime.fromisoformat(str(z['saved_utc']))<freeze_time
    checks=0
    for corpus in CORPORA:
        check_allocation_records(corpus)
        folder=OUT/corpus
        pair_rows=list(csv.DictReader((folder/'per_pair.csv').open(newline='',encoding='utf-8')))
        pair_lookup={(r['cohort'],r['bank'],int(r['n']),int(r['m']),r['method'],int(r['pair'])):r for r in pair_rows}
        assert len(pair_lookup)==len(pair_rows)
        coverage=list(csv.DictReader((folder/'coverage_strata.csv').open(newline='',encoding='utf-8')))
        lookup={(r['cohort'],r['bank'],int(r['n']),int(r['m']),r['method'],r['stratum']):r for r in coverage}
        assert len(lookup)==len(coverage)
        for r in csv.DictReader((folder/'summary.csv').open(newline='',encoding='utf-8')):
            cohort,bank,n,m=r['cohort'],r['bank'],int(r['n']),int(r['m'])
            with np.load(folder/f'{cohort}_n{n}_observed.npz') as z:
                actual=z['point_observed'];saved_observed=z['observed']
                assert datetime.fromisoformat(str(z['saved_utc']))>freeze_time
                rows=z['rows']
            with np.load(folder/f'{bank}_m{m}_n{n}_forecasts.npz') as z:
                index=list(z['methods']).index(r['method'])
                pred=z['point_forecasts'][index];saved_pred=z['forecasts'][index]
                np.testing.assert_array_equal(rows,z['rows'])
                weights=z['point_weight']
                strata=z['strata']
            hidden=np.bincount(rows,minlength=128)
            np.testing.assert_allclose(weights,1/(128*hidden[rows]),rtol=1e-14)
            predicted=float(pred@weights)
            observed_pairs=actual@weights
            np.testing.assert_allclose(saved_observed.mean(axis=1),observed_pairs,rtol=1e-12,atol=1e-14)
            np.testing.assert_allclose(saved_pred.mean(),predicted,rtol=1e-12,atol=1e-14)
            observed=float(observed_pairs.mean())
            for pair,obs in enumerate(observed_pairs,1):
                saved=pair_lookup[cohort,bank,n,m,r['method'],pair]
                np.testing.assert_allclose([predicted,obs,predicted/obs],
                    [float(saved[key]) for key in ['predicted','observed','ratio']],rtol=1e-12,atol=1e-14)
            np.testing.assert_allclose([predicted,observed,predicted/observed],
                [float(r['predicted_mean']),float(r['observed_mean']),float(r['ratio'])],rtol=1e-12,atol=1e-14)
            margin=t.ppf(.975,2)*observed_pairs.std(ddof=1)/np.sqrt(3)
            np.testing.assert_allclose(float(r['lower']),predicted/(observed+margin),rtol=1e-12)
            if observed>margin:
                np.testing.assert_allclose(float(r['upper']),predicted/(observed-margin),rtol=1e-12)
            if r['method']=='source_full' and expectation(n,m):
                lo,hi=expectation(n,m)
                assert (r['within_expected']=='True')==(lo<=predicted/observed<=hi)
            assert set(strata).issubset(range(6))
            np.testing.assert_allclose(sum(float(pred[strata==b]@weights[strata==b]) for b in range(6)),predicted,
                                       rtol=1e-12,atol=1e-14)
            for b,label in enumerate(BIN_NAMES):
                selected=strata==b
                saved=lookup[cohort,bank,n,m,r['method'],label]
                pe=float(pred[selected]@weights[selected])
                oe=float(np.mean(actual[:,selected]@weights[selected]))
                assert int(saved['masked_positions'])==int(selected.sum())
                np.testing.assert_allclose([pe,oe,pe-oe,float(weights[selected].sum()),oe/observed],
                    [float(saved[key]) for key in ['predicted_contribution','observed_contribution',
                     'signed_error_contribution','position_weight','observed_energy_share']],rtol=1e-11,atol=1e-13)
            checks+=1
    from scripts.build_reference_ratio import main as build
    build()
    report=dict(status='passed',checked_method_conditions=checks,
        reconstruction='Saved position-level arrays independently reconcile means, intervals, and expectation checks.',
        scope='Rebuilds numerical summaries, tables, and figures. Source allocation and forecasts are not rerun.',
        forecast_chronology_verified=True,recorded_source_text_chunk_separation_verified=True)
    write_json(OUT/'saved_reproduction.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
