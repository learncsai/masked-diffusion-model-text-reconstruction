"""Batch the existing disagreement laws and retain masked-position traces.

The source algebra is the same as sparse_categorical_influence. A reference
source is counted once, then reused at each target size. The original frozen
implementation is retained as an independent numerical comparator.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from . import sparse_categorical_lag as lag
from .categorical_lag import pattern_weights, per_sequence_squared_error
from .sparse_categorical_influence import sparse_row_inner


def reference_forecasts(reference, groups, evaluation, mask, prior, offsets,
                        gram, cross, sizes, *, smoothing, ridge, progress=None):
    documents, m, classes = len(groups), len(reference), len(prior)
    if documents < 2:
        raise ValueError('At least two reference sources are required')
    counts = lag.pair_counts(reference, offsets, classes)
    rows, _, contexts, visible, patterns = lag.context_design(evaluation, mask, offsets)
    weights = np.zeros((len(rows), len(offsets)))
    for code in np.unique(patterns):
        if code:
            active = tuple(j for j in range(len(offsets)) if code & (1 << j))
            weights[np.ix_(np.flatnonzero(patterns == code), active)] = pattern_weights(
                gram, cross, active, ridge)
    selectors = []
    row_counts = np.full((len(rows), len(offsets)), np.inf)
    ref_tables = lag.conditional_tables(counts, prior, smoothing)
    collisions = {}
    for j, delta in enumerate(offsets):
        selected = np.flatnonzero(visible[j])
        context = contexts[j][selected]
        selectors.append(sparse.csr_matrix((np.ones(len(selected)), (selected, context)),
                                          shape=(len(rows), classes)))
        totals = np.asarray(counts[delta].sum(axis=1)).ravel()
        row_counts[selected, j] = totals[context]
        collisions[delta] = lag.Prediction(*ref_tables[delta], prior).norm2()
    minimum = row_counts.min(axis=1)
    bins = np.select([np.isinf(minimum), minimum == 0, minimum < 5,
                      minimum < 20, minimum < 100], [5, 0, 1, 2, 3], default=4)
    hidden = np.bincount(rows, minlength=len(evaluation))
    point_weight = 1 / (len(evaluation) * hidden[rows])
    states = {}
    for n in sizes:
        baseline = {delta: c * (n/m) for delta, c in counts.items()}
        tables = lag.conditional_tables(baseline, prior, smoothing)
        conditional, scaled_selectors = [], []
        multinomial = np.zeros(len(rows))
        for j, delta in enumerate(offsets):
            denom = np.asarray(baseline[delta].sum(axis=1)).ravel() + smoothing
            conditional.append(lag.Prediction(selectors[j] @ tables[delta][0],
                                              selectors[j] @ tables[delta][1], prior))
            selected = np.flatnonzero(visible[j])
            context = contexts[j][selected]
            scaled_selectors.append(sparse.csr_matrix(
                (weights[selected, j]/denom[context], (selected, context)),
                shape=(len(rows), classes)))
            totals = denom - smoothing
            trace = 2*totals/denom**2*np.maximum(1-collisions[delta], 0)
            multinomial[selected] += weights[selected,j]**2*trace[context]
        kernels = {(j,k):lag.row_inner(conditional[j],conditional[k])
                   for j in range(len(offsets)) for k in range(j+1)}
        ref_parts = [sel @ counts[delta] for sel,delta in zip(scaled_selectors,offsets)]
        ref_totals = [np.asarray(part.sum(axis=1)).ravel() for part in ref_parts]
        ref_outputs = [lag.Prediction(
            (part-p.residual.multiply(total[:,None])).tocsr(),-total*p.prior_weight,prior)
            for part,total,p in zip(ref_parts,ref_totals,conditional)]
        ref_full = lag.Prediction(
            sum((v.residual for v in ref_outputs),sparse.csr_matrix((len(rows),classes))),
            sum((v.prior_weight for v in ref_outputs),np.zeros(len(rows))),prior)
        states[n] = dict(conditional=conditional,selectors=scaled_selectors,kernels=kernels,
            ref_outputs=ref_outputs,ref_full=ref_full,
            ref_kernels_full=[lag.row_inner(p,ref_full) for p in conditional],
            ref_kernels_same=[lag.row_inner(p,r) for p,r in zip(conditional,ref_outputs)],
            ref_norm_full=ref_full.norm2(),
            ref_norm_same=sum((r.norm2() for r in ref_outputs),np.zeros(len(rows))),
            norm_full=np.zeros(len(rows)),norm_same=np.zeros(len(rows)),
            cross_full=np.zeros(len(rows)),cross_same=np.zeros(len(rows)),
            multinomial=multinomial)
    squared_chunks = 0
    for index,group in enumerate(groups):
        source_counts = lag.pair_counts(reference[group],offsets,classes)
        chunks = len(group)
        squared_chunks += chunks**2
        for state in states.values():
            conditional = state['conditional']
            parts = [s @ source_counts[d] for s,d in zip(state['selectors'],offsets)]
            totals = [np.asarray(p.sum(axis=1)).ravel() for p in parts]
            total_part = sum(parts,sparse.csr_matrix((len(rows),classes)))
            state['norm_full'] += np.asarray(total_part.multiply(total_part).sum(axis=1)).ravel()
            cross_full = sparse_row_inner(total_part,state['ref_full'])
            for j,(part,total) in enumerate(zip(parts,totals)):
                state['norm_same'] += np.asarray(part.multiply(part).sum(axis=1)).ravel()
                state['norm_same'] -= 2*total*sparse_row_inner(part,conditional[j])
                state['norm_same'] += total**2*state['kernels'][j,j]
                state['cross_same'] += chunks*(sparse_row_inner(part,state['ref_outputs'][j])
                                               - total*state['ref_kernels_same'][j])
                state['norm_full'] -= 2*total*sparse_row_inner(total_part,conditional[j])
                cross_full -= total*state['ref_kernels_full'][j]
                for k in range(j+1):
                    state['norm_full'] += (1 if j==k else 2)*total*totals[k]*state['kernels'][j,k]
            state['cross_full'] += chunks*cross_full
        if progress and (index+1)%500 == 0:
            progress(index+1,documents)
    result = {}
    for n,state in states.items():
        points = [state['multinomial']]
        for variant in ('same','full'):
            centered = (state['norm_'+variant]-2*state['cross_'+variant]/m
                        +squared_chunks*state['ref_norm_'+variant]/m**2)
            if np.min(centered,initial=0)<-1e-10:
                raise ArithmeticError('Negative source variance')
            points.append(2*(n*documents/m)*np.maximum(centered,0)/(documents-1))
        points = np.array(points)
        forecasts = np.array([per_sequence_squared_error(rows,p,len(evaluation)) for p in points])
        np.testing.assert_allclose(points @ point_weight,forecasts.mean(axis=1),rtol=1e-12,atol=1e-14)
        result[n] = dict(forecasts=forecasts,point_forecasts=points,rows=rows,
                         point_weight=point_weight,strata=bins,minimum_row_count=minimum)
    return result
