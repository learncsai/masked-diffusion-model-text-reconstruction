"""Rebuild Table 3 directly from numerical records and check Appendix E.

This deliberately does not import the sweep summarizer. It independently
recomputes the jackknife and aggregation, then reconciles both paths.
"""
from pathlib import Path
import argparse
import csv
import json
import numpy as np


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def records(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('work', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    work, out = args.work, args.output
    component = work/'13_alpha_sweep'
    data = component/'results/alpha_sweep_v1'
    config = read(component/'configs/alpha_sweep_v1.json')
    expected = read(Path(__file__).with_name('EXPECTED_RESULTS.json'))
    assert config['pairs'] == 20 and config['banks'] == 6
    cells = [(1024, 2048), (1024, 4096), (2048, 4096)]
    assert [(m,n) for m in config['reference_sizes'] for n in config['sizes'] if m<n] == cells
    regimes = records(data/'rebuilt/calibration_by_regime.csv')
    estimates, audit, by_bank = {}, [], []
    for corpus in config['corpora']:
        folder = data/corpus
        observed = []
        for pair in range(config['pairs']):
            with np.load(folder/f'observed_pair{pair}.npz', allow_pickle=False) as array:
                observed.append(array['disagreement'])
        observed_mean = np.mean(observed, axis=0)
        for alpha in (1,5,20):
            ai = config['alphas'].index(alpha)
            for arm, arm_name in enumerate(config['weight_arms']):
                for method, column in [('source_jackknife',2),('multinomial_jackknife',0)]:
                    losses = []
                    for m,n in cells:
                        ni = config['sizes'].index(n)
                        denominator = float(observed_mean[arm,ai,ni])
                        assert denominator>0
                        for bank in range(config['banks']):
                            with np.load(folder/f'analytic_bank{bank}_m{m}.npz', allow_pickle=False) as array:
                                fractions = array['source_fractions']
                                counts = array['source_counts']
                                assert counts[0] == counts[1:].sum()
                                np.testing.assert_allclose(fractions, counts[1:]/counts[0], rtol=1e-13)
                                full, half1, half2 = array['values'][:,arm,ai,ni,column]
                                forecast = float(2*full - fractions[0]*half1 - fractions[1]*half2)
                                np.testing.assert_allclose(forecast,array['corrected'][arm,ai,ni,column],rtol=1e-13,atol=1e-15)
                            assert forecast>0, (corpus,alpha,arm_name,method,m,n,bank)
                            loss = float(abs(np.log(forecast/denominator)))
                            losses.append(loss)
                            if arm_name == 'fixed':
                                by_bank.append(dict(corpus=corpus,alpha=alpha,method=method,m=m,n=n,bank=bank,
                                                    forecast=forecast,observed_pair_mean=denominator,absolute_log_error=loss))
                    value = float(np.mean(losses))
                    summary = next(r for r in regimes if
                        (r['corpus'],r['arm'],float(r['alpha']),r['method'],r['regime']) ==
                        (corpus,arm_name,alpha,method,'m<n'))
                    assert int(summary['cells'])==3 and int(summary['nonpositive_bank_cells'])==0
                    np.testing.assert_allclose(value,float(summary['mean_bank_abs_log_error']),rtol=1e-13,atol=1e-15)
                    estimates[(corpus,arm_name,alpha,method)] = value
                    audit.append(dict(corpus=corpus,arm=arm_name,alpha=alpha,method=method,
                                      value=value,rounded=f'{value:.3f}',bank_cell_count=len(losses),
                                      interval=[float(summary['low']),float(summary['high'])]))
    table_rows, csv_rows = [], []
    for alpha in (1,5,20):
        row={'alpha':alpha}
        for corpus in config['corpora']:
            for method in ('source_jackknife','multinomial_jackknife'):
                row[corpus+'_'+method]=f"{estimates[(corpus,'fixed',alpha,method)]:.3f}"
        csv_rows.append(row)
        table_rows.append(str(alpha)+' & '+' & '.join(list(row.values())[1:]))
    table_folder=out/'tables';table_folder.mkdir(parents=True,exist_ok=True)
    with (table_folder/'table_3.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    with (out/'diagnostics/table_3_bank_cells.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(by_bank[0]));writer.writeheader();writer.writerows(by_bank)
    for arm in config['weight_arms']:
        for corpus in config['corpora']:
            for alpha in (1,5):
                assert estimates[(corpus,arm,alpha,'multinomial_jackknife')]<estimates[(corpus,arm,alpha,'source_jackknife')]
            if corpus!='cnn_dailymail':
                assert estimates[(corpus,arm,20,'source_jackknife')]<estimates[(corpus,arm,20,'multinomial_jackknife')]
    for method in ('source_jackknife','multinomial_jackknife'):
        assert f"{estimates[('cnn_dailymail','fixed',20,method)]:.3f}"=='0.024'
    flattening=[r for r in records(data/'rebuilt/hypotheses.csv') if r['test']=='beta20_minus_beta1']
    assert len(flattening)==6 and all(float(r['low'])>0 for r in flattening)
    occupancy=next(r for r in read(work/'01_original_counts/recomputed.json') if (r['corpus'],r['n'])==('wikitext',2048))
    percentages={name:100*occupancy[name] for name in ('empty_fraction','below_smoothing_fraction')}
    assert [f'{v:.1f}' for v in percentages.values()]==['7.3','20.8']
    assert all(f'{v:.1f}\\%' in expected['reported_percent_strings'] for v in percentages.values())
    decomposition=read(work/'09_reference_correction/results/rare_context_development_v1/missing_row_decomposition/summary.json')
    chosen={r['component']:r['fraction'] for r in decomposition if
            (r['case'],r['n'],r['m'])==('clustered_sparse',1024,128)}
    np.testing.assert_allclose(chosen['coverage_gap'],-chosen['unseen_energy']-chosen['unseen_twice_cross'],rtol=1e-13)
    terms={name:100*chosen[name] for name in ('coverage_gap','seen_reference_error','propagation_error')}
    assert [f'{v:+.1f}' for v in terms.values()]==['-32.6','-38.6','+3.2']
    assert all(f'{v:+.1f}' in expected['reported_signed_numbers'] for v in terms.values())
    report=dict(status='passed',table3_rows=table_rows,table3_values_checked=18,
                aggregation='Mean over six banks and three m<n cells of abs(log(forecast / arithmetic mean of 20 pair disagreements)).',
                corrections='Recomputed as twice full-bank forecast minus source-count-weighted half-bank forecasts.',
                estimates=audit,positive_slope_change_intervals=flattening,wikitext_2048_coverage_percent=percentages,
                synthetic_decomposition_percentage_points=terms,
                scope='Table 3 recomputed independently from scalar full/half reference forecasts and observed pair scores. E.1 checks saved decomposition summaries, not a new simulation. Table 10 exact moments and intervals are rebuilt by component 11. E.4 historical GPU rescoring audits are supplied as records.')
    (out/'APPENDIX_E_CHECKS.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('PASS Table 3: all 18 values independently rebuilt, correction and aggregation reconciled. Appendix E coverage, decomposition and alpha findings checked.')


if __name__=='__main__':
    main()
