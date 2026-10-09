"""Rebuild compact measurements and export the submitted paper's ordered outputs."""
from pathlib import Path
import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def plain(cell):
    cell = cell.strip().rstrip('\\').strip().replace('$', '')
    for command, value in [(r'\times', ' x '), (r'\le', '<='), (r'\%', '%'),
                           (r'\downarrow', ''), (r'\uparrow', '')]:
        cell = cell.replace(command, value)
    cell = re.sub(r'\\(?:text|mathrm|mathbf)\{([^{}]*)\}', r'\1', cell)
    return cell.replace('{', '').replace('}', '').strip()


DEFINITIONS = {
    'tab:data-roles': [
        ['Auxiliary', 'Set the shared unigram and reconstruction weights.'],
        ['Reference', 'Estimate count variation and forecast disagreement.'],
        ['A and B', 'Supply two independent count tables for testing the forecast.'],
        ['Evaluation', 'Supply the same masked inputs to both predictors.']],
    'tab:image-text-map': [
        ['Data sampling', 'Images sampled from a distribution', 'Sources supplying token chunks', 'The same source-disjoint A/B corpora'],
        ['Corruption', 'Additive Gaussian noise', 'Hidden-token probability q', 'Absorbing token masks evaluated at q=0.5'],
        ['Dependence on data', 'Empirical mean and covariance for the linear denoiser', 'Lag counts and conditional rows; auxiliary unigram and weights fixed', 'Transformer parameters learned from masked chunks'],
        ['Predictor', 'Empirical linear denoiser', 'Weighted conditional rows', 'Time-conditioned Transformer probabilities'],
        ['Population target', 'Output variation over independently sampled datasets', 'Disagreement over source collections', 'Disagreement conditional on shared training randomness'],
        ['Shared input', 'Noisy image for denoising; initial noise for sampling', 'Identical text and mask', 'Identical text, mask, and diffusion time'],
        ['Analysis', 'Random-matrix expressions for linear denoiser and sampling-map fluctuations', 'Source-count covariance or multinomial row variance', 'Observed A/B disagreement compared with count-based forecasts'],
        ['Outputs compared', 'Denoiser outputs and generated samples', 'One-step reconstruction scores', 'One-step token probabilities']]
}


def export_table(out, number, title, columns, rows):
    assert rows and all(len(row) == len(columns) for row in rows), (title, rows)
    (out/'tables').mkdir(parents=True, exist_ok=True)
    stem = f'table_{number}'
    with (out / 'tables' / f'{stem}.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)
    escape = lambda value: str(value).replace('|', r'\|')
    lines = [f'# Table {number}: {title}', '', '| ' + ' | '.join(columns) + ' |',
             '| ' + ' | '.join('---' for _ in columns) + ' |']
    lines.extend('| ' + ' | '.join(escape(v) for v in row) + ' |' for row in rows)
    (out / 'tables' / f'{stem}.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export-only', action='store_true', help='Export a previously passed local archive rebuild.')
    args = parser.parse_args()
    env = os.environ.copy()
    env.update(PYTHONUTF8='1', MPLBACKEND='Agg', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    if not args.export_only:
        subprocess.run([sys.executable, str(ROOT/'artifact/reproduce.py')], cwd=ROOT, env=env, check=True)
    replay = ROOT / 'artifact/reproduced'
    report = json.loads((replay/'REPORT.json').read_text(encoding='utf-8'))
    assert report['status'] == 'passed'
    inventory = json.loads((ROOT/'provenance/submitted_outputs.json').read_text())
    out = ROOT / 'results/reproduction'
    for folder in ['tables', 'figures', 'diagnostics']:
        (out/folder).mkdir(parents=True, exist_ok=True)
    index = ['# Reproduced submitted-paper outputs', '',
             'Numbering follows the attached submission. Seven empirical tables are recomputed; two tables describe definitions.', '',
             '## Tables', '', '| Number | Subject | Outputs |', '|---|---|---|']
    table_report = []
    for table in inventory['tables']:
        label, number = table['label'], table['number']
        if label in DEFINITIONS:
            rows = DEFINITIONS[label]
        else:
            source = replay/'tables'/(label.replace(':', '_')+'.tex')
            rows = [[plain(cell) for cell in line.split('&')]
                    for line in source.read_text(encoding='utf-8').splitlines() if '&' in line]
        export_table(out, number, table['title'], table['columns'], rows)
        table_report.append(dict(number=number, label=label, kind=table['kind'], rows=len(rows)))
        index.append(f"| {number} | {table['title']} | [CSV](tables/table_{number}.csv), [Markdown](tables/table_{number}.md) |")
    work = ROOT/'artifact'/report['work_directory']
    png_sources = {
        1: replay/'figures/fig_reference_correction.png',
        2: replay/'figures/fig_reference_correction_all.png',
        3: work/'11_oracle_multiplier/results/oracle_multiplier_v1/fig_multiplier_transfer.png',
        4: work/'06_mdlm/results/mdlm_paper_comparison_v1/mdlm_disagreement.png',
        5: work/'05_topk/results/lag_topk_matched_v1/topk_comparison.png'}
    index += ['', '## Figures', '', '| Number | Subject | PNG |', '|---|---|---|']
    for figure in inventory['figures']:
        shutil.copyfile(png_sources[figure['number']], out/'figures'/f"figure_{figure['number']}.png")
        index.append(f"| {figure['number']} | {figure['title']} | [Figure](figures/figure_{figure['number']}.png) |")
    decisions = replay/'tables/tab_correction-decisions.tex'
    if decisions.exists():
        rows = [[plain(c) for c in line.split('&')] for line in decisions.read_text().splitlines() if '&' in line]
        export_table(out/'diagnostics', 'size_decisions', 'Supporting size decisions',
                     ['Forecast','Success','Failure','Abstain','At minimum','Above minimum'], rows)
    for name in ['PROSE_CLAIMS.json', 'APPENDIX_E_CHECKS.json']:
        shutil.copyfile(replay/name, out/name)
    result = dict(status='passed', mode='saved-measurement numerical reproduction',
        empirical_tables=7, definition_tables=2, figures=5, tables=table_report,
        archive_report='../../artifact/reproduced/REPORT.json',
        paper_reference='provenance/submitted_outputs.json',
        excluded_stages=['Source download','New real-text forecasts','Neural inference','Neural training'])
    (out/'REPORT.json').write_text(json.dumps(result, indent=2)+'\n')
    index += ['', 'Extra archive outputs remain under `artifact/reproduced/diagnostics/`.',
              'See [REPORT.json](REPORT.json) for the validation scope.']
    (out/'INDEX.md').write_text('\n'.join(index)+'\n', encoding='utf-8')
    print('PASS: 9 submitted tables and 5 figures. Open results/reproduction/INDEX.md.')


if __name__ == '__main__':
    main()
