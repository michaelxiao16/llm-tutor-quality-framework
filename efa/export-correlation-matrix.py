#!/usr/bin/env python3
"""Export the item correlation matrix that the EFA is fitted on, for publication."""

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))  # for evaluation_utils
from evaluation_utils import efa_result_dir

# analyze-efa.py has a hyphen, so import its preprocessing via importlib.
_spec = importlib.util.spec_from_file_location(
    'analyze_efa', str(Path(__file__).with_name('analyze-efa.py'))
)
_efa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_efa)


def main():
    parser = argparse.ArgumentParser(
        description='Export the publishable item correlation matrix for one judge')
    parser.add_argument('--input', type=str, nargs='+', required=True,
                        help='The same score CSV(s) passed to the pooled analyze-efa.py run')
    parser.add_argument('--max-error-rate', type=float, default=0.20,
                        help='Must match the pooled run (default 0.20)')
    parser.add_argument('--core-max-na-rate', type=float, default=0.10,
                        help='Must match the pooled run (default 0.10)')
    parser.add_argument('--min-applicable-convos', type=int, default=200,
                        help='Must match the pooled run (default 200)')
    parser.add_argument('--decimals', type=int, default=None,
                        help='Round the exported matrix to this many decimals '
                             '(default: no rounding, full double precision)')
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    input_paths = [Path(p) for p in args.input]
    tag = input_paths[0].stem.replace('efa_evaluation_results_', '')
    if len(input_paths) > 1:
        tag += f'_merged{len(input_paths)}'
    judge = tag.split('_batch_')[0]

    out_dir = (Path(args.output_dir) if args.output_dir
               else efa_result_dir('factor_analysis') / 'correlation_matrices')
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading data from {len(input_paths)} file(s)...')
    matrix = _efa.load_and_pivot_results(
        input_paths,
        max_error_rate=args.max_error_rate,
        core_max_na_rate=args.core_max_na_rate,
        min_applicable_convos=args.min_applicable_convos,
    )

    corr = matrix.corr()
    if args.decimals is not None:
        corr = corr.round(args.decimals)
    corr_path = out_dir / f'item_correlations_{tag}.csv'
    corr.to_csv(corr_path)

    meta = pd.DataFrame([{
        'judge_model': judge,
        'n_obs': int(matrix.shape[0]),
        'n_items': int(matrix.shape[1]),
        'correlation': 'pearson',
        'max_error_rate': args.max_error_rate,
        'core_max_na_rate': args.core_max_na_rate,
        'min_applicable_convos': args.min_applicable_convos,
        'note': ('Core metrics only, after error filtering, applicability tiering, '
                 'mean imputation of residual gaps and zero-variance removal. '
                 'n_obs is required to refit (e.g. psych::fa(r, n.obs = n_obs)).'),
    }])
    meta_path = out_dir / f'item_correlations_{tag}_meta.csv'
    meta.to_csv(meta_path, index=False)

    size_kb = corr_path.stat().st_size / 1024
    print(f'\n{judge}: {matrix.shape[1]} items x {matrix.shape[0]} observations')
    print(f'Wrote {corr_path}  ({size_kb:.0f} KB)')
    print(f'Wrote {meta_path}')


if __name__ == '__main__':
    main()
