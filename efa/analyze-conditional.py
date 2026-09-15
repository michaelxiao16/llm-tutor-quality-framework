#!/usr/bin/env python3
"""Conditional-metric analysis (Part 2 of the EFA pipeline)."""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))  # for evaluation_utils
from evaluation_utils import efa_result_dir

# analyze-efa.py has a hyphen, so import its preprocessing via importlib.
import importlib.util
_spec = importlib.util.spec_from_file_location(
    'analyze_efa', str(Path(__file__).with_name('analyze-efa.py'))
)
_efa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_efa)


def main():
    parser = argparse.ArgumentParser(description='Extension analysis for conditional metrics')
    parser.add_argument('--input', type=str, nargs='+', required=True,
                        help='The same score CSV(s) passed to the pooled analyze-efa.py run')
    parser.add_argument('--n-factors', type=int, required=True,
                        help='Number of core factors (must match the pooled run)')
    parser.add_argument('--rotation', type=str, default='oblimin',
                        choices=['varimax', 'promax', 'oblimin', 'none'])
    parser.add_argument('--threshold', type=float, default=0.4,
                        help='Salient extension-loading threshold (default 0.4)')
    parser.add_argument('--core-max-na-rate', type=float, default=0.10,
                        help='Core cutoff (must match the pooled run; default 0.10)')
    parser.add_argument('--min-applicable-convos', type=int, default=200,
                        help='Applicability floor (must match the pooled run; default 200)')
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    input_paths = [Path(p) for p in args.input]
    rotation = None if args.rotation == 'none' else args.rotation
    output_dir = Path(args.output_dir) if args.output_dir else efa_result_dir('factor_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    tag = input_paths[0].stem.replace('efa_evaluation_results_', '')
    if len(input_paths) > 1:
        tag += f"_merged{len(input_paths)}"
    tag += f"_{rotation or 'unrotated'}_n{args.n_factors}"

    # Same preprocessing as the pooled run -> core matrix + conditional matrix.
    print(f"Loading and tiering {len(input_paths)} file(s)...")
    df = _efa.load_and_pivot_results(
        input_paths,
        core_max_na_rate=args.core_max_na_rate,
        min_applicable_convos=args.min_applicable_convos,
    )
    conditional = df.attrs['conditional_matrix']
    if conditional.shape[1] == 0:
        print("No conditional metrics for this judge — nothing to analyze.")
        return
    print(f"Core metrics: {df.shape[1]}  |  conditional metrics: {conditional.shape[1]}")

    # Route conditional outputs to the purpose-grouped `conditional/` subfolder.
    conditional_dir = output_dir / 'conditional'
    conditional_dir.mkdir(parents=True, exist_ok=True)
    loadings_path = conditional_dir / f'conditional_extension_loadings_{tag}.csv'
    descriptives_path = conditional_dir / f'conditional_descriptives_{tag}.csv'

    script = Path(__file__).with_name('conditional-extension.R')
    with tempfile.TemporaryDirectory(prefix='efa-conditional-') as temp_dir:
        temp = Path(temp_dir)
        core_path = temp / 'core.csv'
        cond_path = temp / 'conditional.csv'
        df.to_csv(core_path, index=False)
        conditional.to_csv(cond_path, index=False)  # NaN preserved where not applicable
        rscript = os.environ.get('RSCRIPT') or shutil.which('Rscript') or 'Rscript'
        command = [
            rscript, str(script.resolve()),
            '--core-data', str(core_path.resolve()),
            '--conditional-data', str(cond_path.resolve()),
            '--n-factors', str(args.n_factors),
            '--rotation', rotation or 'none',
            '--threshold', str(args.threshold),
            '--min-applicable', str(args.min_applicable_convos),
            '--loadings-output', str(loadings_path.resolve()),
            '--descriptives-output', str(descriptives_path.resolve()),
        ]
        # conditional-extension.R has to refit the core solution to hand psych a fitted
        # object.
        committed_pattern = output_dir / 'solution' / f'factor_loadings_{tag}.csv'
        if committed_pattern.exists():
            command += ['--pattern-check', str(committed_pattern.resolve())]
        else:
            print(f"  Note: {committed_pattern.name} not found; skipping the "
                  f"refit-equals-authoritative-solution check.")
        subprocess.run(
            command,
            check=True,
            cwd=str(script.parent),  # cwd -> renv activation
            env={
                **os.environ,
                # Use the pinned project library without waiting on renv's transient sandbox
                # lock, consistent with analyze-efa.py.
                'RENV_CONFIG_SANDBOX_ENABLED': 'FALSE',
            },
            timeout=120,
        )

    desc = pd.read_csv(descriptives_path)
    print(f"\nSaved extension loadings: {loadings_path}")
    print(f"Saved conditional descriptives: {descriptives_path}")
    print("\nWhere each conditional metric attaches on the core factors "
          f"(|loading| >= {args.threshold}):")
    ordered = desc.sort_values('primary_abs_extension_loading', ascending=False, na_position='last')
    for row in ordered.itertuples(index=False):
        where = str(row.attaches_to_core_factor)
        if where.startswith('Factor_'):
            verdict = f"attaches to {where}"
            detail = f"primary={row.primary_core_factor} ({row.primary_extension_loading:+.2f})"
        elif where.startswith('not estimable'):
            verdict = where
            detail = "constant when applicable"
        else:
            verdict = "no salient attachment"
            loading = row.primary_extension_loading
            detail = (f"primary={row.primary_core_factor} ({loading:+.2f})"
                      if pd.notna(loading) else "n/a")
        print(f"  {row.metric_id:<62} n={int(row.n_applicable):<4} {detail:<28} -> {verdict}")
    attach = desc['attaches_to_core_factor'].astype(str)
    n_attach = int(attach.str.startswith('Factor_').sum())
    n_none = int((attach == 'none (< threshold)').sum())
    n_bad = len(desc) - n_attach - n_none
    print(f"\n{n_attach} of {len(desc)} conditional metrics attach to a core factor; "
          f"{n_none} have no salient attachment (distinct / general-factor only)"
          + (f"; {n_bad} not estimable (constant when applicable)." if n_bad else "."))
    print("\nDone!")


if __name__ == '__main__':
    main()
