#!/usr/bin/env python3
"""Exploratory Factor Analysis (EFA) for AI Tutor Evaluation Metrics"""

import argparse
import hashlib
from importlib.metadata import version as package_version
import os
import shutil
import re
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
from scipy.optimize import linear_sum_assignment
from sklearn.impute import SimpleImputer

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times', 'Nimbus Roman', 'Times New Roman', 'Liberation Serif'],
    'mathtext.fontset': 'stix',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

sys.path.insert(0, str(Path(__file__).parent.parent))  # repo root, for evaluation_utils
from evaluation_utils import efa_result_dir


def save_png_and_pdf(fig, output_path: Path, **kwargs) -> None:
    """Save a figure as matching PNG and vector PDF files."""
    output_path = Path(output_path)
    stem = output_path.stem
    fig.savefig(output_path.with_name(f'{stem}.png'), **kwargs)
    fig.savefig(output_path.with_name(f'{stem}.pdf'), **kwargs)

# Purpose-grouped output layout: each result kind is routed to a subfolder of the factor-
# analysis output dir so the three judges' files sit together by purpose, with images beside
# their CSVs.
_SUBFOLDER_BY_PREFIX = {
    'preprocessing_audit': 'preprocessing',
    'factorability_psych_r': 'preprocessing',
    'factorability_table': 'preprocessing',
    'eigenvalues': 'retention',
    'scree_plot': 'retention',
    'parallel_analysis': 'retention',        # csv and parallel_analysis_plot png
    'factor_loadings': 'solution',              # also factor_loadings_heatmap
    'factor_structure_matrix': 'solution',
    'factor_correlations': 'solution',          # also factor_correlations_heatmap
    'pattern_matrix_report': 'solution',
    'psych_r_pattern_matrix': 'solution',
    'psych_r_structure_matrix': 'solution',
    'psych_r_factor_correlations': 'solution',
    'factor_membership': 'membership',          # csv and png
    'factor_summary': 'membership',
    'factor_reliability': 'reliability',
    'schmid_leiman_loadings': 'general_factor',
    'schmid_leiman_summary': 'general_factor',
    'schmid_leiman_omega_by_factor': 'general_factor',
    'schmid_leiman_group_mapping': 'general_factor',
    'cross_library_congruence': 'congruence',
    'model_diagnostics': 'validation',
    'validation_report': 'validation',
    'output_manifest': 'validation',
    'efa_reporting_metadata': 'validation',
}
_SUBFOLDER_PREFIXES_SORTED = sorted(_SUBFOLDER_BY_PREFIX, key=len, reverse=True)


def opath(output_dir: Path, filename: str) -> Path:
    """Route an output filename to its purpose subfolder (created on demand)."""
    for prefix in _SUBFOLDER_PREFIXES_SORTED:
        if filename.startswith(prefix):
            target = Path(output_dir) / _SUBFOLDER_BY_PREFIX[prefix]
            target.mkdir(parents=True, exist_ok=True)
            return target / filename
    return Path(output_dir) / filename


def load_results(csv_paths: list[Path]) -> pd.DataFrame:
    """Load and concatenate one or more result CSVs (the chunks of a batched run)."""
    frames = []
    for p in csv_paths:
        d = pd.read_csv(p)
        print(f"  {p.name}: {len(d)} rows, {d['conversation_id'].nunique()} conversations")
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)

    if len(csv_paths) > 1:
        dups = int(df.duplicated(['conversation_id', 'metric_id']).sum())
        if dups:
            raise ValueError(f"{dups} duplicate (conversation_id, metric_id) rows across inputs "
                             f"— chunks overlap. Check --offset/--limit ranges don't repeat conversations.")
        per_conv = df.groupby('conversation_id').size()
        if per_conv.nunique() > 1:
            n = int(per_conv.mode().iloc[0])
            short = per_conv[per_conv != n]
            print(f"  ⚠️  {len(short)} conversations don't have {n} metrics "
                  f"(range {int(per_conv.min())}-{int(per_conv.max())}) — a partial/failed chunk?")
        print(f"  Combined: {df['conversation_id'].nunique()} conversations, {len(df)} rows")
    return df


def load_and_pivot_results(
    csv_paths,
    max_error_rate: float = 0.2,
    core_max_na_rate: float = 0.10,
    min_applicable_convos: int = 200,
    impute: bool = True,
) -> pd.DataFrame:
    """Load evaluation result CSV(s) and pivot to one row per conversation, tiering each metric
    by how applicable it is (how often it is actually scored vs. error/N-A)."""
    if isinstance(csv_paths, (str, Path)):
        csv_paths = [Path(csv_paths)]
    df = load_results([Path(p) for p in csv_paths])

    # Use metric_id if available, otherwise construct from category + subcategory
    if 'metric_id' in df.columns:
        df['metric'] = df['metric_id']
    else:
        df['metric'] = df['category'] + '_' + df['subcategory']

    all_metrics = sorted(df['metric'].unique())
    category_lookup = (
        df.groupby('metric', sort=False)[['category', 'subcategory']]
        .first()
        .reindex(all_metrics)
    )

    # Keep only metrics present in EVERY chunk.
    absent_from_chunk = set()
    if len(csv_paths) > 1:
        per_chunk = [set(pd.read_csv(p)['metric_id']) for p in csv_paths]
        common = set.intersection(*per_chunk)
        absent_from_chunk = set(all_metrics) - common
        if absent_from_chunk:
            print(f"  Dropping {len(absent_from_chunk)} metric(s) not present in all chunks: "
                  f"{sorted(absent_from_chunk)}")
        df = df[df['metric'].isin(common)].copy()

    # Audit every status and tier metrics by applicability.
    status_counts = df.groupby('metric')['score_status'].value_counts().unstack(fill_value=0)
    n_total = status_counts.sum(axis=1)
    n_scored = status_counts.get('scored', pd.Series(0, index=status_counts.index))
    n_error = status_counts.get('error', pd.Series(0, index=status_counts.index))
    n_not_applicable = status_counts.get(
        'not_applicable', pd.Series(0, index=status_counts.index)
    )
    metric_error_rates = n_error / n_total
    metric_missing_rates = 1 - (n_scored / n_total)

    # Three-way applicability tiering.
    metrics = list(status_counts.index)
    high_error = set(metric_error_rates[metric_error_rates > max_error_rate].index)
    tier = {}
    for m in metrics:
        if m in high_error:
            tier[m] = 'high_error'
        elif metric_missing_rates[m] <= core_max_na_rate:
            tier[m] = 'core'
        elif n_scored[m] >= min_applicable_convos:
            tier[m] = 'conditional'
        else:
            tier[m] = 'rare'
    core_metrics = sorted(m for m in metrics if tier[m] == 'core')
    conditional_metrics = sorted(m for m in metrics if tier[m] == 'conditional')
    rare_metrics = sorted(m for m in metrics if tier[m] == 'rare')
    if high_error:
        print(f"  Dropped {len(high_error)} metric(s) with >{max_error_rate*100:.0f}% error rate: "
              f"{sorted(high_error)}")
    print(f"  Applicability tiers: {len(core_metrics)} core (<= {core_max_na_rate*100:.0f}% "
          f"non-scored), {len(conditional_metrics)} conditional (>= {min_applicable_convos} "
          f"applicable -> extension analysis), {len(rare_metrics)} rare/dropped")
    if conditional_metrics:
        print(f"  Conditional metrics set aside for extension analysis: {conditional_metrics}")
    if rare_metrics:
        print(f"  Dropped {len(rare_metrics)} rare metric(s) (< {min_applicable_convos} "
              f"applicable): {rare_metrics}")

    # Pivot the scored data for core + conditional metrics (rare / high-error excluded).
    kept = set(core_metrics) | set(conditional_metrics)
    scored = df[(df['score_status'] == 'scored') & (df['metric'].isin(kept))]

    # pivot_table's aggfunc silently averages repeated (conversation, metric) cells, which
    # understates that item's variance and contaminates every correlation it enters, with no
    # visible symptom. load_results() already rejects duplicate (conversation_id, metric_id)
    # rows, but only when more than one input file is passed, and only on the metric_id
    # column.
    duplicated = scored.duplicated(['conversation_id', 'metric'], keep=False)
    if duplicated.any():
        offenders = (scored[duplicated]
                     .groupby(['conversation_id', 'metric'])
                     .size()
                     .sort_values(ascending=False))
        preview = '; '.join(f'{c}/{m} x{n}' for (c, m), n in offenders.head(5).items())
        raise ValueError(
            f'{duplicated.sum()} scored rows share a (conversation_id, metric) pair '
            f'across {len(offenders)} distinct pairs. The input chunks for one judge '
            f'must be disjoint; overlapping chunks would be averaged together and '
            f'silently distort the correlation matrix. First offenders: {preview}'
        )

    pivot_all = scored.pivot_table(
        index='conversation_id', columns='metric', values='score', aggfunc='mean'
    )

    # Conditional matrix: raw scores, NaN where not applicable, preserved (never imputed) for
    # downstream conditional extension analysis.
    conditional_matrix = pivot_all.reindex(columns=conditional_metrics)

    # Core matrix: the pooled-EFA analysis matrix; mean-impute the small remaining gaps.
    pivot_df = pivot_all.reindex(columns=core_metrics)
    if impute and pivot_df.shape[1]:
        missing_before = int(pivot_df.isna().sum().sum())
        missing_by_metric = pivot_df.isna().sum()
        imputer = SimpleImputer(strategy='mean')
        pivot_df = pd.DataFrame(
            imputer.fit_transform(pivot_df),
            index=pivot_df.index,
            columns=pivot_df.columns,
        )
        if missing_before > 0:
            print(f"  Imputed {missing_before} missing core values with column means")
    else:
        missing_before = int(pivot_df.isna().sum().sum())
        missing_by_metric = pivot_df.isna().sum()

    # Drop zero-variance core columns (no discriminant information for factor analysis)
    variances = pivot_df.var()
    zero_var_cols = variances[np.isclose(variances, 0.0, atol=1e-12)].index.tolist()
    if zero_var_cols:
        print(f"  Dropped {len(zero_var_cols)} zero-variance core column(s): {zero_var_cols}")
        pivot_df = pivot_df.drop(columns=zero_var_cols)

    # Preserve a complete, machine-readable preprocessing audit for later reporting.
    core_retained = set(pivot_df.columns)
    audit = pd.DataFrame(index=all_metrics)
    audit.index.name = 'metric_id'
    audit[['category', 'subcategory']] = category_lookup[['category', 'subcategory']]
    audit['present_in_all_input_chunks'] = ~audit.index.isin(absent_from_chunk)
    audit['n_present_rows'] = status_counts.reindex(audit.index).fillna(0).sum(axis=1).astype(int)
    audit['n_scored'] = n_scored.reindex(audit.index).fillna(0).astype(int)
    audit['n_error'] = n_error.reindex(audit.index).fillna(0).astype(int)
    audit['n_not_applicable'] = n_not_applicable.reindex(audit.index).fillna(0).astype(int)
    audit['error_rate_among_present_rows'] = metric_error_rates.reindex(audit.index)
    audit['non_scored_rate_among_present_rows'] = metric_missing_rates.reindex(audit.index)
    audit['n_values_mean_imputed'] = missing_by_metric.reindex(audit.index).fillna(0).astype(int)
    audit['variance_after_imputation'] = variances.reindex(audit.index)

    def _tier_label(metric):
        if metric in absent_from_chunk:
            return 'dropped_not_in_all_chunks'
        t = tier.get(metric)
        if t == 'core' and metric in zero_var_cols:
            return 'dropped_zero_variance'
        return t or 'unknown'
    audit['applicability_tier'] = [_tier_label(m) for m in audit.index]
    reasons = []
    for metric in audit.index:
        r = []
        if metric in absent_from_chunk:
            r.append('not_present_in_all_input_chunks')
        t = tier.get(metric)
        if t == 'high_error':
            r.append(f'error_rate_gt_{max_error_rate:g}')
        elif t == 'rare':
            r.append(f'fewer_than_{min_applicable_convos}_applicable_convos')
        elif t == 'conditional':
            r.append('conditional_set_aside_for_subset_efa')
        if metric in zero_var_cols:
            r.append('zero_variance_after_imputation')
        reasons.append(';'.join(r) if r else 'core_retained')
    audit['preprocessing_outcome'] = reasons
    audit['retained_for_pooled_efa'] = audit.index.isin(core_retained)
    audit['analyzed_as_conditional'] = audit.index.isin(conditional_metrics)
    pivot_df.attrs['preprocessing_audit'] = audit.reset_index()
    pivot_df.attrs['conditional_matrix'] = conditional_matrix
    pivot_df.attrs['preprocessing_summary'] = {
        'n_input_metrics_union': len(all_metrics),
        'n_dropped_not_common': len(absent_from_chunk),
        'n_dropped_high_error': len(high_error),
        'n_core_metrics': len(core_retained),
        'n_conditional_metrics': len(conditional_metrics),
        'n_dropped_rare': len(rare_metrics),
        'n_dropped_zero_variance': len(zero_var_cols),
        'n_values_mean_imputed': int(missing_before),
        'max_error_rate': max_error_rate,
        'core_max_na_rate': core_max_na_rate,
        'min_applicable_convos': min_applicable_convos,
    }

    return pivot_df


def surrogate_data_from_correlation(corr: pd.DataFrame, n_obs: int, seed: int = 20260825) -> pd.DataFrame:
    """Build an n_obs x k matrix whose sample Pearson correlation equals `corr` exactly."""
    values = corr.to_numpy(dtype=float)
    if values.shape[0] != values.shape[1] or list(corr.index) != list(corr.columns):
        raise ValueError('Correlation matrix must be square with matching row and column labels.')
    if n_obs <= values.shape[0]:
        raise ValueError(f'n_obs ({n_obs}) must exceed the number of variables ({values.shape[0]}).')
    eigenvalues, eigenvectors = np.linalg.eigh((values + values.T) / 2)
    if eigenvalues.min() <= 0:
        raise ValueError('Correlation matrix is not positive definite.')
    root = eigenvectors * np.sqrt(eigenvalues)  # root @ root.T == corr
    z = np.random.default_rng(seed).standard_normal((n_obs, values.shape[0]))
    q, _ = np.linalg.qr(z - z.mean(axis=0))  # orthonormal, zero-mean columns
    data = np.sqrt(n_obs - 1) * q @ root.T
    return pd.DataFrame(data, columns=pd.Index(corr.columns, name='metric'))


def load_correlation_input(corr_path: Path, audit_path: Path | None = None) -> tuple[pd.DataFrame, str]:
    """Load a published item_correlations_<tag>.csv as the analysis matrix."""
    corr_path = Path(corr_path)
    prefix = 'item_correlations_'
    if not corr_path.stem.startswith(prefix):
        raise ValueError(f'Expected a file named {prefix}<tag>.csv, got {corr_path.name}')
    tag_base = corr_path.stem[len(prefix):]
    corr = pd.read_csv(corr_path, index_col=0)

    meta_path = corr_path.with_name(f'{corr_path.stem}_meta.csv')
    if not meta_path.exists():
        raise ValueError(f'Missing {meta_path.name} (n_obs and preprocessing thresholds).')
    meta = pd.read_csv(meta_path).iloc[0]
    n_obs = int(meta['n_obs'])

    if audit_path is None:
        matches = sorted((corr_path.parent.parent / 'preprocessing').glob(
            f'preprocessing_audit_{tag_base}_*.csv'))
        if len(matches) != 1:
            raise ValueError(f'Expected one preprocessing_audit_{tag_base}_*.csv; found {len(matches)}. '
                             'Pass --preprocessing-audit.')
        audit_path = matches[0]
    audit = pd.read_csv(audit_path)
    core = audit.loc[audit['retained_for_pooled_efa'].astype(bool), 'metric_id'].tolist()
    if core != list(corr.columns):
        raise ValueError(f'{Path(audit_path).name} core metrics do not match the correlation matrix.')

    df = surrogate_data_from_correlation(corr, n_obs)
    tier = audit['applicability_tier']
    df.attrs['preprocessing_audit'] = audit
    df.attrs['preprocessing_audit_path'] = Path(audit_path)
    df.attrs['conditional_matrix'] = pd.DataFrame(index=df.index)
    df.attrs['preprocessing_summary'] = {
        'n_input_metrics_union': len(audit),
        'n_dropped_not_common': int((~audit['present_in_all_input_chunks'].astype(bool)).sum()),
        'n_dropped_high_error': int((tier == 'high_error').sum()),
        'n_core_metrics': len(core),
        'n_conditional_metrics': int(audit['analyzed_as_conditional'].astype(bool).sum()),
        'n_dropped_rare': int((tier == 'rare').sum()),
        'n_dropped_zero_variance': int((tier == 'dropped_zero_variance').sum()),
        'n_values_mean_imputed': int(audit['n_values_mean_imputed'].fillna(0).sum()),
        'max_error_rate': float(meta['max_error_rate']),
        'core_max_na_rate': float(meta['core_max_na_rate']),
        'min_applicable_convos': int(meta['min_applicable_convos']),
    }
    return df, tag_base


def check_factorability(df: pd.DataFrame) -> dict:
    """Check if the data is suitable for factor analysis using KMO and Bartlett's test."""
    # Data should already be imputed; only drop if there are still NaNs
    df_clean = df.dropna() if df.isna().any().any() else df

    if len(df_clean) < 50:
        print(f"Warning: Only {len(df_clean)} complete cases. Factor analysis may be unreliable.")

    # factor_analyzer warns whenever the high-dimensional correlation determinant is
    # numerically tiny, even when the matrix is full-rank.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message='The inverse of the variance-covariance matrix was calculated using.*',
            category=UserWarning,
        )
        kmo_all, kmo_model = calculate_kmo(df_clean)
    chi_square, p_value = calculate_bartlett_sphericity(df_clean)

    return {
        'kmo_per_variable': kmo_all,
        'kmo_overall': kmo_model,
        'bartlett_chi_square': chi_square,
        'bartlett_p_value': p_value,
        'n_complete_cases': len(df_clean),
        'n_variables': len(df_clean.columns)
    }


def save_factorability_table(factorability: dict, output_path: Path, model_label: str) -> None:
    """Write a one-row, APA-ready factorability table (KMO + Bartlett's test of sphericity)."""
    kmo = float(factorability['kmo_overall'])
    chi2 = float(factorability['bartlett_chi_square'])
    dfree = int(factorability['bartlett_df'])
    p = float(factorability['bartlett_p_value'])
    # Kaiser (1974) sampling-adequacy labels.
    interp = ('marvelous' if kmo >= 0.9 else 'meritorious' if kmo >= 0.8
              else 'middling' if kmo >= 0.7 else 'mediocre' if kmo >= 0.6
              else 'miserable' if kmo >= 0.5 else 'unacceptable')
    p_apa = '< .001' if p < 0.001 else '= ' + f'{p:.3f}'.lstrip('0')
    kmo_apa = f'{kmo:.3f}'.lstrip('0')  # bounded 0-1, so omit the leading zero
    row = {
        'judge_model': model_label,
        'n_observations': int(factorability['n_complete_cases']),
        'n_metrics': int(factorability['n_variables']),
        'kmo_overall': round(kmo, 3),
        'kmo_apa': kmo_apa,
        'kmo_interpretation': interp,
        'bartlett_chi_square': round(chi2, 2),
        'bartlett_df': dfree,
        'bartlett_p_value': p,
        'bartlett_p_apa': p_apa,
        'apa_report': f"KMO = {kmo_apa}; Bartlett's χ²({dfree:,}) = {chi2:,.2f}, p {p_apa}",
    }
    pd.DataFrame([row]).to_csv(output_path, index=False)
    print(f"Saved factorability table: {output_path}")


def render_apa_word_tables(membership_csv: Path, output_dir: Path) -> None:
    """Render the editable APA Word tables (factor membership + factorability) for this run."""
    efa_dir = Path(__file__).resolve().parent
    rscript = os.environ.get('RSCRIPT') or shutil.which('Rscript') or 'Rscript'
    env = {
        **os.environ,
        'EFA_RESULTS_DIR': str(output_dir.resolve()),
        'EFA_APA_TABLE_DIR': str((output_dir / 'apa_tables').resolve()),
        # Avoid waiting indefinitely on renv's transient sandbox lock.
        'RENV_CONFIG_SANDBOX_ENABLED': 'FALSE',
    }
    renderers = [
        # membership renderer takes the current run's CSV so it renders only this table
        (efa_dir / 'render' / 'render-apa-efa-tables.R', [str(membership_csv.resolve())]),
        # Pattern matrices are per-run; the remaining tables combine all current judge bundles
        # present in the output directory.
        (efa_dir / 'render' / 'render-apa-pattern-matrices.R', []),
        (efa_dir / 'render' / 'render-apa-factorability.R', []),
        (efa_dir / 'render' / 'render-apa-preprocessing.R', []),
        (efa_dir / 'render' / 'render-apa-metric-status.R', []),
    ]
    for script, extra in renderers:
        if not script.exists():
            continue
        try:
            subprocess.run([rscript, str(script), *extra], cwd=str(efa_dir),
                           env=env, check=True, capture_output=True, text=True,
                           timeout=120)
            print(f"Rendered APA Word table via {script.name}")
        except (subprocess.CalledProcessError, FileNotFoundError,
                subprocess.TimeoutExpired) as exc:
            detail = (getattr(exc, 'stderr', '') or str(exc)).strip().splitlines()
            reason = detail[-1] if detail else str(exc)
            print(f"Warning: {script.name} did not run ({reason[:200]}); "
                  f"CSV outputs are unaffected.")


def determine_n_factors(df: pd.DataFrame, max_factors: int = 10) -> tuple[np.ndarray, int]:
    """Determine optimal number of factors using eigenvalue analysis (Kaiser criterion)."""
    # Data should already be imputed; only drop if there are still NaNs
    df_clean = df.dropna() if df.isna().any().any() else df

    fa = FactorAnalyzer(n_factors=min(max_factors, len(df_clean.columns)), method='minres', rotation=None)
    fa.fit(df_clean)

    eigenvalues, _ = fa.get_eigenvalues()
    n_factors_kaiser = sum(eigenvalues > 1)

    return eigenvalues, n_factors_kaiser


def report_eigenvalues(eigenvalues: np.ndarray, save_csv=None) -> pd.DataFrame:
    """Print AND save the full ranked eigenvalue table (the numbers behind the scree plot)."""
    total = eigenvalues.sum()
    pct = eigenvalues / total * 100
    table = pd.DataFrame({
        'factor': range(1, len(eigenvalues) + 1),
        'eigenvalue': eigenvalues,
        'pct_variance': pct,
        'cumulative_pct_variance': pct.cumsum(),
        'above_kaiser': eigenvalues > 1,
    })

    n_kaiser = int(table['above_kaiser'].sum())
    print(f"\nEigenvalues ({n_kaiser} above Kaiser criterion of 1.0):")
    print(f"  {'Factor':>6}  {'Eigenvalue':>10}  {'% Var':>7}  {'Cum %':>7}")
    for _, r in table.iterrows():
        mark = '' if r['above_kaiser'] else '  (below 1)'
        print(f"  {int(r['factor']):>6}  {r['eigenvalue']:>10.4f}  "
              f"{r['pct_variance']:>6.2f}%  {r['cumulative_pct_variance']:>6.2f}%{mark}")

    if save_csv is not None:
        table.to_csv(save_csv, index=False)
        print(f"Saved eigenvalues: {save_csv}")
    return table


def plot_scree(eigenvalues: np.ndarray, output_path: Path):
    """Create a readable scree plot focused on the potentially retained factors."""
    n_kaiser = int(np.sum(eigenvalues > 1))
    n_display = min(len(eigenvalues), max(25, n_kaiser + 10))
    displayed = eigenvalues[:n_display]
    x = np.arange(1, n_display + 1)

    # The first eigenvalue is often much larger than the rest; a second, magnified panel makes
    # the retention-relevant elbow visible without distorting the overview.
    fig, (ax_overview, ax_detail) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True, height_ratios=[1, 1.35]
    )
    tick_positions = np.unique(np.r_[1, np.arange(5, n_display + 1, 5), n_display])
    for ax in (ax_overview, ax_detail):
        ax.plot(x, displayed, color='#355C7D', marker='o', linewidth=1.8, markersize=4.5)
        ax.axhline(y=1, color='#B64E3B', linestyle='--', linewidth=1.2,
                   label='Kaiser criterion (eigenvalue = 1)')
        ax.axvline(n_kaiser, color='#6D6D6D', linestyle=':', linewidth=1.1,
                   label=f'{n_kaiser} eigenvalues > 1')
        ax.set_xlim(0.5, n_display + 0.5)
        ax.grid(axis='y', alpha=0.25)

    ax_overview.set_ylabel('Eigenvalue')
    ax_overview.legend(frameon=False, loc='upper right')
    ax_overview.text(0.01, 0.87, 'Overview', transform=ax_overview.transAxes,
                     fontsize=9, color='#555555')

    detail_max = max(1.2, float(displayed[1]) * 1.08) if n_display > 1 else 1.2
    ax_detail.set_ylim(0, detail_max)
    ax_detail.set_xticks(tick_positions)
    ax_detail.set_xlabel('Factor number')
    ax_detail.set_ylabel('Eigenvalue')
    ax_detail.text(0.01, 0.87, 'Magnified view: factors after Factor 1',
                   transform=ax_detail.transAxes, fontsize=9, color='#555555')
    ax_detail.text(0.99, 0.03, 'Full eigenvalue spectrum is reported in the companion CSV.',
                   transform=ax_detail.transAxes, ha='right', va='bottom', fontsize=8, color='#555555')

    fig.tight_layout()
    save_png_and_pdf(fig, output_path, dpi=180)
    plt.close(fig)
    plt.close()
    print(f"Saved scree plot: {output_path}")


def plot_parallel_analysis(parallel_path: Path, output_path: Path):
    """Plot Horn's parallel analysis: observed vs simulated random eigenvalues."""
    pa = pd.read_csv(parallel_path)
    required = {
        'factor', 'observed_minres_eigenvalue', 'random_mean_minres_eigenvalue',
        'random_95th_percentile_minres_eigenvalue', 'psych_recommended_n_factors',
        'quantile_criterion', 'iterations',
    }
    missing = required - set(pa.columns)
    if missing:
        raise ValueError(f"{parallel_path} is missing columns: {sorted(missing)}")

    n_recommended = int(pa['psych_recommended_n_factors'].iloc[0])
    quantile = float(pa['quantile_criterion'].iloc[0])

    # Show the retention-relevant range; the full spectrum stays in the companion CSV.
    n_display = int(min(len(pa), max(20, n_recommended + 8)))
    d = pa.iloc[:n_display]
    x = d['factor'].to_numpy()
    observed = d['observed_minres_eigenvalue'].to_numpy()
    random_95 = d['random_95th_percentile_minres_eigenvalue'].to_numpy()
    random_mean = d['random_mean_minres_eigenvalue'].to_numpy()

    # Factor 1 dwarfs the rest, so a magnified second panel makes the crossing visible.
    fig, (ax_overview, ax_detail) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True, height_ratios=[1, 1.35]
    )
    tick_positions = np.unique(np.r_[1, np.arange(5, n_display + 1, 5), n_display])
    for ax in (ax_overview, ax_detail):
        ax.plot(x, observed, color='#355C7D', marker='o', linewidth=1.8, markersize=4.5,
                label='Observed eigenvalues (MINRES)')
        ax.plot(x, random_95, color='#B64E3B', linestyle='--', marker='^', linewidth=1.5,
                markersize=3.5,
                label=f'Simulated random data, {quantile:.0%} percentile')
        ax.plot(x, random_mean, color='#B9A08A', linestyle=':', linewidth=1.3,
                label='Simulated random data, mean')
        ax.axvline(n_recommended, color='#6D6D6D', linestyle=':', linewidth=1.1,
                   label=f'Parallel analysis retains {n_recommended} factors')
        ax.set_xlim(0.5, n_display + 0.5)
        ax.grid(axis='y', alpha=0.25)

    ax_overview.set_ylabel('Eigenvalue')
    ax_overview.legend(frameon=False, loc='upper right', fontsize=8.5)
    ax_overview.text(0.01, 0.87, 'Overview', transform=ax_overview.transAxes,
                     fontsize=9, color='#555555')

    # Scale the detail panel to the crossing region, not to the leading eigenvalues: the point
    # of this panel is to show where observed drops below simulated.
    crossing_scale = float(max(random_95.max(), observed[n_recommended - 1])) if n_recommended >= 1 else 1.0
    detail_max = max(1.5, crossing_scale * 2.0)
    ax_detail.set_ylim(0, detail_max)
    ax_detail.set_xticks(tick_positions)
    ax_detail.set_xlabel('Factor number')
    ax_detail.set_ylabel('Eigenvalue')
    ax_detail.text(0.01, 0.90, 'Magnified view: the retention crossing',
                   transform=ax_detail.transAxes, fontsize=9, color='#555555')

    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.text(0.5, 0.010,
             'Factors are retained while the observed eigenvalue exceeds the simulated '
             f'{quantile:.0%} percentile.\n'
             'Leading eigenvalues are clipped in the lower panel; the full spectrum is in '
             'the companion CSV.',
             ha='center', va='bottom', fontsize=8, color='#555555', linespacing=1.4)
    save_png_and_pdf(fig, output_path, dpi=180)
    plt.close(fig)
    plt.close()
    print(f"Saved parallel analysis plot ({n_recommended} factors retained): {output_path}")


def run_efa(df: pd.DataFrame, n_factors: int, rotation: str = 'oblimin') -> FactorAnalyzer:
    """Fit the Python factor_analyzer cross-check solution (R psych is authoritative)."""
    df_clean = df.dropna()
    fa = FactorAnalyzer(n_factors=n_factors, method='minres', rotation=rotation)
    fa.fit(df_clean)

    # factor_analyzer 0.5.1 sorts pattern and structure columns by pattern SS after oblique
    # rotation but leaves phi_ in its pre-sort order.
    if getattr(fa, 'phi_', None) is not None and getattr(fa, 'structure_', None) is not None:
        pattern = np.asarray(fa.loadings_, dtype=float)
        structure = np.asarray(fa.structure_, dtype=float)
        initial_error = float(np.max(np.abs(pattern @ fa.phi_ - structure)))
        repair_applied = initial_error > 1e-8
        if repair_applied:
            repaired_phi, *_ = np.linalg.lstsq(pattern, structure, rcond=None)
            repaired_phi = (repaired_phi + repaired_phi.T) / 2
            fa.phi_ = repaired_phi
        final_error = float(np.max(np.abs(pattern @ fa.phi_ - structure)))
        if final_error > 1e-8:
            raise ValueError(
                f"Could not reconcile factor_analyzer pattern, Phi, and structure matrices "
                f"(max error {final_error:.3g})."
            )
        fa._phi_order_repair_applied = repair_applied
        fa._phi_order_initial_structure_error = initial_error
        fa._phi_order_final_structure_error = final_error
    return fa


class PsychEfaSolution:
    """Minimal fitted-solution interface backed entirely by R psych outputs."""

    def __init__(self, pattern, phi, structure, communalities):
        self.loadings_ = np.asarray(pattern, dtype=float)
        self.phi_ = np.asarray(phi, dtype=float)
        self.structure_ = np.asarray(structure, dtype=float)
        self.communalities_ = np.asarray(communalities, dtype=float)
        self.primary_engine = 'R psych::fa'

    def get_factor_variance(self):
        """Return pattern SS, SS/p, and cumulative SS/p for display compatibility."""
        pattern_ss = np.sum(self.loadings_ ** 2, axis=0)
        ss_div_p = pattern_ss / self.loadings_.shape[0]
        return pattern_ss, ss_div_p, np.cumsum(ss_div_p)


def efa_model_matrices(fa: FactorAnalyzer) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return pattern, factor-correlation, structure, and common-covariance matrices."""
    pattern = np.asarray(fa.loadings_, dtype=float)
    phi = getattr(fa, 'phi_', None)
    if phi is None:
        phi = np.eye(pattern.shape[1])
    else:
        phi = np.asarray(phi, dtype=float)
    structure = pattern @ phi
    library_structure = getattr(fa, 'structure_', None)
    if library_structure is not None and not np.allclose(
        structure, library_structure, atol=1e-8, rtol=1e-8
    ):
        raise ValueError(
            "The fitted pattern, Phi, and structure matrices are mathematically inconsistent."
        )
    common_covariance = pattern @ phi @ pattern.T
    library_communalities = getattr(fa, 'communalities_', None)
    if library_communalities is not None and not np.allclose(
        np.diag(common_covariance), library_communalities, atol=1e-8, rtol=1e-8
    ):
        raise ValueError(
            "The fitted communalities do not equal diag(pattern @ Phi @ pattern.T)."
        )
    return pattern, phi, structure, common_covariance


def get_loadings_df(fa: FactorAnalyzer, variable_names: list) -> pd.DataFrame:
    """Extract factor loadings as a DataFrame."""
    loadings = fa.loadings_
    factor_names = [f'Factor_{i+1}' for i in range(loadings.shape[1])]
    return pd.DataFrame(loadings, index=variable_names, columns=factor_names)


def run_psych_primary_efa(
    df: pd.DataFrame,
    python_fa: FactorAnalyzer,
    n_factors: int,
    threshold: float,
    iterations: int,
    parallel_output: Path,
    reliability_output: Path,
    r_pattern_output: Path,
    r_phi_output: Path,
    r_structure_output: Path,
    factorability_output: Path,
    congruence_output: Path,
    sl_loadings_output: Path,
    sl_summary_output: Path,
    sl_omega_by_factor_output: Path,
    sl_group_mapping_output: Path,
    rotation: str | None,
) -> dict:
    """Fit the authoritative EFA in R psych and compare it with Python."""
    script = Path(__file__).with_name('psych-diagnostics.R')
    with tempfile.TemporaryDirectory(prefix='efa-psych-') as temp_dir:
        temp = Path(temp_dir)
        data_path = temp / 'scores.csv'
        df.to_csv(data_path, index=False)
        rscript = os.environ.get('RSCRIPT') or shutil.which('Rscript') or 'Rscript'
        # Every path handed to R is absolute so the call is independent of the working
        # directory; we set cwd to the script's directory (the renv project root) so R sources
        # efa/.Rprofile and runs against the pinned renv library.
        command = [
            rscript, str(script.resolve()),
            '--data', str(data_path.resolve()),
            '--n-factors', str(n_factors),
            '--parallel-output', str(parallel_output.resolve()),
            '--reliability-output', str(reliability_output.resolve()),
            '--pattern-output', str(r_pattern_output.resolve()),
            '--phi-output', str(r_phi_output.resolve()),
            '--structure-output', str(r_structure_output.resolve()),
            '--factorability-output', str(factorability_output.resolve()),
            '--sl-loadings-output', str(sl_loadings_output.resolve()),
            '--sl-summary-output', str(sl_summary_output.resolve()),
            '--sl-omega-by-factor-output', str(sl_omega_by_factor_output.resolve()),
            '--threshold', str(threshold),
            '--iterations', str(iterations),
            '--rotation', rotation or 'none',
        ]
        r_env = os.environ.copy()
        # renv's macOS sandbox activation uses a process lock that can strand later analyses
        # after an interrupted R session.
        r_env.setdefault('RENV_CONFIG_SANDBOX_ENABLED', 'FALSE')
        subprocess.run(command, check=True, cwd=str(script.parent), env=r_env)
    parallel = pd.read_csv(parallel_output)
    n_parallel = int(parallel['psych_recommended_n_factors'].iloc[0])

    r_table = pd.read_csv(r_pattern_output).set_index('metric_id').reindex(df.columns)
    r_factor_columns = [c for c in r_table if c.startswith('Factor_')]
    if len(r_factor_columns) != n_factors or r_table[r_factor_columns].isna().any().any():
        raise ValueError('R psych pattern output is incomplete or has unexpected factor columns.')
    r_loadings = r_table[r_factor_columns].to_numpy(dtype=float)
    r_phi_table = pd.read_csv(r_phi_output, index_col=0).loc[
        r_factor_columns, r_factor_columns
    ]
    r_structure_table = pd.read_csv(r_structure_output, index_col=0).loc[
        df.columns, r_factor_columns
    ]
    r_phi = r_phi_table.to_numpy(dtype=float)
    r_structure = r_structure_table.to_numpy(dtype=float)
    r_h2 = r_table['communality_h2_psych'].to_numpy(dtype=float)
    solution = PsychEfaSolution(r_loadings, r_phi, r_structure, r_h2)

    # Match the R-primary factors to Python-check factors by maximum absolute Tucker
    # congruence.
    python_loadings_df = get_loadings_df(python_fa, list(df.columns))
    python_loadings = python_loadings_df.to_numpy()
    numerators = r_loadings.T @ python_loadings
    denominators = np.sqrt(
        np.outer(np.sum(r_loadings ** 2, axis=0), np.sum(python_loadings ** 2, axis=0))
    )
    congruence = np.divide(
        numerators, denominators,
        out=np.full_like(numerators, np.nan, dtype=float),
        where=denominators > 0,
    )
    r_index, python_index = linear_sum_assignment(-np.abs(congruence))
    congruence_table = pd.DataFrame({
        'psych_r_primary_factor': [r_factor_columns[i] for i in r_index],
        'python_factor_analyzer_check_factor': [python_loadings_df.columns[j] for j in python_index],
        'tucker_congruence_signed': [congruence[i, j] for i, j in zip(r_index, python_index)],
        'tucker_congruence_absolute': [abs(congruence[i, j]) for i, j in zip(r_index, python_index)],
    }).sort_values('psych_r_primary_factor', key=lambda s: s.str.extract(r'(\d+)')[0].astype(int))
    congruence_table.to_csv(congruence_output, index=False)

    # psych::omega numbers its residual group factors independently from psych::fa.
    sl_table = pd.read_csv(sl_loadings_output).set_index('metric_id').reindex(df.columns)
    sl_group_columns = [c for c in sl_table if c.startswith('F') and c.endswith('*')]
    if len(sl_group_columns) != n_factors:
        raise ValueError('Unexpected number of Schmid-Leiman group-factor columns.')
    sl_group_loadings = sl_table[sl_group_columns].to_numpy(dtype=float)
    sl_numerators = r_loadings.T @ sl_group_loadings
    sl_denominators = np.sqrt(
        np.outer(np.sum(r_loadings ** 2, axis=0), np.sum(sl_group_loadings ** 2, axis=0))
    )
    sl_congruence = np.divide(
        sl_numerators, sl_denominators,
        out=np.full_like(sl_numerators, np.nan, dtype=float),
        where=sl_denominators > 0,
    )
    primary_index, sl_index = linear_sum_assignment(-np.abs(sl_congruence))
    sl_mapping = pd.DataFrame({
        'psych_r_primary_factor': [r_factor_columns[i] for i in primary_index],
        'schmid_leiman_group_factor': [sl_group_columns[j] for j in sl_index],
        'tucker_congruence_signed': [
            sl_congruence[i, j] for i, j in zip(primary_index, sl_index)
        ],
        'tucker_congruence_absolute': [
            abs(sl_congruence[i, j]) for i, j in zip(primary_index, sl_index)
        ],
    }).sort_values(
        'psych_r_primary_factor', key=lambda s: s.str.extract(r'(\d+)')[0].astype(int)
    )
    sl_mapping.to_csv(sl_group_mapping_output, index=False)

    omega_by_factor = pd.read_csv(sl_omega_by_factor_output)
    mapping_lookup = sl_mapping.set_index('schmid_leiman_group_factor')[
        'psych_r_primary_factor'
    ]
    omega_by_factor['matched_psych_r_primary_factor'] = (
        omega_by_factor['schmid_leiman_factor'].map(mapping_lookup)
    )
    omega_by_factor.to_csv(sl_omega_by_factor_output, index=False)

    return {
        'solution': solution,
        'loadings_df': pd.DataFrame(
            r_loadings, index=df.columns, columns=r_factor_columns
        ),
        'n_parallel': n_parallel,
        'r_loadings_table': r_table,
        'r_factorability': pd.read_csv(factorability_output).iloc[0].to_dict(),
        'congruence': congruence_table,
        'python_loadings_df': python_loadings_df,
        'sl_loadings_table': sl_table,
        'sl_summary': pd.read_csv(sl_summary_output).iloc[0].to_dict(),
        'sl_omega_by_factor': omega_by_factor,
        'sl_group_mapping': sl_mapping,
    }


def report_factor_correlations(fa: FactorAnalyzer, save_csv: Path = None, threshold: float = 0.32) -> pd.DataFrame:
    """Report the inter-factor correlation matrix from an oblique solution."""
    phi = getattr(fa, 'phi_', None)
    if phi is None:
        print("\nFactor correlations: none — orthogonal rotation forces factors independent. "
              "Run an oblique rotation (--rotation oblimin/promax) to apply the .32 test.")
        return None

    n = phi.shape[0]
    names = [f'Factor_{i+1}' for i in range(n)]
    corr = pd.DataFrame(phi, index=names, columns=names)

    # Every unique factor pair (strict upper triangle) with its correlation, strongest first.
    iu = np.triu_indices(n, k=1)
    pairs = sorted(
        ((names[i], names[j], float(phi[i, j])) for i, j in zip(*iu)),
        key=lambda p: abs(p[2]), reverse=True,
    )
    over = [p for p in pairs if abs(p[2]) >= threshold]
    max_abs = abs(pairs[0][2]) if pairs else 0.0

    print(f"\n{'='*60}")
    print("FACTOR CORRELATIONS (oblique) — |r| >= .32 highlighted")
    print(f"{'='*60}")
    print(f"  {'':>9}" + ''.join(f"{c:>9}" for c in names))
    for i, r in enumerate(names):
        print(f"  {r:>9}" + ''.join(f"{corr.iloc[i, j]:>9.3f}" for j in range(n)))

    print(f"\n  Max |off-diagonal correlation|: {max_abs:.3f}")
    print(f"  Pairs with |r| >= {threshold}: {len(over)} of {len(pairs)}")
    for a, b, r in over:
        print(f"    {r:+.3f}  {a} ~ {b}")
    if over:
        print(f"  → The prespecified oblique rotation is empirically supported: the pair(s) "
              f"above share roughly >=10% variance (|r| >= {threshold}).")
    else:
        print(f"  → All factor correlations < {threshold} (<10% shared variance). This does not "
              f"invalidate oblique rotation; consider an orthogonal sensitivity analysis only "
              f"if theory also supports factor independence.")

    if save_csv is not None:
        corr.to_csv(save_csv)
        print(f"  Saved factor correlations: {save_csv}")
    return corr


def plot_factor_correlations(corr: pd.DataFrame, output_path: Path, threshold: float = 0.32):
    """Plot an annotated, lower-triangle inter-factor correlation matrix."""
    n = len(corr)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.7), max(5, n * 0.7)))

    # Since the decision rule uses |r|, encode magnitude rather than sign.
    abs_corr = np.abs(corr.values)
    display_abs_corr = np.round(abs_corr, 2)
    masked = np.ma.masked_where(np.triu(np.ones((n, n), dtype=bool)), display_abs_corr)
    bounds = [0, threshold, 0.50, 1.000001]
    band_colors = ['#FFFFFF', '#F6E7A1', '#E98B73']
    cmap = ListedColormap(band_colors)
    cmap.set_bad('white')
    norm = BoundaryNorm(bounds, cmap.N)
    ax.imshow(masked, cmap=cmap, norm=norm)

    ax.set_xticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(corr.index, fontsize=9)

    legend_handles = [
        Patch(facecolor=band_colors[0], edgecolor='grey', label=f'|r| < {threshold:.2f}'),
        Patch(facecolor=band_colors[1], edgecolor='grey', label=f'|r| {threshold:.2f}–.49'),
        Patch(facecolor=band_colors[2], edgecolor='grey', label='|r| ≥ .50'),
    ]
    ax.legend(handles=legend_handles, title='Correlation strength', loc='upper left',
              bbox_to_anchor=(1.02, 1), frameon=False, borderaxespad=0,
              handlelength=1, handleheight=1)

    # Uniform table dividers on every cell boundary (minor ticks at the half-integers), rather
    # than individual boxes — reads as a table, not a patchwork of outlines.
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which='minor', color='grey', linewidth=0.8)
    ax.tick_params(which='minor', length=0)

    for i in range(n):
        for j in range(n):
            if j > i:
                continue
            if j == i:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       facecolor='#E5E7EB', edgecolor='none'))
                ax.text(j, i, '–', ha='center', va='center', fontsize=8, color='black')
                continue
            val = corr.iloc[i, j]
            # Use the displayed two-decimal value for both color and bolding, so a cell shown
            # as .32 or .50 cannot visually fall into the band below that displayed cutoff.
            over = round(abs(val), 2) >= threshold
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8,
                    fontweight='bold' if over else 'normal',
                    color='black')

    # Reserve explicit margins for long factor labels and the external legend.
    fig.subplots_adjust(left=0.18, right=0.76, bottom=0.18, top=0.90)
    save_png_and_pdf(fig, output_path, dpi=150, bbox_inches='tight', pad_inches=0.16)
    plt.close(fig)
    print(f"Saved factor correlation heatmap: {output_path}")


def report_high_loadings(loadings_df: pd.DataFrame, threshold: float = 0.4, save_csv: Path = None):
    """Print AND save the factor -> questions view: each factor and its high-loading metrics."""
    print(f"\n{'='*60}")
    print(f"HIGH LOADINGS (|loading| >= {threshold})")
    print(f"{'='*60}")

    rows = []
    pattern_ss = loadings_df.pow(2).sum(axis=0)
    pattern_ss_pct = pattern_ss / len(loadings_df) * 100
    for factor in loadings_df.columns:
        print(f"\n{factor} (pattern SS={pattern_ss[factor]:.3f}; "
              f"SS/p={pattern_ss_pct[factor]:.2f}%):")
        factor_loadings = loadings_df[factor].abs().sort_values(ascending=False)
        high_loadings = factor_loadings[factor_loadings >= threshold]

        for var in high_loadings.index:
            loading = loadings_df.loc[var, factor]
            sign = '+' if loading > 0 else '-'
            print(f"  {sign}{abs(loading):.3f}  {var}")
            rows.append({
                'factor': factor,
                'pattern_ss_sum_squared_loadings': float(pattern_ss[factor]),
                'relative_loading_strength_pct_ss_div_p': float(pattern_ss_pct[factor]),
                'metric_id': var,
                'loading': float(loading),
            })

        if len(high_loadings) == 0:
            print("  (no loadings above threshold)")

    if save_csv is not None:
        pd.DataFrame(rows).to_csv(save_csv, index=False)
        print(f"\nSaved factor membership: {save_csv}")


def save_factor_summary(loadings_df: pd.DataFrame, output_path: Path, threshold: float = 0.4):
    """Save one row per factor, including factors with no salient pattern loading."""
    pattern_ss = loadings_df.pow(2).sum(axis=0)
    summary = pd.DataFrame({
        'factor': loadings_df.columns,
        'pattern_ss_sum_squared_loadings': pattern_ss.values,
        'number_of_items_p': len(loadings_df),
        'relative_loading_strength_pct_ss_div_p':
            (pattern_ss.values / len(loadings_df)) * 100,
        'n_loadings_at_threshold': [
            int((loadings_df[factor].abs() >= threshold).sum())
            for factor in loadings_df.columns
        ],
        'max_abs_loading': [
            float(loadings_df[factor].abs().max())
            for factor in loadings_df.columns
        ],
    })
    summary.to_csv(output_path, index=False)
    print(f"Saved factor summary: {output_path}")


def save_apa_efa_outputs(
    df: pd.DataFrame,
    fa: PsychEfaSolution,
    loadings_df: pd.DataFrame,
    factorability: dict,
    python_factorability: dict,
    python_fa: FactorAnalyzer,
    psych_results: dict,
    eigenvalues: np.ndarray,
    n_factors_kaiser: int,
    output_dir: Path,
    tag: str,
    rotation: str | None,
    threshold: float,
    parallel_iterations: int,
):
    """Write the complete reporting set needed for an APA-style EFA results section."""
    p = df.shape[1]
    pattern, phi, structure, common_covariance = efa_model_matrices(fa)
    h2_values = np.diag(common_covariance)
    h2 = pd.Series(h2_values, index=loadings_df.index, name='communality_h2')
    uniqueness = 1.0 - h2_values
    if np.any(h2_values < -1e-8) or np.any(h2_values > 1 + 1e-8):
        bad = loadings_df.index[(h2_values < -1e-8) | (h2_values > 1 + 1e-8)].tolist()
        raise ValueError(
            f"Improper EFA solution: oblique-model communalities outside [0, 1] for {bad}. "
            "Do not report this solution."
        )

    structure_df = pd.DataFrame(
        structure, index=loadings_df.index, columns=loadings_df.columns
    )
    structure_df.to_csv(opath(output_dir, f'factor_structure_matrix_{tag}.csv'))
    abs_loadings = loadings_df.abs()
    primary_factor = abs_loadings.idxmax(axis=1)
    primary_loading = np.array([
        loadings_df.loc[item, factor] for item, factor in primary_factor.items()
    ])
    n_salient = (abs_loadings >= threshold).sum(axis=1)
    sorted_abs = np.sort(abs_loadings.to_numpy(), axis=1)[:, ::-1]
    secondary_abs = sorted_abs[:, 1] if loadings_df.shape[1] > 1 else np.zeros(len(loadings_df))
    primary_abs = abs_loadings.max(axis=1).to_numpy()
    pattern_report = pd.DataFrame({
        'metric_id': loadings_df.index,
        'communality_h2': h2.values,
        'primary_factor': primary_factor.values,
        'primary_pattern_loading': primary_loading,
        'primary_abs_loading': primary_abs,
        'largest_secondary_abs_loading': secondary_abs,
        'primary_secondary_difference': primary_abs - secondary_abs,
        'n_salient_loadings_at_threshold': n_salient.values,
        'cross_loading_at_threshold': n_salient.values >= 2,
        'no_salient_loading_at_threshold': n_salient.values == 0,
    })
    # The full, unsuppressed pattern matrix is preserved in factor_loadings_*.csv; this
    # manuscript-ready version intentionally blanks trivial coefficients.
    for factor in loadings_df.columns:
        pattern_report[f'{factor}_pattern_loading'] = loadings_df[factor].values
        pattern_report[f'{factor}_display_if_abs_ge_{threshold:g}'] = loadings_df[factor].where(
            loadings_df[factor].abs() >= threshold
        ).values
    pattern_report.to_csv(opath(output_dir, f'pattern_matrix_report_{tag}.csv'), index=False)

    congruence_path = opath(output_dir, f'cross_library_congruence_{tag}.csv')
    n_parallel = psych_results['n_parallel']
    congruence_values = psych_results['congruence']['tucker_congruence_absolute']

    _, _, _, python_common_covariance = efa_model_matrices(python_fa)
    python_h2 = np.diag(python_common_covariance)
    h2_difference = h2_values - python_h2
    h2_correlation = float(np.corrcoef(h2_values, python_h2)[0, 1])
    h2_mae = float(np.mean(np.abs(h2_difference)))

    observed_corr = df.corr().to_numpy()
    reproduced_corr = common_covariance.copy()
    np.fill_diagonal(reproduced_corr, h2_values + uniqueness)
    residual_corr = observed_corr - reproduced_corr
    upper = np.triu_indices(p, k=1)
    residual_rms = float(np.sqrt(np.mean(residual_corr[upper] ** 2)))
    residual_max_abs = float(np.max(np.abs(residual_corr[upper])))
    corr_eigenvalues = np.linalg.eigvalsh(observed_corr)
    positive_eigenvalues = corr_eigenvalues[corr_eigenvalues > 1e-12]
    corr_condition_number = (
        float(corr_eigenvalues[-1] / positive_eigenvalues[0])
        if len(positive_eigenvalues) else float('inf')
    )
    phi_eigenvalues = np.linalg.eigvalsh(phi)
    r_factorability = psych_results['r_factorability']
    sl_summary = psych_results['sl_summary']

    diagnostics = pd.DataFrame([{
        'n_observations': len(df),
        'n_metrics': p,
        'fitted_n_factors': loadings_df.shape[1],
        'authoritative_efa_engine': 'R psych::fa',
        'python_check_engine': 'Python factor_analyzer',
        'python_check_phi_order_repair_applied': getattr(
            python_fa, '_phi_order_repair_applied', False
        ),
        'python_check_phi_structure_error_before_repair': getattr(
            python_fa, '_phi_order_initial_structure_error', 0.0
        ),
        'python_check_phi_structure_error_after_repair': getattr(
            python_fa, '_phi_order_final_structure_error', 0.0
        ),
        'correlation_matrix_rank': int(np.linalg.matrix_rank(observed_corr)),
        'correlation_matrix_min_eigenvalue': float(corr_eigenvalues[0]),
        'correlation_matrix_condition_number_positive_spectrum': corr_condition_number,
        'n_item_pairs_abs_correlation_ge_0_95': int(np.sum(np.abs(observed_corr[upper]) >= 0.95)),
        'factor_correlation_matrix_min_eigenvalue': float(phi_eigenvalues[0]),
        'factor_correlation_matrix_max_abs_symmetry_error': float(np.max(np.abs(phi - phi.T))),
        'factor_correlation_matrix_max_abs_diagonal_error': float(np.max(np.abs(np.diag(phi) - 1))),
        'communality_min_oblique_model': float(np.min(h2_values)),
        'communality_max_oblique_model': float(np.max(h2_values)),
        'n_heywood_communalities_gt_1': int(np.sum(h2_values > 1 + 1e-8)),
        'off_diagonal_residual_correlation_rms': residual_rms,
        'off_diagonal_residual_correlation_max_abs': residual_max_abs,
        'r_primary_python_check_pattern_tucker_congruence_min_abs': float(congruence_values.min()),
        'r_primary_python_check_pattern_tucker_congruence_median_abs': float(congruence_values.median()),
        'r_primary_python_check_oblique_communality_correlation': h2_correlation,
        'r_primary_python_check_oblique_communality_mae': h2_mae,
        'kmo_overall_psych_r_primary': r_factorability['kmo_overall_psych'],
        'kmo_overall_factor_analyzer_check': python_factorability['kmo_overall'],
        'kmo_absolute_difference': abs(
            python_factorability['kmo_overall'] - r_factorability['kmo_overall_psych']
        ),
        'bartlett_chi_square_psych_r_primary': r_factorability['bartlett_chi_square_psych'],
        'bartlett_chi_square_factor_analyzer_check': python_factorability['bartlett_chi_square'],
        'bartlett_chi_square_relative_difference': abs(
            python_factorability['bartlett_chi_square'] - r_factorability['bartlett_chi_square_psych']
        ) / abs(r_factorability['bartlett_chi_square_psych']),
        'schmid_leiman_omega_hierarchical': sl_summary['omega_hierarchical'],
        'schmid_leiman_omega_total': sl_summary['omega_total'],
        'schmid_leiman_ecv_general': sl_summary['ecv_general'],
        'schmid_leiman_general_loading_mean': sl_summary['general_loading_mean'],
        'schmid_leiman_general_loading_min': sl_summary['general_loading_min'],
        'schmid_leiman_general_loading_max': sl_summary['general_loading_max'],
    }])
    diagnostics_path = opath(output_dir, f'model_diagnostics_{tag}.csv')
    diagnostics.to_csv(diagnostics_path, index=False)

    validation_rows = [
        ('eigenvalues_sum_to_number_of_metrics', abs(float(np.sum(eigenvalues)) - p) < 1e-8,
         float(np.sum(eigenvalues)), f'within 1e-8 of {p}'),
        ('factor_correlation_matrix_is_symmetric', np.max(np.abs(phi - phi.T)) < 1e-10,
         float(np.max(np.abs(phi - phi.T))), '< 1e-10'),
        ('factor_correlation_diagonal_is_one', np.max(np.abs(np.diag(phi) - 1)) < 1e-10,
         float(np.max(np.abs(np.diag(phi) - 1))), '< 1e-10'),
        ('factor_correlation_matrix_is_positive_definite', phi_eigenvalues[0] > 0,
         float(phi_eigenvalues[0]), '> 0'),
        ('communalities_are_in_unit_interval', bool(np.all((h2_values >= -1e-8) & (h2_values <= 1 + 1e-8))),
         f'{np.min(h2_values):.6g} to {np.max(h2_values):.6g}', '[0, 1] within 1e-8'),
        # Cross-library KMO/Bartlett agreement.
        ('python_r_kmo_agree', diagnostics.iloc[0]['kmo_absolute_difference'] < 1e-3,
         float(diagnostics.iloc[0]['kmo_absolute_difference']), '< 1e-3'),
        ('python_r_bartlett_agree', diagnostics.iloc[0]['bartlett_chi_square_relative_difference'] < 1e-3,
         float(diagnostics.iloc[0]['bartlett_chi_square_relative_difference']), '< 1e-3 relative'),
        ('r_primary_python_check_factor_patterns_are_congruent', congruence_values.min() >= 0.85,
         float(congruence_values.min()), 'minimum absolute Tucker congruence >= 0.85'),
        ('r_primary_python_check_communalities_agree', h2_correlation >= 0.95 and h2_mae <= 0.05,
         f'r={h2_correlation:.6g}; MAE={h2_mae:.6g}', 'r >= 0.95 and MAE <= 0.05'),
        ('schmid_leiman_omega_coefficients_are_ordered_and_bounded',
         0 <= sl_summary['omega_hierarchical'] <= sl_summary['omega_total'] <= 1,
         f"omega_h={sl_summary['omega_hierarchical']:.6g}; omega_total={sl_summary['omega_total']:.6g}",
         '0 <= omega_h <= omega_total <= 1'),
        ('schmid_leiman_ecv_is_bounded', 0 <= sl_summary['ecv_general'] <= 1,
         float(sl_summary['ecv_general']), '[0, 1]'),
        ('schmid_leiman_group_mapping_is_complete',
         len(psych_results['sl_group_mapping']) == loadings_df.shape[1],
         len(psych_results['sl_group_mapping']), loadings_df.shape[1]),
    ]
    validation = pd.DataFrame(
        validation_rows, columns=['validation_check', 'passed', 'observed', 'criterion']
    )
    validation_path = opath(output_dir, f'validation_report_{tag}.csv')
    validation.to_csv(validation_path, index=False)
    failed_validations = validation.loc[~validation['passed'], 'validation_check'].tolist()
    if failed_validations:
        raise ValueError(
            f"EFA validation failed: {failed_validations}. Inspect {validation_path} and do not "
            "treat this output bundle as final."
        )

    rotated_ss = fa.get_factor_variance()
    initial_eigenvalue_variance_pct = float(np.sum(eigenvalues[:loadings_df.shape[1]]) / p * 100)
    preprocessing = df.attrs.get('preprocessing_summary', {})
    metadata = pd.DataFrame([{
        'n_observations': len(df),
        'n_metrics': p,
        'n_input_metrics_union': preprocessing.get('n_input_metrics_union'),
        'n_dropped_not_common_across_chunks': preprocessing.get('n_dropped_not_common'),
        'n_dropped_high_error': preprocessing.get('n_dropped_high_error'),
        'n_core_metrics': preprocessing.get('n_core_metrics'),
        'n_conditional_metrics_set_aside': preprocessing.get('n_conditional_metrics'),
        'n_dropped_rare_below_min_applicable': preprocessing.get('n_dropped_rare'),
        'n_dropped_zero_variance': preprocessing.get('n_dropped_zero_variance'),
        'n_values_mean_imputed': preprocessing.get('n_values_mean_imputed'),
        'max_error_rate': preprocessing.get('max_error_rate'),
        'core_max_na_rate': preprocessing.get('core_max_na_rate'),
        'min_applicable_convos': preprocessing.get('min_applicable_convos'),
        'correlation_type': 'Pearson product-moment correlation',
        'missing_data_handling': (
            f"Core metrics (<= {preprocessing.get('core_max_na_rate', 0.1) * 100:g}% non-scored) "
            f"analyzed pooled with remaining gaps mean-imputed (sklearn SimpleImputer); metrics "
            f"scored in >= {preprocessing.get('min_applicable_convos', 200)} conversations but "
            f"exceeding that non-scored rate held for conditional extension analysis; rarer metrics dropped"
        ),
        'authoritative_efa_engine': 'R psych::fa',
        'extraction_method': 'MINRES (R psych::fa, fm="minres")',
        'python_cross_check': 'factor_analyzer FactorAnalyzer(method="minres")',
        'rotation': rotation or 'none',
        'oblimin_gamma': 0 if rotation == 'oblimin' else np.nan,
        'python_check_factor_analyzer_phi_order_repair': (
            'Applied to the Python cross-check only and validated against structure_ '
            'because factor_analyzer 0.5.1 does not reorder phi_ with pattern/structure columns'
            if getattr(python_fa, '_phi_order_repair_applied', False)
            else 'Not required'
        ),
        'loading_matrix_reported': 'Pattern matrix' if rotation in {'oblimin', 'promax'} else 'Rotated loading matrix',
        'loading_suppression_absolute_threshold': threshold,
        'kmo_overall': factorability['kmo_overall'],
        'bartlett_chi_square': factorability['bartlett_chi_square'],
        'bartlett_df': factorability['bartlett_df'],
        'bartlett_p_value': factorability['bartlett_p_value'],
        'kaiser_n_factors_eigenvalue_gt_1': n_factors_kaiser,
        'parallel_analysis_n_factors': n_parallel,
        'parallel_analysis_method': 'psych::fa.parallel MINRES; observed factor eigenvalues compared with simulated random-data 95th percentiles',
        'parallel_analysis_quantile': 0.95,
        'parallel_analysis_iterations': parallel_iterations,
        'fitted_n_factors': loadings_df.shape[1],
        'schmid_leiman_method': (
            'R psych::omega; MINRES; oblimin; flip=TRUE; full retained metric matrix'
        ),
        'schmid_leiman_omega_hierarchical': sl_summary['omega_hierarchical'],
        'schmid_leiman_omega_total': sl_summary['omega_total'],
        'schmid_leiman_ecv_general': sl_summary['ecv_general'],
        'rotated_pattern_ss_sum': float(np.sum(rotated_ss[0])),
        'rotated_pattern_ss_div_p_sum': float(rotated_ss[2][-1]),
        'initial_correlation_eigenvalue_pct_sum_first_m_factors': initial_eigenvalue_variance_pct,
        'communality_definition': 'diag(pattern @ factor_correlation @ pattern.T)',
        'structure_matrix_definition': 'pattern @ factor_correlation',
        'variance_interpretation_note': (
            'For oblique rotation, rotated pattern SS/p is not unique or additive variance explained.'
            if rotation in {'oblimin', 'promax'} else
            'Orthogonal rotated loading SS/p is reported as the model output.'
        ),
        'factor_analyzer_version': package_version('factor_analyzer'),
        'psych_version': (
            subprocess.check_output([
                os.environ.get('RSCRIPT') or shutil.which('Rscript') or 'Rscript',
                '-e', 'cat(as.character(packageVersion("psych")))',
            ], text=True, cwd=str(Path(__file__).resolve().parent), env={
                **os.environ,
                # This short metadata lookup must use the same pinned R library as the fit and
                # must not wait on renv's transient sandbox lock.
                'RENV_CONFIG_SANDBOX_ENABLED': 'FALSE',
            }, timeout=30).strip()
        ),
        'numpy_version': np.__version__,
        'pandas_version': pd.__version__,
        'scipy_version': package_version('scipy'),
        'scikit_learn_version': package_version('scikit-learn'),
    }])
    metadata.to_csv(opath(output_dir, f'efa_reporting_metadata_{tag}.csv'), index=False)

    print(f"Saved APA EFA reporting metadata: {opath(output_dir, f'efa_reporting_metadata_{tag}.csv')}")
    print(f"Saved pattern matrix / communalities / cross-loadings: {opath(output_dir, f'pattern_matrix_report_{tag}.csv')}")
    print(f"Saved factor reliability (psych::alpha / psych::omega): "
          f"{opath(output_dir, f'factor_reliability_{tag}.csv')}")
    print(f"Saved structure matrix: {opath(output_dir, f'factor_structure_matrix_{tag}.csv')}")
    print(f"Saved R-primary/Python-check cross-library comparison: {congruence_path}")
    print(f"Saved mathematical validation report: {validation_path}")
    print(f"Saved parallel analysis (psych::fa.parallel; {parallel_iterations} simulations; "
          f"{n_parallel} factors retained): {opath(output_dir, f'parallel_analysis_{tag}.csv')}")


def plot_loadings_heatmap(loadings_df: pd.DataFrame, output_path: Path):
    """Create heatmap of factor loadings."""
    # Sort by max absolute loading
    max_loadings = loadings_df.abs().max(axis=1)
    sorted_df = loadings_df.loc[max_loadings.sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(12, max(8, len(sorted_df) * 0.35)))
    im = ax.imshow(sorted_df.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)

    ax.set_xticks(range(len(sorted_df.columns)))
    ax.set_xticklabels(sorted_df.columns, fontsize=10)
    ax.set_yticks(range(len(sorted_df.index)))
    ax.set_yticklabels(sorted_df.index, fontsize=9)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Factor Loading', fontsize=10)

    # Add text for significant loadings
    for i in range(len(sorted_df.index)):
        for j in range(len(sorted_df.columns)):
            val = sorted_df.iloc[i, j]
            if abs(val) >= 0.3:
                color = 'white' if abs(val) > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8, color=color)

    plt.tight_layout()
    save_png_and_pdf(plt.gcf(), output_path, dpi=150)
    plt.close()
    print(f"Saved loadings heatmap: {output_path}")


def plot_factor_membership_table(loadings_df: pd.DataFrame, output_path: Path, threshold: float = 0.4):
    """Render the factor -> questions view as a styled table image."""
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    # This renderer deliberately uses positioned patches and text instead of ``Axes.table``.
    blocks = []
    pattern_ss = loadings_df.pow(2).sum(axis=0)
    pattern_ss_pct = pattern_ss / len(loadings_df) * 100
    metric_categories = (
        'Overall_Rating', 'Accuracy', 'Alignment_to_Constraints',
        'Instructional_Support', 'Assessment', 'Mistake_Handling',
        'Affect_and_Relational_Support', 'Engagement_and_Motivation',
        'Adaptivity', 'Understanding_Learner_Goals', 'Metacognition', 'Safety',
    )

    def readable_metric_label(metric_id: str) -> str:
        """Turn a metric ID into a compact category — metric label for the image only."""
        for category in metric_categories:
            prefix = f'{category}_'
            if metric_id.startswith(prefix):
                category_label = category.replace('_', ' ')
                metric_label = metric_id[len(prefix):].replace('__', ' — ').replace('_', ' ')
                return f'{category_label} — {metric_label}'
        return metric_id.replace('__', ' — ').replace('_', ' ')

    for factor in loadings_df.columns:
        col = loadings_df[factor]
        members = col[col.abs() >= threshold].sort_values(key=lambda s: s.abs(), ascending=False)
        if len(members):
            blocks.append((
                factor.replace('_', ' '), str(len(members)), f'{pattern_ss_pct[factor]:.2f}%',
                [(f'{val:+.2f}', readable_metric_label(metric), val) for metric, val in members.items()],
            ))

    if not blocks:
        print("  (no loadings above threshold — skipping membership table)")
        return

    # Membership is based on absolute loading magnitude.
    norm = Normalize(vmin=threshold, vmax=1)
    cmap = LinearSegmentedColormap.from_list(
        'loading_strength', ['#FFF7F2', '#F2B09A', '#DF7767']
    )

    # Dimensions below are the reference HTML's pixels (at 96 dpi): 46px title, 42px metadata,
    # 36px headings/items, and a 16px inter-factor gap.
    banner_px, metadata_px, row_px, gap_px = 46, 42, 36, 16
    total_px = sum(banner_px + metadata_px + row_px + row_px * len(rows) for *_, rows in blocks)
    total_px += gap_px * (len(blocks) - 1)
    fig, ax = plt.subplots(figsize=(7.92, max(2.0, total_px / 96)))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.99, bottom=0.01)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, total_px)
    ax.invert_yaxis()
    ax.axis('off')

    border, header, metadata, column_bg = '#B8B8B8', '#303030', '#F1F2F3', '#FAFAFA'
    text, muted, divider = '#202124', '#666A70', '#A7AAAE'
    left, width, split = 0.0, 1.0, 105 / 760
    y = 0
    item_row = 0

    def box(x, top, box_width, height, facecolor, edgecolor=border, linewidth=1):
        ax.add_patch(Rectangle((x, top), box_width, height, facecolor=facecolor,
                               edgecolor=edgecolor, linewidth=linewidth, clip_on=False))

    def label(x, top, string, size, weight=400, color=text, ha='left'):
        ax.text(x, top, string, ha=ha, va='center', fontsize=size * 72 / 96,
                fontweight=weight, fontfamily='Inter', color=color)

    for block_index, (factor_title, n_items, strength, rows) in enumerate(blocks):
        # Full-width, left-aligned banner (11px/18px CSS padding in the reference).
        box(left, y, width, banner_px, header)
        label(left + 18 / 760, y + banner_px / 2, factor_title, 16, 700, 'white')
        y += banner_px

        # Metadata row: its divider aligns exactly with the Loading/Metric divider.
        box(left, y, width, metadata_px, metadata)
        ax.plot([split, split], [y, y + metadata_px], color=divider, linewidth=1)
        label(left + 13 / 760, y + metadata_px / 2, 'ITEMS', 12, 600, muted)
        label(left + 13 / 760 + 52 / 760, y + metadata_px / 2, n_items, 14, 700)
        meta_x = split + 13 / 760
        label(meta_x, y + metadata_px / 2, 'RELATIVE LOADING STRENGTH', 12, 600, muted)
        # Approximate the reference's flex gap after the longest label at its native width.
        label(meta_x + 206 / 760, y + metadata_px / 2, strength, 14, 700)
        y += metadata_px

        box(left, y, width, row_px, column_bg)
        ax.plot([split, split], [y, y + row_px], color=divider, linewidth=1)
        label(split / 2, y + row_px / 2, 'LOADING', 11, 700, '#55585D', ha='center')
        label(split + 13 / 760, y + row_px / 2, 'METRIC', 11, 700, '#55585D')
        y += row_px

        for loading_text, metric, value in rows:
            box(left, y, split, row_px, cmap(norm(abs(value))))
            box(split, y, 1 - split, row_px,
                '#F7F7F7' if item_row % 2 else 'white')
            label(split / 2, y + row_px / 2, loading_text, 13, 700, '#1F1F1F', ha='center')
            label(split + 13 / 760, y + row_px / 2, metric, 13, 400)
            y += row_px
            item_row += 1

        if block_index < len(blocks) - 1:
            y += gap_px

    save_png_and_pdf(plt.gcf(), output_path, dpi=150)
    plt.close()
    print(f"Saved factor membership table: {output_path}")


def validate_saved_output_bundle(
    output_dir: Path,
    tag: str,
    df: pd.DataFrame,
    fa: FactorAnalyzer,
    loadings_df: pd.DataFrame,
    threshold: float,
) -> Path:
    """Re-read the saved bundle and verify that every derivative matches its source."""
    checks = []

    def add(name, passed, observed, criterion):
        checks.append((name, bool(passed), observed, criterion))

    loadings_path = opath(output_dir, f'factor_loadings_{tag}.csv')
    saved_loadings = pd.read_csv(loadings_path, index_col=0)
    add(
        'saved_loadings_equal_r_psych_primary_pattern_matrix',
        saved_loadings.shape == loadings_df.shape and np.allclose(saved_loadings, loadings_df),
        str(saved_loadings.shape), str(loadings_df.shape),
    )

    r_pattern_path = opath(output_dir, f'psych_r_pattern_matrix_{tag}.csv')
    r_pattern = pd.read_csv(r_pattern_path).set_index('metric_id').loc[
        loadings_df.index, loadings_df.columns
    ]
    add('final_loadings_equal_raw_r_psych_export', np.allclose(r_pattern, loadings_df),
        float(np.max(np.abs(r_pattern.to_numpy() - loadings_df.to_numpy()))),
        'max absolute difference < numerical tolerance')

    structure_path = opath(output_dir, f'factor_structure_matrix_{tag}.csv')
    saved_structure = pd.read_csv(structure_path, index_col=0)
    _, phi, structure, common = efa_model_matrices(fa)
    add(
        'saved_structure_equals_pattern_times_phi',
        saved_structure.shape == structure.shape and np.allclose(saved_structure, structure),
        str(saved_structure.shape), str(structure.shape),
    )
    r_structure_path = opath(output_dir, f'psych_r_structure_matrix_{tag}.csv')
    raw_r_structure = pd.read_csv(r_structure_path, index_col=0).loc[
        loadings_df.index, loadings_df.columns
    ]
    add('final_structure_equals_raw_r_psych_export', np.allclose(raw_r_structure, saved_structure),
        float(np.max(np.abs(raw_r_structure.to_numpy() - saved_structure.to_numpy()))),
        'max absolute difference < numerical tolerance')

    pattern_path = opath(output_dir, f'pattern_matrix_report_{tag}.csv')
    pattern_report = pd.read_csv(pattern_path).set_index('metric_id')
    saved_h2 = pattern_report.loc[loadings_df.index, 'communality_h2'].to_numpy()
    expected_h2 = np.diag(common)
    add(
        'saved_communalities_equal_diag_pattern_phi_pattern_transpose',
        np.allclose(saved_h2, expected_h2),
        float(np.max(np.abs(saved_h2 - expected_h2))), 'max absolute difference < numerical tolerance',
    )
    pattern_columns_match = all(
        np.allclose(
            pattern_report.loc[loadings_df.index, f'{factor}_pattern_loading'],
            loadings_df[factor],
        )
        for factor in loadings_df.columns
    )
    add('pattern_report_coefficients_equal_loadings_csv', pattern_columns_match,
        pattern_columns_match, True)

    summary_path = opath(output_dir, f'factor_summary_{tag}.csv')
    summary = pd.read_csv(summary_path).set_index('factor')
    expected_ss = (loadings_df ** 2).sum(axis=0)
    expected_counts = (loadings_df.abs() >= threshold).sum(axis=0)
    add(
        'factor_summary_pattern_ss_matches_loadings',
        np.allclose(summary.loc[loadings_df.columns, 'pattern_ss_sum_squared_loadings'], expected_ss),
        float(np.max(np.abs(
            summary.loc[loadings_df.columns, 'pattern_ss_sum_squared_loadings'].to_numpy()
            - expected_ss.to_numpy()
        ))), 'max absolute difference < numerical tolerance',
    )
    add(
        'factor_summary_salient_counts_match_loadings',
        np.array_equal(
            summary.loc[loadings_df.columns, 'n_loadings_at_threshold'].to_numpy(),
            expected_counts.to_numpy(),
        ),
        summary['n_loadings_at_threshold'].sum(), int(expected_counts.sum()),
    )

    membership_path = opath(output_dir, f'factor_membership_{tag}.csv')
    membership = pd.read_csv(membership_path)
    expected_membership = {
        (factor, metric): float(loadings_df.loc[metric, factor])
        for factor in loadings_df.columns
        for metric in loadings_df.index
        if abs(loadings_df.loc[metric, factor]) >= threshold
    }
    saved_membership = {
        (row.factor, row.metric_id): float(row.loading)
        for row in membership.itertuples(index=False)
    }
    membership_keys_match = saved_membership.keys() == expected_membership.keys()
    membership_values_match = membership_keys_match and all(
        np.isclose(saved_membership[key], value)
        for key, value in expected_membership.items()
    )
    add('membership_rows_equal_all_salient_pattern_coefficients', membership_values_match,
        len(saved_membership), len(expected_membership))

    correlation_path = opath(output_dir, f'factor_correlations_{tag}.csv')
    if getattr(fa, 'phi_', None) is not None:
        saved_phi = pd.read_csv(correlation_path, index_col=0).to_numpy()
        add('saved_factor_correlations_equal_r_psych_primary_phi', np.allclose(saved_phi, phi),
            float(np.max(np.abs(saved_phi - phi))), 'max absolute difference < numerical tolerance')
        r_phi_path = opath(output_dir, f'psych_r_factor_correlations_{tag}.csv')
        raw_r_phi = pd.read_csv(r_phi_path, index_col=0).loc[
            loadings_df.columns, loadings_df.columns
        ].to_numpy()
        add('final_factor_correlations_equal_raw_r_psych_export', np.allclose(raw_r_phi, saved_phi),
            float(np.max(np.abs(raw_r_phi - saved_phi))),
            'max absolute difference < numerical tolerance')

    eigen_path = opath(output_dir, f'eigenvalues_{tag}.csv')
    eigen_table = pd.read_csv(eigen_path)
    add('eigenvalue_table_has_one_row_per_metric', len(eigen_table) == df.shape[1],
        len(eigen_table), df.shape[1])
    add('eigenvalue_percentages_sum_to_100', np.isclose(eigen_table['pct_variance'].sum(), 100),
        float(eigen_table['pct_variance'].sum()), 100)

    metadata_path = opath(output_dir, f'efa_reporting_metadata_{tag}.csv')
    metadata = pd.read_csv(metadata_path).iloc[0]
    add('metadata_dimensions_match_analysis_matrix',
        int(metadata.n_observations) == len(df) and int(metadata.n_metrics) == df.shape[1],
        f"{int(metadata.n_observations)} x {int(metadata.n_metrics)}",
        f"{len(df)} x {df.shape[1]}")
    add('metadata_fitted_factor_count_matches_loadings',
        int(metadata.fitted_n_factors) == loadings_df.shape[1],
        int(metadata.fitted_n_factors), loadings_df.shape[1])
    add('metadata_identifies_r_psych_as_authoritative_engine',
        metadata.authoritative_efa_engine == 'R psych::fa',
        metadata.authoritative_efa_engine, 'R psych::fa')

    sl_loadings_path = opath(output_dir, f'schmid_leiman_loadings_{tag}.csv')
    sl_loadings = pd.read_csv(sl_loadings_path)
    add('schmid_leiman_has_one_loading_row_per_analysis_metric',
        len(sl_loadings) == df.shape[1] and set(sl_loadings.metric_id) == set(df.columns),
        len(sl_loadings), df.shape[1])
    add('schmid_leiman_general_loadings_are_finite',
        np.isfinite(sl_loadings['g']).all(),
        int(np.isfinite(sl_loadings['g']).sum()), df.shape[1])
    sl_mapping_path = opath(output_dir, f'schmid_leiman_group_mapping_{tag}.csv')
    sl_mapping = pd.read_csv(sl_mapping_path)
    add('schmid_leiman_group_mapping_is_one_to_one',
        len(sl_mapping) == loadings_df.shape[1]
        and sl_mapping['psych_r_primary_factor'].nunique() == loadings_df.shape[1]
        and sl_mapping['schmid_leiman_group_factor'].nunique() == loadings_df.shape[1],
        len(sl_mapping), loadings_df.shape[1])

    preprocessing_path = opath(output_dir, f'preprocessing_audit_{tag}.csv')
    preprocessing = pd.read_csv(preprocessing_path)
    add('preprocessing_retained_count_matches_analysis_matrix',
        int(preprocessing['retained_for_pooled_efa'].sum()) == df.shape[1],
        int(preprocessing['retained_for_pooled_efa'].sum()), df.shape[1])

    expected_files = [
        preprocessing_path, eigen_path, opath(output_dir, f'scree_plot_{tag}.png'),
        membership_path, opath(output_dir, f'factor_membership_{tag}.png'), summary_path,
        pattern_path, opath(output_dir, f'parallel_analysis_{tag}.csv'),
        opath(output_dir, f'factor_reliability_{tag}.csv'), structure_path,
        r_pattern_path, r_structure_path,
        opath(output_dir, f'psych_r_factor_correlations_{tag}.csv'),
        opath(output_dir, f'factorability_psych_r_{tag}.csv'),
        opath(output_dir, f'cross_library_congruence_{tag}.csv'),
        sl_loadings_path,
        opath(output_dir, f'schmid_leiman_summary_{tag}.csv'),
        opath(output_dir, f'schmid_leiman_omega_by_factor_{tag}.csv'),
        sl_mapping_path,
        opath(output_dir, f'model_diagnostics_{tag}.csv'),
        opath(output_dir, f'validation_report_{tag}.csv'), metadata_path,
        loadings_path, opath(output_dir, f'factor_loadings_heatmap_{tag}.png'),
    ]
    if getattr(fa, 'phi_', None) is not None:
        expected_files.extend([
            correlation_path, opath(output_dir, f'factor_correlations_heatmap_{tag}.png')
        ])
    missing_or_empty = [str(path) for path in expected_files if not path.exists() or path.stat().st_size == 0]
    add('all_expected_output_files_exist_and_are_nonempty', not missing_or_empty,
        ';'.join(missing_or_empty) if missing_or_empty else 'all present', 'all present')

    validation_path = opath(output_dir, f'validation_report_{tag}.csv')
    previous = pd.read_csv(validation_path)
    bundle_checks = pd.DataFrame(
        checks, columns=['validation_check', 'passed', 'observed', 'criterion']
    )
    # Replace prior bundle-level rows when validation is rerun after regenerating a derivative
    # file; retain the independent mathematical checks from save_apa_efa_outputs.
    previous = previous[~previous['validation_check'].isin(bundle_checks['validation_check'])]
    validation = pd.concat([previous, bundle_checks], ignore_index=True)
    validation.to_csv(validation_path, index=False)

    manifest_rows = []
    for path in expected_files:
        if path.exists():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest_rows.append({
                'filename': path.name,
                'size_bytes': path.stat().st_size,
                'sha256': digest,
            })
    manifest_path = opath(output_dir, f'output_manifest_{tag}.csv')
    pd.DataFrame(manifest_rows).sort_values('filename').to_csv(manifest_path, index=False)

    failed = validation.loc[~validation['passed'], 'validation_check'].tolist()
    if failed:
        raise ValueError(
            f"Saved EFA bundle failed validation: {failed}. Inspect {validation_path}."
        )
    print(f"Validated {len(expected_files)} output files; manifest: {manifest_path}")
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description='Run EFA on AI Tutor evaluation results')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--input', type=str, nargs='+',
                        help='Evaluation results CSV(s) — one file, or several chunk files to '
                             'concatenate (e.g. --input results/efa_*_batch_*.csv, shell-expanded)')
    source.add_argument('--correlation', type=str,
                        help='Published item_correlations_<tag>.csv to refit instead of raw scores '
                             '(reads the sibling _meta.csv and the matching preprocessing audit)')
    parser.add_argument('--preprocessing-audit', type=str, default=None,
                        help='With --correlation: preprocessing audit CSV (default: found by tag)')
    parser.add_argument('--n-factors', type=int, default=None, help='Number of factors (default: auto via Kaiser)')
    parser.add_argument('--rotation', type=str, default='oblimin',
                        choices=['varimax', 'promax', 'oblimin', 'none'],
                        help='Rotation method (default: oblimin, oblique). Use varimax for orthogonal.')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory (default: same as input)')
    parser.add_argument('--threshold', type=float, default=0.4, help='Loading threshold for reporting (default: 0.4)')
    parser.add_argument('--max-error-rate', type=float, default=0.20,
                        help='Drop metrics with more than this proportion of error statuses (default: 0.20)')
    parser.add_argument('--core-max-na-rate', type=float, default=0.10,
                        help='Max non-scored rate (incl. N/A) for a metric to stay in the pooled EFA '
                             'and be mean-imputed (default: 0.10). Metrics above this but scored in '
                             '>= --min-applicable-convos conversations are held for conditional extension analysis.')
    parser.add_argument('--min-applicable-convos', type=int, default=200,
                        help='Min scored conversations for a conditional metric to be analyzable in a '
                             'conditional extension analysis; below this it is dropped as rare (default: 200)')
    parser.add_argument('--parallel-iterations', type=int, default=100,
                        help='Iterations for psych::fa.parallel (default: 100; must be positive)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else efa_result_dir('factor_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    rotation = None if args.rotation == 'none' else args.rotation

    # Tag outputs with the input's model/timestamp (like the result CSVs) so analysis
    # artifacts identify their source and don't overwrite each other across runs.
    if args.correlation:
        tag = Path(args.correlation).stem[len('item_correlations_'):]
    else:
        input_paths = [Path(p) for p in args.input]  # one file, or several chunks (shell-expanded glob)
        tag = input_paths[0].stem.replace('efa_evaluation_results_', '')
        if len(input_paths) > 1:
            tag += f"_merged{len(input_paths)}"
    tag += f"_{rotation or 'unrotated'}"
    # Preserve the fitted factor count in explicitly constrained solutions so that, for
    # example, a parallel-analysis rerun cannot be confused with the Kaiser run.
    if args.n_factors is not None:
        tag += f"_n{args.n_factors}"

    if not 0 <= args.max_error_rate < 1:
        parser.error('--max-error-rate must be in [0, 1)')
    if not 0 <= args.core_max_na_rate < 1:
        parser.error('--core-max-na-rate must be in [0, 1)')
    if args.min_applicable_convos < 1:
        parser.error('--min-applicable-convos must be a positive integer')
    if args.parallel_iterations <= 0:
        parser.error('--parallel-iterations must be greater than 0 for the R-primary pipeline')

    # Load and prepare data
    if args.correlation:
        print(f"Loading correlation matrix {args.correlation}...")
        df, _ = load_correlation_input(
            Path(args.correlation),
            audit_path=Path(args.preprocessing_audit) if args.preprocessing_audit else None,
        )
    else:
        print(f"Loading data from {len(input_paths)} file(s)...")
        df = load_and_pivot_results(
            input_paths,
            max_error_rate=args.max_error_rate,
            core_max_na_rate=args.core_max_na_rate,
            min_applicable_convos=args.min_applicable_convos,
        )
    preprocessing_audit_path = opath(output_dir, f'preprocessing_audit_{tag}.csv')
    source_audit = df.attrs.get('preprocessing_audit_path')
    if source_audit is None:
        df.attrs['preprocessing_audit'].to_csv(preprocessing_audit_path, index=False)
    elif source_audit.resolve() != preprocessing_audit_path.resolve():
        shutil.copyfile(source_audit, preprocessing_audit_path)  # byte-for-byte copy of the input
    print(f"Saved preprocessing audit: {preprocessing_audit_path}")
    print(f"Loaded {len(df)} conversations with {len(df.columns)} metrics")

    # These Python values are retained only as a cross-check.
    print("\nComputing Python factorability cross-check...")
    python_factorability = check_factorability(df)

    # Determine number of factors
    print("\nAnalyzing eigenvalues...")
    eigenvalues, n_factors_kaiser = determine_n_factors(df)
    print(f"  Kaiser criterion suggests {n_factors_kaiser} factors")

    # Report the full ranked eigenvalue table (print + CSV) and save the scree plot
    report_eigenvalues(eigenvalues, save_csv=opath(output_dir, f'eigenvalues_{tag}.csv'))
    plot_scree(eigenvalues, opath(output_dir, f'scree_plot_{tag}.png'))

    # Python is fitted only as an independent check.
    n_factors = args.n_factors if args.n_factors else n_factors_kaiser
    print(f"\nRunning Python factor_analyzer cross-check with {n_factors} factors, "
          f"{rotation or 'no'} rotation...")
    python_fa = run_efa(df, n_factors, rotation)

    print(f"\nRunning authoritative R psych::fa EFA with {n_factors} factors, "
          f"{rotation or 'no'} rotation...")
    psych_results = run_psych_primary_efa(
        df=df,
        python_fa=python_fa,
        n_factors=n_factors,
        threshold=args.threshold,
        iterations=args.parallel_iterations,
        parallel_output=opath(output_dir, f'parallel_analysis_{tag}.csv'),
        reliability_output=opath(output_dir, f'factor_reliability_{tag}.csv'),
        r_pattern_output=opath(output_dir, f'psych_r_pattern_matrix_{tag}.csv'),
        r_phi_output=opath(output_dir, f'psych_r_factor_correlations_{tag}.csv'),
        r_structure_output=opath(output_dir, f'psych_r_structure_matrix_{tag}.csv'),
        factorability_output=opath(output_dir, f'factorability_psych_r_{tag}.csv'),
        congruence_output=opath(output_dir, f'cross_library_congruence_{tag}.csv'),
        sl_loadings_output=opath(output_dir, f'schmid_leiman_loadings_{tag}.csv'),
        sl_summary_output=opath(output_dir, f'schmid_leiman_summary_{tag}.csv'),
        sl_omega_by_factor_output=opath(output_dir, f'schmid_leiman_omega_by_factor_{tag}.csv'),
        sl_group_mapping_output=opath(output_dir, f'schmid_leiman_group_mapping_{tag}.csv'),
        rotation=rotation,
    )

    # Parallel analysis is the operative retention criterion, so plot it now that
    # psych::fa.parallel has written its observed/simulated eigenvalues.
    plot_parallel_analysis(
        opath(output_dir, f'parallel_analysis_{tag}.csv'),
        opath(output_dir, f'parallel_analysis_plot_{tag}.png'),
    )

    fa = psych_results['solution']
    loadings_df = psych_results['loadings_df']
    r_factorability = psych_results['r_factorability']
    factorability = {
        'kmo_overall': float(r_factorability['kmo_overall_psych']),
        'bartlett_chi_square': float(r_factorability['bartlett_chi_square_psych']),
        'bartlett_df': int(r_factorability['bartlett_df_psych']),
        'bartlett_p_value': float(r_factorability['bartlett_p_value_psych']),
        'n_complete_cases': len(df),
        'n_variables': len(df.columns),
    }
    kmo = factorability['kmo_overall']
    print(f"  Authoritative R KMO overall: {kmo:.3f}", end='')
    print(" (good)" if kmo >= 0.8 else " (acceptable)" if kmo >= 0.6 else " (poor)")
    print(f"  Authoritative R Bartlett's p-value: {factorability['bartlett_p_value']:.2e}")
    print(f"  Authoritative R Bartlett's chi-square: {factorability['bartlett_chi_square']:.2f} "
          f"(df={factorability['bartlett_df']})")
    # Model name for the table: strip the batch id / timestamp / rotation-count run tag.
    model_label = re.sub(r'(_batch_.*|_\d{8}_\d{6}.*)$', '', tag) or tag
    save_factorability_table(
        factorability, opath(output_dir, f'factorability_table_{tag}.csv'), model_label)
    print(f"  Minimum absolute R/Python Tucker congruence: "
          f"{psych_results['congruence']['tucker_congruence_absolute'].min():.3f}")

    variance = fa.get_factor_variance()
    print("\nR psych rotated pattern-loading SS summary:")
    print(f"  Pattern SS per factor: {variance[0].round(3)}")
    print(f"  Pattern SS/p: {variance[1].round(3)}")
    print(f"  Sum of pattern SS/p: {variance[2][-1]:.1%}")
    if rotation in {'oblimin', 'promax'}:
        print("  Note: for oblique rotation, these are not unique/additive shares of explained "
              "variance because the factors correlate.")

    factor_corr = report_factor_correlations(fa, save_csv=opath(output_dir, f'factor_correlations_{tag}.csv'))
    if factor_corr is not None:
        plot_factor_correlations(factor_corr, opath(output_dir, f'factor_correlations_heatmap_{tag}.png'))

    report_high_loadings(loadings_df, threshold=args.threshold,
                         save_csv=opath(output_dir, f'factor_membership_{tag}.csv'))
    save_factor_summary(loadings_df, opath(output_dir, f'factor_summary_{tag}.csv'),
                        threshold=args.threshold)
    save_apa_efa_outputs(
        df, fa, loadings_df, factorability, python_factorability, python_fa,
        psych_results, eigenvalues, n_factors_kaiser, output_dir, tag, rotation,
        args.threshold, args.parallel_iterations,
    )

    # Save outputs
    plot_loadings_heatmap(loadings_df, opath(output_dir, f'factor_loadings_heatmap_{tag}.png'))
    plot_factor_membership_table(loadings_df, opath(output_dir, f'factor_membership_{tag}.png'),
                                 threshold=args.threshold)

    loadings_csv = opath(output_dir, f'factor_loadings_{tag}.csv')
    loadings_df.to_csv(loadings_csv)
    print(f"\nSaved: {loadings_csv}")

    validate_saved_output_bundle(
        output_dir, tag, df, fa, loadings_df, args.threshold
    )

    # Presentation layer: editable APA Word tables (does not affect the validated bundle).
    render_apa_word_tables(opath(output_dir, f'factor_membership_{tag}.csv'), output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
