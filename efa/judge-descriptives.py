#!/usr/bin/env python3
"""Per-judge item distribution descriptives (reporting companion to the EFA pipeline)."""

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times', 'Nimbus Roman', 'Times New Roman', 'Liberation Serif'],
    'mathtext.fontset': 'stix',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

sys.path.insert(0, str(Path(__file__).parent.parent))  # for evaluation_utils
from evaluation_utils import efa_result_dir

# analyze-efa.py has a hyphen, so import its preprocessing via importlib.
_spec = importlib.util.spec_from_file_location(
    'analyze_efa', str(Path(__file__).with_name('analyze-efa.py'))
)
_efa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_efa)

SCALE_MIN, SCALE_MAX = 1, 5
SCALE_POINTS = list(range(SCALE_MIN, SCALE_MAX + 1))

# --- Figure palette ------------------------------------------------------------- The 1-5
# judge scale is an ordered (Likert-type) scale, so the response distribution takes a
# DIVERGING ramp centred on the neutral midpoint (3), not categorical hues: blue <-> red poles
# with a gray midpoint.
DIVERGING = {
    1: '#d04241',   # L .583  - low pole, dark
    2: '#ef9694',   # L .763  - low pole, light
    3: '#f0efec',   # L .952  - neutral midpoint
    4: '#86b6ef',   # L .764  - high pole, light
    5: '#2a78d6',   # L .575  - high pole, dark
}
SURFACE = '#fcfcfb'
INK_PRIMARY = '#0b0b0b'
INK_SECONDARY = '#52514e'
INK_MUTED = '#898781'
GRIDLINE = '#e1e0d9'
ACCENT = '#2a78d6'
# Categorical slots 1-3 of the reference palette, for the grouped form where judges are a
# nominal series.
CATEGORICAL = ['#2a78d6', '#eb6834', '#1baf7a']


def fmt_count(n: float, pct: float) -> str:
    """Bar label: raw response count with its share in parentheses, on two lines."""
    return f'{int(round(n)):,}\n({pct:.0f}%)'

# Judge identity comes from the axis row, so panel B needs no second hue.
DISPLAY_NAMES = {
    'gpt-5.4-mini': 'GPT-5.4 mini',
    'claude-haiku-4-5': 'Haiku 4.5',
    'gemma4-26b': 'Gemma 4 26B',
}


def display_name(judge_model: str) -> str:
    """Map a run's judge label onto the short name used in the paper."""
    for key, pretty in DISPLAY_NAMES.items():
        if judge_model.startswith(key):
            return pretty
    return judge_model


def item_descriptives(df: pd.DataFrame) -> pd.DataFrame:
    """Per-metric distribution statistics over scored rows only."""
    scored = df[df['score_status'] == 'scored']
    g = scored.groupby('metric_id')['score']

    out = pd.DataFrame({
        'n_scored_values': g.size(),
        'mean': g.mean(),
        'sd': g.std(ddof=1),
        'median': g.median(),
        'min': g.min(),
        'max': g.max(),
        # pandas .skew()/.kurt() are the sample (G1/G2) estimators; kurtosis is EXCESS, so the
        # Curran et al. threshold of 7 applies directly with no adjustment.
        'skewness': g.skew(),
        'kurtosis_excess': g.apply(pd.Series.kurt),
    })

    # Full response distribution.
    counts = (
        scored.groupby(['metric_id', 'score']).size()
        .unstack(fill_value=0)
        .reindex(columns=SCALE_POINTS, fill_value=0)
    )
    pct = counts.div(counts.sum(axis=1), axis=0) * 100
    for point in SCALE_POINTS:
        out[f'n_{point}'] = counts[point]
        out[f'pct_{point}'] = pct[point]

    out['pct_floor'] = pct[SCALE_MIN]
    out['pct_ceiling'] = pct[SCALE_MAX]
    # How much of the scale the judge actually exercises on this item.
    out['n_scale_points_used'] = (counts > 0).sum(axis=1)

    return out


def judge_summary(items: pd.DataFrame, judge_label: str,
                  skew_cut: float, kurt_cut: float,
                  ceiling_cut: float,
                  status_counts: dict[str, int] | None = None) -> pd.DataFrame:
    """Collapse the per-item table into one row describing this judge's scale use."""
    n = len(items)
    # Response-weighted overall distribution: how this judge used the scale across every
    # scored response, not the average of per-item percentages (which would weight a rarely-
    # applicable item the same as a universal one).
    weights = items['n_scored_values']
    overall_pct = {
        f'overall_pct_{point}': (items[f'pct_{point}'] * weights).sum() / weights.sum()
        for point in SCALE_POINTS
    }
    # Raw response counts behind those shares, so figures can report N and % together.
    overall_pct.update({
        f'overall_n_{point}': int(items[f'n_{point}'].sum()) for point in SCALE_POINTS
    })

    grand_mean = (items['mean'] * weights).sum() / weights.sum()
    counts = np.array([overall_pct[f'overall_n_{p}'] for p in SCALE_POINTS], float)
    points = np.array(SCALE_POINTS, float)
    total = counts.sum()
    pooled_sd = float(np.sqrt((counts * (points - grand_mean) ** 2).sum() / (total - 1)))
    # Pooled shape statistics for the response distribution as a whole, the same distribution
    # M and SD describe.
    _m2 = (counts * (points - grand_mean) ** 2).sum() / total
    _m3 = (counts * (points - grand_mean) ** 3).sum() / total
    _m4 = (counts * (points - grand_mean) ** 4).sum() / total
    pooled_skewness = float(_m3 / _m2 ** 1.5)
    pooled_kurtosis_excess = float(_m4 / _m2 ** 2 - 3)

    status_counts = status_counts or {}
    n_judgments = sum(status_counts.values()) or None

    row = {
        'judge_model': judge_label,
        'pooled_sd': pooled_sd,
        'pooled_skewness': pooled_skewness,
        'pooled_kurtosis_excess': pooled_kurtosis_excess,
        'n_judgments_total': n_judgments,
        'n_not_applicable': status_counts.get('not_applicable', 0),
        'n_error': status_counts.get('error', 0),
        'pct_not_applicable': (
            status_counts.get('not_applicable', 0) / n_judgments * 100
            if n_judgments else float('nan')
        ),
        'n_metrics': n,
        'n_scored_values_total': int(weights.sum()),
        'grand_mean': (items['mean'] * weights).sum() / weights.sum(),
        'mean_item_sd': items['sd'].mean(),
        'median_item_sd': items['sd'].median(),
        'median_skewness': items['skewness'].median(),
        'median_kurtosis_excess': items['kurtosis_excess'].median(),
        'pct_items_abs_skew_gt_cut': (items['skewness'].abs() > skew_cut).mean() * 100,
        'pct_items_kurtosis_gt_cut': (items['kurtosis_excess'] > kurt_cut).mean() * 100,
        'pct_items_nonnormal_either': (
            (items['skewness'].abs() > skew_cut) | (items['kurtosis_excess'] > kurt_cut)
        ).mean() * 100,
        # Threshold-free ceiling statistic: the median item's share of top scores.
        'median_pct_ceiling': items['pct_ceiling'].median(),
        'pct_items_ceiling_gt_cut': (items['pct_ceiling'] > ceiling_cut).mean() * 100,
        'pct_items_le_2_scale_points': (items['n_scale_points_used'] <= 2).mean() * 100,
        'skew_cut': skew_cut,
        'kurtosis_cut': kurt_cut,
        'ceiling_cut_pct': ceiling_cut,
    }
    row.update(overall_pct)
    return pd.DataFrame([row])


def _save(fig, output_stem: Path) -> None:
    """Write both raster and vector copies (PDF for LaTeX, PNG for quick viewing)."""
    for ext in ('png', 'pdf'):
        fig.savefig(f'{output_stem}.{ext}', dpi=200, facecolor=SURFACE,
                    bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote figure -> {output_stem}.png / .pdf')


def plot_small_multiples(summary: pd.DataFrame, output_stem: Path) -> None:
    """One panel per judge, stacked vertically, over the ordered 1-5 scale."""
    summary = summary.sort_values('grand_mean').reset_index(drop=True)
    n = len(summary)

    fig, axes = plt.subplots(n, 1, figsize=(3.4, 1.52 * n + 0.6), sharex=True)
    fig.patch.set_facecolor(SURFACE)
    ymax = max(summary[f'overall_pct_{p}'].max() for p in SCALE_POINTS) * 1.22

    for ax, (_, row) in zip(axes, summary.iterrows()):
        pct = [row[f'overall_pct_{p}'] for p in SCALE_POINTS]
        ax.bar(SCALE_POINTS, pct, width=0.68, color=ACCENT, zorder=3)
        for point, value in zip(SCALE_POINTS, pct):
            ax.text(point, value + ymax * 0.03, f'{value:.0f}%', ha='center',
                    va='bottom', fontsize=7.5, color=INK_SECONDARY, zorder=4)
        ax.set_title(
            f"{display_name(row['judge_model'])}\n"
            f"($M$ = {row['grand_mean']:.2f}, $SD$ = {row['pooled_sd']:.2f}, "
            f"$N$ = {int(row['n_scored_values_total']):,})",
            fontsize=8.5, color=INK_PRIMARY, loc='center', pad=4, linespacing=1.35)
        ax.set_ylim(0, ymax)
        ax.set_ylabel('% of responses', fontsize=7.5, color=INK_SECONDARY)
        ax.set_facecolor(SURFACE)
        ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color('#c3c2b7')
        ax.set_xticks(SCALE_POINTS)
        ax.set_xticklabels([str(p) for p in SCALE_POINTS], fontsize=8)
        ax.tick_params(colors=INK_MUTED, labelsize=8, length=0, labelbottom=True)

    axes[-1].set_xlabel('Judge score (1 = lowest, 5 = highest)',
                        fontsize=8, color=INK_SECONDARY)
    fig.tight_layout()
    _save(fig, output_stem)


def plot_grouped(summary: pd.DataFrame, output_stem: Path) -> None:
    """All three judges side by side at each score point."""
    summary = summary.sort_values('grand_mean').reset_index(drop=True)
    x = list(range(len(SCALE_POINTS)))
    width = 0.26

    fig, ax = plt.subplots(figsize=(6.5, 2.2))
    fig.patch.set_facecolor(SURFACE)
    ymax = max(summary[f'overall_pct_{p}'].max() for p in SCALE_POINTS) * 1.55

    for i, (_, row) in enumerate(summary.iterrows()):
        pct = [row[f'overall_pct_{p}'] for p in SCALE_POINTS]
        offset = (i - (len(summary) - 1) / 2) * width
        ax.bar([xi + offset for xi in x], pct, width=width * 0.9,
               color=CATEGORICAL[i], zorder=3,
               label=f"{display_name(row['judge_model'])}  ($M$ = {row['grand_mean']:.2f})")
        for xi, value in zip(x, pct):
            ax.text(xi + offset, value + ymax * 0.02,
                    fmt_count(row[f'overall_n_{SCALE_POINTS[xi]}'], value).replace('\n', ' '),
                    ha='left', va='bottom', fontsize=6, color=INK_SECONDARY,
                    zorder=4, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in SCALE_POINTS], fontsize=8)
    ax.set_xlabel('Judge score (1 = lowest, 5 = highest)', fontsize=8, color=INK_SECONDARY)
    ax.set_ylabel('% of scored\nresponses', fontsize=8, color=INK_SECONDARY)
    ax.set_ylim(0, ymax)
    ax.set_facecolor(SURFACE)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color('#c3c2b7')
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
    ax.legend(frameon=False, fontsize=7.5, ncol=len(summary), loc='upper center',
              bbox_to_anchor=(0.5, 1.24), columnspacing=1.2, handlelength=1.0)

    fig.tight_layout()
    _save(fig, output_stem)
    print(f'Wrote figure -> {output_stem}.png / .pdf')


def main():
    parser = argparse.ArgumentParser(
        description='Per-judge item distribution descriptives for the EFA metric set')
    parser.add_argument('--input', type=str, nargs='*',
                        help='The same score CSV(s) passed to the pooled analyze-efa.py run')
    parser.add_argument('--judge-label', type=str, default=None,
                        help='Judge name for the summary row (default: derived from filename)')
    parser.add_argument('--compare', action='store_true',
                        help='Stack every judge_distribution_summary_*.csv already written')
    parser.add_argument('--grouped', action='store_true',
                        help='With --compare, also emit the grouped-bar variant. Off by '
                             'default: grouping forces the eye across a spatial gap to '
                             'trace one judge, which the small multiples avoid.')
    parser.add_argument('--max-error-rate', type=float, default=0.20,
                        help='Must match the pooled run (default 0.20)')
    parser.add_argument('--core-max-na-rate', type=float, default=0.10,
                        help='Must match the pooled run (default 0.10)')
    parser.add_argument('--min-applicable-convos', type=int, default=200,
                        help='Must match the pooled run (default 200)')
    parser.add_argument('--skew-cut', type=float, default=2.0,
                        help='|skew| above this counts as non-normal (Curran et al. 1996)')
    parser.add_argument('--kurtosis-cut', type=float, default=7.0,
                        help='Excess kurtosis above this counts as non-normal')
    parser.add_argument('--ceiling-cut', type=float, default=50.0,
                        help='Percent-at-ceiling above this counts as a ceiling item')
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    out_dir = (Path(args.output_dir) if args.output_dir
               else efa_result_dir('factor_analysis') / 'descriptives')
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.compare:
        summaries = sorted(out_dir.glob('judge_distribution_summary_*.csv'))
        if not summaries:
            parser.error(f'No judge_distribution_summary_*.csv in {out_dir} — run per judge first')
        stacked = pd.concat([pd.read_csv(p) for p in summaries], ignore_index=True)
        # A summary written before a new statistic was added concatenates as NaN and would
        # render silently as "SD = nan" in the figure.
        required = ['grand_mean', 'pooled_sd', 'pooled_skewness',
                    'pooled_kurtosis_excess', 'mean_item_sd',
                    'pct_not_applicable'] + \
                   [f'overall_pct_{p}' for p in SCALE_POINTS] + \
                   [f'overall_n_{p}' for p in SCALE_POINTS]
        stale = {
            col: sorted(stacked.loc[stacked[col].isna(), 'judge_model'])
            for col in required
            if col not in stacked.columns or stacked[col].isna().any()
        }
        if stale:
            details = '; '.join(f'{c} missing for {v or "all judges"}'
                                for c, v in stale.items())
            parser.error(
                f'Stale judge summaries in {out_dir} ({details}). Re-run '
                'judge-descriptives.py --input ... for every judge, then --compare.')
        dest = out_dir / 'judge_distribution_comparison.csv'
        stacked.to_csv(dest, index=False)
        print(f'\nStacked {len(summaries)} judge summaries -> {dest}')
        print(stacked.to_string(index=False))

        # Panel B needs the per-item means, so pair each summary with its item table.
        item_frames = {}
        for p in sorted(out_dir.glob('item_descriptives_*.csv')):
            d = pd.read_csv(p)
            item_frames[d['judge_model'].iloc[0]] = d
        missing = set(stacked['judge_model']) - set(item_frames)
        if missing:
            print(f'  Skipping figure — no item_descriptives_*.csv for: {sorted(missing)}')
        else:
            plot_small_multiples(stacked, out_dir / 'judge_distributions')
            if args.grouped:
                plot_grouped(stacked, out_dir / 'judge_distributions_grouped')
        return

    if not args.input:
        parser.error('--input is required unless --compare is given')

    input_paths = [Path(p) for p in args.input]
    tag = input_paths[0].stem.replace('efa_evaluation_results_', '')
    if len(input_paths) > 1:
        tag += f'_merged{len(input_paths)}'
    judge_label = args.judge_label or tag.split('_batch_')[0]

    # Take tiering from the EFA pipeline itself so this table can never disagree with it about
    # which metrics are core / conditional / rare.
    print(f'Loading data from {len(input_paths)} file(s)...')
    pooled = _efa.load_and_pivot_results(
        input_paths,
        max_error_rate=args.max_error_rate,
        core_max_na_rate=args.core_max_na_rate,
        min_applicable_convos=args.min_applicable_convos,
    )
    # analyze-efa.py stores the audit with reset_index(), so metric_id arrives as a column;
    # restore it as the index so the descriptives join on metric, not position.
    audit = pooled.attrs['preprocessing_audit'].set_index('metric_id')

    raw = _efa.load_results(input_paths)
    if 'metric_id' not in raw.columns:
        raw['metric_id'] = raw['category'] + '_' + raw['subcategory']

    # Describe only the metrics the pipeline analyses: those present in every input chunk.
    if 'present_in_all_input_chunks' in audit.columns:
        common = audit.index[audit['present_in_all_input_chunks'].astype(bool)]
        excluded = sorted(set(raw['metric_id']) - set(common))
        raw = raw[raw['metric_id'].isin(common)]
        audit = audit.loc[common]
        if excluded:
            print(f'Excluded {len(excluded)} metric(s) not present in every input chunk: {excluded}')

    items = item_descriptives(raw)

    # Left-join onto the audit so every metric the pipeline knows about appears, in the
    # pipeline's own order, with its tier — including any with no scored values at all.
    audit_cols = [c for c in ['category', 'subcategory', 'n_scored', 'n_error',
                              'n_not_applicable', 'non_scored_rate_among_present_rows',
                              'applicability_tier', 'retained_for_pooled_efa',
                              'analyzed_as_conditional'] if c in audit.columns]
    table = audit[audit_cols].join(items, how='left')
    table.insert(0, 'judge_model', judge_label)

    item_dest = out_dir / f'item_descriptives_{tag}.csv'
    table.to_csv(item_dest)
    print(f'\nWrote {len(table)} item rows -> {item_dest}')

    summary = judge_summary(items, judge_label, args.skew_cut,
                            args.kurtosis_cut, args.ceiling_cut,
                            status_counts=raw['score_status'].value_counts().to_dict())
    summary_dest = out_dir / f'judge_distribution_summary_{tag}.csv'
    summary.to_csv(summary_dest, index=False)
    print(f'Wrote judge summary -> {summary_dest}\n')
    print(summary.T.to_string(header=False))


if __name__ == '__main__':
    main()
