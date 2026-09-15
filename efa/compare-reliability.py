#!/usr/bin/env python3
"""Compare EFA evaluation result CSVs to assess test-retest reliability."""

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import cohen_kappa_score

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times', 'Nimbus Roman', 'Times New Roman', 'Liberation Serif'],
    'mathtext.fontset': 'stix',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

sys.path.insert(0, str(Path(__file__).parent.parent))  # repo root, for evaluation_utils


def save_png_and_pdf(output_path: Path, **kwargs) -> None:
    """Save the current Matplotlib figure as matching PNG and vector PDF files."""
    output_path = Path(output_path)
    stem = output_path.stem
    plt.savefig(output_path.with_name(f'{stem}.png'), **kwargs)
    plt.savefig(output_path.with_name(f'{stem}.pdf'), **kwargs)
from evaluation_utils import efa_result_dir


def load_results(csv_path: Path, id_mapping: dict | None = None) -> pd.DataFrame:
    """Load evaluation results, optionally mapping conversation IDs."""
    df = pd.read_csv(csv_path)
    if id_mapping:
        df['conversation_id'] = df['conversation_id'].map(id_mapping).fillna(df['conversation_id'])
    return df


def build_id_mapping(sample_dir: Path) -> dict:
    """Build mapping from filename (efa-XXXX) to conversation UUID."""
    mapping = {}
    for f in sample_dir.glob('*.json'):
        with open(f) as fp:
            data = json.load(fp)
        uuid = data.get('metadata', {}).get('conversation_id')
        if uuid:
            mapping[f.stem] = uuid
    return mapping


def compare_results(run1: pd.DataFrame, run2: pd.DataFrame) -> pd.DataFrame:
    """Merge two result sets on conversation_id + metric_id."""
    merged = run1.merge(run2, on=['conversation_id', 'metric_id'], suffixes=('_run1', '_run2'))

    # Filter to valid scores only
    valid = merged[
        (merged['score_run1'].notna()) &
        (merged['score_run2'].notna()) &
        (merged['score_status_run1'] != 'error') &
        (merged['score_status_run2'] != 'error')
    ].copy()

    return valid


def calculate_reliability(valid: pd.DataFrame) -> dict:
    """Calculate reliability metrics."""
    exact_match = (valid['score_run1'] == valid['score_run2']).mean()
    within_1 = (abs(valid['score_run1'] - valid['score_run2']) <= 1).mean()
    correlation = valid['score_run1'].corr(valid['score_run2'])
    # Quadratic weighted kappa: the standard ordinal-rater-agreement metric.
    qwk = cohen_kappa_score(valid['score_run1'], valid['score_run2'], weights='quadratic')

    return {
        'n': len(valid),
        'exact_match': exact_match,
        'within_1': within_1,
        'correlation': correlation,
        'qwk': qwk
    }


def plot_single_heatmap(ax, valid: pd.DataFrame, metrics: dict, xlabel: str, ylabel: str,
                        compact_vertical: bool = False):
    """Plot a single heatmap on the given axis."""
    scores = [1, 2, 3, 4, 5]
    confusion = pd.crosstab(valid['score_run1'], valid['score_run2'], dropna=False)
    confusion = confusion.reindex(index=scores, columns=scores, fill_value=0)

    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=scores, yticklabels=scores, cbar=False,
                square=not compact_vertical,
                annot_kws={'fontsize': 16 if compact_vertical else 10})
    label_size = 16 if compact_vertical else 10
    ax.set_xlabel(xlabel, fontsize=label_size)
    ax.set_ylabel(ylabel, fontsize=label_size)
    ax.tick_params(axis='both', labelsize=15 if compact_vertical else 9)

    if compact_vertical:
        stats_text = (f"n = {metrics['n']:,}   Exact = {metrics['exact_match']:.0%}   "
                      f"Within ±1 = {metrics['within_1']:.0%}   "
                      f"r = {metrics['correlation']:.2f}   QWK = {metrics['qwk']:.2f}")
        ax.text(0.5, 1.015, stats_text, transform=ax.transAxes, fontsize=15,
                ha='center', va='bottom')
    else:
        stats_text = f"n={metrics['n']}\nExact: {metrics['exact_match']:.0%}\nWithin ±1: {metrics['within_1']:.0%}\nr={metrics['correlation']:.2f}\nQWK={metrics['qwk']:.2f}"
        ax.text(1.05, 0.5, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def plot_agreement_heatmap(valid: pd.DataFrame, metrics: dict, output_path: Path, title: str = None,
                           xlabel: str = 'Run 2 Score', ylabel: str = 'Run 1 Score'):
    """Create score agreement heatmap for a single comparison."""
    scores = [1, 2, 3, 4, 5]
    confusion = pd.crosstab(valid['score_run1'], valid['score_run2'], dropna=False)
    confusion = confusion.reindex(index=scores, columns=scores, fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=scores, yticklabels=scores, cbar_kws={'shrink': 0.8}, square=True)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    stats_text = f"n = {metrics['n']}\nExact match: {metrics['exact_match']:.1%}\nWithin ±1: {metrics['within_1']:.1%}\nCorrelation: {metrics['correlation']:.3f}\nQWK: {metrics['qwk']:.3f}"
    fig.text(0.60, 0.5, stats_text, fontsize=10, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0, 0.58, 1])
    save_png_and_pdf(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')


def plot_multi_comparison(comparisons: list[tuple], output_path: Path, title: str = None):
    """Create a grid of heatmaps for multiple pairwise comparisons."""
    n = len(comparisons)

    if n <= 2:
        nrows, ncols = 1, n
        figsize = (6 * n + 2, 6)
    elif n == 3:
        # The paper places this image beside the vertically stacked judge distributions.
        nrows, ncols = 3, 1
        figsize = (7.2, 10.5)
    elif n <= 4:
        nrows, ncols = 2, 2
        figsize = (14, 12)
    else:
        nrows, ncols = 2, 3
        figsize = (18, 12)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, (valid, metrics, label1, label2) in enumerate(comparisons):
        plot_single_heatmap(
            axes[i], valid, metrics, f'{label2} Score', f'{label1} Score',
            compact_vertical=(n == 3),
        )
        axes[i].set_title(f'{label1} vs {label2}', fontsize=18 if n == 3 else 11,
                          pad=28 if n == 3 else 6)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout(h_pad=2.0)
    save_png_and_pdf(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')


def main():
    parser = argparse.ArgumentParser(description='Compare EFA evaluation runs for reliability')
    parser.add_argument('--runs', type=str, nargs='+', required=True, help='Results CSVs to compare (2-4 files)')
    parser.add_argument('--labels', type=str, nargs='+', default=None, help='Labels for each run (e.g., "5.4-mini" "4.1-nano")')
    parser.add_argument('--output', type=str, default=None, help='Output PNG path')
    parser.add_argument('--sample-dir', type=str, default=None,
                        help='Sample directory for ID mapping (if runs use filenames instead of UUIDs)')
    parser.add_argument('--title', type=str, default=None, help='Chart title')
    args = parser.parse_args()

    if len(args.runs) < 2:
        print('ERROR: Need at least 2 runs to compare')
        return

    if len(args.runs) > 4:
        print('ERROR: Maximum 4 runs supported')
        return

    # Build ID mapping if sample dir provided
    id_mapping = None
    if args.sample_dir:
        id_mapping = build_id_mapping(Path(args.sample_dir))
        print(f'Loaded {len(id_mapping)} ID mappings from {args.sample_dir}')

    # Load all runs
    run_paths = [Path(r) for r in args.runs]
    runs = [load_results(p, id_mapping) for p in run_paths]

    # Generate labels
    if args.labels:
        if len(args.labels) != len(args.runs):
            print(f'ERROR: Number of labels ({len(args.labels)}) must match number of runs ({len(args.runs)})')
            return
        labels = args.labels
    else:
        labels = [p.stem.split('_')[3] if len(p.stem.split('_')) > 3 else f'Run{i+1}' for i, p in enumerate(run_paths)]

    for i, (run, label) in enumerate(zip(runs, labels)):
        print(f'{label}: {len(run)} rows, {run["conversation_id"].nunique()} conversations')

    # Generate all pairwise comparisons
    comparisons = []
    print('\nPairwise cross-tabulations:')
    print('-' * 50)

    for (i, run1), (j, run2) in combinations(enumerate(runs), 2):
        valid = compare_results(run1, run2)
        if len(valid) == 0:
            print(f'{labels[i]} vs {labels[j]}: No matching scores')
            continue

        metrics = calculate_reliability(valid)
        comparisons.append((valid, metrics, labels[i], labels[j]))

        print(f'{labels[i]} vs {labels[j]}:')
        print(f'  n = {metrics["n"]}, Exact: {metrics["exact_match"]:.1%}, Within ±1: {metrics["within_1"]:.1%}, r = {metrics["correlation"]:.3f}, QWK = {metrics["qwk"]:.3f}')

    if not comparisons:
        print('ERROR: No valid comparisons found')
        return

    # Generate output path
    if args.output:
        output_path = Path(args.output)
    else:
        label_str = '_vs_'.join(labels)
        output_path = efa_result_dir('reliability') / f'reliability_{label_str}.png'

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Plot
    if len(comparisons) == 1:
        valid, metrics, label1, label2 = comparisons[0]
        plot_agreement_heatmap(valid, metrics, output_path, args.title,
                               xlabel=f'{label2} Score', ylabel=f'{label1} Score')
    else:
        plot_multi_comparison(comparisons, output_path, args.title)


if __name__ == '__main__':
    main()
