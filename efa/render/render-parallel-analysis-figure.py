#!/usr/bin/env python3
"""Three-panel Horn's parallel analysis figure, one panel per judge."""

import argparse
import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times', 'Nimbus Roman', 'Times New Roman', 'Liberation Serif'],
    'mathtext.fontset': 'stix',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation_utils import efa_result_dir

SURFACE = '#fcfcfb'
INK_PRIMARY = '#0b0b0b'
INK_SECONDARY = '#52514e'
INK_MUTED = '#898781'
GRIDLINE = '#e1e0d9'
OBSERVED = '#2a78d6'   # categorical slot 1
SIMULATED = '#d04241'  # contrasting pole; the two series are also dash-coded

JUDGE_ORDER = ['gpt-5.4-mini', 'claude-haiku-4-5', 'gemma4-26b']
DISPLAY_NAMES = {
    'gpt-5.4-mini': 'GPT-5.4 mini',
    'claude-haiku-4-5': 'Haiku 4.5',
    'gemma4-26b': 'Gemma 4 26B',
}


def judge_key(name: str) -> str | None:
    for key in JUDGE_ORDER:
        if key in name:
            return key
    return None


def main():
    parser = argparse.ArgumentParser(description="Render the parallel analysis figure")
    parser.add_argument('--max-factor', type=int, default=15,
                        help='Highest factor number to plot (default 15)')
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    results_dir = efa_result_dir('factor_analysis')
    out_dir = (Path(args.output_dir) if args.output_dir
               else results_dir / 'descriptives')
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = {}
    for path in sorted(glob.glob(str(results_dir / 'retention' / 'parallel_analysis_*.csv'))):
        key = judge_key(Path(path).name)
        if key:
            frames[key] = pd.read_csv(path)
    ordered = [k for k in JUDGE_ORDER if k in frames]
    if not ordered:
        raise SystemExit(f'No parallel_analysis_*.csv under {results_dir}/retention.')

    fig, axes = plt.subplots(1, len(ordered), figsize=(7.0, 2.5), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    cut = args.max_factor

    for ax, key in zip(axes, ordered):
        d = frames[key].head(cut)
        x = d['factor'].values
        obs = d['observed_minres_eigenvalue'].values
        sim = d['random_95th_percentile_minres_eigenvalue'].values
        n_ret = int(frames[key]['psych_recommended_n_factors'].iloc[0])

        ax.plot(x, obs, color=OBSERVED, linewidth=1.6, marker='o', markersize=3.4,
                label='Observed', zorder=3)
        ax.plot(x, sim, color=SIMULATED, linewidth=1.4, linestyle='--', marker='^',
                markersize=3.2, label='Simulated (95th pct)', zorder=3)
        # The crossing is the decision; mark it rather than leaving it to be read off.
        ax.axvline(n_ret + 0.5, color=INK_MUTED, linewidth=0.9, linestyle=':', zorder=2)
        ax.set_title(f'{DISPLAY_NAMES[key]}\n{n_ret} factors retained',
                     fontsize=8.5, color=INK_PRIMARY, pad=6)
        ax.set_xlim(0.5, cut + 0.5)
        ax.set_xticks([1, 5, 10, 15][:len([t for t in [1, 5, 10, 15] if t <= cut])])
        ax.set_facecolor(SURFACE)
        ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color('#c3c2b7')
        ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)

    # Magnified: the first eigenvalue is off-scale by design, so the crossing is legible.
    axes[0].set_ylim(0, 2.0)
    axes[0].set_ylabel('Eigenvalue', fontsize=8, color=INK_SECONDARY)
    for ax in axes:
        ax.set_xlabel('Factor number', fontsize=8, color=INK_SECONDARY, labelpad=2)
    # Anchored rather than loc='upper right': the observed curve leaves the panel through the
    # top-right corner, so an automatic placement sits on top of it.
    axes[0].legend(frameon=False, fontsize=7.5, loc='upper right',
                   bbox_to_anchor=(1.0, 0.72), handlelength=1.8, borderaxespad=0.2)

    fig.tight_layout()
    stem = out_dir / 'parallel_analysis'
    for ext in ('png', 'pdf'):
        fig.savefig(f'{stem}.{ext}', dpi=200, facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)

    meta = frames[ordered[0]]
    print(f"iterations = {meta['iterations'].iloc[0]}, seed = {meta['random_seed'].iloc[0]}")
    for key in ordered:
        print(f"  {DISPLAY_NAMES[key]:14s} retains "
              f"{int(frames[key]['psych_recommended_n_factors'].iloc[0])}")
    print(f'Wrote figure -> {stem}.png / .pdf')


if __name__ == '__main__':
    main()
