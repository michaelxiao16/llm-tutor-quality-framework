"""Explainer diagram: what the Schmid-Leiman transformation does."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # for evaluation_utils
from evaluation_utils import efa_result_dir

SURFACE, INK, INK2, MUTED = '#fcfcfb', '#0b0b0b', '#52514e', '#898781'
BLUE, RED, GRAY = '#2a78d6', '#d04241', '#e1e0d9'


def only_match(folder: Path, pattern: str) -> Path:
    matches = sorted(folder.glob(pattern))
    if len(matches) != 1:
        raise SystemExit(f'Expected exactly one {pattern} in {folder}; found {len(matches)}.')
    return matches[0]


results = efa_result_dir('factor_analysis')
Phi = pd.read_csv(only_match(results / 'solution', 'factor_correlations_gpt*.csv'), index_col=0)
SL = pd.read_csv(only_match(results / 'general_factor', 'schmid_leiman_loadings_gpt*.csv'))
w, v = np.linalg.eigh(Phi.values)
i = np.argmax(w)
b = np.abs(v[:, i] * np.sqrt(w[i]))

fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5.2))
fig.patch.set_facecolor(SURFACE)
for ax in (axL, axR):
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off'); ax.set_facecolor(SURFACE)


def node(ax, x, y, r, label, color, fs=9):
    ax.add_patch(Circle((x, y), r, facecolor='white', edgecolor=color, linewidth=1.8, zorder=3))
    ax.text(x, y, label, ha='center', va='center', fontsize=fs, color=INK, zorder=4)


def link(ax, p0, p1, color=MUTED, lw=1.0, label=None, ls='-'):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=8,
                                 color=color, linewidth=lw, shrinkA=1, shrinkB=1,
                                 zorder=2, linestyle=ls))
    if label:
        ax.text((p0[0] + p1[0]) / 2 - 2, (p0[1] + p1[1]) / 2, label, fontsize=6.8,
                color=INK2, ha='right', va='center', zorder=4)


# ---------------- LEFT: hierarchical (before) ----------------
axL.set_title('Before: hierarchical model', fontsize=11, color=INK, loc='left', pad=12)
node(axL, 50, 88, 7, '$g$', RED, fs=12)
fx = np.linspace(10, 90, 7)
for k, x in enumerate(fx):
    node(axL, x, 55, 5.4, f'F{k+1}', BLUE, fs=8)
    link(axL, (50, 81), (x, 60.5))
    axL.text(x, 63.5, f'{b[k]:.2f}', ha='center', va='bottom',
             fontsize=6.8, color=INK2, zorder=4)
    axL.add_patch(Rectangle((x - 5.5, 18), 11, 16, facecolor='white',
                            edgecolor=GRAY, linewidth=1.0, zorder=3))
    axL.text(x, 26, 'items', ha='center', va='center', fontsize=6.5, color=INK2, zorder=4)
    link(axL, (x, 49.6), (x, 34.5))
axL.text(50, 73, 'second-order loadings $b$', ha='center', fontsize=7.5,
         color=INK2, style='italic')
axL.text(50, 8, 'the 7 factors are CORRELATED ($r$ = .31 – .69),\n'
                'so their variance overlaps and cannot simply be added',
         ha='center', fontsize=7.5, color=INK2)

# ---------------- RIGHT: bifactor (after) ----------------
axR.set_title('After Schmid–Leiman: orthogonal bifactor', fontsize=11, color=INK, loc='left', pad=12)
node(axR, 12, 55, 8, '$g$', RED, fs=12)
ys = np.linspace(84, 22, 7)
for k, y in enumerate(ys):
    axR.add_patch(Rectangle((44, y - 4), 14, 8, facecolor='white',
                            edgecolor=GRAY, linewidth=1.0, zorder=3))
    axR.text(51, y, 'items', ha='center', va='center', fontsize=6.3, color=INK2, zorder=4)
    link(axR, (20, 55), (43.5, y), color=RED, lw=0.8)
    node(axR, 76, y, 4.6, f'F{k+1}*', BLUE, fs=7)
    link(axR, (71.4, y), (58.5, y), color=BLUE, lw=0.8)
axR.text(76, 12, 'residualised: $F_k^* = F_k\\times\\sqrt{1-b_k^2}$\nscaling .58 – .78',
         ha='center', fontsize=7.5, color=INK2, style='italic')
axR.text(12, 38, 'every item\nloads on $g$', ha='center', fontsize=7.5, color=INK2)

# variance partition bar
g2 = (SL['g'] ** 2).sum()
grp2 = (SL[[c for c in SL.columns if c.startswith('F') and c.endswith('*')]] ** 2).values.sum()
uniq = len(SL) - SL['h2'].sum()
tot = g2 + grp2 + uniq
x0 = 8
for val, col, lab in [(g2, RED, f'general\n{g2/tot*100:.0f}%'),
                      (grp2, BLUE, f'group\n{grp2/tot*100:.0f}%'),
                      (uniq, GRAY, f'unique\n{uniq/tot*100:.0f}%')]:
    wdt = val / tot * 84
    axR.add_patch(Rectangle((x0, 2), wdt, 5, facecolor=col, edgecolor=SURFACE,
                            linewidth=1.2, alpha=0.85 if col != GRAY else 1, zorder=3))
    axR.text(x0 + wdt / 2, 4.5, lab.split('\n')[1], ha='center', va='center',
             fontsize=6.5, color='white' if col != GRAY else INK2, zorder=4)
    x0 += wdt
axR.text(8, 9.5, 'g $\\perp$ F*, so total variance now partitions cleanly:',
         fontsize=7.5, color=INK2)

fig.tight_layout()
out_dir = results / 'descriptives'
out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / 'schmid_leiman_explainer'
fig.savefig(out.with_suffix('.png'), dpi=170, facecolor=SURFACE, bbox_inches='tight')
fig.savefig(out.with_suffix('.pdf'), dpi=170, facecolor=SURFACE, bbox_inches='tight')
print(f'Wrote figure -> {out}.png / .pdf')
