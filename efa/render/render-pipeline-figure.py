#!/usr/bin/env python3
"""Figure 1: the end-to-end study pipeline, from candidate metrics to final framework."""

import argparse
import csv
import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Match the ACL LaTeX template's Times body font.
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times', 'Nimbus Roman', 'Times New Roman', 'Liberation Serif'],
    'mathtext.fontset': 'stix',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation_utils import efa_result_dir

# --- Analyst-maintained values (not derivable from any output file) ---------------- The
# inductive category count comes from the qualitative coding pass; the final counts come from
# the crosswalk.
FRAMEWORK = {
    'inductive_categories': 12,
    'final_categories': 10,
    'final_items': 41,
    # Safety is retained in the recommended framework on a-priori grounds but carries no
    # factor-analytic evidence here: ConvoLearn contains almost no safety-triggering content,
    # so those items were non-applicable throughout.
    'apriori_only_categories': 1,
    'apriori_only_items': 7,
    # Constructs reaching fair congruence in every judge pair (see the crosswalk).
    'replicated_constructs': 5,
}

# Shared with judge-descriptives.py so the two figures read as one set.
SURFACE = '#fcfcfb'
INK_PRIMARY = '#0b0b0b'
INK_SECONDARY = '#52514e'
INK_MUTED = '#898781'
ACCENT = '#2a78d6'
ACCENT_SOFT = '#dce9fa'
NEUTRAL_FILL = '#f0efec'
EDGE = '#c3c2b7'

JUDGE_ORDER = ['gpt-5.4-mini', 'claude-haiku-4-5', 'gemma4-26b']
JUDGE_NAMES = {
    'gpt-5.4-mini': 'GPT-5.4 mini',
    'claude-haiku-4-5': 'Haiku 4.5',
    'gemma4-26b': 'Gemma 4 26B',
}


def judge_key(filename: str) -> str | None:
    for key in JUDGE_ORDER:
        if key in filename:
            return key
    return None


def read_pipeline_counts(root: Path, results_dir: Path) -> dict:
    """Pull every derivable number out of the committed analysis outputs."""
    counts = {}

    # Candidate pool: rows of the source metric inventory.
    metrics_csv = root / 'Tutor Metrics.csv'
    with metrics_csv.open(newline='', encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh))
    counts['candidate_metrics'] = len(rows)
    counts['n_sources'] = len({r.get('Source', '') for r in rows if r.get('Source')})

    # Analysed pool + per-judge tiers, from each judge's preprocessing audit.
    audits = sorted(glob.glob(str(results_dir / 'preprocessing' / 'preprocessing_audit_*.csv')))
    if not audits:
        raise SystemExit(f'No preprocessing_audit_*.csv under {results_dir}. Run analyze-efa.py first.')
    per_judge, pools, convs = {}, set(), set()
    for path in audits:
        key = judge_key(Path(path).name)
        if key is None:
            continue
        with open(path, newline='', encoding='utf-8') as fh:
            audit = list(csv.DictReader(fh))
        in_pool = [r for r in audit if r['present_in_all_input_chunks'] == 'True']
        core = [r for r in in_pool if r['retained_for_pooled_efa'] == 'True']
        conditional = [r for r in in_pool if r['analyzed_as_conditional'] == 'True']
        per_judge[key] = {
            'core': len(core),
            'conditional': len(conditional),
            'dropped': len(in_pool) - len(core) - len(conditional),
        }
        pools.add(len(in_pool))
        # n_present_rows / n_metrics recovers the conversation count.
        rows_in_pool = sum(int(r['n_present_rows']) for r in in_pool)
        convs.add(round(rows_in_pool / len(in_pool)))
    if len(pools) != 1:
        raise SystemExit(f'Judges disagree on the analysed metric pool: {sorted(pools)}')
    if len(convs) != 1:
        raise SystemExit(f'Judges disagree on the conversation count: {sorted(convs)}')
    counts['analysed_metrics'] = pools.pop()
    counts['n_conversations'] = convs.pop()
    counts['per_judge'] = per_judge

    # Retained factor counts, from the Schmid-Leiman summaries.
    for path in sorted(glob.glob(str(results_dir / 'general_factor' / 'schmid_leiman_summary_*.csv'))):
        key = judge_key(Path(path).name)
        if key is None or key not in per_judge:
            continue
        with open(path, newline='', encoding='utf-8') as fh:
            row = next(csv.DictReader(fh))
        per_judge[key]['factors'] = int(row['n_group_factors'])
        per_judge[key]['omega_h'] = float(row['omega_hierarchical'])
        per_judge[key]['ecv'] = float(row['ecv_general'])
    missing = [k for k in per_judge if 'factors' not in per_judge[k]]
    if missing:
        raise SystemExit(f'No schmid_leiman_summary_*.csv for: {missing}')
    return counts


def box(ax, x, y, w, h, title, body, *, fill=SURFACE, edge=EDGE, bold_edge=False):
    """One labelled stage of the pipeline."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1.2',
        facecolor=fill, edgecolor=ACCENT if bold_edge else edge,
        linewidth=1.6 if bold_edge else 1.0, zorder=3))
    ax.text(x + w / 2, y + h - 2.8, title, ha='center', va='top', fontsize=13,
            color=INK_PRIMARY, fontweight='bold', zorder=4)
    ax.text(x + w / 2, y + h - 8.8, body, ha='center', va='top', fontsize=11.5,
            color=INK_SECONDARY, linespacing=1.25, zorder=4)


def arrow(ax, start, end, label=None, *, shrink_a=0.55, shrink_b=0.55):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle='-|>', mutation_scale=9, color=INK_MUTED,
        linewidth=1.0, shrinkA=shrink_a, shrinkB=shrink_b, zorder=4,
        connectionstyle='arc3,rad=0'))
    if label:
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 1.8, label,
                ha='center', va='bottom', fontsize=11, color=INK_MUTED,
                style='italic', zorder=4)


# Layout grid.
COL_W, COL_PITCH, COL_X0 = 48, 64, 2
ROW_TOP, ROW_MID, ROW_BOT = 86, 54, 22


def col(i: int) -> float:
    return COL_X0 + i * COL_PITCH


def build_figure(c: dict, out_stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    fig.patch.set_alpha(0)
    ax.set_facecolor('none')
    ax.set_xlim(0, 330)
    ax.set_ylim(0, 110)
    ax.axis('off')

    pj = c['per_judge']
    ordered = [k for k in JUDGE_ORDER if k in pj]
    factors = '/'.join(str(pj[k]['factors']) for k in ordered)
    ecvs = [pj[k]['ecv'] for k in ordered]
    ecv_txt = f"ECV {min(ecvs):.2f}\u2013{max(ecvs):.2f}".replace('0.', '.')

    def place(i, centre_y, h, title, body, *, width=COL_W, x_offset=0, **kw):
        box(ax, col(i) + x_offset, centre_y - h / 2, width, h, title, body, **kw)

    # Every stage carries the same weight; only the terminal result is emphasised, so the eye
    # lands on the output rather than on an arbitrary intermediate step.
    place(0, ROW_MID, 27, 'Metric pool',
          f"{c['candidate_metrics']} candidates,\n{c['n_sources']} sources\n"
          f"$\\rightarrow$ {c['analysed_metrics']} conversation-\nlevel, rubric-scored")
    place(1, ROW_TOP, 14, 'Inductive coding',
          f"$\\rightarrow$ {FRAMEWORK['inductive_categories']} categories")
    place(1, ROW_BOT, 18, 'LLM-as-judge',
          f"{c['n_conversations']:,} conversations\n\u00d7 3 judges")
    place(2, ROW_BOT, 14, 'EFA', f"$\\rightarrow$ {factors} factors")
    place(3, 62, 18, 'Schmid\u2013Leiman', f"general factor\n{ecv_txt}")
    tucker_width = COL_W + 4
    tucker_offset = -2
    place(3, 22, 18, 'Tucker congruence',
          f"{FRAMEWORK['replicated_constructs']} coherent\nconstructs",
          width=tucker_width, x_offset=tucker_offset)
    place(4, ROW_TOP, 18, 'Crosswalk',
          f"{FRAMEWORK['inductive_categories']} categories\n\u00d7 EFA evidence")

    place(4, 30, 18, 'Final framework',
          f"{FRAMEWORK['final_categories']} categories,\n{FRAMEWORK['final_items']} items")

    r0, r1, r2, r3, r4 = (col(i) + COL_W for i in range(5))
    tucker_left = col(3) + tucker_offset
    tucker_right = tucker_left + tucker_width
    arrow_gap = 0
    crosswalk_target_x = col(4) - arrow_gap
    arrow(ax, (r0, 64), (col(1) - arrow_gap, 82))
    arrow(ax, (r0, 43), (col(1) - arrow_gap, 30.5))
    arrow(ax, (r1, ROW_BOT), (col(2) - arrow_gap, ROW_BOT))
    arrow(ax, (r2, 28), (col(3) - arrow_gap, 62))
    arrow(ax, (r2, 22), (tucker_left - arrow_gap, 22))
    arrow(ax, (r1, ROW_TOP), (crosswalk_target_x, ROW_TOP))
    arrow(ax, (r3, 70), (crosswalk_target_x, ROW_TOP))
    arrow(ax, (tucker_right, 30), (crosswalk_target_x, ROW_TOP))
    arrow(ax, (col(4) + COL_W / 2, 77),
          (col(4) + COL_W / 2, 39))

    fig.tight_layout(pad=0.2)
    for ext in ('png', 'pdf'):
        fig.savefig(f'{out_stem}.{ext}', dpi=200, transparent=True, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote figure -> {out_stem}.png / .pdf')


def main():
    parser = argparse.ArgumentParser(description='Render Figure 1, the study pipeline')
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    results_dir = efa_result_dir('factor_analysis')
    out_dir = Path(args.output_dir) if args.output_dir else results_dir / 'descriptives'
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = read_pipeline_counts(root, results_dir)
    print('Derived from committed outputs:')
    print(f"  candidate metrics : {counts['candidate_metrics']} ({counts['n_sources']} sources)")
    print(f"  analysed pool     : {counts['analysed_metrics']}")
    print(f"  conversations     : {counts['n_conversations']:,}")
    for key in JUDGE_ORDER:
        if key in counts['per_judge']:
            d = counts['per_judge'][key]
            print(f"  {JUDGE_NAMES[key]:14s}: {d['core']} core, {d['conditional']} conditional, "
                  f"{d['dropped']} dropped, {d['factors']} factors")
    print(f"  framework (manual): {FRAMEWORK['final_categories']} categories, "
          f"{FRAMEWORK['final_items']} items")

    build_figure(counts, out_dir / 'pipeline_figure')


if __name__ == '__main__':
    main()
