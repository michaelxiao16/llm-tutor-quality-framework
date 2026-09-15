"""Cross-judge factor congruence for the AI Tutor EFA study."""

from __future__ import annotations

import argparse
import itertools
import os
import shutil
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

DEFAULT_DIR = Path(__file__).resolve().parent / "results" / "factor_analysis"

FAIR = 0.85
IDENTICAL = 0.95
SALIENT = 0.40  # |loading| threshold used elsewhere in the pipeline
TOP_K = 5       # defining metrics listed per factor
EPS = 1e-9


def derive_label(path: Path) -> str:
    """Short judge name from a factor_loadings_{tag}.csv filename."""
    stem = path.name
    stem = re.sub(r"^factor_loadings_", "", stem)
    stem = re.sub(r"\.csv$", "", stem)
    # Cut the run tag (batch id / timestamp / rotation / n) off the model name.
    for cut in (r"_batch_", r"_\d{8}_\d{6}", r"_(?:oblimin|promax|varimax|unrotated)(?:_n\d+)?$"):
        m = re.search(cut, stem)
        if m:
            stem = stem[: m.start()]
            break
    # Drop a trailing model date suffix (e.g. -2026-03-17 or -20251001).
    stem = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", stem)
    stem = re.sub(r"-\d{8}$", "", stem)
    return stem


def load_loadings(path: Path) -> pd.DataFrame:
    """Read a pattern matrix indexed by metric, columns Factor_1..Factor_k."""
    df = pd.read_csv(path)
    if df.columns[0] != "metric":
        df = df.rename(columns={df.columns[0]: "metric"})
    df = df.set_index("metric")
    factor_cols = [c for c in df.columns if c.startswith("Factor_")]
    if not factor_cols:
        raise ValueError(f"No Factor_* columns in {path.name}")
    return df[factor_cols].astype(float)


def congruence_matrix(la: pd.DataFrame, lb: pd.DataFrame) -> np.ndarray:
    """Tucker congruence for every factor of `la` vs every factor of `lb`."""
    a = la.to_numpy()
    b = lb.to_numpy()
    a_norm = np.sqrt((a**2).sum(axis=0))  # per-factor norms
    b_norm = np.sqrt((b**2).sum(axis=0))
    cross = a.T @ b                        # k_a x k_b dot products
    denom = np.outer(a_norm, b_norm)
    with np.errstate(invalid="ignore", divide="ignore"):
        phi = cross / denom
    phi[np.outer(a_norm < EPS, np.ones_like(b_norm, dtype=bool))] = np.nan
    phi[np.outer(np.ones_like(a_norm, dtype=bool), b_norm < EPS)] = np.nan
    return phi


def top_metrics(full: pd.DataFrame, factor: str, k: int = TOP_K) -> str:
    """The k highest |loading| metrics defining a factor in its own solution."""
    col = full[factor].reindex(full[factor].abs().sort_values(ascending=False).index)
    picked = col.head(k)
    return "; ".join(f"{m} ({v:+.2f})" for m, v in picked.items())


def band(abs_phi: float) -> str:
    if np.isnan(abs_phi):
        return "undefined"
    if abs_phi >= IDENTICAL:
        return "identical"
    if abs_phi >= FAIR:
        return "fair"
    return "not_replicated"


def compare_pair(name_a, full_a, name_b, full_b) -> pd.DataFrame:
    """Hungarian-matched congruence rows for one judge pair."""
    common = full_a.index.intersection(full_b.index)
    la = full_a.loc[common]
    lb = full_b.loc[common]
    phi = congruence_matrix(la, lb)

    # Match on absolute congruence; unmatchable (NaN) treated as 0 cost.
    cost = -np.nan_to_num(np.abs(phi), nan=0.0)
    rows_idx, cols_idx = linear_sum_assignment(cost)

    factors_a = list(full_a.columns)
    factors_b = list(full_b.columns)
    records = []
    for i, j in zip(rows_idx, cols_idx):
        signed = phi[i, j]
        records.append(
            {
                "judge_a": name_a,
                "factor_a": factors_a[i],
                "judge_b": name_b,
                "factor_b": factors_b[j],
                "n_common_metrics": int(len(common)),
                "tucker_signed": round(float(signed), 4),
                "tucker_abs": round(float(abs(signed)), 4),
                "replication_band": band(abs(signed)),
                "defining_metrics_a": top_metrics(full_a, factors_a[i]),
                "defining_metrics_b": top_metrics(full_b, factors_b[j]),
            }
        )
    out = pd.DataFrame(records).sort_values("tucker_abs", ascending=False, ignore_index=True)

    # Note any factor of the larger solution that went unmatched.
    unmatched_a = sorted(set(range(len(factors_a))) - set(rows_idx))
    unmatched_b = sorted(set(range(len(factors_b))) - set(cols_idx))
    for i in unmatched_a:
        out = pd.concat([out, pd.DataFrame([{
            "judge_a": name_a, "factor_a": factors_a[i], "judge_b": name_b,
            "factor_b": "(none)", "n_common_metrics": int(len(common)),
            "tucker_signed": np.nan, "tucker_abs": np.nan,
            "replication_band": "unmatched",
            "defining_metrics_a": top_metrics(full_a, factors_a[i]),
            "defining_metrics_b": "",
        }])], ignore_index=True)
    for j in unmatched_b:
        out = pd.concat([out, pd.DataFrame([{
            "judge_a": name_a, "factor_a": "(none)", "judge_b": name_b,
            "factor_b": factors_b[j], "n_common_metrics": int(len(common)),
            "tucker_signed": np.nan, "tucker_abs": np.nan,
            "replication_band": "unmatched",
            "defining_metrics_a": "",
            "defining_metrics_b": top_metrics(full_b, factors_b[j]),
        }])], ignore_index=True)
    return out, pd.DataFrame(phi, index=factors_a, columns=factors_b)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--loadings", nargs="*", type=Path,
                    help="factor_loadings_*.csv files (>=2). Default: all in the results dir.")
    ap.add_argument("--labels", nargs="*", default=None,
                    help="Optional judge labels, one per --loadings file, in order.")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    args = ap.parse_args()

    if args.loadings:
        files = args.loadings
    else:
        # Pattern matrices live in the purpose-grouped `solution/` subfolder.
        files = sorted((DEFAULT_DIR / "solution").glob("factor_loadings_*.csv"))
    if len(files) < 2:
        raise SystemExit(f"Need >=2 factor_loadings files; found {len(files)} in {DEFAULT_DIR / 'solution'}")

    labels = args.labels if args.labels else [derive_label(Path(f)) for f in files]
    if len(labels) != len(files):
        raise SystemExit("--labels count must match --loadings count")

    solutions = {lab: load_loadings(Path(f)) for lab, f in zip(labels, files)}
    print("Loaded judges:")
    for lab, df in solutions.items():
        print(f"  {lab:20s} {df.shape[1]} factors, {df.shape[0]} metrics")

    # Congruence outputs go in the purpose-grouped `congruence/` subfolder.
    cong_dir = args.output_dir / "congruence"
    cong_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for name_a, name_b in itertools.combinations(labels, 2):
        rows, matrix = compare_pair(name_a, solutions[name_a], name_b, solutions[name_b])
        all_rows.append(rows)
        mpath = cong_dir / f"cross_judge_congruence_matrix_{name_a}_vs_{name_b}.csv"
        matrix.round(4).to_csv(mpath)

        print(f"\n=== {name_a}  vs  {name_b}   ({rows['n_common_metrics'].dropna().iloc[0]} common metrics) ===")
        for _, r in rows[rows["replication_band"] != "unmatched"].iterrows():
            print(f"  {r['factor_a']:>9} ~ {r['factor_b']:<9} "
                  f"phi={r['tucker_signed']:+.3f} |{r['tucker_abs']:.3f}|  [{r['replication_band']}]")
            print(f"        A: {r['defining_metrics_a']}")
            print(f"        B: {r['defining_metrics_b']}")

    combined = pd.concat(all_rows, ignore_index=True)
    cpath = cong_dir / "cross_judge_congruence_pairs.csv"
    combined.to_csv(cpath, index=False)
    print(f"\nWrote {cpath}")
    print("Wrote per-pair full congruence matrices to", cong_dir)

    render_apa_congruence_table(args.output_dir)


def render_apa_congruence_table(output_dir: Path) -> None:
    """Render the editable APA Word congruence table for this run (presentation layer)."""
    efa_dir = Path(__file__).resolve().parent
    script = efa_dir / "render" / "render-apa-congruence.R"
    labels = output_dir / "congruence" / "congruence_construct_labels.csv"
    if not script.exists():
        return
    if not labels.exists():
        print(f"Skipping APA Word table: create {labels.name} "
              f"(construct -> factor mapping) to enable it.")
        return
    rscript = os.environ.get("RSCRIPT") or shutil.which("Rscript") or "Rscript"
    env = {
        **os.environ,
        "EFA_RESULTS_DIR": str(output_dir.resolve()),
        "EFA_APA_TABLE_DIR": str((output_dir / "apa_tables").resolve()),
        # Match analyze-efa.py: use the pinned project library without waiting indefinitely on
        # renv's transient sandbox lock.
        "RENV_CONFIG_SANDBOX_ENABLED": "FALSE",
    }
    try:
        subprocess.run([rscript, str(script)], cwd=str(efa_dir),
                       env=env, check=True, capture_output=True, text=True,
                       timeout=120)
        print("Rendered APA Word table via render-apa-congruence.R")
    except (subprocess.CalledProcessError, FileNotFoundError,
            subprocess.TimeoutExpired) as exc:
        detail = (getattr(exc, "stderr", "") or str(exc)).strip().splitlines()
        reason = detail[-1] if detail else str(exc)
        print(f"Warning: render-apa-congruence.R did not run ({reason[:200]}); "
              f"CSV outputs are unaffected.")


if __name__ == "__main__":
    main()
