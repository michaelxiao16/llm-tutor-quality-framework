# LLM Tutor Quality Evaluation Scale

Code, correlation matrices, and paper outputs for **"Developing an LLM Tutor Quality Evaluation Scale"** (Xiao, Tian, Liu, Esbenshade, & Sun, 2026).

We coded 177 literature-derived tutoring metrics and scored 1,000 ConvoLearn tutoring dialogues on every metric with three LLM judges: GPT-5.4 mini, Claude Haiku 4.5, and Gemma 4 26B. We then ran an exploratory factor analysis (EFA) for each judge. This repository lets you:

1. **Reproduce the paper's factor analysis from the published correlation matrices.** No API keys and no conversation data are needed.
2. **Rerun scoring and analysis on the ConvoLearn dialogues.** This needs the dataset and API access, and LLM judges are not deterministic, so expect close but not identical results.

No conversation text or per-conversation judge scores are included.

---

## Contents

```
.
├── README.md
├── LICENSE                         Apache-2.0
├── CITATION.cff
├── requirements.txt                Python packages (analysis and scoring)
├── ProposedMetricFramework.csv     the final framework: 10 categories, 41 items, prompts, references (Table E1)
├── Tutor Metrics.csv               the 241-row metric inventory (177 are scored)
├── evaluate-metrics.py             LLM-as-judge scoring, one call per conversation × metric
├── evaluation_utils.py             judge prompt, response parsing, ConvoLearn loader
├── convolearn/
│   └── split_convolearn.py         seeded EFA / CFA / human-validation split
├── efa/
│   ├── analyze-efa.py              EFA pipeline (scores or a correlation matrix as input)
│   ├── psych-diagnostics.R         R psych fit: MINRES/oblimin, parallel analysis, Schmid–Leiman
│   ├── analyze-conditional.py      extension analysis for conditional metrics
│   ├── conditional-extension.R
│   ├── cross-judge-congruence.py   Tucker congruence between judges
│   ├── export-correlation-matrix.py
│   ├── judge-descriptives.py       judge scale-use statistics
│   ├── compare-reliability.py      pairwise judge agreement
│   ├── render/                     Word-table (render-apa-*.R) and figure (render-*.py) renderers
│   ├── test_efa_math.py            regression tests
│   ├── renv.lock, .Rprofile, renv/ pinned R environment
│   └── results/factor_analysis/    published inputs (below); analysis outputs are written here too
└── paper/
    ├── figures/                    Figures 1–2 exactly as in the paper (PDF)
    └── tables/                     Tables 1–3, B1, C1–C3, D1–D3, S1–S3 (Word)
```

### Published data (`efa/results/factor_analysis/`)

| File | Contents | Used for |
|---|---|---|
| `correlation_matrices/item_correlations_<judge>.csv` | Pearson correlations among each judge's core metrics, at full double precision: the exact matrix each factor analysis was fitted on | All EFA results |
| `correlation_matrices/item_correlations_<judge>_meta.csv` | N = 1,000 conversations and the preprocessing thresholds | Refitting |
| `preprocessing/preprocessing_audit_<judge>_oblimin_n<k>.csv` | Per-metric score, N/A, and error counts, and each metric's tier (core, conditional, or dropped) | Table 1 (right), Table B1, Figure 1 |
| `conditional/conditional_extension_loadings_<judge>_oblimin_n<k>.csv` | Each conditional metric's extension loading on each core factor | Tables D1–D3 |
| `conditional/conditional_descriptives_<judge>_oblimin_n<k>.csv` | Applicable N, mean, SD, and strongest factor for each conditional metric | Tables D1–D3 |
| `congruence/congruence_construct_labels.csv` | Construct names mapped to factor numbers per judge (analyst decision) | Tables 3, D1–D3 |

`<judge>` is `gpt-5.4-mini`, `claude-haiku-4-5`, or `gemma4-26b`.

### Judge models

| Judge | Model version | Access |
|---|---|---|
| GPT-5.4 mini | `gpt-5.4-mini-2026-03-17` | OpenAI Batch API |
| Claude Haiku 4.5 | `claude-haiku-4-5-20251001` | Anthropic Message Batches API |
| Gemma 4 26B | `gemma4:26b` (open weights) | Local OpenAI-compatible server (Ollama) |

---

## Setup

Requirements: **Python 3.11** and **R 4.5.3**. Run all commands from the repository root unless a step says otherwise.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# R packages (psych, GPArotation, flextable, officer) at the exact pinned versions
cd efa && Rscript -e 'install.packages("renv", repos = "https://cloud.r-project.org"); renv::restore()' && cd ..
```

The Python scripts call R from `efa/`, so `efa/.Rprofile` loads the pinned library automatically. If `Rscript` isn't on your `PATH`, set `RSCRIPT=/path/to/Rscript`. Run the table renderers from inside `efa/` for the same reason.

Check the installation:

```bash
python -m unittest efa/test_efa_math.py -v
```

---

## 1. Reproduce the paper from the correlation matrices

`analyze-efa.py --correlation` refits a judge directly from its published matrix. It builds an in-memory data matrix whose correlation matrix equals the published one exactly, then runs the same pipeline used in the paper. Every factor-analytic result depends only on the correlation matrix and N, so the outputs match the published results.

```bash
C=efa/results/factor_analysis/correlation_matrices

python efa/analyze-efa.py --n-factors 7 \
  --correlation $C/item_correlations_gpt-5.4-mini.csv
python efa/analyze-efa.py --n-factors 9 \
  --correlation $C/item_correlations_claude-haiku-4-5.csv
python efa/analyze-efa.py --n-factors 7 \
  --correlation $C/item_correlations_gemma4-26b.csv

python efa/cross-judge-congruence.py                     # Table 3
(cd efa && Rscript render/render-apa-schmid-leiman.R)           # Table 2, general-factor panel
(cd efa && Rscript render/render-apa-conditional.R)             # Tables D1–D3
python efa/render/render-pipeline-figure.py                     # Figure 1
```

Each judge takes a few minutes, mostly for parallel analysis. The factor counts (7, 9, 7) are the Horn parallel-analysis recommendations, and the refit recomputes them.

### What you should get

Outputs go to `efa/results/factor_analysis/`. They are gitignored, so only the published inputs above are tracked. Each judge's run validates itself and stops with an error if any of its 33 mathematical checks fails (`validation/validation_report_*.csv`).

| Paper result | Regenerated file(s) | Compare with |
|---|---|---|
| Table 1 (right): core / conditional / dropped | `apa_tables/apa_preprocessing_tiers.docx` | `paper/tables/table1b_metric_applicability.docx` |
| Table 2: KMO, Bartlett χ² | `preprocessing/factorability_table_*.csv`, `apa_tables/apa_factorability.docx` | `table2a_factorability.docx` |
| Table 2: ωH, ωt, ECV, PUC | `general_factor/schmid_leiman_summary_*.csv`, `apa_tables/apa_schmid_leiman.docx` | `table2b_general_factor.docx` |
| §3.1: Kaiser 10/11/9, Horn 7/9/7 | `retention/eigenvalues_*.csv`, `retention/parallel_analysis_*.csv` | paper text |
| §3.2: factor correlations | `solution/factor_correlations_*.csv` | paper text |
| Table 3: cross-judge congruence | `congruence/cross_judge_congruence_pairs.csv`, `apa_tables/apa_cross_judge_congruence.docx` | `table3_congruence.docx` |
| Table B1: metric applicability by judge | `apa_tables/apa_metric_status.docx` | `tableB1_metric_applicability_by_judge.docx` |
| Tables C1–C3: loadings | `membership/factor_membership_*.csv`, `apa_tables/apa_factor_membership_*.docx` | `tableC1–C3_*.docx` |
| Tables S1–S3: complete pattern matrices | `solution/factor_loadings_*.csv`, `apa_tables/apa_pattern_matrix_*.docx` | `tableS1–S3_*.docx` |
| Tables D1–D3: conditional-metric mappings | `apa_tables/apa_conditional_mappings.docx` (run `cd efa && Rscript render/render-apa-conditional.R`) | `tablesD1-D3_conditional_mappings.docx` |
| Figure 1 | `descriptives/pipeline_figure.{png,pdf}` | `paper/figures/figure1_pipeline.pdf` |

We checked this ourselves:
- The text of every regenerated Word table above is identical to the file in `paper/tables/`.
- Figure 1 renders with identical content.
- Loadings, factor correlations, communalities, KMO, Bartlett's χ², ωH, ωt, and ECV agree with the original raw-score analysis to within 1e-8.

Tables D1–D3 are rendered from the published extension loadings rather than refit: extension analysis uses correlations between each conditional metric and the core metrics over the conversations where that metric applied, which are not part of the published matrices. Refitting it requires the scores (Section 2).

**Not reproducible from the published data** (committed in `paper/` only): Table 1 (left) and Figure 2, the judge score distributions and agreement. These need per-conversation scores (Section 2). Also, `psych::fa.parallel` isn't bit-reproducible, so the simulated eigenvalues in `parallel_analysis_*.csv` vary by about .01 between runs. The retained factor counts do not change.

---

## 2. Rerun scoring and analysis from the dialogues

This reruns the computational pipeline: scoring, factor analysis, and rendering. LLM judges aren't deterministic and hosted model versions can be retired, so expect close, not identical, results. The inductive categories (Table A1), the crosswalk (Table 4), and the final framework (Table E1) came from qualitative coding and analyst review, not from code.

### 2a. Get ConvoLearn and make the EFA split

The paper used the ConvoLearn dataset (Sharma et al., 2026, arXiv:2601.08950) in the 2,155-dialogue version provided by its authors. Save it as `convolearn/convolearn-full.csv`, with a `conversationText` column, then run:

```bash
python convolearn/split_convolearn.py
```

This writes `convolearn_efa.csv` (1,000 dialogues), `convolearn_human.csv` (500), and `convolearn_cfa.csv` (the rest), using seed 42. Conversation IDs are row numbers in the full file, so the open 2,134-dialogue release produces a different split.

### 2b. Score with the three judges

```bash
export OPENAI_API_KEY=...        # GPT-5.4 mini
export ANTHROPIC_API_KEY=...     # Claude Haiku 4.5
```

`evaluate-metrics.py` selects the 177 scored metrics from `Tutor Metrics.csv` by source and metric filters. Each judge scores all 177 metrics independently, one judge call per conversation × metric. The prompt is in Appendix F of the paper (`SCORING_PREAMBLE` and `build_messages` in `evaluation_utils.py`). Results go to `efa/results/scores/`.

The hosted judges run through batch APIs in four 250-conversation chunks. Submit, status, and retrieve are separate commands; `<batch_id>` is the ID printed at submit.

**GPT-5.4 mini** (OpenAI Batch API):

```bash
for off in 0 250 500 750; do
  python evaluate-metrics.py --batch submit --input convolearn/convolearn_efa.csv \
    --offset $off --limit 250 --model openai:/gpt-5.4-mini
done
python evaluate-metrics.py --batch status   --batch-id <batch_id>
python evaluate-metrics.py --batch retrieve --batch-id <batch_id>    # once per chunk, when completed
```

**Claude Haiku 4.5** (Anthropic Message Batches API). Submit prints the batch ID and the path of an id-map file; pass that id-map back when you retrieve.

```bash
for off in 0 250 500 750; do
  python evaluate-metrics.py --batch submit --input convolearn/convolearn_efa.csv \
    --offset $off --limit 250 --model anthropic:/claude-haiku-4-5-20251001
done
python evaluate-metrics.py --batch retrieve --batch-id <msgbatch_id> \
  --model anthropic:/claude-haiku-4-5-20251001 --id-map <idmap.jsonl printed at submit>
```

**Gemma 4 26B** (local OpenAI-compatible server, such as Ollama; synchronous):

```bash
python evaluate-metrics.py --input convolearn/convolearn_efa.csv \
  --model local:/gemma4:26b --base-url http://localhost:11434/v1
```

### 2c. Factor analysis

For each judge, pass all of its score chunks. The pipeline keeps metrics present in every chunk, drops high-error metrics, and assigns each metric a tier:
- **Core** (≤ 10% not scored): mean-imputed and included in the EFA.
- **Conditional** (scored in ≥ 200 conversations): analyzed separately with extension analysis.
- **Dropped:** everything else.

```bash
S=efa/results/scores
python efa/analyze-efa.py --input $S/efa_evaluation_results_gpt-5.4-mini*.csv   --n-factors 7
python efa/analyze-efa.py --input $S/efa_evaluation_results_claude-haiku*.csv  --n-factors 9
python efa/analyze-efa.py --input $S/efa_evaluation_results_gemma4-26b_<timestamp>.csv --n-factors 7
```

Sync-mode scoring (Gemma) also writes a `_metadata.csv` next to the results file, so pass the results file explicitly rather than a glob. Run once without `--n-factors` to see the parallel-analysis recommendation (`retention/parallel_analysis_*.csv`), then refit with that count. Rerun `analyze-conditional.py`, `judge-descriptives.py`, and `export-correlation-matrix.py` with the same `--input` for each judge:

```bash
python efa/analyze-conditional.py        --input <same files> --n-factors <n>   # Tables D1–D3 inputs
python efa/judge-descriptives.py         --input <same files>                   # Table 1 (left) inputs
python efa/export-correlation-matrix.py  --input <same files>                   # publishable matrix
```

After all three judges:

```bash
python efa/cross-judge-congruence.py      # Table 3
python efa/judge-descriptives.py --compare  # Figure 2 (left) and judge_distribution_comparison.csv
python efa/render/render-pipeline-figure.py      # Figure 1
cd efa
Rscript render/render-apa-judge-distributions.R  # Table 1 (left)
Rscript render/render-apa-schmid-leiman.R        # Table 2 (right)
Rscript render/render-apa-conditional.R          # Tables D1–D3
cd ..
```

Figure 2 (right) compares judges pairwise on the same conversation × metric pairs. First combine each judge's chunks into one file (for example with `pandas.concat`), then run:

```bash
python efa/compare-reliability.py --runs gpt.csv haiku.csv gemma.csv \
  --labels "GPT-5.4 mini" "Haiku 4.5" "Gemma 4 26B" \
  --output efa/results/reliability/compare_3judges_all1000.png
```

`analyze-efa.py` renders Tables 1 (right), 2 (left), B1, C1–C3, and S1–S3 automatically, and `cross-judge-congruence.py` renders Table 3. After rerunning, check that every run's `validation_report_*.csv` passes. Output names from a rerun include the batch ID and timestamp of the score files; rename them to the short judge names to use them with Section 1.

---

## Using the proposed framework

`ProposedMetricFramework.csv` is the paper's final framework (Table E1): 10 categories and 41 items, each with the judge prompt and supporting references. It uses the same `Category`, `Subcategory`, `Prompt` columns the scorer reads, so you can score tutoring conversations on it directly:

```bash
python evaluate-metrics.py --metrics ProposedMetricFramework.csv \
  --input <conversations.csv or folder of JSON conversations> --model openai:/gpt-5.4-mini
```

The framework is proposed for confirmatory factor analysis and human validation; it has not been validated as a scale yet.

## Method summary

- **Correlation:** Pearson, on core metrics with residual gaps mean-imputed
- **Factorability:** KMO and Bartlett's test (`psych`)
- **Retention:** Horn parallel analysis (`psych::fa.parallel`, MINRES, 95th percentile of simulated eigenvalues, 100 iterations); the Kaiser criterion is reported for reference
- **Extraction and rotation:** MINRES with oblimin rotation (`psych::fa`); salient loadings are |λ| ≥ .40
- **Cross-check:** Python `factor_analyzer` independently fits the same model; its solution must match R (Tucker congruence ≥ .85)
- **General factor:** Schmid–Leiman transformation (`psych::omega`), reporting ωH, ωt, ECV, and PUC
- **Conditional metrics:** extension analysis onto the core factors (`psych::fa.extension`)
- **Cross-judge replication:** Tucker's congruence with Hungarian matching of factors

R `psych` is the authoritative engine; all reported values come from it.

## Citation

See `CITATION.cff`.

## License

Apache License 2.0. See `LICENSE`.
