#!/usr/bin/env Rscript

# Render an editable APA-style table of each judge model's scale-use profile as a Word
# document, from judge-descriptives.py's judge_distribution_summary_*.csv outputs.

required_packages <- c("flextable", "officer")
missing <- required_packages[!vapply(required_packages, requireNamespace,
                                    quietly = TRUE, FUN.VALUE = logical(1))]
if (length(missing)) {
  stop("Missing R package(s): ", paste(missing, collapse = ", "),
       ". Install them with install.packages(c(\"flextable\", \"officer\")).", call. = FALSE)
}

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_path <- if (length(script_arg)) sub("^--file=", "", script_arg[1]) else "efa/render/render-apa-judge-distributions.R"
root <- normalizePath(file.path(dirname(script_path), "..", ".."), mustWork = FALSE)

results_dir <- Sys.getenv("EFA_RESULTS_DIR", unset = file.path(root, "efa", "results", "factor_analysis"))
output_dir <- Sys.getenv("EFA_APA_TABLE_DIR", unset = file.path(results_dir, "apa_tables"))
# Paper Table 1 (left panel, "Scale use"); the right panel is render-apa-preprocessing.R.
table_number <- Sys.getenv("EFA_TABLE_NUMBER", unset = "Table 1")
results_dir <- normalizePath(results_dir, mustWork = TRUE)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

summary_paths <- list.files(
  file.path(results_dir, "descriptives"),
  pattern = "^judge_distribution_summary_.*\\.csv$",
  full.names = TRUE
)
if (!length(summary_paths)) {
  stop("No judge_distribution_summary_*.csv files found in ",
       file.path(results_dir, "descriptives"),
       ". Run efa/judge-descriptives.py for each judge first.", call. = FALSE)
}

format_model_name <- function(slug) {
  known <- c(
    "gpt-5.4-mini"     = "GPT-5.4 mini",
    "claude-haiku-4-5" = "Claude Haiku 4.5",
    "gemma4-26b"       = "Gemma 4 26B"
  )
  for (key in names(known)) if (startsWith(slug, key)) return(known[[key]])
  gsub("-", " ", slug)
}

model_rank <- function(name) {
  if (grepl("GPT", name)) return(1)
  if (grepl("Haiku", name)) return(2)
  if (grepl("Gemma", name)) return(3)
  4
}

rows <- lapply(summary_paths, function(p) {
  d <- read.csv(p, check.names = FALSE, stringsAsFactors = FALSE)[1, ]
  model <- format_model_name(d$judge_model)
  # Kept deliberately narrow so the table fits a portrait text block; item counts are omitted
  # (already in the applicability-tier table).
  data.frame(
    Model     = model,
    Responses = format(as.integer(d$n_scored_values_total), big.mark = ","),
    PctNA     = sprintf("%.1f", as.numeric(d$pct_not_applicable)),
    # A 1-5 mean and a skew/kurtosis statistic can all exceed 1, so APA's drop-the-leading-
    # zero rule does not apply to any column here.
    M         = sprintf("%.2f", as.numeric(d$grand_mean)),
    SD        = sprintf("%.2f", as.numeric(d$pooled_sd)),
    Skew      = sprintf("%.2f", as.numeric(d$pooled_skewness)),
    Kurt      = sprintf("%.2f", as.numeric(d$pooled_kurtosis_excess)),
    rank      = model_rank(model),
    stringsAsFactors = FALSE
  )
})
table_data <- do.call(rbind, rows)
table_data <- table_data[order(table_data$rank), ]
table_data$rank <- NULL

ft <- flextable::flextable(table_data)
ft <- flextable::set_header_labels(
  ft,
  Model = "Judge model", Responses = "Scored", PctNA = "% N/A",
  M = "M", SD = "SD", Skew = "Skew", Kurt = "Kurtosis"
)
ft <- flextable::theme_booktabs(ft)
ft <- flextable::font(ft, fontname = "Arial", part = "all")
ft <- flextable::fontsize(ft, size = 10, part = "all")
ft <- flextable::bold(ft, part = "header")
# APA: italicize statistical symbols (M, SD, Mdn) but not plain words or percentages.
ft <- flextable::italic(ft, j = c("M", "SD"), part = "header")
ft <- flextable::align(ft, j = "Model", align = "left", part = "all")
ft <- flextable::align(ft, j = setdiff(names(table_data), "Model"),
                       align = "right", part = "all")
ft <- flextable::padding(ft, padding.top = 3, padding.bottom = 3,
                         padding.left = 3, padding.right = 3, part = "all")
# Fixed layout with explicit widths: autofit() sizes to content and overflows the page for a
# table this wide, so the widths below are set to total 6.35 in, inside a 6.5 in portrait text
# block.
ft <- flextable::set_table_properties(ft, layout = "fixed")
ft <- flextable::width(ft, j = "Model",     width = 1.60)
ft <- flextable::width(ft, j = "Responses", width = 0.80)
ft <- flextable::width(ft, j = "PctNA",     width = 0.62)
ft <- flextable::width(ft, j = "M",         width = 0.50)
ft <- flextable::width(ft, j = "SD",        width = 0.50)
ft <- flextable::width(ft, j = "Skew",      width = 0.80)
ft <- flextable::width(ft, j = "Kurt",      width = 0.85)
ft <- flextable::add_footer_lines(
  ft,
  values = paste(
    "Note. Each judge produced 177,000 metric-level judgments (177 metrics x 1,000",
    "conversations). Scored = judgments returning a 1-5 rating; % N/A = judgments the judge",
    "declined to rate because the rubric's situation did not arise. All remaining columns",
    "are computed on scored responses only, which are never mean-imputed here, so they",
    "describe each judge's actual use of the 1-5 scale. M = response-weighted grand mean,",
    "indexing judge severity/leniency; SD = standard deviation of the pooled response",
    "distribution; Skew and Kurtosis likewise describe that pooled distribution, so all",
    "four shape statistics share one unit of analysis. Kurtosis is excess kurtosis,",
    "for which 0 is the normal distribution: positive values indicate heavier tails",
    "and negative values lighter tails. All three judges spread responses broadly across",
    "the scale rather than concentrating them. The shape of each judge's response",
    "distribution is shown in Figure 2."
  )
)
ft <- flextable::fontsize(ft, size = 9, part = "footer")
ft <- flextable::italic(ft, part = "footer")

doc <- officer::read_docx()
doc <- officer::body_add_par(doc, table_number, style = "Normal")
doc <- officer::body_add_par(doc,
  "Judge Model Scale-Use and Item Distribution Profile", style = "Normal")
doc <- flextable::body_add_flextable(doc, value = ft)
doc <- officer::body_end_section_continuous(doc)

output_path <- file.path(output_dir, "apa_judge_distributions.docx")
print(doc, target = output_path)
message("Wrote ", output_path)
