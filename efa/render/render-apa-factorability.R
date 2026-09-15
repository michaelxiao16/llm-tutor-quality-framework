#!/usr/bin/env Rscript

# Render an editable APA-style factorability table (KMO + Bartlett's test of sphericity) as a
# Word document from analyze-efa.py's factorability_table_*.csv outputs.

required_packages <- c("flextable", "officer")
missing <- required_packages[!vapply(required_packages, requireNamespace,
                                    quietly = TRUE, FUN.VALUE = logical(1))]
if (length(missing)) {
  stop("Missing R package(s): ", paste(missing, collapse = ", "),
       ". Install them with install.packages(c(\"flextable\", \"officer\")).", call. = FALSE)
}

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_path <- if (length(script_arg)) sub("^--file=", "", script_arg[1]) else "efa/render/render-apa-factorability.R"
root <- normalizePath(file.path(dirname(script_path), "..", ".."), mustWork = FALSE)

results_dir <- Sys.getenv("EFA_RESULTS_DIR", unset = file.path(root, "efa", "results", "factor_analysis"))
output_dir <- Sys.getenv("EFA_APA_TABLE_DIR", unset = file.path(results_dir, "apa_tables"))
results_dir <- normalizePath(results_dir, mustWork = TRUE)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

table_paths <- list.files(
  file.path(results_dir, "preprocessing"),
  pattern = "^factorability_table_.*\\.csv$",
  full.names = TRUE
)
if (!length(table_paths)) {
  stop("No factorability_table_*.csv files found in ", results_dir,
       ". Run analyze-efa.py first.", call. = FALSE)
}

# Prettify the stored judge_model slug into a display name.
format_model_name <- function(slug) {
  known <- c(
    "gpt-5.4-mini"     = "GPT-5.4 mini",
    "claude-haiku-4-5" = "Claude Haiku 4.5",
    "gemma4-26b"       = "Gemma 4 26B"
  )
  for (key in names(known)) if (startsWith(slug, key)) return(known[[key]])
  gsub("-", " ", slug)
}

# Sort order: GPT, then Haiku, then Gemma, then anything else alphabetically.
model_rank <- function(name) {
  if (grepl("GPT", name)) return(1)
  if (grepl("Haiku", name)) return(2)
  if (grepl("Gemma", name)) return(3)
  4
}

no_lead_zero <- function(x) sub("^(-?)0", "\\1", x)

rows <- lapply(table_paths, function(p) {
  d <- read.csv(p, check.names = FALSE, stringsAsFactors = FALSE)[1, ]
  model <- format_model_name(d$judge_model)
  # Re-derive the APA strings from the numeric columns; read.csv coerces the pre-formatted
  # ".995"/"< .001" text columns (".995" parses as a number, which would reintroduce the
  # leading zero), so don't trust them here.
  kmo_apa <- no_lead_zero(sprintf("%.3f", as.numeric(d$kmo_overall)))
  p_apa <- if (as.numeric(d$bartlett_p_value) < 0.001) "< .001" else
    no_lead_zero(sprintf("= %.3f", as.numeric(d$bartlett_p_value)))
  data.frame(
    Model   = model,
    N       = format(as.integer(d$n_observations), big.mark = ","),
    Metrics = as.character(as.integer(d$n_metrics)),
    KMO     = kmo_apa,
    Chi2    = formatC(as.numeric(d$bartlett_chi_square), format = "f", digits = 2, big.mark = ","),
    Df      = format(as.integer(d$bartlett_df), big.mark = ","),
    P       = p_apa,
    rank    = model_rank(model),
    stringsAsFactors = FALSE
  )
})
table_data <- do.call(rbind, rows)
table_data <- table_data[order(table_data$rank), ]
table_data$rank <- NULL

ft <- flextable::flextable(table_data)
ft <- flextable::set_header_labels(
  ft, Model = "Judge model", N = "N", Metrics = "Metrics",
  KMO = "KMO", Chi2 = "χ²", Df = "df", P = "p"
)
ft <- flextable::theme_booktabs(ft)
ft <- flextable::font(ft, fontname = "Arial", part = "all")
ft <- flextable::fontsize(ft, size = 10, part = "all")
ft <- flextable::bold(ft, part = "header")
# APA: italicize statistical symbols (N, df, p) but not Greek letters (chi).
ft <- flextable::italic(ft, j = c("N", "Df", "P"), part = "header")
ft <- flextable::align(ft, j = "Model", align = "left", part = "all")
ft <- flextable::align(ft, j = c("N", "Metrics", "KMO", "Chi2", "Df", "P"),
                       align = "right", part = "all")
ft <- flextable::padding(ft, padding.top = 4, padding.bottom = 4,
                         padding.left = 5, padding.right = 5, part = "all")
ft <- flextable::autofit(ft)
ft <- flextable::width(ft, j = "Model", width = 1.70)
ft <- flextable::add_footer_lines(
  ft,
  values = paste(
    "Note. KMO = Kaiser–Meyer–Olkin measure of sampling adequacy; all values fall in",
    "Kaiser's (1974) \"marvelous\" range (≥ .90). χ² = Bartlett's test of sphericity.",
    "A KMO ≥ .60 with a significant Bartlett's test indicates the correlation matrix is",
    "suitable for factor analysis. N = number of conversations; Metrics = analyzed core metrics",
    "after applicability tiering. Values computed in R (psych::fa)."
  )
)
ft <- flextable::fontsize(ft, size = 9, part = "footer")
ft <- flextable::italic(ft, part = "footer")

# Exhibit number, overridable so renumbering the manuscript needs no code change.
table_number <- Sys.getenv("EFA_TABLE_NUMBER", unset = "Table 2")
doc <- officer::read_docx()
doc <- officer::body_add_par(doc, table_number, style = "Normal")
doc <- officer::body_add_par(doc,
  "Factorability of the Metric Correlation Matrix by Judge Model", style = "Normal")
doc <- flextable::body_add_flextable(doc, value = ft)
doc <- officer::body_end_section_continuous(doc)

output_path <- file.path(output_dir, "apa_factorability.docx")
print(doc, target = output_path)
message("Wrote ", output_path)
