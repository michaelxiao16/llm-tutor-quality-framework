#!/usr/bin/env Rscript

# Render an editable APA-style metric-tiering summary table as a Word document from analyze-
# efa.py's preprocessing_audit_*.csv outputs.

required_packages <- c("flextable", "officer")
missing <- required_packages[!vapply(required_packages, requireNamespace,
                                    quietly = TRUE, FUN.VALUE = logical(1))]
if (length(missing)) {
  stop("Missing R package(s): ", paste(missing, collapse = ", "),
       ". Install them with install.packages(c(\"flextable\", \"officer\")).", call. = FALSE)
}

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_path <- if (length(script_arg)) sub("^--file=", "", script_arg[1]) else "efa/render/render-apa-preprocessing.R"
root <- normalizePath(file.path(dirname(script_path), "..", ".."), mustWork = FALSE)

results_dir <- Sys.getenv("EFA_RESULTS_DIR", unset = file.path(root, "efa", "results", "factor_analysis"))
output_dir <- Sys.getenv("EFA_APA_TABLE_DIR", unset = file.path(results_dir, "apa_tables"))
results_dir <- normalizePath(results_dir, mustWork = TRUE)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

audit_paths <- list.files(
  file.path(results_dir, "preprocessing"),
  pattern = "^preprocessing_audit_.*\\.csv$",
  full.names = TRUE
)
if (!length(audit_paths)) {
  stop("No preprocessing_audit_*.csv files found in ", results_dir,
       ". Run analyze-efa.py first.", call. = FALSE)
}

format_model_name <- function(path) {
  name <- sub("^preprocessing_audit_", "", basename(path))
  name <- sub("_oblimin(_n[0-9]+)?\\.csv$", "", name)
  name <- sub("_merged[0-9]+$", "", name)
  name <- sub("_batch_.*$", "", name)
  name <- sub("(_\\d{8}_\\d{6}.*)$", "", name)
  known <- c(
    "gpt-5.4-mini"     = "GPT-5.4 mini",
    "claude-haiku-4-5" = "Claude Haiku 4.5",
    "gemma4-26b"       = "Gemma 4 26B"
  )
  for (key in names(known)) if (startsWith(name, key)) return(known[[key]])
  gsub("-", " ", name)
}

model_rank <- function(name) {
  if (grepl("GPT", name)) return(1)
  if (grepl("Haiku", name)) return(2)
  if (grepl("Gemma", name)) return(3)
  4
}

rows <- lapply(audit_paths, function(p) {
  a <- read.csv(p, check.names = FALSE, stringsAsFactors = FALSE)
  a <- a[as.logical(a$present_in_all_input_chunks), ]  # metrics scored in every chunk (177)
  tier <- a$applicability_tier
  core <- sum(tier == "core")
  conditional <- sum(tier == "conditional")
  dropped <- sum(!tier %in% c("core", "conditional"))
  model <- format_model_name(p)
  data.frame(
    Model = model,
    Total = format(nrow(a), big.mark = ","),
    Core = format(core, big.mark = ","),
    Conditional = format(conditional, big.mark = ","),
    Dropped = format(dropped, big.mark = ","),
    rank = model_rank(model),
    stringsAsFactors = FALSE
  )
})
table_data <- do.call(rbind, rows)
table_data <- table_data[order(table_data$rank), ]
table_data$rank <- NULL

ft <- flextable::flextable(table_data)
ft <- flextable::set_header_labels(
  ft, Model = "Judge model", Total = "Total metrics",
  # "Pooled" is pipeline vocabulary that appears nowhere in the manuscript; the note below
  # carries the joint-versus-per-metric distinction the parenthetical was doing.
  Core = "Core", Conditional = "Conditional", Dropped = "Dropped"
)
ft <- flextable::theme_booktabs(ft)
ft <- flextable::font(ft, fontname = "Arial", part = "all")
ft <- flextable::fontsize(ft, size = 10, part = "all")
ft <- flextable::bold(ft, part = "header")
ft <- flextable::align(ft, j = "Model", align = "left", part = "all")
ft <- flextable::align(ft, j = c("Total", "Core", "Conditional", "Dropped"),
                       align = "right", part = "all")
ft <- flextable::padding(ft, padding.top = 4, padding.bottom = 4,
                         padding.left = 5, padding.right = 5, part = "all")
ft <- flextable::autofit(ft)
ft <- flextable::width(ft, j = "Model", width = 1.70)
ft <- flextable::add_footer_lines(
  ft,
  values = paste(
    "Note. The six MinorBench safety metrics were excluded a priori from all judges (harm-topic",
    "gates that are non-applicable in nearly all tutoring conversations), leaving 177 candidate",
    "metrics, which were then tiered by applicability. Core = non-applicable in ≤ 10% of",
    "conversations; analyzed together in a single factor analysis across all conversations, with",
    "sporadic gaps mean-imputed. Conditional = non-applicable in > 10% but scored in ≥ 200",
    "conversations; held out of that joint matrix because the missingness is structural (MNAR) and",
    "instead related to the core factors by extension analysis, each metric fitted over only the",
    "conversations in which it was scored. Dropped = scored in fewer",
    "than 200 conversations. Judges analyze different numbers of core metrics because they mark",
    "conversations non-applicable at different rates."
  )
)
ft <- flextable::fontsize(ft, size = 9, part = "footer")
ft <- flextable::italic(ft, part = "footer")

# Exhibit number, overridable so renumbering the manuscript needs no code change.
table_number <- Sys.getenv("EFA_TABLE_NUMBER", unset = "Table 1")
doc <- officer::read_docx()
doc <- officer::body_add_par(doc, table_number, style = "Normal")
doc <- officer::body_add_par(doc,
  "Metric Applicability Tiers by Judge Model", style = "Normal")
doc <- flextable::body_add_flextable(doc, value = ft)
doc <- officer::body_end_section_continuous(doc)

output_path <- file.path(output_dir, "apa_preprocessing_tiers.docx")
print(doc, target = output_path)
message("Wrote ", output_path)
