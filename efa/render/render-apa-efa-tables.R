#!/usr/bin/env Rscript

# Render editable APA-style EFA membership tables from analyze-efa.py CSV outputs.

required_packages <- c("flextable", "officer")
missing <- required_packages[!vapply(required_packages, requireNamespace,
                                    quietly = TRUE, FUN.VALUE = logical(1))]
if (length(missing)) {
  stop("Missing R package(s): ", paste(missing, collapse = ", "),
       ". Install them with install.packages(c(\"flextable\", \"officer\")).", call. = FALSE)
}

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_path <- if (length(script_arg)) sub("^--file=", "", script_arg[1]) else "efa/render/render-apa-efa-tables.R"
root <- normalizePath(file.path(dirname(script_path), "..", ".."), mustWork = FALSE)

results_dir <- Sys.getenv("EFA_RESULTS_DIR", unset = file.path(root, "efa", "results", "factor_analysis"))
output_dir <- Sys.getenv("EFA_APA_TABLE_DIR", unset = file.path(results_dir, "apa_tables"))
results_dir <- normalizePath(results_dir, mustWork = TRUE)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Optional single-file argument: `Rscript render-apa-efa-tables.R <membership.csv>` renders
# just that file (used by analyze-efa.py to render the current run); with no argument it
# renders every oblimin membership CSV in results_dir.
cli_args <- commandArgs(trailingOnly = TRUE)
if (length(cli_args) >= 1 && nzchar(cli_args[1])) {
  membership_paths <- normalizePath(cli_args[1], mustWork = TRUE)
} else {
  membership_paths <- list.files(
    file.path(results_dir, "membership"),
    pattern = "^factor_membership_.*_oblimin(_n[0-9]+)?\\.csv$",
    full.names = TRUE
  )
}
if (!length(membership_paths)) {
  stop("No oblimin factor-membership CSVs found in ", results_dir, call. = FALSE)
}

clean_metric <- function(metric_id) {
  categories <- c(
    "Overall_Rating", "Accuracy", "Alignment_to_Constraints", "Instructional_Support",
    "Assessment", "Mistake_Handling", "Affect_and_Relational_Support",
    "Engagement_and_Motivation", "Adaptivity", "Understanding_Learner_Goals",
    "Metacognition", "Safety"
  )
  for (category in categories) {
    prefix <- paste0(category, "_")
    if (startsWith(metric_id, prefix)) {
      metric <- substring(metric_id, nchar(prefix) + 1)
      return(paste0(gsub("_", " ", category), " - ", gsub("_", " ", metric)))
    }
  }
  gsub("_", " ", metric_id)
}

# Exhibit numbers for the per-judge loading tables (C1/C2/C3 in the paper), keyed by judge so
# the three documents cannot collide.
table_number_for <- function(model_name) {
  prefix <- Sys.getenv("EFA_TABLE_NUMBER_PREFIX", unset = "C")
  n <- if (grepl("GPT", model_name)) 1 else if (grepl("Haiku", model_name)) 2 else
       if (grepl("Gemma", model_name)) 3 else NA_integer_
  if (is.na(n)) paste0("Table ", prefix, "#") else paste0("Table ", prefix, n)
}

format_model_name <- function(path) {
  name <- sub("^factor_membership_", "", basename(path))
  name <- sub("_oblimin(_n[0-9]+)?\\.csv$", "", name)
  name <- sub("_merged[0-9]+$", "", name)
  name <- sub("_batch_.*$", "", name)
  name <- sub("(_\\d{8}_\\d{6}.*)$", "", name)  # Gemma: strip trailing timestamp
  known <- c(
    "gpt-5.4-mini"     = "GPT-5.4 mini",
    "claude-haiku-4-5" = "Claude Haiku 4.5",
    "gemma4-26b"       = "Gemma 4 26B"
  )
  for (key in names(known)) if (startsWith(name, key)) return(known[[key]])
  gsub("-", " ", name)
}

make_table_data <- function(input) {
  input$factor_number <- as.integer(sub("Factor_", "", input$factor))
  input <- input[order(input$factor_number, -abs(input$loading)), ]
  rows <- list()
  for (factor in unique(input$factor)) {
    block <- input[input$factor == factor, ]
    strength <- unique(block$relative_loading_strength_pct_ss_div_p)[1]
    header <- data.frame(
      Factor = paste0(gsub("_", " ", factor), " (", nrow(block), " salient metrics; relative loading strength ",
                      sprintf("%.2f%%", strength), ")"),
      Loading = "",
      Metric = "",
      kind = "factor",
      stringsAsFactors = FALSE
    )
    metrics <- data.frame(
      Factor = "",
      Loading = sprintf("%+.2f", block$loading),
      Metric = vapply(block$metric_id, clean_metric, character(1)),
      kind = "metric",
      stringsAsFactors = FALSE
    )
    rows[[length(rows) + 1]] <- rbind(header, metrics)
  }
  do.call(rbind, rows)
}

for (membership_path in membership_paths) {
  membership <- read.csv(membership_path, check.names = FALSE, stringsAsFactors = FALSE)
  required_columns <- c("factor", "metric_id", "loading", "relative_loading_strength_pct_ss_div_p")
  if (!all(required_columns %in% names(membership))) {
    stop("Unexpected membership CSV columns in ", membership_path, call. = FALSE)
  }
  table_data <- make_table_data(membership)
  model_name <- format_model_name(membership_path)

  ft <- flextable::flextable(table_data[, c("Factor", "Loading", "Metric")])
  ft <- flextable::set_header_labels(ft, Factor = "Factor", Loading = "Pattern loading", Metric = "Metric")
  ft <- flextable::theme_booktabs(ft)
  ft <- flextable::font(ft, fontname = "Arial", part = "all")
  ft <- flextable::fontsize(ft, size = 10, part = "all")
  ft <- flextable::align(ft, j = "Loading", align = "right", part = "all")
  ft <- flextable::align(ft, j = c("Factor", "Metric"), align = "left", part = "all")
  ft <- flextable::bold(ft, part = "header")
  factor_rows <- which(table_data$kind == "factor")
  ft <- flextable::bold(ft, i = factor_rows, part = "body")
  for (row in factor_rows) {
    ft <- flextable::merge_at(ft, i = row, j = 1:3, part = "body")
  }
  ft <- flextable::padding(ft, padding.top = 4, padding.bottom = 4,
                           padding.left = 5, padding.right = 5, part = "all")
  ft <- flextable::autofit(ft)
  ft <- flextable::width(ft, j = 1, width = 1.30)
  ft <- flextable::width(ft, j = 2, width = 1.05)
  ft <- flextable::width(ft, j = 3, width = 4.90)
  ft <- flextable::set_table_properties(ft, layout = "fixed")
  ft <- flextable::add_footer_lines(
    ft,
    values = "Note. Pattern loadings from an exploratory factor analysis using MINRES extraction and oblimin rotation. Only absolute loadings >= .40 are displayed."
  )
  ft <- flextable::fontsize(ft, size = 9, part = "footer")
  ft <- flextable::italic(ft, part = "footer")

  doc <- officer::read_docx()
  doc <- officer::body_add_par(doc, table_number_for(model_name), style = "Normal")
  doc <- officer::body_add_par(doc,
    paste0("Oblimin pattern loadings for ", model_name), style = "Normal")
  doc <- flextable::body_add_flextable(doc, value = ft)
  doc <- officer::body_end_section_continuous(doc)

  output_path <- file.path(output_dir, paste0("apa_", sub("\\.csv$", ".docx", basename(membership_path))))
  print(doc, target = output_path)
  message("Wrote ", output_path)
}
