#!/usr/bin/env Rscript

# Produce manuscript-ready, editable APA-style pattern-matrix tables from the reporting bundle
# written by analyze-efa.py.

if (!requireNamespace("flextable", quietly = TRUE) || !requireNamespace("officer", quietly = TRUE)) {
  stop("This renderer requires flextable and officer.", call. = FALSE)
}

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_path <- if (length(script_arg)) sub("^--file=", "", script_arg[1]) else "efa/render/render-apa-pattern-matrices.R"
project_root <- normalizePath(file.path(dirname(script_path), "..", ".."), mustWork = FALSE)
results_dir <- normalizePath(Sys.getenv("EFA_RESULTS_DIR", file.path(project_root, "efa", "results", "factor_analysis")), mustWork = TRUE)
output_dir <- Sys.getenv("EFA_APA_TABLE_DIR", file.path(results_dir, "apa_tables"))
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Complete per-judge pattern matrices.
table_number_for <- function(path) {
  prefix <- Sys.getenv("EFA_TABLE_NUMBER_PREFIX", unset = "S")
  label <- model_label(path)
  n <- if (grepl("GPT", label)) 1 else if (grepl("Haiku", label)) 2 else
       if (grepl("Gemma", label)) 3 else NA_integer_
  if (is.na(n)) paste0("Table ", prefix, "#") else paste0("Table ", prefix, n)
}

clean_metric <- function(x) {
  x <- gsub("_", " ", x)
  sub("([A-Za-z]+) ([A-Za-z]+) ([A-Za-z]+) (.*)", "\\1 \\2 \\3 - \\4", x)
}

model_label <- function(path) {
  name <- sub("^pattern_matrix_report_", "", basename(path))
  known <- c("gpt-5.4-mini" = "GPT-5.4 mini",
             "claude-haiku-4-5" = "Claude Haiku 4.5",
             "gemma4-26b" = "Gemma 4 26B")
  for (key in names(known)) if (startsWith(name, key)) return(known[[key]])
  sub("\\.csv$", "", name)
}

factor_number <- function(x) as.integer(sub("Factor_", "", x))

make_pattern_table <- function(path) {
  d <- read.csv(path, check.names = FALSE, stringsAsFactors = FALSE)
  display_cols <- grep("^Factor_[0-9]+_display_if_abs_ge_", names(d), value = TRUE)
  factors <- sub("_display_if_abs_ge_.*$", "", display_cols)
  factor_order <- order(vapply(factors, factor_number, integer(1)))
  display_cols <- display_cols[factor_order]
  factors <- factors[factor_order]
  d$factor_order <- factor_number(d$primary_factor)
  d <- d[order(d$factor_order, -d$primary_abs_loading, d$metric_id), ]

  body <- data.frame(
    Item = vapply(d$metric_id, clean_metric, character(1)),
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
  for (i in seq_along(display_cols)) {
    values <- d[[display_cols[i]]]
    body[[paste0("F", factor_number(factors[i]))]] <- ifelse(
      is.na(values), "", sprintf("%.2f", values)
    )
  }
  body[["h2"]] <- sprintf("%.2f", d$communality_h2)
  list(body = body, factor_names = names(body)[-c(1, ncol(body))])
}

# Explicit factor-count runs carry the `_n{count}` suffix (for example, `_oblimin_n7.csv`);
# diagnostic runs do not.
pattern_paths <- list.files(
  file.path(results_dir, "solution"),
  "^pattern_matrix_report_.*_oblimin(_n[0-9]+)?\\.csv$",
  full.names = TRUE
)
path_filter <- Sys.getenv("EFA_PATTERN_FILTER", unset = "")
if (nzchar(path_filter)) pattern_paths <- pattern_paths[grepl(path_filter, basename(pattern_paths), fixed = TRUE)]
if (!length(pattern_paths)) stop("No pattern-matrix reporting CSVs found.", call. = FALSE)

for (path in pattern_paths) {
  table_data <- make_pattern_table(path)
  body <- table_data$body
  factor_names <- table_data$factor_names
  n_factors <- length(factor_names)
  # Factor columns shrink only as needed; the table remains readable in landscape.
  factor_width <- min(0.58, 7.0 / n_factors)

  ft <- flextable::flextable(body)
  ft <- flextable::set_header_labels(ft, Item = "Item", h2 = "h²")
  ft <- flextable::theme_booktabs(ft)
  ft <- flextable::font(ft, fontname = "Arial", part = "all")
  ft <- flextable::fontsize(ft, size = if (n_factors > 15) 7 else 8, part = "all")
  ft <- flextable::bold(ft, part = "header")
  ft <- flextable::align(ft, j = "Item", align = "left", part = "all")
  ft <- flextable::align(ft, j = c(factor_names, "h2"), align = "right", part = "all")
  ft <- flextable::padding(ft, padding.top = 2, padding.bottom = 2,
                           padding.left = 3, padding.right = 3, part = "all")
  ft <- flextable::width(ft, j = "Item", width = 2.65)
  ft <- flextable::width(ft, j = factor_names, width = factor_width)
  ft <- flextable::width(ft, j = "h2", width = 0.42)
  ft <- flextable::set_table_properties(ft, layout = "fixed")
  ft <- flextable::add_footer_lines(
    ft,
    values = "Note. Pattern coefficients from MINRES exploratory factor analysis with oblimin rotation. Values with |loading| < .40 are suppressed. h² = communality. Factor correlations are reported separately."
  )
  ft <- flextable::fontsize(ft, size = 7, part = "footer")
  ft <- flextable::italic(ft, part = "footer")

  section <- officer::prop_section(
    page_size = officer::page_size(orient = "landscape"),
    page_margins = officer::page_mar(top = 0.55, bottom = 0.55, left = 0.55, right = 0.55)
  )
  doc <- officer::read_docx()
  doc <- officer::body_set_default_section(doc, section)
  doc <- officer::body_add_fpar(
    doc,
    officer::fpar(officer::ftext(table_number_for(path), officer::fp_text(font.family = "Arial", font.size = 10, bold = TRUE)))
  )
  doc <- officer::body_add_fpar(
    doc,
    officer::fpar(officer::ftext(
      paste0("Pattern Matrix for ", model_label(path)),
      officer::fp_text(font.family = "Arial", font.size = 10, italic = TRUE)
    ))
  )
  doc <- flextable::body_add_flextable(doc, ft)

  output <- file.path(output_dir, paste0("apa_pattern_matrix_", sub("^pattern_matrix_report_", "", sub("\\.csv$", ".docx", basename(path)))))
  print(doc, target = output)
  message("Wrote ", output)
}
