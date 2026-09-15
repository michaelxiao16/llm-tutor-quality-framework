#!/usr/bin/env Rscript

# Render an editable APA-style Schmid–Leiman / general-factor summary table as a Word document
# from analyze-efa.py's schmid_leiman_summary_*.csv outputs.

required_packages <- c("flextable", "officer")
missing <- required_packages[!vapply(required_packages, requireNamespace,
                                    quietly = TRUE, FUN.VALUE = logical(1))]
if (length(missing)) {
  stop("Missing R package(s): ", paste(missing, collapse = ", "),
       ". Install them with install.packages(c(\"flextable\", \"officer\")).", call. = FALSE)
}

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_path <- if (length(script_arg)) sub("^--file=", "", script_arg[1]) else "efa/render/render-apa-schmid-leiman.R"
root <- normalizePath(file.path(dirname(script_path), "..", ".."), mustWork = FALSE)

results_dir <- Sys.getenv("EFA_RESULTS_DIR", unset = file.path(root, "efa", "results", "factor_analysis"))
output_dir <- Sys.getenv("EFA_APA_TABLE_DIR", unset = file.path(results_dir, "apa_tables"))
results_dir <- normalizePath(results_dir, mustWork = TRUE)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

summary_paths <- list.files(file.path(results_dir, "general_factor"), pattern = "^schmid_leiman_summary_.*\\.csv$", full.names = TRUE)
if (!length(summary_paths)) {
  stop("No schmid_leiman_summary_*.csv files found in ", results_dir,
       ". Run analyze-efa.py first.", call. = FALSE)
}

format_model_name <- function(path) {
  name <- sub("^schmid_leiman_summary_", "", basename(path))
  name <- sub("_oblimin(_n[0-9]+)?\\.csv$", "", name)
  name <- sub("_merged[0-9]+$", "", name)
  name <- sub("_batch_.*$", "", name)
  name <- sub("(_\\d{8}_\\d{6}.*)$", "", name)
  known <- c("gpt-5.4-mini" = "GPT-5.4 mini",
             "claude-haiku-4-5" = "Claude Haiku 4.5",
             "gemma4-26b" = "Gemma 4 26B")
  for (key in names(known)) if (startsWith(name, key)) return(known[[key]])
  gsub("-", " ", name)
}
model_rank <- function(name) {
  if (grepl("GPT", name)) return(1); if (grepl("Haiku", name)) return(2)
  if (grepl("Gemma", name)) return(3); 4
}
no_lead_zero <- function(x) sub("^(-?)0", "\\1", x)
f2 <- function(x) no_lead_zero(sprintf("%.2f", as.numeric(x)))  # bounded [0,1]: drop leading zero
f3 <- function(x) no_lead_zero(sprintf("%.3f", as.numeric(x)))  # omega-total sits near 1; 3 dp avoids a misleading 1.00

rows <- lapply(summary_paths, function(p) {
  d <- read.csv(p, check.names = FALSE, stringsAsFactors = FALSE)[1, ]
  model <- format_model_name(p)
  data.frame(
    Model   = model,
    Items   = format(as.integer(d$n_items), big.mark = ","),
    Factors = as.character(as.integer(d$n_group_factors)),
    OmegaH  = f3(d$omega_hierarchical),  # 3 dp so the near-equal values are distinguishable
    OmegaT  = f3(d$omega_total),
    ECV     = f2(d$ecv_general),
    PUC     = f2(d$puc_general),
    rank    = model_rank(model),
    stringsAsFactors = FALSE
  )
})
table_data <- do.call(rbind, rows)
table_data <- table_data[order(table_data$rank), ]
table_data$rank <- NULL

ft <- flextable::flextable(table_data)
ft <- flextable::set_header_labels(
  ft, Model = "Judge model", Items = "Items", Factors = "Group factors",
  OmegaH = "ωH", OmegaT = "ωt", ECV = "ECV", PUC = "PUC"
)
ft <- flextable::theme_booktabs(ft)
ft <- flextable::font(ft, fontname = "Arial", part = "all")
ft <- flextable::fontsize(ft, size = 10, part = "all")
ft <- flextable::bold(ft, part = "header")
ft <- flextable::align(ft, j = "Model", align = "left", part = "all")
ft <- flextable::align(ft, j = c("Items", "Factors", "OmegaH", "OmegaT", "ECV", "PUC"),
                       align = "right", part = "all")
ft <- flextable::padding(ft, padding.top = 4, padding.bottom = 4,
                         padding.left = 5, padding.right = 5, part = "all")
ft <- flextable::autofit(ft)
ft <- flextable::width(ft, j = "Model", width = 1.70)
ft <- flextable::add_footer_lines(
  ft,
  values = paste(
    "Note. Bifactor indices from a Schmid–Leiman transformation of the oblique (MINRES, oblimin)",
    "solution (psych::omega), fit on each judge's full retained core-metric matrix. ωH =",
    "omega-hierarchical, the proportion of total-score variance attributable to the general factor;",
    "ωt = omega-total; ECV = explained common variance of the general factor; PUC = percent of",
    "uncontaminated correlations (item pairs spanning different group factors, whose covariance",
    "reflects only the general factor). Per Reise, Scheines, Widaman, and Haviland (2013), a high",
    "PUC (> .80) means multidimensionality has little effect on parameter bias; combined with",
    "ωH > .80 and ECV > .70, the general factor is interpretable as essentially unidimensional",
    "(see also Rodriguez, Reise, & Haviland, 2016). ωt approaches 1.0 because of the large item",
    "pool and should not be read as scale precision."
  )
)
ft <- flextable::fontsize(ft, size = 9, part = "footer")
ft <- flextable::italic(ft, part = "footer")

# Exhibit number, overridable so renumbering the manuscript needs no code change.
table_number <- Sys.getenv("EFA_TABLE_NUMBER", unset = "Table 2")
doc <- officer::read_docx()
doc <- officer::body_add_par(doc, table_number, style = "Normal")
doc <- officer::body_add_par(doc,
  "General-Factor (Schmid–Leiman Bifactor) Indices by Judge Model", style = "Normal")
doc <- flextable::body_add_flextable(doc, value = ft)
doc <- officer::body_end_section_continuous(doc)

output_path <- file.path(output_dir, "apa_schmid_leiman.docx")
print(doc, target = output_path)
message("Wrote ", output_path)
