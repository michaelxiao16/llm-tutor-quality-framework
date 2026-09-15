#!/usr/bin/env Rscript

# Render an editable APA-style table ranking metrics by their mean general-factor (g) loading
# across the judge models, from the Schmid–Leiman solutions (schmid_leiman_loadings_*.csv, `g`
# column).

required_packages <- c("flextable", "officer")
missing <- required_packages[!vapply(required_packages, requireNamespace,
                                    quietly = TRUE, FUN.VALUE = logical(1))]
if (length(missing)) {
  stop("Missing R package(s): ", paste(missing, collapse = ", "),
       ". Install them with install.packages(c(\"flextable\", \"officer\")).", call. = FALSE)
}

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_path <- if (length(script_arg)) sub("^--file=", "", script_arg[1]) else "efa/render/render-apa-general-factor.R"
root <- normalizePath(file.path(dirname(script_path), "..", ".."), mustWork = FALSE)

results_dir <- Sys.getenv("EFA_RESULTS_DIR", unset = file.path(root, "efa", "results", "factor_analysis"))
output_dir <- Sys.getenv("EFA_APA_TABLE_DIR", unset = file.path(results_dir, "apa_tables"))
results_dir <- normalizePath(results_dir, mustWork = TRUE)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

loading_paths <- list.files(file.path(results_dir, "general_factor"),
                            pattern = "^schmid_leiman_loadings_.*\\.csv$", full.names = TRUE)
if (!length(loading_paths)) {
  stop("No schmid_leiman_loadings_*.csv found in ", file.path(results_dir, "general_factor"),
       ". Run analyze-efa.py first.", call. = FALSE)
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
      return(paste0(gsub("_", " ", category), " – ", gsub("_", " ", metric)))
    }
  }
  gsub("_", " ", metric_id)
}

# Collect g per metric across models.
g_by_metric <- list()
for (p in loading_paths) {
  d <- read.csv(p, check.names = FALSE, stringsAsFactors = FALSE)
  for (i in seq_len(nrow(d))) {
    m <- d$metric_id[i]
    g_by_metric[[m]] <- c(g_by_metric[[m]], as.numeric(d$g[i]))
  }
}

no_lead_zero <- function(x) sub("^(-?)0", "\\1", x)
metrics <- names(g_by_metric)
mean_g <- vapply(g_by_metric, mean, numeric(1))
n_models <- vapply(g_by_metric, length, integer(1))
ord <- order(-mean_g)
metrics <- metrics[ord]; mean_g <- mean_g[ord]; n_models <- n_models[ord]

table_data <- data.frame(
  Rank = seq_along(metrics),
  Metric = vapply(metrics, clean_metric, character(1)),
  MeanG = no_lead_zero(sprintf("%.2f", mean_g)),
  N = as.integer(n_models),
  stringsAsFactors = FALSE
)

top_n <- suppressWarnings(as.integer(Sys.getenv("EFA_GF_TOP_N", unset = "")))
if (!is.na(top_n) && top_n > 0) table_data <- head(table_data, top_n)

ft <- flextable::flextable(table_data)
ft <- flextable::set_header_labels(ft, Rank = "Rank", Metric = "Metric",
                                   MeanG = "Mean g", N = "Models")
ft <- flextable::theme_booktabs(ft)
ft <- flextable::font(ft, fontname = "Arial", part = "all")
ft <- flextable::fontsize(ft, size = 9, part = "all")
ft <- flextable::bold(ft, part = "header")
ft <- flextable::italic(ft, j = "MeanG", part = "header")  # g is a statistical symbol
ft <- flextable::align(ft, j = "Metric", align = "left", part = "all")
ft <- flextable::align(ft, j = c("Rank", "MeanG", "N"), align = "right", part = "all")
ft <- flextable::padding(ft, padding.top = 2, padding.bottom = 2,
                         padding.left = 4, padding.right = 4, part = "all")
ft <- flextable::autofit(ft)
ft <- flextable::width(ft, j = "Metric", width = 4.2)
ft <- flextable::add_footer_lines(
  ft,
  values = paste(
    "Note. g = general-factor loading from the Schmid–Leiman bifactor solution",
    "(psych::omega), averaged across the models in which the metric was a core item;",
    "Models = number of judges contributing (of three). Higher values mark the general",
    "\"overall tutoring quality\" factor most strongly."
  )
)
ft <- flextable::fontsize(ft, size = 8, part = "footer")
ft <- flextable::italic(ft, part = "footer")

# Exhibit number, overridable so renumbering the manuscript needs no code change.
table_number <- Sys.getenv("EFA_TABLE_NUMBER", unset = "Table S4")
doc <- officer::read_docx()
doc <- officer::body_add_par(doc, table_number, style = "Normal")
doc <- officer::body_add_par(doc,
  "Metrics Ranked by Mean General-Factor Loading Across Judge Models", style = "Normal")
doc <- flextable::body_add_flextable(doc, value = ft)
doc <- officer::body_end_section_continuous(doc)

output_path <- file.path(output_dir, "apa_general_factor_ranking.docx")
print(doc, target = output_path)
message("Wrote ", output_path, " (", nrow(table_data), " metrics)")
