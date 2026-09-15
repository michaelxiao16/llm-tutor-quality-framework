#!/usr/bin/env Rscript
# Render Tables D1-D3 (one per judge: GPT-5.4 mini, Claude Haiku 4.5, Gemma 4 26B) from saved
# conditional extension results; does not rerun EFA.

for (pkg in c("flextable", "officer")) {
  if (!requireNamespace(pkg, quietly = TRUE)) stop("Missing R package: ", pkg)
}
script <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])
results <- Sys.getenv("EFA_RESULTS_DIR", file.path(dirname(script), "..", "results", "factor_analysis"))
out <- Sys.getenv("EFA_APA_TABLE_DIR", file.path(results, "apa_tables"))
table_prefix <- Sys.getenv("EFA_TABLE_NUMBER_PREFIX", "D")
threshold <- as.numeric(Sys.getenv("EFA_CONDITIONAL_THRESHOLD", "0.40"))
if (!is.finite(threshold) || threshold <= 0) stop("Threshold must be positive and finite")
labels <- read.csv(file.path(results, "congruence", "congruence_construct_labels.csv"),
                   stringsAsFactors = FALSE, check.names = FALSE)
judges <- data.frame(key = c("gpt-5.4-mini", "claude-haiku-4-5", "gemma4-26b"),
                     label = c("GPT-5.4 mini", "Claude Haiku 4.5", "Gemma 4 26B"),
                     column = c("gpt_factor", "haiku_factor", "gemma_factor"))
stopifnot(all(c("construct", judges$column) %in% names(labels)))
fmt <- function(x) ifelse(is.finite(x), sub("^(-?)0", "\\1", sprintf("%.2f", x)), "—")
clean <- function(id) {
  categories <- c("Overall_Rating", "Accuracy", "Alignment_to_Constraints",
    "Instructional_Support", "Assessment", "Mistake_Handling",
    "Affect_and_Relational_Support", "Engagement_and_Motivation", "Adaptivity",
    "Understanding_Learner_Goals", "Metacognition", "Safety")
  for (category in categories) {
    prefix <- paste0(category, "_")
    if (startsWith(id, prefix)) return(paste(gsub("_", " ", category), "–",
      gsub("_", " ", substring(id, nchar(prefix) + 1))))
  }
  gsub("_", " ", id)
}
panels <- list()
summary_rows <- list()
for (j in seq_len(nrow(judges))) {
  judge <- judges[j, ]
  candidates <- list.files(file.path(results, "conditional"), full.names = TRUE)
  paths <- candidates[startsWith(basename(candidates),
    paste0("conditional_extension_loadings_", judge$key)) & endsWith(candidates, ".csv")]
  if (length(paths) != 1) stop("Expected exactly one loading file for ", judge$label,
    "; found ", length(paths), ". Select a results directory with one run per judge.")
  path <- paths[1]
  desc_path <- file.path(dirname(path), sub("^conditional_extension_loadings_",
    "conditional_descriptives_", basename(path)))
  loadings <- read.csv(path, check.names = FALSE)
  desc <- read.csv(desc_path, check.names = FALSE)
  if (anyDuplicated(loadings$metric_id) || anyDuplicated(desc$metric_id) ||
      !setequal(loadings$metric_id, desc$metric_id)) stop("Metric IDs do not match: ", path)
  desc <- desc[match(loadings$metric_id, desc$metric_id), ]
  factors <- grep("^Factor_[0-9]+$", names(loadings), value = TRUE)
  if (!length(factors)) stop("No factor columns: ", path)
  factor_name <- function(f) {
    ix <- which(!is.na(labels[[judge$column]]) & labels[[judge$column]] == f)
    if (length(ix) > 1) stop("Ambiguous factor label: ", judge$label, " ", f)
    short <- sub("Factor_", "F", f)
    if (length(ix)) paste0(labels$construct[ix], " (", short, ")") else short
  }
  rows <- lapply(seq_len(nrow(loadings)), function(i) {
    v <- as.numeric(loadings[i, factors]); names(v) <- factors
    finite <- which(is.finite(v))
    salient <- finite[abs(v[finite]) >= threshold]
    salient <- salient[order(-abs(v[salient]))]
    primary <- if (length(finite)) finite[which.max(abs(v[finite]))] else integer(0)
    mapping <- if (length(salient)) paste(vapply(salient, function(k)
      paste0(factor_name(factors[k]), ": ", fmt(v[k])), character(1)), collapse = "\n") else
      if (!length(finite)) "Not estimable" else "No salient mapping"
    data.frame(Judge = judge$label, metric_id = loadings$metric_id[i],
      Metric = clean(loadings$metric_id[i]), N = desc$n_applicable[i],
      Mapping = mapping,
      Strongest = if (length(primary)) paste0(sub("Factor_", "F", factors[primary]),
        ": ", fmt(v[primary])) else "—",
      Status = if (length(salient)) "Salient" else if (length(finite)) "Below threshold" else "Not estimable",
      Strongest_abs_loading = if (length(primary)) abs(v[primary]) else NA_real_,
      stringsAsFactors = FALSE)
  })
  panel <- do.call(rbind, rows)
  # Sort on unrounded numeric loadings, not the formatted display strings.
  status_order <- match(panel$Status, c("Salient", "Below threshold", "Not estimable"))
  panel <- panel[order(status_order, -panel$Strongest_abs_loading,
                       panel$Metric, na.last = TRUE), ]
  panels[[j]] <- panel
  summary_rows[[j]] <- data.frame(Judge = judge$label, Conditional = nrow(panel),
    Salient = sum(panel$Status == "Salient"),
    Below_threshold = sum(panel$Status == "Below threshold"),
    Not_estimable = sum(panel$Status == "Not estimable"))
}
dir.create(out, recursive = TRUE, showWarnings = FALSE)
doc <- officer::read_docx()
doc <- officer::body_set_default_section(doc, officer::prop_section(
  page_size = officer::page_size(width = 8.5, height = 11),
  page_margins = officer::page_mar(top = .7, bottom = .7, left = .7, right = .7)))
note <- paste0("Note. N = conversations with an applicable score. Mappings list all extension ",
  "loadings. Rows with salient mappings appear first, ordered by strongest absolute loading, ",
  "followed by below-threshold and non-estimable results. Mappings display only ",
  "loadings with absolute values ≥ ", fmt(threshold), ", ordered by magnitude; signs are preserved. ",
  "Strongest reports the largest absolute loading even when below threshold. ",
  "Factor numbers are specific to each judge; names use the analyst-supplied congruence labels. ",
  "No salient mapping does not establish a separate construct. Not estimable indicates ",
  "undefined loadings (the current GPT case has constant applicable scores).")
section_border <- officer::fp_border(color = "#7F7F7F", width = 1.25)
for (j in seq_along(panels)) {
  if (j > 1) doc <- officer::body_add_break(doc)
  table_rows <- panels[[j]][c("Metric", "N", "Mapping", "Strongest")]
  doc <- officer::body_add_par(doc, paste0("Table ", table_prefix, j))
  doc <- officer::body_add_par(doc, paste0(
    "Conditional-Metric Mappings to Core EFA Factors: ", judges$label[j]))
  ft <- flextable::flextable(table_rows)
  ft <- flextable::set_header_labels(ft, Mapping = "Salient factor mappings (λ)", Strongest = "Strongest λ")
  ft <- flextable::theme_booktabs(ft)
  ft <- flextable::font(ft, fontname = "Arial", part = "all")
  ft <- flextable::fontsize(ft, size = 9, part = "all")
  ft <- flextable::bold(ft, part = "header")
  # Separate the salient block from the below-threshold / not-estimable rows.
  status <- panels[[j]]$Status
  last_salient <- max(c(0, which(status == "Salient")))
  if (last_salient > 0 && last_salient < nrow(table_rows)) {
    ft <- flextable::border(ft, i = last_salient, border.bottom = section_border, part = "body")
  }
  ft <- flextable::padding(ft, padding = 4, part = "all")
  ft <- flextable::valign(ft, valign = "center", part = "all")
  ft <- flextable::align(ft, j = c("N", "Strongest"), align = "center", part = "all")
  ft <- flextable::width(ft, j = "Metric", width = 3.0)
  ft <- flextable::width(ft, j = "N", width = .45)
  ft <- flextable::width(ft, j = "Mapping", width = 2.75)
  ft <- flextable::width(ft, j = "Strongest", width = .8)
  ft <- flextable::set_table_properties(ft, layout = "fixed",
    opts_word = list(split = FALSE, repeat_headers = TRUE))
  doc <- flextable::body_add_flextable(doc, ft)
  doc <- officer::body_add_par(doc, note)
}
target <- file.path(out, "apa_conditional_mappings.docx")
print(doc, target = target)
write.csv(do.call(rbind, panels), file.path(out, "conditional_mappings.csv"), row.names = FALSE)
write.csv(do.call(rbind, summary_rows), file.path(out, "conditional_mapping_summary.csv"), row.names = FALSE)
print(do.call(rbind, summary_rows), row.names = FALSE)
message("Wrote ", target)
