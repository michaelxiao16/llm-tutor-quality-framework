#!/usr/bin/env Rscript

# Render an editable APA-style metric-disposition table as a Word document from analyze-
# efa.py's preprocessing_audit_*.csv outputs.

required_packages <- c("flextable", "officer")
missing <- required_packages[!vapply(required_packages, requireNamespace,
                                    quietly = TRUE, FUN.VALUE = logical(1))]
if (length(missing)) {
  stop("Missing R package(s): ", paste(missing, collapse = ", "),
       ". Install them with install.packages(c(\"flextable\", \"officer\")).", call. = FALSE)
}

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_path <- if (length(script_arg)) sub("^--file=", "", script_arg[1]) else "efa/render/render-apa-metric-status.R"
root <- normalizePath(file.path(dirname(script_path), "..", ".."), mustWork = FALSE)

results_dir <- Sys.getenv("EFA_RESULTS_DIR", unset = file.path(root, "efa", "results", "factor_analysis"))
output_dir <- Sys.getenv("EFA_APA_TABLE_DIR", unset = file.path(results_dir, "apa_tables"))
results_dir <- normalizePath(results_dir, mustWork = TRUE)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

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

format_model_name <- function(path) {
  name <- sub("^preprocessing_audit_", "", basename(path))
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

audit_paths <- list.files(file.path(results_dir, "preprocessing"), pattern = "^preprocessing_audit_.*\\.csv$", full.names = TRUE)
if (!length(audit_paths)) stop("No preprocessing_audit_*.csv files found in ", results_dir, call. = FALSE)

# tier[[model]] is a named vector metric_id -> tier
models <- character(0); tier <- list()
for (p in audit_paths) {
  a <- read.csv(p, check.names = FALSE, stringsAsFactors = FALSE)
  a <- a[as.logical(a$present_in_all_input_chunks), ]  # metrics scored in every chunk (177)
  m <- format_model_name(p)
  models <- c(models, m)
  v <- a$applicability_tier; names(v) <- a$metric_id
  tier[[m]] <- v
}
models <- models[order(vapply(models, model_rank, numeric(1)))]

# metrics that were non-core in at least one judge
all_ids <- unique(unlist(lapply(tier, names)))
# Show ALL applicability-tiered metrics (the six a-priori safety metrics remain omitted; they
# are noted separately).
metric_ids <- sort(all_ids)

status_word <- function(t) {
  if (is.null(t) || is.na(t)) return("—")
  if (t == "core") return("Core")
  if (t == "conditional") return("Conditional")
  "Dropped"  # rare or any other non-core disposition
}
rank_of <- c(Core = 0, Conditional = 1, Dropped = 2, "—" = 3)

# Build: Metric + one status column per judge.
df <- data.frame(Metric = vapply(metric_ids, clean_metric, character(1)),
                 stringsAsFactors = FALSE)
jkeys <- paste0("J", seq_along(models))
for (k in seq_along(models)) {
  df[[jkeys[k]]] <- vapply(metric_ids, function(id) status_word(tier[[models[k]]][[id]]), character(1))
}

# Sort by disposition pattern: all-Core first, grading toward all-Dropped, so similar rows
# cluster. severity = sum of per-judge ranks; then the pattern; then name.
sev <- rowSums(sapply(jkeys, function(k) rank_of[df[[k]]]))
band <- ifelse(sev == 0, 1L,                              # core in all judges
        ifelse(vapply(seq_len(nrow(df)), function(i)      # any Dropped -> band 3
                 any(df[i, jkeys] == "Dropped"), logical(1)), 3L, 2L))
patt <- apply(df[jkeys], 1, paste, collapse = "|")
ord <- order(band, sev, patt, df$Metric)
df <- df[ord, ]; band <- band[ord]

ft <- flextable::flextable(df)
header_labels <- setNames(as.list(models), jkeys)
header_labels[["Metric"]] <- "Metric"
ft <- do.call(flextable::set_header_labels, c(list(ft), header_labels))
ft <- flextable::theme_booktabs(ft)
ft <- flextable::font(ft, fontname = "Arial", part = "all")
ft <- flextable::fontsize(ft, size = 8, part = "all")
ft <- flextable::bold(ft, part = "header")
ft <- flextable::align(ft, j = "Metric", align = "left", part = "all")
ft <- flextable::align(ft, j = jkeys, align = "center", part = "all")
ft <- flextable::valign(ft, valign = "center", part = "all")

# Redundant color shading (words still carry the information for grayscale/CVD).
bg_core <- "#E4F1E1"; bg_cond <- "#FBEED0"; bg_drop <- "#F6DEDC"
for (k in jkeys) {
  ft <- flextable::bg(ft, i = which(df[[k]] == "Core"),        j = k, bg = bg_core)
  ft <- flextable::bg(ft, i = which(df[[k]] == "Conditional"), j = k, bg = bg_cond)
  ft <- flextable::bg(ft, i = which(df[[k]] == "Dropped"),     j = k, bg = bg_drop)
}

# Dividers: vertical rules between every column; horizontal rules between bands.
thin <- officer::fp_border(color = "grey60", width = 0.5)
ft <- flextable::border_inner_v(ft, border = thin, part = "all")
ft <- flextable::vline_left(ft, border = thin, part = "all")
ft <- flextable::vline_right(ft, border = thin, part = "all")
band_starts <- which(diff(band) != 0) + 1
if (length(band_starts)) {
  ft <- flextable::hline(ft, i = band_starts - 1,
                         border = officer::fp_border(color = "grey40", width = 1), part = "body")
}
ft <- flextable::padding(ft, padding.top = 1.5, padding.bottom = 1.5,
                         padding.left = 4, padding.right = 4, part = "all")
ft <- flextable::width(ft, j = "Metric", width = 3.20)
ft <- flextable::width(ft, j = jkeys, width = 1.35)
ft <- flextable::set_table_properties(ft, layout = "fixed")
ft <- flextable::add_footer_lines(
  ft,
  values = paste(
    "Note. Rows are ordered by disposition pattern and separated into blocks: core in all judges,",
    "conditional in at least one judge, and dropped in at least one judge. Cell shading is redundant",
    "with the status label (green = Core, amber = Conditional, rose = Dropped) so the table remains",
    "legible in grayscale. Core = pooled into the EFA. Conditional = held out and related to the core",
    "factors via extension analysis (structural, MNAR missingness). Dropped = scored in fewer than",
    "200 conversations. The six a-priori-excluded MinorBench safety metrics are not shown."
  )
)
ft <- flextable::fontsize(ft, size = 8, part = "footer")
ft <- flextable::italic(ft, part = "footer")

# Exhibit number, overridable so renumbering the manuscript needs no code change.
table_number <- Sys.getenv("EFA_TABLE_NUMBER", unset = "Table B1")
doc <- officer::read_docx()
doc <- officer::body_add_par(doc, table_number, style = "Normal")
doc <- officer::body_add_par(doc, "Metric Disposition Across Judge Models", style = "Normal")
doc <- flextable::body_add_flextable(doc, value = ft)
doc <- officer::body_end_section_landscape(doc)

output_path <- file.path(output_dir, "apa_metric_status.docx")
print(doc, target = output_path)
message("Wrote ", output_path, " (", length(metric_ids), " metrics)")
