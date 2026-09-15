#!/usr/bin/env Rscript

# Render an editable APA-style cross-judge factor-congruence table (Tucker's coefficient of
# congruence) as a Word document.

required_packages <- c("flextable", "officer")
missing <- required_packages[!vapply(required_packages, requireNamespace,
                                    quietly = TRUE, FUN.VALUE = logical(1))]
if (length(missing)) {
  stop("Missing R package(s): ", paste(missing, collapse = ", "),
       ". Install them with install.packages(c(\"flextable\", \"officer\")).", call. = FALSE)
}

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_path <- if (length(script_arg)) sub("^--file=", "", script_arg[1]) else "efa/render/render-apa-congruence.R"
root <- normalizePath(file.path(dirname(script_path), "..", ".."), mustWork = FALSE)

results_dir <- Sys.getenv("EFA_RESULTS_DIR", unset = file.path(root, "efa", "results", "factor_analysis"))
output_dir <- Sys.getenv("EFA_APA_TABLE_DIR", unset = file.path(results_dir, "apa_tables"))
results_dir <- normalizePath(results_dir, mustWork = TRUE)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

labels_path <- file.path(results_dir, "congruence", "congruence_construct_labels.csv")
if (!file.exists(labels_path)) {
  stop("Missing ", labels_path, ". Create it with columns: ",
       "construct, gpt_factor, haiku_factor, gemma_factor.", call. = FALSE)
}
labels <- read.csv(labels_path, check.names = FALSE, stringsAsFactors = FALSE)

read_matrix <- function(name) {
  p <- file.path(results_dir, "congruence", name)
  if (!file.exists(p)) stop("Missing congruence matrix ", p,
                            ". Run cross-judge-congruence.py first.", call. = FALSE)
  m <- read.csv(p, check.names = FALSE, row.names = 1)
  as.matrix(m)
}
hg <- read_matrix("cross_judge_congruence_matrix_claude-haiku-4-5_vs_gemma4-26b.csv")
hp <- read_matrix("cross_judge_congruence_matrix_claude-haiku-4-5_vs_gpt-5.4-mini.csv")
gp <- read_matrix("cross_judge_congruence_matrix_gemma4-26b_vs_gpt-5.4-mini.csv")

blank <- function(x) is.null(x) || is.na(x) || !nzchar(x)   # judge has no such factor
# sign is arbitrary in EFA; return NA when either factor is absent for this pair.
phi <- function(mat, r, c) if (blank(r) || blank(c)) NA_real_ else abs(mat[r, c])
no_lead_zero <- function(x) sub("^(-?)0", "\\1", x)
# "Factor_2" -> "2"; absent factors show an em dash.
fac_no <- function(x) if (blank(x)) "\u2014" else sub("^Factor_", "", x)
fmt <- function(x) if (is.na(x)) "—" else no_lead_zero(sprintf("%.2f", x))  # APA: drop leading zero

# Order retained constructs first, then non-retained; within each block, sort by mean
# congruence descending (constructs with no computable mean sort last).
if (!"retained" %in% names(labels)) labels$retained <- "yes"
is_ret <- tolower(labels$retained) %in% c("yes", "true", "1")
mean_by_label <- vapply(seq_len(nrow(labels)), function(i) {
  lab <- labels[i, ]
  vals <- c(phi(hg, lab$haiku_factor, lab$gemma_factor),
            phi(hp, lab$haiku_factor, lab$gpt_factor),
            phi(gp, lab$gemma_factor, lab$gpt_factor))
  if (all(is.na(vals))) NA_real_ else mean(vals, na.rm = TRUE)
}, numeric(1))
sort_mean <- ifelse(is.na(mean_by_label), -Inf, mean_by_label)
labels <- labels[order(!is_ret, -sort_mean), ]
is_ret <- tolower(labels$retained) %in% c("yes", "true", "1")

rows <- lapply(seq_len(nrow(labels)), function(i) {
  lab <- labels[i, ]
  hg_v <- phi(hg, lab$haiku_factor, lab$gemma_factor)
  hp_v <- phi(hp, lab$haiku_factor, lab$gpt_factor)
  gp_v <- phi(gp, lab$gemma_factor, lab$gpt_factor)
  vals <- c(hg_v, hp_v, gp_v)
  mean_v <- if (all(is.na(vals))) NA_real_ else mean(vals, na.rm = TRUE)
  min_v <- if (all(is.na(vals))) NA_real_ else min(vals, na.rm = TRUE)
  data.frame(
    Construct = lab$construct,
    FacG = fac_no(lab$gpt_factor),
    FacH = fac_no(lab$haiku_factor),
    FacM = fac_no(lab$gemma_factor),
    HG = fmt(hg_v), HP = fmt(hp_v), GP = fmt(gp_v),
    Min = fmt(min_v), Mean = fmt(mean_v),
    stringsAsFactors = FALSE
  )
})
table_data <- do.call(rbind, rows)

ft <- flextable::flextable(table_data)
ft <- flextable::set_header_labels(
  ft, Construct = "Factor name",
  FacG = "GPT", FacH = "Haiku", FacM = "Gemma",
  HG = "Haiku–Gemma", HP = "Haiku–GPT", GP = "Gemma–GPT", Min = "Min",
  # APA: a sample mean is italic M; the Greek mu denotes a population mean.
  Mean = "M"
)
ft <- flextable::add_header_row(
  ft, top = TRUE,
  values = c("", "Factor number", "Tucker's congruence (rc)", ""),
  colwidths = c(1, 3, 3, 2)
)
ft <- flextable::theme_booktabs(ft)
ft <- flextable::align(ft, i = 1, align = "center", part = "header")
ft <- flextable::font(ft, fontname = "Arial", part = "all")
ft <- flextable::fontsize(ft, size = 10, part = "all")
ft <- flextable::fontsize(ft, size = 9, part = "header")
ft <- flextable::bold(ft, part = "header")
ft <- flextable::align(ft, j = "Construct", align = "left", part = "all")
ft <- flextable::align(ft, j = c("FacG", "FacH", "FacM"), align = "center", part = "all")
ft <- flextable::align(ft, j = c("HG", "HP", "GP", "Min", "Mean"), align = "right", part = "all")
ft <- flextable::italic(ft, j = "Mean", part = "header")  # italic mu
# Divider between the retained block and the non-retained block.
n_ret <- sum(is_ret)
if (n_ret > 0 && n_ret < nrow(table_data)) {
  ft <- flextable::hline(ft, i = n_ret,
                         border = officer::fp_border(color = "grey40", width = 1), part = "body")
}
ft <- flextable::padding(ft, padding.top = 4, padding.bottom = 4,
                         padding.left = 3, padding.right = 3, part = "all")
# Fixed widths totalling 6.25 in so the extra column still fits a portrait text block.
ft <- flextable::set_table_properties(ft, layout = "fixed")
ft <- flextable::width(ft, j = "Construct", width = 1.46)
ft <- flextable::width(ft, j = "FacG",      width = 0.36)
ft <- flextable::width(ft, j = "FacH",      width = 0.44)
ft <- flextable::width(ft, j = "FacM",      width = 0.58)
ft <- flextable::width(ft, j = "HG",        width = 1.00)
ft <- flextable::width(ft, j = "HP",        width = 0.86)
ft <- flextable::width(ft, j = "GP",        width = 0.90)
ft <- flextable::width(ft, j = "Min",       width = 0.40)
ft <- flextable::width(ft, j = "Mean",      width = 0.40)
ft <- flextable::add_footer_lines(
  ft,
  values = paste(
    "Note. Factor number gives the index each construct occupied in that judge's solution. Because",
    "factor order is arbitrary in exploratory factor analysis, factors were matched across judges by",
    "maximum absolute congruence using the Hungarian assignment algorithm rather than by index. Values",
    "are Tucker's coefficient of congruence (rc) between the matched factors of each judge pair, computed",
    "over the metrics common to that pair (n = 138–143). M = mean of the pairwise values; Min = the smallest available",
    "pairwise value. rc ≥ .95 indicates essentially identical factors and rc ≥ .85 fair similarity",
    "(Lorenzo-Seva & ten Berge, 2006). Constructs with a mean congruence ≥ .85 across all three judge",
    "pairs (above the line) replicated and were retained; those below did not. A dash indicates the",
    "construct did not form a distinct factor for that judge (style and presentation was absorbed into",
    "GPT-5.4 mini's general first factor; mistake handling emerged only for GPT-5.4 mini), so no",
    "congruence could be computed. Congruence between Accuracy and GPT-5.4 mini is attenuated because",
    "accuracy items load on that model's broad general first factor."
  )
)
ft <- flextable::fontsize(ft, size = 9, part = "footer")
ft <- flextable::italic(ft, part = "footer")

# Exhibit number, overridable so renumbering the manuscript needs no code change.
table_number <- Sys.getenv("EFA_TABLE_NUMBER", unset = "Table 3")
doc <- officer::read_docx()
doc <- officer::body_add_par(doc, table_number, style = "Normal")
doc <- officer::body_add_par(doc,
  "Cross-Judge Factor Congruence by Construct", style = "Normal")
doc <- flextable::body_add_flextable(doc, value = ft)
doc <- officer::body_end_section_continuous(doc)

output_path <- file.path(output_dir, "apa_cross_judge_congruence.docx")
print(doc, target = output_path)
message("Wrote ", output_path)
