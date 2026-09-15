#!/usr/bin/env Rscript

# Extension analysis for conditional metrics.

args <- commandArgs(trailingOnly = TRUE)
arg <- function(name) {
  i <- match(name, args)
  if (is.na(i) || i == length(args)) stop("Missing argument: ", name, call. = FALSE)
  args[i + 1]
}
# Optional argument: returns NULL when absent rather than stopping.
arg_opt <- function(name) {
  i <- match(name, args)
  if (is.na(i) || i == length(args)) NULL else args[i + 1]
}
if (!requireNamespace("psych", quietly = TRUE)) {
  stop("The R package psych is required.", call. = FALSE)
}

core <- read.csv(arg("--core-data"), check.names = FALSE)
cond <- read.csv(arg("--conditional-data"), check.names = FALSE)
nfactors <- as.integer(arg("--n-factors"))
rotation <- arg("--rotation")
threshold <- as.numeric(arg("--threshold"))
min_applicable <- as.integer(arg("--min-applicable"))
loadings_out <- arg("--loadings-output")
descriptives_out <- arg("--descriptives-output")

# Re-fit the pooled core solution with the same specification as the main pipeline, so the
# extension is projected onto exactly the reported core factors.
fo <- psych::fa(core, nfactors = nfactors, rotate = rotation, fm = "minres", warnings = FALSE)

# This is a second fit, so it must be proven equal to the committed authoritative solution
# before anything is projected onto it -- otherwise the extension loadings would describe
# factors the paper never reports.
pattern_check <- arg_opt("--pattern-check")
if (!is.null(pattern_check)) {
  if (!file.exists(pattern_check)) {
    stop("--pattern-check file not found: ", pattern_check, call. = FALSE)
  }
  committed <- read.csv(pattern_check, check.names = FALSE)
  committed_cols <- grep("^Factor_", names(committed), value = TRUE)
  refit <- unclass(fo$loadings)
  if (length(committed_cols) != ncol(refit) ||
      nrow(committed) != nrow(refit)) {
    stop("--pattern-check shape ", nrow(committed), "x", length(committed_cols),
         " does not match the refit ", nrow(refit), "x", ncol(refit),
         ". Pass the loadings file from the same pooled run.", call. = FALSE)
  }
  max_diff <- max(abs(refit - as.matrix(committed[, committed_cols])))
  if (max_diff > 1e-8) {
    stop("The core refit does not reproduce the authoritative pattern matrix ",
         "(max absolute difference ", format(max_diff), " > 1e-8). Conditional ",
         "extension loadings would be projected onto a different solution than ",
         "the one reported.", call. = FALSE)
  }
  cat(sprintf("Core refit matches committed pattern matrix (max diff %.2e).\n",
              max_diff))
}

# Correlations of the core items (original) with the conditional items (extension).
# use="pairwise.complete.obs" => each conditional item's correlations are computed only over
# conversations where it was scored (core is fully observed).
Roe <- stats::cor(core, cond, use = "pairwise.complete.obs")

extension <- psych::fa.extension(Roe, fo)
ext_load <- unclass(extension$loadings)
colnames(ext_load) <- paste0("Factor_", seq_len(ncol(ext_load)))
rownames(ext_load) <- colnames(cond)

loadings_table <- data.frame(metric_id = colnames(cond), ext_load,
                             check.names = FALSE, row.names = NULL)
write.csv(loadings_table, loadings_out, row.names = FALSE)

# Per-metric descriptives over the applicable subset + attachment interpretation.
abs_load <- abs(ext_load)
primary <- max.col(replace(abs_load, is.na(abs_load), -1), ties.method = "first")
primary_abs <- abs_load[cbind(seq_len(nrow(abs_load)), primary)]
signed <- ext_load[cbind(seq_len(nrow(ext_load)), primary)]
estimable <- is.finite(primary_abs)
n_applicable <- vapply(cond, function(x) sum(!is.na(x)), integer(1))
descriptives <- data.frame(
  metric_id = colnames(cond),
  n_applicable = n_applicable,
  applicable_rate = vapply(cond, function(x) mean(!is.na(x)), numeric(1)),
  mean_when_applicable = vapply(cond, function(x) mean(x, na.rm = TRUE), numeric(1)),
  sd_when_applicable = vapply(cond, function(x) stats::sd(x, na.rm = TRUE), numeric(1)),
  primary_core_factor = ifelse(estimable, paste0("Factor_", primary), NA_character_),
  primary_extension_loading = signed,
  primary_abs_extension_loading = primary_abs,
  attaches_to_core_factor = ifelse(
    !estimable, "not estimable (constant when applicable)",
    ifelse(primary_abs >= threshold, paste0("Factor_", primary), "none (< threshold)")
  ),
  n_salient_core_factors = rowSums(abs_load >= threshold, na.rm = TRUE),
  below_min_applicable = n_applicable < min_applicable,
  row.names = NULL
)
write.csv(descriptives, descriptives_out, row.names = FALSE)

cat(sprintf("Extension analysis: %d conditional metrics projected onto %d core factors.\n",
            ncol(cond), nfactors))
