#!/usr/bin/env Rscript

# Authoritative EFA engine for analyze-efa.py.

args <- commandArgs(trailingOnly = TRUE)
arg_value <- function(name) {
  index <- match(name, args)
  if (is.na(index) || index == length(args)) stop("Missing argument: ", name, call. = FALSE)
  args[index + 1]
}

if (!requireNamespace("psych", quietly = TRUE)) {
  stop("The R package psych is required. Install it with install.packages(\"psych\").", call. = FALSE)
}

data_path <- arg_value("--data")
n_factors <- as.integer(arg_value("--n-factors"))
parallel_output <- arg_value("--parallel-output")
reliability_output <- arg_value("--reliability-output")
pattern_output <- arg_value("--pattern-output")
phi_output <- arg_value("--phi-output")
structure_output <- arg_value("--structure-output")
factorability_output <- arg_value("--factorability-output")
sl_loadings_output <- arg_value("--sl-loadings-output")
sl_summary_output <- arg_value("--sl-summary-output")
sl_omega_by_factor_output <- arg_value("--sl-omega-by-factor-output")
threshold <- as.numeric(arg_value("--threshold"))
iterations <- as.integer(arg_value("--iterations"))
rotation <- arg_value("--rotation")

scores <- read.csv(data_path, check.names = FALSE)
metric_ids <- names(scores)
if (anyNA(scores)) stop("The authoritative EFA matrix must be complete before R is called.", call. = FALSE)
if (n_factors < 1 || n_factors >= ncol(scores)) stop("Invalid number of factors.", call. = FALSE)

score_cor <- stats::cor(scores, use = "everything")
r_kmo <- psych::KMO(score_cor)
r_bartlett <- psych::cortest.bartlett(score_cor, n = nrow(scores))
factorability_table <- data.frame(
  kmo_overall_psych = unname(r_kmo$MSA),
  bartlett_chi_square_psych = unname(r_bartlett$chisq),
  bartlett_df_psych = unname(r_bartlett$df),
  bartlett_p_value_psych = unname(r_bartlett$p.value)
)
write.csv(factorability_table, factorability_output, row.names = FALSE)

set.seed(20260825)
parallel <- psych::fa.parallel(
  scores, fa = "fa", fm = "minres", n.iter = iterations,
  plot = FALSE, error.bars = FALSE, sim = TRUE, quant = 0.95,
  cor = "cor", use = "pairwise"
)
n_parallel <- parallel$nfact
simulated_factor_columns <- paste0("Fsim", seq_len(ncol(scores)))
if (!all(simulated_factor_columns %in% colnames(parallel$values))) {
  stop("psych::fa.parallel did not return the expected Fsim columns.", call. = FALSE)
}
random_95th <- apply(
  parallel$values[, simulated_factor_columns, drop = FALSE], 2,
  stats::quantile, probs = 0.95, na.rm = TRUE
)
parallel_table <- data.frame(
  factor = seq_along(parallel$fa.values),
  observed_minres_eigenvalue = parallel$fa.values,
  random_mean_minres_eigenvalue = parallel$fa.sim,
  random_95th_percentile_minres_eigenvalue = random_95th,
  observed_exceeds_random_95th_percentile = parallel$fa.values > random_95th,
  retained_by_psych_sequential_rule = seq_along(parallel$fa.values) <= n_parallel,
  psych_recommended_n_factors = n_parallel,
  quantile_criterion = 0.95,
  iterations = iterations,
  random_seed = 20260825
)
write.csv(parallel_table, parallel_output, row.names = FALSE)

r_rotation <- if (rotation == "none") "none" else rotation
r_fit <- psych::fa(
  score_cor, n.obs = nrow(scores), nfactors = n_factors,
  rotate = r_rotation, fm = "minres", SMC = TRUE, warnings = FALSE
)

pattern <- unclass(r_fit$loadings)
factor_names <- paste0("Factor_", seq_len(ncol(pattern)))
rownames(pattern) <- metric_ids
colnames(pattern) <- factor_names

phi <- if (is.null(r_fit$Phi)) diag(n_factors) else unclass(r_fit$Phi)
rownames(phi) <- factor_names
colnames(phi) <- factor_names
structure <- pattern %*% phi
rownames(structure) <- metric_ids
colnames(structure) <- factor_names
common_covariance <- pattern %*% phi %*% t(pattern)
model_h2 <- diag(common_covariance)
psych_h2 <- unname(r_fit$communality)

if (max(abs(model_h2 - psych_h2)) > 1e-7) {
  stop("psych communalities are inconsistent with diag(pattern %*% Phi %*% t(pattern)).", call. = FALSE)
}
if (!is.null(r_fit$Structure) && max(abs(structure - unclass(r_fit$Structure))) > 1e-7) {
  stop("psych Structure is inconsistent with pattern %*% Phi.", call. = FALSE)
}

pattern_table <- data.frame(
  metric_id = metric_ids,
  communality_h2_psych = psych_h2,
  pattern,
  check.names = FALSE
)
write.csv(pattern_table, pattern_output, row.names = FALSE)
write.csv(phi, phi_output, row.names = TRUE)
write.csv(structure, structure_output, row.names = TRUE)

absolute_loadings <- abs(pattern)
primary_index <- max.col(absolute_loadings, ties.method = "first")
primary_factor <- factor_names[primary_index]
primary_strength <- absolute_loadings[cbind(seq_len(nrow(absolute_loadings)), primary_index)]
primary_signed_loading <- pattern[cbind(seq_len(nrow(pattern)), primary_index)]

reliability_rows <- lapply(factor_names, function(factor) {
  members <- metric_ids[primary_factor == factor & primary_strength >= threshold]
  n_members <- length(members)
  alpha_raw <- NA_real_
  alpha_standardized <- NA_real_
  omega_total <- NA_real_
  reversed_members <- character(0)
  status <- "not estimable: fewer than 2 primary metrics"
  if (n_members >= 2) {
    member_signs <- sign(primary_signed_loading[match(members, metric_ids)])
    member_signs[member_signs == 0] <- 1
    keyed_scores <- sweep(scores[, members, drop = FALSE], 2, member_signs, `*`)
    reversed_members <- members[member_signs < 0]
    alpha_result <- suppressWarnings(psych::alpha(
      keyed_scores, check.keys = FALSE, warnings = FALSE
    ))
    alpha_raw <- alpha_result$total$raw_alpha
    alpha_standardized <- alpha_result$total$std.alpha
    status <- if (n_members == 2) "caution: 2-item reliability is limited" else "estimated"
  }
  if (n_members >= 3) {
    omega_result <- suppressWarnings(psych::omega(
      keyed_scores, nfactors = 1, plot = FALSE, warnings = FALSE
    ))
    omega_total <- omega_result$omega.tot
  }
  data.frame(
    factor = factor,
    n_primary_metrics = n_members,
    cronbach_alpha_raw = alpha_raw,
    cronbach_alpha_standardized = alpha_standardized,
    omega_total = omega_total,
    n_sign_keyed_metrics = length(reversed_members),
    sign_keyed_metric_ids = paste(reversed_members, collapse = ";"),
    reliability_status = status
  )
})
write.csv(do.call(rbind, reliability_rows), reliability_output, row.names = FALSE)

# Full Schmid-Leiman/general-factor analysis across all retained metrics.
omega_fit <- suppressWarnings(psych::omega(
  scores,
  nfactors = n_factors,
  fm = "minres",
  rotate = "oblimin",
  flip = TRUE,
  plot = FALSE,
  warnings = FALSE,
  title = "AI Tutor metric Schmid-Leiman analysis"
))
if (is.null(omega_fit$schmid$sl) || is.null(omega_fit$omega.group)) {
  stop("psych::omega did not return the Schmid-Leiman matrices.", call. = FALSE)
}

sl <- as.data.frame(unclass(omega_fit$schmid$sl), check.names = FALSE)
if (nrow(sl) != length(metric_ids) || !"g" %in% names(sl)) {
  stop("Unexpected Schmid-Leiman loading matrix dimensions.", call. = FALSE)
}
omega_key <- omega_fit$key[metric_ids]
if (anyNA(omega_key)) stop("psych::omega did not return a key for every metric.", call. = FALSE)
sl_table <- data.frame(
  metric_id = metric_ids,
  sign_key = as.numeric(omega_key),
  sign_flipped_for_omega = as.numeric(omega_key) < 0,
  sl,
  check.names = FALSE
)
write.csv(sl_table, sl_loadings_output, row.names = FALSE)

g_loadings <- sl_table$g

# Percent of Uncontaminated Correlations (PUC): the share of item-pair correlations that
# reflect only the general factor (pairs whose two items sit in different group factors).
group_cols <- grep("^F[0-9]+\\*$", names(sl), value = TRUE)
dominant_group <- apply(abs(sl[, group_cols, drop = FALSE]), 1, which.max)
group_sizes <- as.numeric(table(dominant_group))
total_pairs <- length(metric_ids) * (length(metric_ids) - 1) / 2
within_pairs <- sum(group_sizes * (group_sizes - 1) / 2)
puc_general <- 1 - within_pairs / total_pairs

sl_summary <- data.frame(
  n_items = length(metric_ids),
  n_group_factors = n_factors,
  extraction_method = "minres",
  first_order_rotation = "oblimin",
  omega_item_keying = "psych::omega flip=TRUE; sign_key and sign_flipped_for_omega saved per item",
  omega_hierarchical = unname(omega_fit$omega_h),
  omega_total = unname(omega_fit$omega.tot),
  ecv_general = unname(omega_fit$ECV[1]),
  puc_general = puc_general,
  general_loading_mean = mean(g_loadings, na.rm = TRUE),
  general_loading_min = min(g_loadings, na.rm = TRUE),
  general_loading_max = max(g_loadings, na.rm = TRUE),
  n_general_loadings_abs_ge_0_40 = sum(abs(g_loadings) >= 0.40, na.rm = TRUE)
)
write.csv(sl_summary, sl_summary_output, row.names = FALSE)

omega_by_factor <- data.frame(
  schmid_leiman_factor = rownames(omega_fit$omega.group),
  omega_total = unname(omega_fit$omega.group[, "total"]),
  omega_general = unname(omega_fit$omega.group[, "general"]),
  omega_group_specific = unname(omega_fit$omega.group[, "group"]),
  check.names = FALSE
)
write.csv(omega_by_factor, sl_omega_by_factor_output, row.names = FALSE)
