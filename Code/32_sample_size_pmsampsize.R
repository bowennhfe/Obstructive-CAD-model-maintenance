# =============================================================================
# 32_sample_size_pmsampsize.R
# -----------------------------------------------------------------------------
# Minimum sample size for developing the CAD-RADS 3-5 (obstructive coronary
# artery disease) prediction model, estimated with the four-step procedure of
# Riley et al. implemented in the pmsampsize package.
#
# References:
#   Riley RD, Ensor J, Snell KIE, et al. Calculating the sample size required
#     for developing a clinical prediction model. BMJ 2020;368:m441.
#   Riley RD, Snell KIE, Ensor J, et al. Minimum sample size for developing a
#     multivariable prediction model: PART I - binary outcomes.
#     Stat Med 2018;37:1266-76. doi:10.1002/sim.7992
#   Riley RD, Van Calster B, Collins GS. A note on estimating the Cox-Snell
#     R2 from a reported C statistic (AUROC). Stat Med 2021;40:859-64.
#   Ensor J. pmsampsize: Sample Size for Development of a Prediction Model.
#     R package v1.1.3. https://CRAN.R-project.org/package=pmsampsize
#
# Inputs (verified against Tables/*.csv and Code/common.py::load_cohorts):
#   type         "b"   binary outcome: obstructive CAD (CAD-RADS 3-5 vs 0-2)
#   parameters   19    19 candidate predictors (FEATURES = NUM + BIN), each
#                      entered as one linear term -> 19 predictor df
#   prevalence   0.470588   events/N of the derivation cohort (208/442)
#   cstatistic   0.832      internal 50-fold cross-validated AUC of the random
#                           forest model (Table11_algo_internal.csv)
#   shrinkage    0.90       minimum acceptable shrinkage factor
#   seed         123456     pmsampsize default (C-statistic to Cox-Snell R2
#                           conversion is simulation based)
#
# The three criteria reported by pmsampsize map onto the four steps as:
#   Step 1  specify the model                 -> type, parameters, prevalence
#   Step 2  specify anticipated performance   -> cstatistic (Cox-Snell R2)
#   Step 3  sample size for model development -> Criteria 1 (global shrinkage
#                                                >= 0.90) and Criteria 2
#                                                (optimism in R2 <= 0.05)
#   Step 4  sample size for model validation  -> Criteria 3 (margin of error
#                                                <= 0.05 for the overall risk)
#   Nmin = max(Criteria 1, Criteria 2, Criteria 3)
#
# Cohorts (Table1_baseline_characteristics.csv, TableS10_discrimination_cohorts.csv):
#   Derivation   442 / 208 events  (centres 1-2, 2017-2022)
#   Temporal     364 / 172 events  (centre 1, 2023-2026)
#   Pooled dev   806 / 380 events  (derivation + temporal; the geographical
#                                   cohort is external evaluation only and is
#                                   NOT part of the development pool)
#   Geographical 177 /  63 events  (third centre, 2024-2026, evaluation only)
#
# Usage:
#   Rscript 32_sample_size_pmsampsize.R "<output directory>"
# The output directory is passed on the command line because Rscript in this
# environment reads UTF-8 source files as GBK when the locale is C; keeping the
# script ASCII-only and passing the path as an argument avoids any mangling.
# The UTF-8 codepage locale set below is required for R to create files under
# the non-ASCII project path.
#
# Outputs (written to the given output directory):
#   sample_size_pmsampsize.log    full console log (shell redirect)
#   sample_size_criteria.csv      primary run, criteria table
#   sample_size_cohorts.csv       requirement vs the actual cohorts
#   sample_size_sensitivity.csv   sensitivity grids
#   sample_size_pmsampsize.png / .pdf / .tiff    figure
# =============================================================================

Sys.setlocale("LC_ALL", "Chinese (Simplified)_China.65001")

.libPaths("C:/Users/lenovo/AppData/Local/R/win-library/4.5")
suppressPackageStartupMessages(library(pmsampsize))
options(width = 110)

args <- commandArgs(trailingOnly = TRUE)
OUTDIR <- if (length(args) >= 1) args[1] else tempdir()
if (!dir.exists(OUTDIR)) {
  created <- tryCatch(dir.create(OUTDIR, recursive = TRUE), error = function(e) FALSE)
  if (!created || !dir.exists(OUTDIR)) {
    OUTDIR <- tempdir()
    cat("NOTE: could not create the requested output directory; using", OUTDIR, "\n")
  }
}
cat("Output directory:", OUTDIR, "\n")

hr <- function(t) cat("\n", strrep("=", 76), "\n", t, "\n", strrep("=", 76), "\n", sep = "")

# ------------------------------------------------------------------ inputs --
DEV_N <- 442; DEV_EV <- 208          # derivation cohort, 2017-2022
TEM_N <- 364; TEM_EV <- 172          # temporal cohort, 2023-2026
GEO_N <- 177; GEO_EV <- 63           # geographical cohort, evaluation only
POOL_N <- DEV_N + TEM_N              # 806, pooled development cohort
POOL_EV <- DEV_EV + TEM_EV          # 380
PARAM  <- 19                         # candidate predictors (FEATURES = NUM + BIN)
PREV   <- DEV_EV / DEV_N             # 0.470588
CSTAT  <- 0.832                      # internal 50-fold CV AUC, RF model
SHRINK <- 0.9
SEED   <- 123456L                    # pmsampsize default

hr("INPUTS (verified against Tables/*.csv and Code/common.py::load_cohorts)")
cat(sprintf("Outcome                : obstructive CAD (CAD-RADS 3-5 vs 0-2), binary\n"))
cat(sprintf("Candidate predictors   : %d, each linear -> %d predictor df\n", PARAM, PARAM))
cat(sprintf("Derivation cohort      : n = %d, events = %d, prevalence = %.6f\n", DEV_N, DEV_EV, PREV))
cat(sprintf("Temporal cohort        : n = %d, events = %d, prevalence = %.6f\n", TEM_N, TEM_EV, TEM_EV / TEM_N))
cat(sprintf("Geographical cohort    : n = %d, events = %d, prevalence = %.6f (external evaluation only)\n", GEO_N, GEO_EV, GEO_EV / GEO_N))
cat(sprintf("Pooled development     : n = %d, events = %d, prevalence = %.6f (derivation + temporal)\n", POOL_N, POOL_EV, POOL_EV / POOL_N))
cat(sprintf("C-statistic (CV AUC)   : %.3f (internal 50-fold CV, RF, Table11)\n", CSTAT))
cat(sprintf("Shrinkage target       : %.2f ; seed = %d\n", SHRINK, SEED))
cat(sprintf("pmsampsize             : %s ; %s\n", as.character(packageVersion("pmsampsize")), R.version.string))

# ------------------------------------------------------------- primary run --
hr("PRIMARY RUN")
cat("Call: pmsampsize(type = 'b', parameters = 19, prevalence = 208/442,\n")
cat("                 cstatistic = 0.832, shrinkage = 0.9, seed = 123456)\n\n")
res <- pmsampsize(type = "b", parameters = PARAM, prevalence = PREV,
                  cstatistic = CSTAT, shrinkage = SHRINK, seed = SEED)
print(res)

rt <- res$results_table
N1 <- rt["Criteria 1", "Samp_size"]
N2 <- rt["Criteria 2", "Samp_size"]
N3 <- rt["Criteria 3", "Samp_size"]
REQ_N <- res$sample_size
REQ_EV <- ceiling(res$events)

hr("FOUR-STEP MAPPING (Riley et al., BMJ 2020;368:m441)")
cat("Step 1  specify the model                 : binary outcome, 19 predictor df\n")
cat("Step 2  specify anticipated performance   : C-statistic 0.832 -> Cox-Snell R2 =", res$rsquared, "\n")
cat("Step 3  sample size for model development : Criteria 1 (shrinkage >= 0.90) N =", N1, "\n")
cat("                                              Criteria 2 (R2 optimism <= 0.05) N =", N2, "\n")
cat("Step 4  sample size for model validation  : Criteria 3 (MoE <= 0.05 overall risk) N =", N3, "\n")
cat(sprintf("Nmin = max(criteria) = %d participants = %d events (prevalence %.4f), EPP = %.2f\n",
            REQ_N, REQ_EV, PREV, res$EPP))

crit_df <- data.frame(
  step = c("Step 3a (development)", "Step 3b (development)", "Step 4 (validation)", "Minimum"),
  criterion = c("Criteria 1: global shrinkage >= 0.90",
                "Criteria 2: optimism in Nagelkerke R2 <= 0.05",
                "Criteria 3: margin of error <= 0.05 for overall risk",
                "max(Criteria 1-3)"),
  shrinkage_target = c(SHRINK, rt["Criteria 2", "Shrinkage"], NA, res$final_shrinkage),
  sample_size = c(N1, N2, N3, REQ_N),
  events = c(ceiling(N1 * PREV), ceiling(N2 * PREV), ceiling(N3 * PREV), REQ_EV),
  epp = round(c(N1, N2, N3, REQ_N) * PREV / PARAM, 2),
  stringsAsFactors = FALSE)
write.csv(crit_df, file.path(OUTDIR, "sample_size_criteria.csv"), row.names = FALSE)

# ------------------------------------------- requirement vs actual cohorts --
hr("REQUIREMENT vs ACTUAL COHORTS")
# invert criterion 1 for the achieved shrinkage at a given N
shrink_at_N <- function(N, p, r2) {
  f <- function(S) p / ((S - 1) * log(1 - r2 / S)) - N
  uniroot(f, lower = r2 + 1e-8, upper = 1 - 1e-8)$root
}
moe_at_N <- function(N, prev) 1.96 * sqrt(prev * (1 - prev) / N)

coh <- data.frame(
  cohort = c("Derivation (2017-2022)", "Temporal (2023-2026)",
             "Pooled development (derivation + temporal)",
             "Geographical (external evaluation)"),
  n = c(DEV_N, TEM_N, POOL_N, GEO_N),
  events = c(DEV_EV, TEM_EV, POOL_EV, GEO_EV),
  prevalence = c(DEV_EV / DEV_N, TEM_EV / TEM_N, POOL_EV / POOL_N, GEO_EV / GEO_N),
  stringsAsFactors = FALSE)
coh$epp <- round(coh$events / PARAM, 2)
shr_raw <- sapply(coh$n, shrink_at_N, p = PARAM, r2 = res$rsquared)
moe_raw <- moe_at_N(coh$n, coh$prevalence)
coh$shrinkage_achieved <- round(shr_raw, 4)
coh$moe_overall_risk <- round(moe_raw, 4)
coh$meets_crit1 <- shr_raw >= SHRINK      # compare unrounded values
coh$meets_crit3 <- moe_raw <= 0.05
coh$meets_nmin <- coh$n >= REQ_N
cat("NOTE: achieved shrinkage assumes the same anticipated Cox-Snell R2 =", res$rsquared,
    "for every cohort; a cohort-specific R2 would shift the values.\n\n")
print(coh, row.names = FALSE, digits = 4)
write.csv(coh, file.path(OUTDIR, "sample_size_cohorts.csv"), row.names = FALSE)

# --------------------------------------------------------- sensitivity grid --
hr("SENSITIVITY ANALYSES")
run_ss <- function(cstat = NA, r2 = NA, prev, p, sh) {
  r <- tryCatch(
    if (!is.na(cstat)) {
      pmsampsize(type = "b", parameters = p, prevalence = prev, cstatistic = cstat,
                 shrinkage = sh, seed = SEED)
    } else {
      pmsampsize(type = "b", parameters = p, prevalence = prev, csrsquared = r2,
                 shrinkage = sh)
    },
    error = function(e) NULL)
  if (is.null(r)) {
    return(data.frame(cstatistic = cstat, csrsquared = r2, prevalence = prev,
                      parameters = p, shrinkage = sh, n_crit1 = NA, n_crit2 = NA,
                      n_crit3 = NA, n_final = NA, events = NA, epp = NA,
                      stringsAsFactors = FALSE))
  }
  data.frame(cstatistic = cstat,
             csrsquared = ifelse(is.na(r2), round(r$rsquared, 4), round(r2, 4)),
             prevalence = prev, parameters = p, shrinkage = sh,
             n_crit1 = r$results_table["Criteria 1", "Samp_size"],
             n_crit2 = r$results_table["Criteria 2", "Samp_size"],
             n_crit3 = r$results_table["Criteria 3", "Samp_size"],
             n_final = r$sample_size, events = ceiling(r$events), epp = r$EPP,
             stringsAsFactors = FALSE)
}

gA <- do.call(rbind, lapply(c(0.75, 0.80, 0.832, 0.85, 0.90), function(cs) {
  do.call(rbind, lapply(c(0.9, 0.95), function(sh) {
    cbind(scenario = "cstat x shrinkage",
          run_ss(cstat = cs, prev = PREV, p = PARAM, sh = sh))
  }))
}))
gB <- do.call(rbind, lapply(
  list(c("prevalence: geographical 0.3559", GEO_EV / GEO_N),
       c("prevalence: temporal 0.4725", TEM_EV / TEM_N),
       c("prevalence: derivation 0.4706", DEV_EV / DEV_N),
       c("prevalence: pooled development 0.4715", POOL_EV / POOL_N)),
  function(x) cbind(scenario = x[1],
                    run_ss(cstat = CSTAT, prev = as.numeric(x[2]), p = PARAM, sh = SHRINK))))
gC <- do.call(rbind, lapply(c(15, 19, 24), function(pp) {
  cbind(scenario = sprintf("predictor df %d", pp),
        run_ss(cstat = CSTAT, prev = PREV, p = pp, sh = SHRINK))
}))
r2max_alt <- 0.15 * res$max_r2a
gD <- cbind(scenario = "R2 = 15% of max R2 (no C-statistic)",
            run_ss(r2 = r2max_alt, prev = PREV, p = PARAM, sh = SHRINK))

sens <- rbind(gA, gB, gC, gD)
sens$binding <- apply(sens[, c("n_crit1", "n_crit2", "n_crit3")], 1,
                      function(x) c("Criteria 1", "Criteria 2", "Criteria 3")[which.max(x)])
print(sens, row.names = FALSE, digits = 4)
write.csv(sens, file.path(OUTDIR, "sample_size_sensitivity.csv"), row.names = FALSE)

# ------------------------------------------------------------------- figure --
hr("FIGURE")

draw_fig <- function() {
  op <- par(mfrow = c(1, 3), mar = c(4.4, 5.2, 2.8, 1.0), mgp = c(3, 1, 0),
            cex = 0.72, family = "Arial", las = 1)
  on.exit(par(op))

  BLUE <- "#1f4e79"; GREY <- "#6e6e6e"; RED <- "#c00000"; GREEN <- "#2e7d32"

  # Manual axis helper. The PNG/TIFF device of this R build silently drops
  # some tick labels when they are drawn rotated by axis()/mtext(); drawing
  # tick marks with axis(labels = FALSE) and labels with mtext() under
  # las = 1 renders every label reliably (verified against the PDF text layer).
  manual_axis <- function(side, at, labels, line = 1.1) {
    axis(side, at = at, labels = FALSE)
    mtext(labels, side = side, line = line, at = at, cex = 0.72)
  }

  # ---- Panel A: minimum N and events vs assumed C-statistic ----
  ymaxA <- ceiling(max(dA$n_final, dA$events) / 500) * 500
  plot(dA$cstatistic, dA$n_final, type = "l", lwd = 1.7, col = BLUE,
       ylim = c(0, ymaxA), xlim = c(0.70, 0.90),
       xlab = "Assumed C-statistic", ylab = "Minimum sample size",
       main = "A  vs assumed C-statistic", cex.main = 0.85,
       axes = FALSE, frame.plot = FALSE)
  manual_axis(1, seq(0.70, 0.90, 0.05), sprintf("%.2f", seq(0.70, 0.90, 0.05)))
  manual_axis(2, seq(0, ymaxA, 500), as.character(seq(0, ymaxA, 500)))
  lines(dA$cstatistic, dA$events, lwd = 1.5, lty = 2, col = GREY)
  abline(h = N3, lty = 3, col = "grey60", lwd = 1.1)
  abline(v = CSTAT, lty = 2, col = RED, lwd = 1.1)
  abline(h = DEV_N, lty = 4, col = GREEN, lwd = 1.1)
  abline(h = POOL_N, lty = 4, col = GREEN, lwd = 1.1)
  text(0.707, DEV_N + 25, "Derivation n = 442", cex = 0.58, col = GREEN, adj = 0)
  text(0.707, POOL_N + 25, "Pooled n = 806", cex = 0.58, col = GREEN, adj = 0)
  legend("topright", legend = c("Minimum N", "Minimum events",
                                "Criterion 3 (MoE <= 0.05)", "C = 0.832 (assumed)"),
         lty = c(1, 2, 3, 2), lwd = c(1.7, 1.5, 1.1, 1.1),
         col = c(BLUE, GREY, "grey60", RED), cex = 0.58, bty = "n", inset = 0.02)

  # ---- Panel B: minimum N and events vs assumed shrinkage ----
  ymaxB <- ceiling(max(dB$n_final, dB$events) / 200) * 200
  plot(dB$shrinkage, dB$n_final, type = "l", lwd = 1.7, col = BLUE,
       ylim = c(0, ymaxB), xlim = c(0.80, 0.95),
       xlab = "Assumed shrinkage factor", ylab = "Minimum sample size",
       main = "B  vs assumed shrinkage", cex.main = 0.85,
       axes = FALSE, frame.plot = FALSE)
  manual_axis(1, seq(0.80, 0.95, 0.05), sprintf("%.2f", seq(0.80, 0.95, 0.05)))
  manual_axis(2, seq(0, ymaxB, 200), as.character(seq(0, ymaxB, 200)))
  lines(dB$shrinkage, dB$events, lwd = 1.5, lty = 2, col = GREY)
  abline(h = N3, lty = 3, col = "grey60", lwd = 1.1)
  abline(v = SHRINK, lty = 2, col = RED, lwd = 1.1)
  abline(h = DEV_N, lty = 4, col = GREEN, lwd = 1.1)
  abline(h = POOL_N, lty = 4, col = GREEN, lwd = 1.1)
  text(0.803, DEV_N + 22, "Derivation n = 442", cex = 0.58, col = GREEN, adj = 0)
  text(0.803, POOL_N + 22, "Pooled n = 806", cex = 0.58, col = GREEN, adj = 0)
  legend("bottomright", legend = c("Minimum N", "Minimum events",
                                   "Criterion 3 (MoE <= 0.05)", "S = 0.90 (assumed)"),
         lty = c(1, 2, 3, 2), lwd = c(1.7, 1.5, 1.1, 1.1),
         col = c(BLUE, GREY, "grey60", RED), cex = 0.58, bty = "n", inset = 0.02)

  # ---- Panel C: actual cohorts vs the requirement ----
  met <- coh$n >= REQ_N
  cols <- ifelse(met, "#a9d18e", "#9dc3e6")
  bp <- barplot(coh$n, horiz = TRUE,
                names.arg = c("Derivation", "Temporal", "Pooled", "Geographical"),
                xlim = c(0, 1200), col = cols, border = NA, axes = FALSE,
                xlab = "Sample size (N)", main = "C  cohorts vs requirement",
                cex.main = 0.85, cex.names = 0.58, axisnames = TRUE)
  manual_axis(1, seq(0, 1200, 200), as.character(seq(0, 1200, 200)))
  abline(v = REQ_N, lty = 2, col = RED, lwd = 1.4)
  text(REQ_N + 12, par("usr")[4] - 0.15, sprintf("Required N = %d", REQ_N),
       cex = 0.6, col = RED, adj = 0)
  text(coh$n + 12, bp, sprintf("%d events", coh$events), cex = 0.55, adj = 0)
}

csA <- seq(0.70, 0.90, by = 0.01)
dA <- do.call(rbind, lapply(csA, function(cs) run_ss(cstat = cs, prev = PREV,
                                                    p = PARAM, sh = SHRINK)))
shB <- seq(0.80, 0.95, by = 0.005)
dB <- do.call(rbind, lapply(shB, function(sh) run_ss(r2 = res$rsquared, prev = PREV,
                                                    p = PARAM, sh = sh)))

windowsFonts(Arial = windowsFont("Arial"))
tmp_png <- tempfile(pattern = "ssfig", fileext = ".png")
png(tmp_png, width = 7.4, height = 2.75, units = "in", res = 600,
    type = "windows", family = "Arial")
draw_fig()
dev.off()

tmp_pdf <- tempfile(pattern = "ssfig", fileext = ".pdf")
cairo_pdf(tmp_pdf, width = 7.4, height = 2.75, family = "Arial")
draw_fig()
dev.off()

tmp_tif <- tempfile(pattern = "ssfig", fileext = ".tif")
tiff(tmp_tif, width = 7.4, height = 2.75, units = "in", res = 600,
     type = "windows", compression = "lzw", family = "Arial")
draw_fig()
dev.off()

for (f in c(tmp_png, tmp_pdf, tmp_tif)) {
  ext <- tools::file_ext(f)
  dest <- file.path(OUTDIR, paste0("sample_size_pmsampsize.", ext))
  ok <- file.copy(f, dest, overwrite = TRUE)
  cat(sprintf("figure %-4s -> %s (%s, %.0f KB)\n", ext, dest,
              ifelse(ok && file.exists(dest), "ok", "FAILED"),
              ifelse(file.exists(dest), file.size(dest) / 1024, 0)))
}

# ------------------------------------------------------------------ summary --
hr("SUMMARY")
cat(sprintf("Minimum sample size (Riley four-step, pmsampsize %s) = %d participants\n",
            as.character(packageVersion("pmsampsize")), REQ_N))
cat(sprintf("  = %d events at prevalence %.4f, EPP = %.2f\n", REQ_EV, PREV, res$EPP))
cat(sprintf("Binding criterion: Criteria 1 (global shrinkage >= 0.90), N = %d\n", N1))
cat(sprintf("Derivation cohort (%d participants, %d events) is %d participant(s) below Nmin;\n",
            DEV_N, DEV_EV, REQ_N - DEV_N))
cat(sprintf("pooled development cohort (%d participants, %d events) exceeds it by %d participants.\n",
            POOL_N, POOL_EV, POOL_N - REQ_N))
cat("\nOutput files:\n")
for (f in c("sample_size_criteria.csv", "sample_size_cohorts.csv",
            "sample_size_sensitivity.csv", "sample_size_pmsampsize.png",
            "sample_size_pmsampsize.pdf", "sample_size_pmsampsize.tif")) {
  p <- file.path(OUTDIR, f)
  cat(sprintf("  %-38s %s\n", f, ifelse(file.exists(p), "written", "MISSING")))
}
