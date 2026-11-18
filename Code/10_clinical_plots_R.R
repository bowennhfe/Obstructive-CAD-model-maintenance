# =============================================================================
# 10_clinical_plots_R.R - 关键临床图（R / ggplot2 版）
# =============================================================================
# 用户指令（2026-09-20）：照「旧记录/Code/07_Clinical_Plots_R.R」的规范与算法，
# 用 R 再出一版关键临床图（生长曲线、校准、DCA、亚组森林图、风险分层、
# 临床影响曲线、连续净收益、NIA）。
#
# 与旧 07 的对应关系（逐条对齐）
#   R01 Growth_Curve            <- 07 plot_growth_curve（AUC / Brier 双分面）
#   R02 Calibration_Curves      <- 07 plot_calibration_curves（分位数分箱 + 局部合并）
#   R03 DCA_Curves              <- 07 plot_dca_curves（含同色临床价值填充带）
#   R04 Subgroup_Forest_Plot    <- 07 plot_subgroup_forest（右侧 N / AUC / O:E 列）
#   R05 Risk_Stratification     <- 07 plot_risk_stratification（Cleveland 点图 + Wilson CI）
#   R06 Clinical_Impact_Curve   <- 07 plot_clinical_impact（双 y 轴）
#   R07 Continuous_Net_Benefit  <- 07 plot_continuous_nb（标准化净收益 + AUC-NB）
#   R08 Net_Intervention_Avoided<- 07 plot_net_intervention_avoided
#
# 与旧 07 的两处有意差异（已在交付说明中列出）
#   1) 配色沿用本项目 Figure 2 的口径（Derivation 蓝 / 地理-原模型 绿 /
#      地理-更新模型 紫 / 地理-对照臂 灰），使全套图件颜色一致；
#   2) 输出为 tif(LZW) + pdf + jpg（本项目约定，jpg 用于 Word 内嵌），
#      而非旧 07 的 tif + pdf + png。tif 沿用旧 07 的「线图 1200 dpi /
#      彩图 600 dpi」；jpg 固定 600 dpi（Word 内嵌足够，避免 60 Mpx 级文件）。
#
# 输入：Outputs/r_input/{analysis_set,growth_steps,risk_cutoffs}.csv（由 09 导出）
# 输出：Figures/r/{tif,pdf,jpg}/R0x_*.{tif,pdf,jpg}
# =============================================================================

# --- 1. 依赖包 ---
required_pkgs <- c("ggplot2", "patchwork", "dplyr", "tidyr", "scales",
                   "pROC", "grid", "gridExtra")
for (pkg in required_pkgs) {
  if (!suppressWarnings(require(pkg, character.only = TRUE, quietly = TRUE))) {
    suppressWarnings(install.packages(pkg, repos = "https://cloud.r-project.org",
                                     type = "binary", quiet = TRUE))
    if (!suppressWarnings(require(pkg, character.only = TRUE, quietly = TRUE))) {
      stop(sprintf("R 包 %s 自动安装/加载失败，请手动 install.packages(\"%s\") 后重跑。",
                   pkg, pkg))
    }
  }
}
options(bitmapType = "cairo", stringsAsFactors = FALSE)
set.seed(123)

# --- 2. 路径与全局参数 ---
# ★ Windows + 中文目录：R 的 normalizePath() 对**相对路径**会经 ANSI API 返回
#   被破坏的字节（本项目铁律：.R 勿写中文绝对路径）。因此这里直接用 getwd()
#   定位项目根，并对 --file= 只在其本身为绝对路径时才尝试解析。
find_base <- function() {
  probe <- file.path("Outputs", "r_input", "analysis_set.csv")
  cands <- c(getwd(), file.path(getwd(), ".."))
  for (p in cands) if (file.exists(file.path(p, probe))) return(p)
  fa <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(fa) > 0) {
    rp <- sub("^--file=", "", fa[1])
    if (grepl("^([A-Za-z]:|/)", rp)) {
      d <- dirname(rp)
      for (p in c(d, file.path(d, ".."))) {
        if (file.exists(file.path(p, probe))) return(p)
      }
    }
  }
  stop("找不到项目根目录（需含 Outputs/r_input/analysis_set.csv）；",
       "请在 <项目根>/Code 目录下运行本脚本。")
}
BASE_DIR <- find_base()
IN_DIR  <- file.path(BASE_DIR, "Outputs", "r_input")
OUT_DIR <- file.path(BASE_DIR, "Figures", "r")
for (sub in c("tif", "pdf", "jpg")) dir.create(file.path(OUT_DIR, sub),
                                               showWarnings = FALSE, recursive = TRUE)

# 配色（与论文 Figure 2 一致）
COL_DERIV  <- "#1f77b4"   # 推导队列（交叉验证袋外）
COL_GEO    <- "#2ca02c"   # 地理队列 - 原模型
COL_UPDATE <- "#9467bd"   # 地理队列 - 更新模型
COL_CTRL   <- "#7f7f7f"   # 地理队列 - 对照臂
COL_RED    <- "#d62728"
COL_GRAY   <- "gray50"

FONT_TITLE  <- 14
FONT_LABEL  <- 12
FONT_LEGEND <- 10
FONT_TICK   <- 10
FONT_SMALL  <- 8

GROUP_LEVELS <- c("Derivation (CV)",
                  "Geographical, original model",
                  "Geographical, updated model",
                  "Geographical, control arm")
GROUP_COLORS <- c("Derivation (CV)"                  = COL_DERIV,
                  "Geographical, original model"     = COL_GEO,
                  "Geographical, updated model"      = COL_UPDATE,
                  "Geographical, control arm"        = COL_CTRL)
GROUP_SHAPES <- c("Derivation (CV)" = 16, "Geographical, original model" = 15,
                  "Geographical, updated model" = 17,
                  "Geographical, control arm" = 4)
GEO_MODELS <- GROUP_LEVELS[2:4]        # 地理队列上的三条模型曲线
GROUP_SHORT <- setNames(as.character(GROUP_LEVELS), GROUP_LEVELS)  # 具名映射，供 scale_*_manual(labels=)

# ★ 论文主图归位（2026-09-20 用户拍板：主图收敛 + R 为主）
#   主图 6 = R02 校准（四队列同一张）；主图 7 = R01 生长曲线；
#   主图 8 = R03+R06+R07+R08 合成四联（临床效用）；主图 9 = R04+R05 合成上下叠放（亚组与风险分层）。
#   本表把 R0x 绘图对象额外以「论文主图文件名」再导出一份到 Figures/{pdf,tif,jpg}/，使主图集中。
MAIN_NAME <- c(R01_Growth_Curve   = "Fig7_external_validity_growth",
               R02_Calibration_Curves = "Fig6_calibration")
MAIN_FIG_DIR <- file.path(BASE_DIR, "Figures")   # 主图目录（与 Figures/r/ 并列）
for (sub in c("tif", "pdf", "jpg"))
  dir.create(file.path(MAIN_FIG_DIR, sub), showWarnings = FALSE, recursive = TRUE)

# --- 3. 读入数据 ---
read_in <- function(f) read.csv(file.path(IN_DIR, f), fileEncoding = "UTF-8-BOM",
                                check.names = FALSE)
dat    <- read_in("analysis_set.csv")
steps  <- read_in("growth_steps.csv")
cutoff <- read_in("risk_cutoffs.csv")

dat$group <- factor(dat$group, levels = GROUP_LEVELS)
LOW_CUT  <- cutoff$low_cutoff[1]
HIGH_CUT <- cutoff$high_cutoff[1]

geo  <- dat[dat$cohort == "Geographical", ]
geo_wide <- reshape(geo[, c("id", "group", "p")], idvar = "id",
                    timevar = "group", direction = "wide")
names(geo_wide) <- sub("^p\\.", "", names(geo_wide))
geo_base <- geo[geo$group == GEO_MODELS[1],
                c("id", "y", "age", "male", "female", "htn", "dm", "smoke", "dyslip")]
geo_wide <- merge(geo_base, geo_wide, by = "id", sort = FALSE)
stopifnot(nrow(geo_wide) == 177, all(GEO_MODELS %in% names(geo_wide)))

# --- 4. 统一保存函数（照旧 07 save_sci_plot） ---
# ★ Windows 中文目录：tiff()/pdf()/jpeg() 的设备名必须能转成 native encoding，
#   否则报 "unable to translate ... to native encoding"。故先写到 tempfile()
#   （ASCII 临时目录），再用 file.copy() 搬到项目内的中文路径。
save_sci_plot <- function(plot_obj, base_name, width, height, dpi_type = "line") {
  dpi_val <- ifelse(dpi_type == "line", 1200, 600)
  # 目标文件名：R0x 名写 Figures/r/；名字以 Fig 开头（论文主图）写 Figures/。
  out_names <- c(base_name, unname(MAIN_NAME[base_name]))
  out_names <- out_names[!is.na(out_names) & nzchar(out_names)]
  tgt_dirs <- ifelse(grepl("^Fig", out_names), MAIN_FIG_DIR, OUT_DIR)
  cat(sprintf("  saving %s (%.1f x %.1f in, %.0f dpi)\n", base_name, width, height, dpi_val))

  specs <- list(
    list(ext = "tif", d = dpi_val, fn = function(p, w, h, d)
      tiff(p, width = w, height = h, units = "in", res = d,
           compression = "lzw", family = "sans")),
    list(ext = "pdf", d = dpi_val, fn = function(p, w, h, d)
      pdf(p, width = w, height = h, family = "sans")),
    list(ext = "jpg", d = min(dpi_val, 600), fn = function(p, w, h, d)
      jpeg(p, width = w, height = h, units = "in", res = d,
           quality = 95, family = "sans")))

  for (spec in specs) {
    tmp <- tempfile(fileext = paste0(".", spec$ext))
    tryCatch({
      spec$fn(tmp, width, height, spec$d)
      on.exit({ if (dev.cur() > 1) dev.off(); unlink(tmp) }, add = TRUE)
      print(plot_obj)
      if (dev.cur() > 1) dev.off()
      for (i in seq_along(out_names)) {
        target <- file.path(tgt_dirs[i], spec$ext, paste0(out_names[i], ".", spec$ext))
        ok <- file.copy(tmp, target, overwrite = TRUE)
        cat(sprintf("    -> %s%s\n", basename(target),
                    if (isTRUE(ok)) "" else "  (COPY FAILED)"))
      }
      unlink(tmp)
    }, error = function(e) {
      if (dev.cur() > 1) dev.off()
      unlink(tmp)
      cat(sprintf("    ERROR %s: %s\n", spec$ext, conditionMessage(e)))
    })
  }
}

# 主题（照旧 07：theme_bw + 只留左/下轴线 + 细网格）
theme_sci <- function(base_size = FONT_LABEL) {
  theme_bw(base_size = base_size, base_family = "sans") +
    theme(plot.title = element_blank(),
          axis.title = element_text(size = FONT_LABEL),
          axis.text = element_text(size = FONT_TICK),
          panel.grid.minor = element_blank(),
          panel.grid.major = element_line(linewidth = 0.3, color = "gray90"),
          panel.border = element_blank(),
          axis.line = element_line(color = "black", linewidth = 0.5))
}
theme_leg <- theme(
  legend.background = element_rect(fill = "white", color = "gray80", linewidth = 0.3),
  legend.text = element_text(size = FONT_LEGEND - 1),
  legend.title = element_blank(),
  legend.key.size = unit(0.45, "cm"),
  legend.spacing.y = unit(0.02, "cm"))

# Wilson 95% CI
wilson_ci <- function(x, n, conf = 0.95) {
  if (n == 0) return(c(NA_real_, NA_real_))
  z <- qnorm((1 + conf) / 2)
  ph <- x / n
  den <- 1 + z^2 / n
  ctr <- (ph + z^2 / (2 * n)) / den
  mar <- z * sqrt((ph * (1 - ph) + z^2 / (4 * n)) / n) / den
  c(max(ctr - mar, 0), min(ctr + mar, 1))
}

# =============================================================================
# R01 生长曲线（AUC / Brier 双分面，时间臂 vs 对照臂）
# =============================================================================
plot_growth_curve <- function() {
  cat("[R01] 生长曲线...\n")
  d <- steps
  d$arm_label <- ifelse(d$arm == "temporal", "Contemporary data added",
                        "Row-count-matched resampled original")
  d$pct_label <- factor(sprintf("%.0f%%", d$pct * 100),
                        levels = sprintf("%.0f%%", sort(unique(d$pct)) * 100))
  agg <- function(col) {
    d %>% group_by(pct_label, arm_label) %>%
      summarise(n_derive = mean(n_derive), m = mean(.data[[col]]),
                se = sd(.data[[col]]) / sqrt(n()), .groups = "drop") %>%
      mutate(lo = m - 1.96 * se, hi = m + 1.96 * se)
  }
  a_auc   <- agg("geo_auc")
  a_brier <- agg("geo_brier")
  base_auc <- mean(d$geo_auc[d$arm == "temporal" & d$pct == 0])
  arm_cols <- c("Contemporary data added" = COL_UPDATE,
                "Row-count-matched resampled original" = COL_CTRL)

  p_auc <- ggplot(a_auc, aes(x = pct_label, y = m, group = arm_label,
                             color = arm_label)) +
    geom_ribbon(aes(ymin = lo, ymax = hi, fill = arm_label), alpha = 0.18,
                color = NA, show.legend = FALSE) +
    geom_line(linewidth = 1.1) + geom_point(size = 2.2) +
    geom_hline(yintercept = base_auc, linetype = "dotted", color = COL_RED,
               linewidth = 0.6) +
    # ★ 注释移到右上（原置于左端 vjust=-0.6 会压在时间臂紫色置信带上）
    annotate("text", x = Inf, y = base_auc, vjust = -0.6, hjust = 1.1,
             label = sprintf("baseline %.3f", base_auc), size = 2.8,
             color = COL_RED) +
    annotate("text", x = -Inf, y = Inf, hjust = -0.05, vjust = 1.4,
             label = "y-axis truncated", size = 2.5, color = "gray45",
             fontface = "italic") +
    scale_color_manual(values = arm_cols) +
    scale_fill_manual(values = arm_cols) +
    scale_y_continuous(labels = scales::label_number(accuracy = 0.01)) +
    labs(y = "AUC (geographical validation)", x = NULL,
         color = NULL, fill = NULL) +
    theme_sci() + theme_leg +
    theme(legend.position = c(0.72, 0.16),
          legend.background = element_rect(fill = alpha("white", 0.85),
                                          color = "gray80", linewidth = 0.3))

  p_brier <- ggplot(a_brier, aes(x = pct_label, y = m, group = arm_label,
                                 color = arm_label)) +
    geom_ribbon(aes(ymin = lo, ymax = hi, fill = arm_label), alpha = 0.18,
                color = NA, show.legend = FALSE) +
    geom_line(linewidth = 1.1) + geom_point(size = 2.2) +
    annotate("text", x = -Inf, y = Inf, hjust = -0.05, vjust = 1.4,
             label = "y-axis truncated", size = 2.5, color = "gray45",
             fontface = "italic") +
    scale_color_manual(values = arm_cols) +
    scale_fill_manual(values = arm_cols) +
    scale_y_continuous(labels = scales::label_number(accuracy = 0.005)) +
    labs(x = "Contemporary data added (%)", y = "Brier score") +
    theme_sci() + theme_leg + theme(legend.position = "none")

  p <- (p_auc / p_brier) +
    plot_layout(heights = c(1, 1)) +
    plot_annotation(tag_levels = "A") &
    theme(plot.tag = element_text(face = "bold", size = FONT_LABEL))
  save_sci_plot(p, "R01_Growth_Curve", width = 7.0, height = 6.0, dpi_type = "line")
  invisible(p)
}

# =============================================================================
# R02 校准曲线（分位数分箱 + 局部合并；图例含 Brier）
# =============================================================================
calib_bins <- function(df, prob_col, y_col, n_bins = 10, min_pos = 5) {
  sub <- df[order(df[[prob_col]]), ]
  edges <- unique(quantile(sub[[prob_col]], probs = seq(0, 1, length.out = n_bins + 1),
                           na.rm = TRUE))
  repeat {
    bin <- cut(sub[[prob_col]], breaks = edges, include.lowest = TRUE, labels = FALSE)
    nb <- length(unique(bin[!is.na(bin)]))
    if (nb <= 1 || length(edges) <= 4) break
    tab <- data.frame(ok = !is.na(bin), bin = bin, y = sub[[y_col]])
    tab <- tab[tab$ok, ]
    agg <- aggregate(y ~ bin, data = tab, FUN = function(v) c(npos = sum(v), n = length(v)))
    n_pos <- agg$y[, "npos"]; n_bin <- agg$y[, "n"]
    bad <- which(n_pos < min_pos)
    if (length(bad) == 0) break
    tgt <- bad[which.min(n_pos[bad])]
    nb_sel <- c(tgt - 1, tgt + 1); nb_sel <- nb_sel[nb_sel >= 1 & nb_sel <= nb]
    if (length(nb_sel) == 0) break
    nbr <- nb_sel[which.min(n_bin[nb_sel])]
    edges <- edges[-(min(tgt, nbr) + 1)]
  }
  sub$.bin <- cut(sub[[prob_col]], breaks = edges, include.lowest = TRUE, labels = FALSE)
  out <- sub %>% filter(!is.na(.bin)) %>% group_by(.bin) %>%
    summarise(mean_pred = mean(.data[[prob_col]]), obs_prop = mean(.data[[y_col]]),
              n_bin = n(), n_pos = sum(.data[[y_col]]), .groups = "drop") %>%
    arrange(.bin)
  out
}

plot_calibration_curves <- function() {
  cat("[R02] 校准曲线...\n")
  bins <- do.call(rbind, lapply(GROUP_LEVELS, function(m) {
    d <- dat[dat$group == m, ]
    b <- calib_bins(d, "p", "y")
    b$group <- m
    b
  }))
  brier <- dat %>% group_by(group) %>%
    summarise(b = mean((y - p)^2), .groups = "drop")
  labs_map <- setNames(sprintf("%s (Brier = %.3f)",
                               GROUP_SHORT[brier$group], brier$b),
                       brier$group)
  bins$group <- factor(bins$group, levels = GROUP_LEVELS)
  p <- ggplot(bins, aes(x = mean_pred, y = obs_prop, color = group)) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed",
                color = "gray40", linewidth = 0.7) +
    geom_line(linewidth = 0.8) + geom_point(shape = 16, size = 2.6) +
    coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
    scale_color_manual(values = GROUP_COLORS, breaks = GROUP_LEVELS,
                       labels = labs_map) +
    labs(x = "Mean predicted probability", y = "Observed proportion") +
    theme_sci() + theme_leg + theme(legend.position = c(0.78, 0.16),
                                    legend.background = element_rect(
                                      fill = alpha("white", 0.88),
                                      color = "gray80", linewidth = 0.3))
  save_sci_plot(p, "R02_Calibration_Curves", width = 7.0, height = 6.0,
                dpi_type = "color")
  invisible(p)
}

# =============================================================================
# R03 决策曲线 DCA（含同色临床价值填充带）
# =============================================================================
nb_curve <- function(y, p, thr) sapply(thr, function(t) {
  tp <- sum(p >= t & y == 1); fp <- sum(p >= t & y == 0)
  (tp / length(y)) - (fp / length(y)) * (t / (1 - t))
})

plot_dca_curves <- function() {
  cat("[R03] 决策曲线 DCA...\n")
  thr <- seq(0.01, 0.80, by = 0.01)
  y <- geo_wide$y
  prev <- mean(y)
  mods <- c("Geographical, original model", "Geographical, updated model",
            "Geographical, control arm")
  cols <- GROUP_COLORS[mods]
  short <- setNames(c("Geographical, original model", "Geographical, updated model",
                      "Geographical, control arm"), mods)
  nb_df <- do.call(rbind, lapply(mods, function(m) {
    data.frame(Threshold = thr, Net_Benefit = nb_curve(y, geo_wide[[m]], thr),
               Strategy = m)
  }))
  treat_all <- prev - (1 - prev) * (thr / (1 - thr))
  base_df <- rbind(
    data.frame(Threshold = thr, Net_Benefit = 0, Strategy = "Treat None"),
    data.frame(Threshold = thr, Net_Benefit = rep(prev, length(thr)),
               Strategy = "Ideal (perfect)"),
    data.frame(Threshold = thr, Net_Benefit = treat_all,
               Strategy = "Treat All (Geographical)"))
  region <- do.call(rbind, lapply(mods, function(m) {
    s <- nb_df[nb_df$Strategy == m, ]
    base <- pmax(treat_all, 0)
    data.frame(Threshold = thr, ymin = base,
               ymax = pmax(s$Net_Benefit, base), Region = m)
  }))
  lv <- c("Ideal (perfect)", "Treat None", "Treat All (Geographical)", mods)
  all_lines <- rbind(nb_df, base_df)
  all_lines$Strategy <- factor(all_lines$Strategy, levels = lv)
  line_cols <- c("Ideal (perfect)" = "darkred", "Treat None" = "black",
                 "Treat All (Geographical)" = "gray60", cols)
  line_types <- c("Ideal (perfect)" = "dotted", "Treat None" = "dashed",
                  "Treat All (Geographical)" = "dashed",
                  setNames(rep("solid", length(mods)), mods))
  line_w <- c("Ideal (perfect)" = 0.6, "Treat None" = 0.5,
              "Treat All (Geographical)" = 0.7,
              setNames(rep(1.1, length(mods)), mods))
  leg_labels <- c("Ideal (perfect)", "Treat none", "Treat all (geographical)",
                  short[mods])
  p <- ggplot() +
    geom_ribbon(data = region, aes(x = Threshold, ymin = ymin, ymax = ymax,
                                   fill = Region), alpha = 0.20, color = NA) +
    geom_line(data = all_lines, aes(x = Threshold, y = Net_Benefit,
                                    color = Strategy, linetype = Strategy,
                                    linewidth = Strategy)) +
    scale_color_manual(name = NULL, values = line_cols, breaks = lv,
                       labels = leg_labels) +
    scale_linetype_manual(name = NULL, values = line_types, breaks = lv,
                          labels = leg_labels) +
    scale_linewidth_manual(name = NULL, values = line_w, breaks = lv,
                           labels = leg_labels) +
    scale_fill_manual(name = NULL, values = cols, breaks = mods) +
    coord_cartesian(xlim = c(0, 0.80), ylim = c(-0.12, max(0.45, prev + 0.12))) +
    scale_x_continuous(breaks = seq(0, 0.8, by = 0.1),
                       labels = scales::label_number(accuracy = 0.1)) +
    labs(x = "Threshold probability", y = "Net benefit") +
    guides(color = guide_legend(order = 1, override.aes = list(linewidth = 1.2)),
           fill = guide_legend(order = 2, override.aes = list(alpha = 0.35)),
           linetype = "none", linewidth = "none") +
    theme_sci() + theme_leg +
    theme(legend.position = c(0.78, 0.80),
          legend.text = element_text(size = FONT_SMALL),
          legend.background = element_rect(fill = alpha("white", 0.88),
                                          color = "gray80", linewidth = 0.3))
  save_sci_plot(p, "R03_DCA_Curves", width = 7.0, height = 6.0, dpi_type = "line")
  invisible(p)
}

# =============================================================================
# R04 亚组森林图（地理队列；Age / Sex / Hypertension；原模型 vs 更新模型）
# =============================================================================
plot_subgroup_forest <- function() {
  cat("[R04] 亚组森林图...\n")
  g <- geo_wide
  g$age_grp <- cut(g$age, breaks = c(-Inf, 50, 70, Inf),
                   labels = c("Age: <50", "Age: 50-69", "Age: >=70"), right = FALSE)
  g$sex_grp <- factor(ifelse(g$female == 1, "Sex: Female", "Sex: Male"),
                      levels = c("Sex: Female", "Sex: Male"))
  g$htn_grp <- factor(ifelse(g$htn == 1, "Hypertension: Yes", "Hypertension: No"),
                      levels = c("Hypertension: No", "Hypertension: Yes"))
  grp_vars <- list(Age = "age_grp", Sex = "sex_grp", Hypertension = "htn_grp")
  mods <- c("Geographical, original model", "Geographical, updated model")

  rows <- list()
  for (gn in names(grp_vars)) {
    v <- grp_vars[[gn]]
    for (lv in levels(g[[v]])) {
      sub <- g[!is.na(g[[v]]) & as.character(g[[v]]) == lv, ]
      if (nrow(sub) < 10 || length(unique(sub$y)) < 2) next
      for (m in mods) {
        r <- suppressMessages(pROC::roc(sub$y, sub[[m]], quiet = TRUE,
                                        direction = "<", levels = c(0, 1)))
        ci <- as.numeric(pROC::ci.auc(r, method = "delong"))
        rows[[length(rows) + 1]] <- data.frame(
          Subgroup = lv, Group = gn, Model = m,
          N = nrow(sub), N_Pos = sum(sub$y),
          Metric_Value = as.numeric(pROC::auc(r)),
          CI_Lower = ci[1], CI_Upper = ci[3])
      }
    }
  }
  df <- do.call(rbind, rows)
  rank_map <- c("Age: <50" = 1, "Age: 50-69" = 2, "Age: >=70" = 3,
                "Sex: Female" = 1, "Sex: Male" = 2,
                "Hypertension: No" = 1, "Hypertension: Yes" = 2)
  df$level_rank <- rank_map[df$Subgroup]
  df <- df[order(factor(df$Group, levels = c("Age", "Sex", "Hypertension")),
                 df$level_rank, decreasing = TRUE), ]
  subs <- unique(df[, c("Subgroup", "Group", "level_rank")])
  subs <- subs[order(factor(subs$Group, levels = c("Age", "Sex", "Hypertension")),
                     subs$level_rank, decreasing = TRUE), ]
  subs$y_pos <- nrow(subs):1
  subs$N_label <- sprintf("%d/%d",
                          df$N_Pos[match(paste(subs$Subgroup, mods[2]),
                                         paste(df$Subgroup, df$Model))],
                          df$N[match(paste(subs$Subgroup, mods[2]),
                                     paste(df$Subgroup, df$Model))])
  df$y_pos <- subs$y_pos[match(df$Subgroup, subs$Subgroup)]
  df$y_off <- ifelse(df$Model == mods[1], 0.16, -0.16)
  # 亚组相对事件率（更新模型）：亚组事件率 / 队列总体事件率，并非模型 O:E
  upd <- df[df$Model == mods[2], ]
  upd$obs <- upd$N_Pos / upd$N
  overall_prev <- sum(upd$N_Pos) / sum(upd$N)
  oe <- sapply(seq_len(nrow(upd)), function(i) {
    ci <- wilson_ci(upd$N_Pos[i], upd$N[i])
    c(oe = upd$obs[i] / overall_prev, lo = ci[1] / overall_prev,
      hi = ci[2] / overall_prev)
  })
  # ★ 版式修正（相对旧 07）：旧 07 把 N/AUC/O:E 三列文本写在数据坐标 x=1.03/1.20/1.42，
  #   同时用 coord_cartesian(xlim=c(x_lo,0.95)) 把面板收窄——文本列因此被映射到
  #   面板右侧约 5 英寸处，远超 plot.margin 预留的 2.4 英寸，在 8 英寸设备上必然被裁掉；
  #   且图例置于 (0.30,0.04) 会压住最末一行。
  #   这里改为 patchwork 双面板：A 面板 = 森林图（full 宽度、自带 y 轴与图例），
  #   B 面板 = 纯文本列（N / AUC / O:E），两面板共用同一 y 映射并对齐，
  #   从根本上消除裁切与重叠。
  sep_y <- sapply(seq(2, nrow(subs)), function(i)
    if (as.character(subs$Group[i]) != as.character(subs$Group[i - 1]))
      subs$y_pos[i] + 0.5 else NA)
  sep_y <- sep_y[!is.na(sep_y)]
  df$auc_lab <- sprintf("%.3f (%.3f-%.3f)",
                        df$Metric_Value, df$CI_Lower, df$CI_Upper)
  upd$oe_lab <- sprintf("%.2f (%.2f-%.2f)", oe["oe", ], oe["lo", ], oe["hi", ])
  x_lo <- min(0.30, min(df$CI_Lower, na.rm = TRUE) - 0.03)
  x_hi <- 1.00
  y_lo <- 0.35
  y_hi <- max(subs$y_pos) + 1.05
  loc_short <- setNames(c("Original model", "Updated model"), mods)

  # ---- A 面板：森林图（仅图形，无文本列）----
  p_main <- ggplot(df, aes(x = Metric_Value, y = y_pos + y_off)) +
    geom_vline(xintercept = 0.5, linetype = "dashed", color = COL_GRAY,
               linewidth = 0.5) +
    geom_segment(data = data.frame(y = sep_y),
                 aes(x = -Inf, xend = Inf, y = y, yend = y),
                 linetype = "dotted", color = "gray70", linewidth = 0.4,
                 inherit.aes = FALSE) +
    geom_linerange(aes(xmin = CI_Lower, xmax = CI_Upper, color = Model),
                   linewidth = 0.8) +
    geom_point(aes(color = Model, shape = Model), size = 2.6) +
    scale_color_manual(values = GROUP_COLORS, breaks = mods,
                       labels = loc_short[mods]) +
    scale_shape_manual(values = GROUP_SHAPES, breaks = mods,
                       labels = loc_short[mods]) +
    scale_x_continuous(limits = c(x_lo, x_hi),
                       breaks = seq(ceiling(x_lo * 10) / 10, x_hi, by = 0.1),
                       expand = expansion(mult = 0)) +
    scale_y_continuous(limits = c(y_lo, y_hi), breaks = subs$y_pos,
                       labels = subs$Subgroup, expand = expansion(mult = 0)) +
    labs(x = "AUC (95% CI)", y = NULL) +
    theme_sci() + theme_leg +
    theme(axis.text.y = element_text(size = FONT_TICK - 1, hjust = 1),
          panel.grid.major.y = element_blank(),
          legend.position = "bottom",
          legend.direction = "horizontal",
          legend.text = element_text(size = FONT_LEGEND - 1),
          legend.margin = margin(2, 4, 2, 4),
          legend.box.margin = margin(0, 0, 0, 0),
          plot.margin = margin(12, 4, 2, 6))

  # ---- B 面板：文本列（N events / AUC 95% CI / O:E 95% CI）----
  # 列位置用面板 npc 比例（0-1），不受 x 数据范围影响，故绝不会被裁切。
  xc <- c(N = 0.00, AUC = 0.335, OE = 0.675)
  tx_body <- data.frame(
    x = c(rep(xc["N"], nrow(subs)), rep(xc["AUC"], nrow(df)),
          rep(xc["OE"], nrow(upd))),
    y = c(subs$y_pos, df$y_pos + df$y_off, upd$y_pos),
    lab = c(subs$N_label, df$auc_lab, upd$oe_lab))
  tx_head <- data.frame(
    x = unname(xc), y = y_hi - 0.22,
    lab = c("N (events)", "AUC (95% CI)", "Rel. event rate"))
  p_txt <- ggplot() +
    geom_segment(data = data.frame(y = sep_y),
                 aes(x = -Inf, xend = Inf, y = y, yend = y),
                 linetype = "dotted", color = "gray70", linewidth = 0.4) +
    geom_text(data = tx_body, aes(x = x, y = y, label = lab),
              hjust = 0, size = 2.55, color = "black", family = "sans") +
    geom_text(data = tx_head, aes(x = x, y = y, label = lab),
              hjust = 0, size = 2.75, fontface = "bold", color = "black",
              family = "sans") +
    scale_x_continuous(limits = c(0, 1), expand = expansion(mult = c(0.01, 0.03))) +
    scale_y_continuous(limits = c(y_lo, y_hi), expand = expansion(mult = 0)) +
    theme_void() +
    theme(plot.margin = margin(12, 8, 2, 0))

  p <- p_main + p_txt + plot_layout(widths = c(1, 0.58))
  # 导出亚组表（供论文 Table S11，与图 9A 同源）
  out <- data.frame(Subgroup = df$Subgroup, Model = df$Model, N = df$N,
                    Events = df$N_Pos, AUC = df$Metric_Value,
                    AUC_95CI_lower = df$CI_Lower, AUC_95CI_upper = df$CI_Upper,
                    OE = NA_real_, OE_95CI_lower = NA_real_, OE_95CI_upper = NA_real_)
  idx <- match(paste(upd$Subgroup, upd$Model), paste(out$Subgroup, out$Model))
  out$OE[idx] <- oe["oe", ]; out$OE_95CI_lower[idx] <- oe["lo", ]
  out$OE_95CI_upper[idx] <- oe["hi", ]
  write.csv(out, file.path(BASE_DIR, "Outputs", "r_subgroup_auc.csv"),
            row.names = FALSE, fileEncoding = "UTF-8")
  h <- max(5.0, nrow(subs) * 0.56 + 2.1)
  save_sci_plot(p, "R04_Subgroup_Forest_Plot", width = 9.2, height = h,
                dpi_type = "line")
  invisible(p)
}

# =============================================================================
# R05 风险分层点图（Cleveland 点图 + Wilson CI）
# =============================================================================
plot_risk_stratification <- function() {
  cat("[R05] 风险分层点图...\n")
  d <- dat
  d$Risk_Group <- ifelse(d$p < LOW_CUT, "Low",
                         ifelse(d$p < HIGH_CUT, "Intermediate", "High"))
  d$Risk_Group <- factor(d$Risk_Group, levels = c("Low", "Intermediate", "High"))
  agg <- d %>% group_by(group, Risk_Group) %>%
    summarise(N = n(), Pos = sum(y), Observed_Rate = mean(y), .groups = "drop") %>%
    rowwise() %>%
    mutate(lo = wilson_ci(Pos, N)[1], hi = wilson_ci(Pos, N)[2]) %>%
    ungroup()
  agg$group <- factor(agg$group, levels = GROUP_LEVELS)
  agg$y_pos <- as.numeric(agg$Risk_Group) +
    c("Derivation (CV)" = 0.27, "Geographical, original model" = 0.09,
      "Geographical, updated model" = -0.09,
      "Geographical, control arm" = -0.27)[as.character(agg$group)]
  deriv_rate <- with(agg, sum(agg$Pos[group == GROUP_LEVELS[1]]) /
                       sum(agg$N[group == GROUP_LEVELS[1]]))
  # ★ 修正（旧版三行 y 轴标签均打印同一串截断值）：改为各风险组各自的真实区间
  rg_levels <- c("Low", "Intermediate", "High")
  rg_labels <- c(sprintf("Low\n(p < %.2f)", LOW_CUT),
                 sprintf("Intermediate\n(%.2f-%.2f)", LOW_CUT, HIGH_CUT),
                 sprintf("High\n(p >= %.2f)", HIGH_CUT))
  agg$rg_lab <- sprintf("%s\n(%.2f / %.2f)", as.character(agg$Risk_Group),
                        LOW_CUT, HIGH_CUT)
  p <- ggplot(agg, aes(x = Observed_Rate * 100, y = y_pos)) +
    geom_line(aes(group = Risk_Group), color = "gray70", linewidth = 0.5,
              alpha = 0.5) +
    geom_vline(xintercept = deriv_rate * 100, linetype = "dashed",
               color = COL_GRAY, linewidth = 0.5) +
    geom_linerange(aes(xmin = lo * 100, xmax = hi * 100, color = group),
                   linewidth = 0.8) +
    geom_point(aes(color = group, shape = group), size = 3.0, stroke = 0.7) +
    # ★ 修正：百分比标签改为贴着 CI 右端外侧，避免压在点标记上
    geom_text(aes(x = hi * 100, label = sprintf("%.0f%%", Observed_Rate * 100),
                  color = group),
              hjust = -0.18, vjust = 0.5, size = 2.7, show.legend = FALSE) +
    scale_color_manual(values = GROUP_COLORS, breaks = GROUP_LEVELS,
                       labels = GROUP_SHORT[GROUP_LEVELS]) +
    scale_shape_manual(values = GROUP_SHAPES, breaks = GROUP_LEVELS,
                       labels = GROUP_SHORT[GROUP_LEVELS]) +
    scale_y_continuous(breaks = 1:3, labels = rg_labels,
                       limits = c(0.4, 3.6)) +
    scale_x_continuous(limits = c(0, 100), labels = scales::label_number(suffix = "%")) +
    labs(x = "Observed event rate", y = "Risk group (predicted probability)") +
    theme_sci() + theme_leg +
    theme(legend.position = c(0.80, 0.16),
          legend.text = element_text(size = FONT_SMALL),
          legend.background = element_rect(fill = alpha("white", 0.88),
                                          color = "gray80", linewidth = 0.3))
  # 导出风险分层表（供论文 Table S12，与图 9B 同源）
  write.csv(agg[, c("group", "Risk_Group", "N", "Pos", "Observed_Rate", "lo", "hi")],
            file.path(BASE_DIR, "Outputs", "r_risk_strata.csv"),
            row.names = FALSE, fileEncoding = "UTF-8")
  save_sci_plot(p, "R05_Risk_Stratification_Dotplot", width = 7.4, height = 4.6,
                dpi_type = "line")
  invisible(p)
}

# =============================================================================
# R06 临床影响曲线（双 y 轴）
# =============================================================================
plot_clinical_impact <- function() {
  cat("[R06] 临床影响曲线...\n")
  thr <- seq(0.01, 0.50, by = 0.01)
  # ★ 复核轮：本面板只画地理队列上的三个模型。原先并入的 "Derivation (CV)"
  #   在外部队列的临床影响曲线里没有对照意义（不同队列、不同患病率），
  #   图注也无法一句话解释清楚，故移除。
  IMPACT_LEVELS <- GEO_MODELS
  rows <- do.call(rbind, lapply(IMPACT_LEVELS, function(m) {
    d <- dat[dat$group == m, ]
    n_hr <- sapply(thr, function(t) sum(d$p >= t))
    n_pos <- sapply(thr, function(t) sum(d$p >= t & d$y == 1))
    ppv <- ifelse(n_hr > 0, n_pos / n_hr, 0)
    data.frame(Threshold = thr, N_High_Risk = n_hr, N_High_Risk_Pos = n_pos,
               PPV = ppv, group = m)
  }))
  rows$group <- factor(rows$group, levels = IMPACT_LEVELS)
  rows$lab <- as.character(rows$group)
  cols <- setNames(unname(GROUP_COLORS[IMPACT_LEVELS]), IMPACT_LEVELS)
  sf <- max(rows$N_High_Risk) / 1.0
  p <- ggplot() +
    geom_bar(data = rows, aes(x = Threshold, y = N_High_Risk, fill = lab),
             stat = "identity", alpha = 0.18, width = 0.008,
             position = "dodge") +
    geom_line(data = rows, aes(x = Threshold, y = N_High_Risk_Pos, color = lab),
              linewidth = 1.0) +
    geom_line(data = rows, aes(x = Threshold, y = PPV * sf, color = lab),
              linewidth = 1.1, linetype = "dashed") +
    scale_y_continuous(name = "Number of patients (high risk / true positives)",
                       sec.axis = sec_axis(~ . / sf, name = "Positive predictive value",
                                           labels = scales::percent_format(accuracy = 1))) +
    scale_color_manual(values = cols) +
    scale_fill_manual(values = cols) +
    coord_cartesian(xlim = c(0, 0.50)) +
    scale_x_continuous(limits = c(0, 0.50), breaks = seq(0, 0.5, by = 0.1),
                       labels = scales::label_number(accuracy = 0.1)) +
    labs(x = "Threshold probability") +
    theme_sci() + theme_leg +
    theme(legend.position = "bottom", legend.box = "horizontal",
          legend.text = element_text(size = FONT_SMALL),
          axis.title.y.left = element_text(size = FONT_LABEL),
          axis.title.y.right = element_text(size = FONT_LABEL))
  save_sci_plot(p, "R06_Clinical_Impact_Curve", width = 8.0, height = 6.0,
                dpi_type = "line")
  invisible(p)
}

# =============================================================================
# R07 连续净收益（标准化净收益 + AUC-NB）
# =============================================================================
std_nb <- function(y, p, thr, prev) {
  nb <- nb_curve(y, p, thr)
  if (prev > 0) nb / prev else rep(0, length(nb))
}
plot_continuous_nb <- function() {
  cat("[R07] 连续净收益...\n")
  thr <- seq(0.01, 0.80, by = 0.01)
  y <- geo_wide$y; prev <- mean(y)
  mods <- c("Geographical, original model", "Geographical, updated model",
            "Geographical, control arm")
  snb_df <- do.call(rbind, lapply(mods, function(m) {
    data.frame(Threshold = thr, Value = std_nb(y, geo_wide[[m]], thr, prev),
               Series = m)
  }))
  ta <- (prev - (1 - prev) * (thr / (1 - thr))) / prev
  all_lines <- rbind(snb_df,
                     data.frame(Threshold = thr, Value = ta,
                                Series = "Treat All (Geographical)"),
                     data.frame(Threshold = thr, Value = 0, Series = "Treat None"))
  auc_nb <- sapply(mods, function(m) {
    v <- snb_df$Value[snb_df$Series == m]
    sum(diff(thr) * (head(v, -1) + tail(v, -1)) / 2)
  })
  lv <- c(mods, "Treat All (Geographical)", "Treat None")
  lbl <- c(sprintf("%s (AUC-NB = %.3f)", GROUP_SHORT[mods], auc_nb),
           "Treat all (geographical)", "Treat none")
  line_cols <- c(setNames(unname(GROUP_COLORS[mods]), mods),
                 "Treat All (Geographical)" = "gray60", "Treat None" = "black")
  line_types <- c(setNames(rep("solid", length(mods)), mods),
                  "Treat All (Geographical)" = "dashed", "Treat None" = "dashed")
  line_w <- c(setNames(rep(1.1, length(mods)), mods),
              "Treat All (Geographical)" = 0.7, "Treat None" = 0.5)
  all_lines$Series <- factor(all_lines$Series, levels = lv)
  p <- ggplot(all_lines, aes(x = Threshold, y = Value, color = Series,
                             linetype = Series, linewidth = Series)) +
    geom_line() +
    scale_color_manual(name = NULL, values = line_cols, breaks = lv, labels = lbl) +
    scale_linetype_manual(name = NULL, values = line_types, breaks = lv, labels = lbl) +
    scale_linewidth_manual(name = NULL, values = line_w, breaks = lv, labels = lbl) +
    coord_cartesian(xlim = c(0, 0.80),
                    ylim = c(max(-0.20, min(all_lines$Value)), 1.05)) +
    scale_x_continuous(breaks = seq(0, 0.8, by = 0.1),
                       labels = scales::label_number(accuracy = 0.1)) +
    labs(x = "Threshold probability", y = "Standardized net benefit") +
    guides(color = guide_legend(override.aes = list(linewidth = 1.2)),
           linetype = "none", linewidth = "none") +
    theme_sci() + theme_leg +
    theme(legend.position = c(0.76, 0.86),
          legend.text = element_text(size = FONT_SMALL),
          legend.background = element_rect(fill = alpha("white", 0.88),
                                          color = "gray80", linewidth = 0.3))
  save_sci_plot(p, "R07_Continuous_Net_Benefit", width = 7.0, height = 6.0,
                dpi_type = "color")
  invisible(p)
}

# =============================================================================
# R08 避免不必要干预数（NIA，每 100 名被治疗者）
# =============================================================================
plot_nia <- function() {
  cat("[R08] NIA...\n")
  thr <- seq(0.01, 0.80, by = 0.01)
  y <- geo_wide$y; prev <- mean(y)
  mods <- c("Geographical, original model", "Geographical, updated model",
            "Geographical, control arm")
  odds <- thr / (1 - thr)
  ta <- prev - (1 - prev) * odds
  nia <- do.call(rbind, lapply(mods, function(m) {
    data.frame(Threshold = thr,
               NIA = 100 * (nb_curve(y, geo_wide[[m]], thr) - ta) / odds,
               Series = m)
  }))
  nia$Series <- factor(nia$Series, levels = mods)
  p <- ggplot(nia, aes(x = Threshold, y = NIA, color = Series)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = COL_GRAY,
               linewidth = 0.5) +
    geom_line(linewidth = 1.2) +
    scale_color_manual(values = GROUP_COLORS[mods],
                       breaks = mods, labels = GROUP_SHORT[mods]) +
    coord_cartesian(xlim = c(0, 0.80)) +
    scale_x_continuous(breaks = seq(0, 0.8, by = 0.1),
                       labels = scales::label_number(accuracy = 0.1)) +
    labs(x = "Threshold probability",
         y = "Net interventions avoided (per 100 patients)") +
    theme_sci() + theme_leg +
    theme(legend.position = c(0.74, 0.20),
          legend.text = element_text(size = FONT_SMALL),
          legend.background = element_rect(fill = alpha("white", 0.88),
                                          color = "gray80", linewidth = 0.3))
  save_sci_plot(p, "R08_Net_Intervention_Avoided", width = 7.0, height = 6.0,
                dpi_type = "line")
  invisible(p)
}

# =============================================================================
# 主图版式合成（论文主图 8 / 主图 9）
# =============================================================================
# 主图 8：临床效用四联图（A DCA / B 临床影响曲线 / C 连续净收益 / D 避免干预数）
compose_clinical_utility <- function() {
  cat("[MAIN-8] 临床效用四联图...\n")
  p1 <- plot_dca_curves()
  p2 <- plot_clinical_impact()
  p3 <- plot_continuous_nb(); p4 <- plot_nia()
  # 拼版内 B 面板改为面板内图例（旧 07 的 "bottom" 在四联版式里会把图例顶到两行之间，
  # 造成 A 面板下方大片空白），仍保留独立出图时的 bottom 版式。
  p2 <- p2 + theme(legend.position = c(0.55, 0.115),
                   legend.direction = "horizontal",
                   legend.key.size = unit(0.34, "cm"),
                   legend.text = element_text(size = 6.6),
                   legend.background = element_rect(fill = alpha("white", 0.88),
                                                   color = "gray80", linewidth = 0.3)) +
    guides(color = guide_legend(nrow = 2, byrow = TRUE),
           fill  = guide_legend(nrow = 2, byrow = TRUE))
  plate <- (p1 | p2) / (p3 | p4) +
    plot_annotation(tag_levels = "A") &
    theme(plot.tag = element_text(face = "bold", size = FONT_LABEL))
  save_sci_plot(plate, "Fig8_clinical_utility", width = 12.0, height = 10.0,
                dpi_type = "line")
  invisible(plate)
}

# 主图 9：亚组森林图 + 风险分层点图（上下叠放，A / B）
compose_subgroups_risk <- function() {
  cat("[MAIN-9] 亚组与风险分层合成图...\n")
  pf <- plot_subgroup_forest()
  pr <- plot_risk_stratification()
  # ★ R04 自身是「森林图 + 文本列」二层 patchwork；若直接嵌套，patchwork 会把文本列
  #   也当成一个独立 panel 而多打一个标签（变成 A/B/C）。这里显式取回两个子面板，
  #   只给森林图与风险分层图打 A/B，文本列不参与编号。
  p_forest <- pf[[1]] + labs(tag = "A")
  p_text   <- pf[[2]]
  p_risk   <- pr + labs(tag = "B")
  plate <- ((p_forest | p_text) / p_risk) +
    plot_layout(heights = c(1.45, 1)) &
    theme(plot.tag = element_text(face = "bold", size = FONT_LABEL),
          plot.tag.position = c(0.005, 0.99))
  save_sci_plot(plate, "Fig9_subgroups_and_risk", width = 9.2, height = 10.6,
                dpi_type = "line")
  invisible(plate)
}

# =============================================================================
# 主函数
# =============================================================================
main <- function() {
  cat(strrep("=", 78), "\n")
  cat("  10_clinical_plots_R.R - 关键临床图（R 版，照旧 07 规范）\n")
  cat(strrep("=", 78), "\n")
  cat(sprintf("INPUT : %s\nOUTPUT: %s\n", IN_DIR, OUT_DIR))
  cat(sprintf("n = %d 行；风险分层截断值 %.4f / %.4f\n\n", nrow(dat), LOW_CUT, HIGH_CUT))

  plots <- list(R01_Growth_Curve = plot_growth_curve,
                R02_Calibration_Curves = plot_calibration_curves,
                R03_DCA_Curves = plot_dca_curves,
                R04_Subgroup_Forest_Plot = plot_subgroup_forest,
                R05_Risk_Stratification_Dotplot = plot_risk_stratification,
                R06_Clinical_Impact_Curve = plot_clinical_impact,
                R07_Continuous_Net_Benefit = plot_continuous_nb,
                R08_Net_Intervention_Avoided = plot_nia,
                MAIN8_Fig8_clinical_utility = compose_clinical_utility,
                MAIN9_Fig9_subgroups_and_risk = compose_subgroups_risk)
  only <- Sys.getenv("CLINPLOT_ONLY")
  if (nzchar(only)) plots <- plots[grepl(only, names(plots))]
  DEBUG <- identical(Sys.getenv("CLINPLOT_DEBUG"), "1")

  for (nm in names(plots)) {
    if (DEBUG) {
      plots[[nm]]()
    } else {
      tryCatch(plots[[nm]](), error = function(e)
        cat(sprintf("  ERROR in %s: %s\n", nm, conditionMessage(e))))
    }
  }
  cat("\n", strrep("=", 78), "\n  所有 R 临床图生成完成\n", strrep("=", 78), "\n", sep = "")
}

main()
