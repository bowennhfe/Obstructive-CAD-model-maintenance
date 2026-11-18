# -*- coding: utf-8 -*-
"""
08_roc_shap_original.py — 按旧记录 05 的**原始方法**重画 ROC 与 SHAP
=============================================================================
背景
    用户指令（2026-09-20）：ROC 曲线与 SHAP 改用「旧记录/Code/05_Model_Development.py」
    的原始实现方法，替换论文原 Figure 2 / Figure 4 / Figure 5 的对应内容。
    （原 Figure 2 的逐折 AUC 分布与袋外校准迁至补充图 Figure S1；
      原 Figure 5 的 ROC 栏因与本文多队列 ROC 重复而移除。）

与原脚本的对应关系（逐条对齐，只改画法、不动任何建模/统计）
    ROC   <- 05_Model_Development.py :: plot_roc_curves
              单图叠加多队列；横轴 1 - Specificity、纵轴 Sensitivity；
              黑色对角虚线 'k--'（lw 0.8, alpha 0.5）；图例右下、含 AUC 与 95% CI；
              网格 alpha=0.3；等比例坐标。
    SHAP  <- 05_Model_Development.py :: plot_shap
              shap.TreeExplainer(clf) → explainer.shap_values(X)（二分类取正类）；
              shap.summary_plot(plot_type="dot") 与 (plot_type="bar")；
              前 6 位特征依赖图：二分变量 → 小提琴图（四分位内线）、
              连续变量 → 散点（c = 特征值, cmap='coolwarm'）+ y = 0 灰虚线。

产出
    Figures/*/Fig2_roc_multicohort.*        多队列 ROC（正文 Figure 2）
    Figures/*/Fig4_derivation_shap.*        SHAP beeswarm（正文 Figure 4）
    Figures/*/FigS2_shap_bar.*              SHAP 条形图（补充图）
    Figures/*/FigS3_shap_dependence.*       前 6 位特征依赖图（补充图）
    Outputs/s08_roc_multicohort.csv         各队列 AUC 与 95% CI

★ 本脚本不拟合任何模型：推导队列袋外概率取自 01 的产物，地理队列概率取自 02 与 03 的产物。
=============================================================================
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

import common as C

# ---------------------------------------------------------------------------
# 队列配色（与旧 05 的 COLORBLIND_PALETTE 一致：Derivation 蓝 / Geographic 绿）
# ---------------------------------------------------------------------------
ROC_COLORS = {
    "derivation": "#1f77b4",   # 蓝
    "geo_frozen": "#2ca02c",   # 绿
    "geo_updated": "#9467bd",  # 紫
    "geo_control": "#7f7f7f",  # 灰
}
ENDPOINT_PCT = 1.00            # 数据积累端点（全部同期数据并入）


# =============================================================================
# 数据装配
# =============================================================================
def load_prediction_sets():
    """返回 [(label, color, ls, y, p, n_derive)]，顺序即图例顺序。"""
    oof = C.read_csv_out("s01_oof_rf")
    y_d, p_d = oof["y"].values.astype(int), oof["oof_rf"].values.astype(float)

    gp = C.read_csv_out("s02_geo_pred_rf")
    y_g, p_g = gp["y"].values.astype(int), gp["p_rf"].values.astype(float)

    g = pd.read_pickle(os.path.join(C.OUT_DIR, "s03_growth_full.pkl"))
    rep = int(g["repeat"].min())                      # 取第 1 次重复，作为一次真实实现
    end = g[(g["pct_temporal"] == ENDPOINT_PCT) & (g["repeat"] == rep)]
    row_t = end[end["arm"] == "temporal"].iloc[0]
    row_c = end[end["arm"] == "control"].iloc[0]
    y_t = np.asarray(row_t["y_geo"]).astype(int)
    p_t = np.asarray(row_t["p_geo"]).astype(float)
    y_c = np.asarray(row_c["y_geo"]).astype(int)
    p_c = np.asarray(row_c["p_geo"]).astype(float)

    sets = [
        (f"Derivation cohort (n = {len(y_d)}), 10-fold CV", "derivation", "-",
         y_d, p_d, len(y_d)),
        (f"Geographical cohort (n = {len(y_g)}) - model developed on the original cohort",
         "geo_frozen", "-", y_g, p_g, len(y_d)),
        (f"Geographical cohort (n = {len(y_g)}) - model developed after data accumulation",
         "geo_updated", "-", y_t, p_t, int(row_t["n_derive"])),
        (f"Geographical cohort (n = {len(y_g)}) - row-count-matched control arm",
         "geo_control", "--", y_c, p_c, int(row_c["n_derive"])),
    ]
    return sets


# =============================================================================
# Figure 2 —— 多队列 ROC（旧 05 plot_roc_curves 方法）
# =============================================================================
def fig_roc_multicohort(sets, path_out):
    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    rows = []
    for label, key, ls, y, p, n_dev in sets:
        auc = roc_auc_score(y, p)
        lo, hi = C.boot_ci(y, p, lambda a, b: roc_auc_score(a, b), n_iter=2000)
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, lw=1.8, ls=ls, color=ROC_COLORS[key],
                label=f"{label}\n    AUC {auc:.3f} (95% CI {lo:.3f}-{hi:.3f})")
        rows.append({"cohort": label, "n": int(len(y)),
                     "events": int(np.sum(y)), "n_derive": int(n_dev),
                     "auc": float(auc), "ci_lo": float(lo), "ci_hi": float(hi),
                     "role": {"derivation": "internal cross-validation",
                              "geo_frozen": "external, frozen model",
                              "geo_updated": "external, updated model",
                              "geo_control": "external, row-count-matched control"}[key]})
    # 旧 05 的参考线：黑色对角虚线
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("1 - Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_aspect("equal")
    C.style_axes(ax, grid=True, alpha=0.30)
    C.style_legend(ax, loc="lower right", fontsize=8.4, handlelength=1.8)
    C.save_figure(fig, path_out)
    return pd.DataFrame(rows)


# =============================================================================
# SHAP 值（TreeExplainer，与 01 的 Table 4 同源）
# =============================================================================
def compute_shap(cohorts):
    import shap
    obj = C.load_pickle("m01_rf_original.pkl")
    pipe = obj["pipeline"]
    X, y = C.derivation_label(cohorts["original"])
    X = X.reset_index(drop=True)
    Ximp = pipe.named_steps["imp"].transform(X)
    explainer = shap.TreeExplainer(pipe.named_steps["clf"])
    raw = explainer.shap_values(Ximp)
    if isinstance(raw, list):
        sv = np.asarray(raw[1])
    elif np.ndim(raw) == 3:
        sv = np.asarray(raw)[:, :, 1]
    else:
        sv = np.asarray(raw)
    names = [C.LABEL_EN[f] for f in C.FEATURES]
    C.log(f"  SHAP 矩阵 {sv.shape}  平均 |SHAP| 前 3："
          + ", ".join(f"{names[j]} {np.abs(sv[:, j]).mean():.4f}"
                      for j in np.argsort(-np.abs(sv).mean(axis=0))[:3]))
    return sv, X, Ximp, names


# =============================================================================
# Figure 4 —— SHAP beeswarm（旧 05 plot_shap 的 plot_type="dot"）
# =============================================================================
def fig_shap_beeswarm(sv, Ximp, names, path_out):
    import shap
    shap.summary_plot(sv, Ximp, feature_names=names, plot_type="dot",
                      max_display=len(names), show=False,
                      rng=np.random.RandomState(C.SEED))
    fig = plt.gcf()
    fig.set_size_inches(7.4, 5.8)
    if fig.axes:
        fig.axes[0].set_xlabel("SHAP value (impact on model output)", fontsize=10)
    C.save_figure(fig, path_out)


# =============================================================================
# Figure S2 —— SHAP bar（旧 05 plot_shap 的 plot_type="bar"，原脚本列为补充）
# =============================================================================
def fig_shap_bar(sv, Ximp, names, path_out):
    import shap
    shap.summary_plot(sv, Ximp, feature_names=names, plot_type="bar",
                      max_display=len(names), show=False,
                      rng=np.random.RandomState(C.SEED))
    fig = plt.gcf()
    fig.set_size_inches(7.4, 5.8)
    if fig.axes:
        fig.axes[0].set_xlabel("Mean |SHAP value|", fontsize=10)
    C.save_figure(fig, path_out)


# =============================================================================
# Figure S3 —— 前 6 位特征依赖图（旧 05 plot_shap 的 dependence 段）
# =============================================================================
def fig_shap_dependence(sv, X, path_out, top=6):
    mean_abs = np.abs(sv).mean(axis=0)
    idx = list(np.argsort(mean_abs)[-top:][::-1])
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.9))
    axes = np.asarray(axes).ravel()
    letters = "ABCDEF"
    for k, (ax, j) in enumerate(zip(axes, idx)):
        name = C.FEATURES[j]
        xv = np.asarray(X.iloc[:, j], float)
        yv = np.asarray(sv[:, j], float)
        uv = np.unique(xv[~np.isnan(xv)])
        binary = len(uv) <= 2
        if binary and all(np.sum(xv == v) >= 3 for v in uv):
            groups = [yv[xv == v] for v in uv]
            parts = ax.violinplot(groups, showmeans=False, showextrema=False,
                                  showmedians=False, widths=0.8)
            for body, col in zip(parts["bodies"],
                                 ["#1f77b4", "#d62728"][:len(groups)]):
                body.set_facecolor(col)
                body.set_alpha(0.82)
                body.set_edgecolor("#333333")
                body.set_linewidth(0.6)
            for i, gp in enumerate(groups):
                q1, med, q3 = np.percentile(gp, [25, 50, 75])
                ax.vlines(i + 1, q1, q3, color="#222222", lw=1.0)
                ax.plot([i + 1 - 0.09, i + 1 + 0.09], [med, med],
                        color="white", lw=1.8)
            ax.set_xticks(np.arange(1, len(groups) + 1))
            ax.set_xticklabels([f"{int(v)}" for v in uv])
        else:
            sc = ax.scatter(xv, yv, c=xv, cmap="coolwarm", alpha=0.7, s=16,
                            edgecolors="none")
            cb = fig.colorbar(sc, ax=ax, pad=0.025, fraction=0.042)
            cb.ax.tick_params(labelsize=7)
        ax.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.6)
        ax.set_xlabel(C.LABEL_EN[name], fontsize=9)
        # 纵轴标题只在每行最左面板标注，避免紧贴左侧面板的色标造成误读
        ax.set_ylabel("SHAP value" if k in (0, 3) else "", fontsize=9)
        ax.set_title(f"{letters[k]}  {C.LABEL_EN[name]}", loc="left",
                     fontsize=9.5, fontweight="bold")
        ax.tick_params(labelsize=8.5)
        C.style_axes(ax, grid=True, alpha=0.25)
    fig.subplots_adjust(wspace=0.58, hspace=0.72)
    C.save_figure(fig, path_out)
    return [C.FEATURES[j] for j in idx]


# =============================================================================
# main
# =============================================================================
def main():
    C.log_init("08_roc_shap_original")
    t0 = time.time()
    only = sys.argv[1] if len(sys.argv) > 1 else "all"

    C.section("Figure 2  多队列 ROC（旧 05 plot_roc_curves 方法）")
    sets = load_prediction_sets()
    t8 = fig_roc_multicohort(sets, "Fig2_roc_multicohort")
    C.save_csv(t8, "s08_roc_multicohort")
    for _, r in t8.iterrows():
        C.log(f"  {r['role']:32s}  AUC {r['auc']:.3f} "
              f"(95% CI {r['ci_lo']:.3f}-{r['ci_hi']:.3f})  n={r['n']}")

    if only in ("all", "figs"):
        C.section("SHAP（旧 05 plot_shap 方法：TreeExplainer + summary_plot）")
        cohorts, _ = C.load_cohorts(verbose=False)
        sv, X, Ximp, names = compute_shap(cohorts)
        fig_shap_beeswarm(sv, Ximp, names, "Fig4_derivation_shap")
        fig_shap_bar(sv, Ximp, names, "FigS2_shap_bar")
        top6 = fig_shap_dependence(sv, X, "FigS3_shap_dependence")
        C.log("  依赖图前 6 位特征：" + ", ".join(C.LABEL_EN[f] for f in top6))

    C.section(f"脚本 08 完成，总耗时 {C.timer(t0)}")


if __name__ == "__main__":
    main()
