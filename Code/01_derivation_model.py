# -*- coding: utf-8 -*-
"""
01 —— 推导队列建模（复现旧文）
=============================================================================
队列：original data.xlsx，全部 442 例（不剔除任何受试者，含 15 岁 1 例）
设计：10 折交叉验证 × 5 次重复（RepeatedStratifiedKFold, n=50 折）
模型：**只建立随机森林（RF）一个模型**（用户指令 2026-09-20）。
      RF 即旧文报告的最佳模型，故作为预先指定的唯一模型贯穿全流程，
      不再建立/比较 SVM、NN、KNN、DTC、LDA。
输出：
    Fig1  研究流程图
    Fig2  推导队列判别力（CV 平均 ROC / 逐折 AUC 分布 / 袋外校准）
    Fig3  特征重要性（MDI + 折外置换）
    Fig4  SHAP（beeswarm + 平均 |SHAP|）
    Table 1  三队列基线特征
    Table 2  RF 的 10 折 CV×5 性能
    Table 3  特征重要性
    TableS1  越界值（策略 D）置缺失明细
    TableS2  网格搜索结果（RF）
★ 抗泄露：插补封装在 Pipeline 内，只在训练折 fit；阈值来自推导集袋外
=============================================================================
"""
import os
import sys
import time
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

from sklearn.model_selection import (StratifiedKFold, RepeatedStratifiedKFold,
                                     GridSearchCV, cross_val_predict)
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_curve, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


# =============================================================================
# 绘图小工具
# =============================================================================
def panel_tag(ax, tag, dx=-0.13, dy=1.06):
    ax.text(dx, dy, tag, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top", ha="left")


def swarm_y(vals, nbins=120, spread=0.42):
    """按数值局部密度把点垂直铺开（beeswarm 的简化实现）。"""
    vals = np.asarray(vals, float)
    if vals.max() <= vals.min():
        return np.zeros(len(vals))
    edges = np.linspace(vals.min(), vals.max(), nbins + 1)
    idx = np.clip(np.digitize(vals, edges) - 1, 0, nbins - 1)
    y = np.zeros(len(vals))
    for b in np.unique(idx):
        pos = np.where(idx == b)[0]
        order = pos[np.argsort(vals[pos])]
        k = len(order)
        y[order] = (np.linspace(-spread, spread, k) if k > 1 else 0.0)
    return y


# =============================================================================
# Fig 1 研究流程
# =============================================================================
def fig_study_flow(cohorts, n_derive, path_out):
    """研究流程图。

    版式按英寸精确布置（1 数据单位 = 0.01 英寸）：框高由正文行数推出，
    避免文案增删后文字溢出方框。字体 9.5 pt（框标题）/ 7.6 pt（正文），
    均为印刷可读且不溢出的组合。
    """
    W, H = 7.2, 5.2
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    AW, AH = W * 0.96 * 100.0, H * 0.96 * 100.0
    ax.set_xlim(0, AW); ax.set_ylim(0, AH); ax.axis("off")
    # 不可见外框：保证 bbox_inches="tight" 不裁掉留白，物理尺寸可控
    ax.add_patch(Rectangle((0, 0), AW, AH, facecolor="none",
                           edgecolor="none", linewidth=0))

    FS_T, FS_B, FS_N = 9.5, 7.6, 7.5
    OFF_T, OFF_B, PAD_B = 13.0, 30.0, 12.0            # 距框顶（标题/正文）与框底留白
    LH = FS_B * 1.42 / 72 * 100.0                     # 正文行高
    H3 = OFF_B + 3 * LH + PAD_B                       # 3 行正文所需框高
    H5 = OFF_B + 5 * LH + PAD_B                       # 5 行正文所需框高

    def box(x, y, w, h, title, body, fc, ec, tc="#111111"):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0,rounding_size=6",
                     linewidth=1.2, facecolor=fc, edgecolor=ec))
        ax.text(x + w / 2, y + h - OFF_T, title, ha="center", va="top",
                fontsize=FS_T, fontweight="bold", color=tc)
        ax.text(x + w / 2, y + h - OFF_B, body, ha="center", va="top",
                fontsize=FS_B, color=tc, linespacing=1.42)

    def arrow(x1, y1, x2, y2, color="#444444"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                     mutation_scale=11, linewidth=1.1, color=color,
                     shrinkA=2, shrinkB=2))

    o = cohorts["original"]; t = cohorts["temporal"]; g = cohorts["geo"]

    # ---- 纵向：4 行，行距按剩余空间均分（下 → 上） ----
    gap = (AH - 14 - 22 - (H3 + H3 + H3 + H5)) / 3.0
    yE = 22.0
    yD = yE + H5 + gap
    yBC = yD + H3 + gap
    yA = yBC + H3 + gap

    cx = AW / 2.0

    # ---- 行 1：推导队列 ----
    box(cx - 190, yA, 380, H3, "Derivation cohort (single centre, 2017-2022)",
        f"Original cohort: n = {len(o)}; obstructive "
        f"{int(o['label'].sum())} ({o['label'].mean()*100:.1f}%)\n"
        f"19 candidate variables; 10-fold CV x {C.CV_REPEAT} repeats\n"
        "Random forest pre-specified as the only model",
        "#dbe9f6", C.C_ORIG)

    # ---- 行 2：Step 1（左） / Step 2（右） ----
    box(25, yBC, 300, H3, "Step 1  Model development",
        "Random forest only (pre-specified)\n"
        "Grid search within 10-fold CV\n"
        "-> carried forward unchanged",
        "#eef3f9", C.C_ORIG)

    box(AW - 325, yBC, 300, H3, "Step 2  Geographical validation",
        "Independent centre (different hospital)\n"
        f"n = {len(g)}; obstructive "
        f"{int(g['label'].sum())} ({g['label'].mean()*100:.1f}%)\n"
        "Pipeline transferred frozen (no refitting)",
        "#e6f4ea", C.C_GEO)

    # ---- 行 3：Step 3 ----
    box(cx - 190, yD, 380, H3, "Step 3  Contemporary data accumulate",
        "Temporal cohort (same centre, later period)\n"
        f"n = {len(t)}; obstructive "
        f"{int(t['label'].sum())} ({t['label'].mean()*100:.1f}%)\n"
        "Merged in increments, re-tuned at every step",
        "#fdf0e3", C.C_TEMP)

    # ---- 行 4：结论 ----
    box(cx - 300, yE, 600, H5, "Maturation of external validity",
        "Re-evaluated on the same geographical cohort, and on\n"
        "a contemporary hold-out never used for fitting\n"
        "Always compared with a row-count-matched control arm\n"
        "-> calibration repaired early and could overshoot;\n"
        "discrimination still rising when the data ran out",
        "#f1eaf7", C.C_PUR)

    # ---- 箭头：A→B、B→C、C→D、D→E ----
    arrow(cx, yA, 175, yBC + H3)
    arrow(325, yBC + H3 / 2, AW - 325, yBC + H3 / 2)
    arrow(AW - 150, yBC, cx + 95, yD + H3)
    arrow(cx, yD, cx, yE + H5)

    ax.text(25, 8,
            "The geographical cohort is never used for fitting, tuning or "
            "threshold definition.",
            fontsize=FS_N, color="#666666", va="bottom", ha="left")
    C.save_figure(fig, path_out)


# =============================================================================
# Fig 2 判别力（单一模型：随机森林）
# =============================================================================
def fig_discrimination(curves, aucs, oof_mean, y, path_out):
    aucs = np.asarray(aucs, float)
    fig = plt.figure(figsize=(7.4, 8.6))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.30, 0.95, 1.05], hspace=0.62)

    # -- A 平均 ROC（50 折） -------------------------------------------------
    ax = fig.add_subplot(gs[0])
    gx, mu, sd, _ = C.mean_roc(curves)
    auc_mu = aucs.mean()
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    # 阴影 = ±1 SD（已在图注中说明，不再单列图例，避免图例过宽压住横轴刻度）
    ax.fill_between(gx, np.clip(mu - sd, 0, 1), np.clip(mu + sd, 0, 1),
                    color=C.C_RED, alpha=0.15, lw=0)
    ax.plot(gx, mu, color=C.C_RED, lw=1.9,
            label=f"RF  AUC {auc_mu:.3f}\n    fold 2.5-97.5 pct {lo:.3f}-{hi:.3f}")
    C.ref_line(ax, diag=True)
    ax.set_xlabel("1 - Specificity"); ax.set_ylabel("Sensitivity")
    ax.set_title("A  Mean ROC of 10-fold cross-validation "
                 f"({C.CV_REPEAT} repeats, {len(aucs)} folds)", loc="left",
                 fontweight="bold")
    C.style_legend(ax, loc="lower right", fontsize=8.4, handlelength=1.6)
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01); ax.set_aspect("equal")

    # -- B 逐折 AUC 分布 ----------------------------------------------------
    ax = fig.add_subplot(gs[1])
    ax.hist(aucs, bins=11, color=C.C_ORIG, alpha=0.80, edgecolor="white",
            linewidth=0.6)
    ax.axvline(auc_mu, color=C.C_RED, lw=1.5,
               label=f"Mean {auc_mu:.3f}")
    ax.axvline(np.median(aucs), color=C.C_GRAY, lw=1.3, ls="--",
               label=f"Median {np.median(aucs):.3f}")
    rng = np.random.RandomState(0)
    ax.plot(aucs, np.full(len(aucs), -0.9) + rng.normal(0, 0.35, len(aucs)),
            "o", ms=2.6, color=C.C_RED, alpha=0.55)
    ax.set_ylim(-3.0, ax.get_ylim()[1])
    ax.set_yticks([t for t in ax.get_yticks() if t >= 0])
    ax.set_xlabel("AUC per fold")
    ax.set_ylabel("Number of folds")
    ax.set_title("B  Fold-to-fold variability "
                 f"(SD {aucs.std(ddof=1):.3f})", loc="left", fontweight="bold")
    C.style_legend(ax, loc="upper left", fontsize=9)

    # -- C 袋外校准 ---------------------------------------------------------
    ax = fig.add_subplot(gs[2])
    cc = C.calibration_curve_bins(y, oof_mean, n_bins=8)
    ax.errorbar(cc["pred"], cc["obs"], yerr=1.96 * cc["se"], fmt="o-",
                color=C.C_PUR, lw=1.6, ms=5, capsize=3, ecolor="#666666",
                label="Cross-validated out-of-fold")
    C.ref_line(ax, diag=True, label="Ideal")
    cm = C.calibration_metrics(y, oof_mean)
    C.stat_box(ax,
               f"Calibration slope = {cm['Calibration slope']:.2f}\n"
               f"Intercept (in the large) = {cm['Calibration intercept']:+.2f}\n"
               f"Brier score = {cm['Brier']:.3f}\n"
               f"ICI = {cm['ICI']:.3f}\n"
               f"O:E = {cm['O:E']:.2f}",
               loc="lower right", fontsize=8.5)
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed proportion")
    ax.set_title("C  Calibration of the cross-validated predictions "
                 f"(threshold {C.youden_threshold(y, oof_mean):.3f})",
                 loc="left", fontweight="bold")
    C.style_legend(ax, loc="upper left", fontsize=9)
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.05, 1.05)

    C.save_figure(fig, path_out)


# =============================================================================
# Fig 3 特征重要性
# =============================================================================
def fig_importance(mdi_df, perm_df, path_out):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.6))

    for ax, df, tag, title, col in [
            (axes[0], mdi_df, "A", "A  Mean decrease in impurity", C.C_ORIG),
            (axes[1], perm_df, "B", "B  Permutation importance (test folds)", C.C_RED)]:
        d = df.sort_values("mean")
        y = np.arange(len(d))
        # 误差棒 = 跨折均值的 95% CI（与图注一致；lo/hi 即 mean ± 1.96·SD/√折数）
        err = np.nan_to_num(np.vstack([d["mean"] - d["lo"],
                                       d["hi"] - d["mean"]]), nan=0.0)
        ax.barh(y, d["mean"], xerr=err, color=col, alpha=0.85,
                error_kw=dict(lw=0.9, capsize=2, ecolor="#555555"))
        ax.set_yticks(y)
        ax.set_yticklabels([C.LABEL_EN[v] for v in d["var"]], fontsize=9)
        for yi, (v, lo, hi) in enumerate(zip(d["mean"], d["lo"], d["hi"])):
            if not np.isnan(hi):
                ax.text(hi + d["mean"].max() * 0.03, yi,
                        f"{v:.3f}", va="center", fontsize=8, color="#333333")
        ax.set_xlabel("Importance" if tag == "A" else "Decrease in AUC")
        ax.set_title(title, loc="left", fontweight="bold")
        lo_x = min(0.0, float(np.nanmin(d["lo"])))
        hi_x = float(np.nanmax(d["hi"]))
        ax.set_xlim(lo_x - 0.02 * (hi_x - lo_x), hi_x + 0.24 * (hi_x - lo_x))
        if lo_x < 0:
            ax.axvline(0, color="#999999", lw=0.8, ls=":")

    fig.subplots_adjust(wspace=0.72)
    C.save_figure(fig, path_out)


# =============================================================================
# Fig 4 SHAP
# =============================================================================
def fig_shap(sv, X, path_out):
    order = np.argsort(-np.abs(sv).mean(axis=0))
    top = order[:15][::-1]
    names = [C.LABEL_EN[C.FEATURES[j]] for j in top]

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 5.2),
                             gridspec_kw=dict(width_ratios=[1.45, 1.0], wspace=0.55))

    # A beeswarm
    ax = axes[0]
    cmap = plt.get_cmap("coolwarm")
    for row, j in enumerate(top):
        v = sv[:, j]
        xj = X.iloc[:, j].values.astype(float)
        rng = np.nanmax(xj) - np.nanmin(xj)
        cval = (xj - np.nanmin(xj)) / rng if rng > 0 else np.zeros_like(xj)
        cval = np.nan_to_num(cval, nan=0.5)
        ax.scatter(v, row + swarm_y(v), c=cval, cmap=cmap, vmin=0, vmax=1,
                   s=7, alpha=0.85, linewidths=0)
    ax.axvline(0, color="#888888", lw=0.9)
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("SHAP value (impact on model output)")
    ax.set_title("A  SHAP summary", loc="left", fontweight="bold")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.045)
    cb.set_ticks([0, 1]); cb.set_ticklabels(["low", "high"])
    cb.ax.tick_params(labelsize=8)

    # B mean |SHAP|
    ax = axes[1]
    d = pd.DataFrame({"var": [C.FEATURES[j] for j in top],
                      "m": np.abs(sv).mean(axis=0)[top]})
    y = np.arange(len(d))
    ax.barh(y, d["m"], color=C.C_PUR, alpha=0.9)
    ax.set_yticks(y); ax.set_yticklabels([""] * len(y))
    for yi, v in enumerate(d["m"]):
        ax.text(v + d["m"].max() * 0.02, yi, f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("B  Global importance", loc="left", fontweight="bold")
    ax.set_xlim(0, d["m"].max() * 1.45)

    C.save_figure(fig, path_out)
    return d


# =============================================================================
# Table 1 基线特征
# =============================================================================
def baseline_table(cohorts):
    from scipy.stats import mannwhitneyu, chi2_contingency, fisher_exact

    def cont_row(d, c):
        s = d[c].dropna().astype(float)
        return (f"{s.median():.2f} ({s.quantile(.25):.2f}-{s.quantile(.75):.2f})"
                + (f" [{int(s.isna().sum())} missing]" if s.isna().any() else ""))

    def bin_row(d, c):
        s = d[c].dropna().astype(float)
        return f"{int(s.sum())} ({s.mean()*100:.1f})"

    rows = []
    for c in C.FEATURES:
        r = {"Variable": C.LABEL_EN[c],
             "Type": "Continuous" if c in C.NUM else "Binary"}
        for k, d in cohorts.items():
            r[k] = cont_row(d, c) if c in C.NUM else bin_row(d, c)
        for k in ("temporal", "geo"):
            a = cohorts["original"]
            if c in C.NUM:
                sa, sb = a[c].dropna(), cohorts[k][c].dropna()
                r[f"p_vs_{k}"] = mannwhitneyu(sa, sb).pvalue
                r[f"SMD_vs_{k}"] = C.smd_cont(sa, sb)
            else:
                sa, sb = a[c].dropna(), cohorts[k][c].dropna()
                tab = np.array([[int(sa.sum()), int(len(sa) - sa.sum())],
                                [int(sb.sum()), int(len(sb) - sb.sum())]])
                if tab.min() < 5:
                    r[f"p_vs_{k}"] = fisher_exact(tab)[1]
                else:
                    r[f"p_vs_{k}"] = chi2_contingency(tab)[1]
                r[f"SMD_vs_{k}"] = C.smd_bin(sa.mean(), sb.mean())
        rows.append(r)

    # 结局行
    r = {"Variable": "Obstructive CAD", "Type": "Binary"}
    for k, d in cohorts.items():
        r[k] = f"{int(d['label'].sum())} ({d['label'].mean()*100:.1f})"
    a = cohorts["original"]
    for k in ("temporal", "geo"):
        sb = cohorts[k]["label"]
        sa = a["label"]
        tab = np.array([[int(sa.sum()), int(len(sa) - sa.sum())],
                        [int(sb.sum()), int(len(sb) - sb.sum())]])
        r[f"p_vs_{k}"] = chi2_contingency(tab)[1]
        r[f"SMD_vs_{k}"] = C.smd_bin(sa.mean(), sb.mean())
    rows.append(r)

    t = pd.DataFrame(rows)
    for k in ("temporal", "geo"):
        t[f"SMD_vs_{k}"] = t[f"SMD_vs_{k}"].astype(float).round(3)
        t[f"p_vs_{k}"] = t[f"p_vs_{k}"].astype(float).apply(
            lambda v: "<0.001" if v < 0.001 else f"{v:.3f}")
    return t


# =============================================================================
# main
# =============================================================================
def main():
    C.log_init("01_derivation")
    t0 = time.time()

    cohorts, marks = C.load_cohorts()
    if len(marks):
        C.save_table(marks, "TableS1_outlier_handling")

    # ---------------- Table 1 ----------------
    C.section("Table 1  三队列基线特征")
    tb1 = baseline_table(cohorts)
    C.save_table(tb1, "Table1_baseline_characteristics")
    log_cols = ["Variable", "original", "temporal", "geo"]
    C.log(tb1[log_cols].to_string(index=False))

    X, y = C.derivation_label(cohorts["original"])
    C.section(f"推导队列：n={len(X)}，事件={int(y.sum())}（{y.mean()*100:.1f}%）")

    rskf = RepeatedStratifiedKFold(n_splits=C.CV_FOLDS, n_repeats=C.CV_REPEAT,
                                   random_state=C.SEED)

    # ---------------- Step 1: RF 网格搜索 ----------------
    C.section("Step 1  随机森林：网格搜索（10 折 CV，只在推导集内部）")
    params_rows, roc_by_model, fold_auc = [], None, {}
    for name in C.MODEL_NAMES:
        ts = time.time()
        gs = C.tune_model(X, y, name, cv=C.CV_FOLDS, seed=C.SEED, n_jobs=-1)
        bp = {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}
        C.log(f"  {name:4s} best CV AUC = {gs.best_score_:.4f}   params = {bp}"
              f"   [{C.timer(ts)}]")
        params_rows.append({"model": name, "cv_auc_tuned": gs.best_score_,
                            "params": json.dumps(bp, default=str)})

    params_df = pd.DataFrame(params_rows)
    C.save_table(params_df, "TableS2_hyperparameters")

    # ---------------- Step 1b: 用最优超参做 10 折 CV × 5 ----------------
    C.section("Step 1b  固定最优超参，10 折 CV × 5 重复（50 折）")
    best_params = {r["model"]: json.loads(r["params"]) for _, r in params_df.iterrows()}
    names_by_model = C.MODEL_NAMES

    oof_by_repeat = {m: [] for m in names_by_model}     # 每次重复的袋外预测（整队列）
    mdi_by_model = {m: [] for m in names_by_model}
    perm_by_model = {m: [] for m in names_by_model}

    Xv = X.reset_index(drop=True)

    for m in names_by_model:
        ts = time.time()
        curves, aucs = [], []
        oof_rep = []
        for rep in range(C.CV_REPEAT):
            skf = StratifiedKFold(C.CV_FOLDS, shuffle=True,
                                  random_state=C.SEED + rep)
            oof = np.full(len(y), np.nan)
            for k, (tr, te) in enumerate(skf.split(Xv, y)):
                pipe = C.build_estimator(m, best_params[m], seed=C.SEED + rep * 10 + k)
                pipe.fit(Xv.iloc[tr], y[tr])
                p = pipe.predict_proba(Xv.iloc[te])[:, 1]
                oof[te] = p
                auc = roc_auc_score(y[te], p)
                aucs.append(auc)
                fpr, tpr, _ = roc_curve(y[te], p)
                curves.append((fpr, tpr))
                # MDI / 折外置换重要性
                clf = pipe.named_steps["clf"]
                if hasattr(clf, "feature_importances_"):
                    mdi_by_model[m].append(clf.feature_importances_)
                r = permutation_importance(pipe, Xv.iloc[te], y[te],
                                           n_repeats=3, random_state=C.SEED,
                                           scoring="roc_auc", n_jobs=1)
                perm_by_model[m].append(r.importances_mean)
            oof_rep.append(oof)
        roc_by_model = (curves, aucs)
        fold_auc[m] = aucs
        oof_by_repeat[m] = np.array(oof_rep)
        C.log(f"  {m:4s} AUC = {np.mean(aucs):.4f} ± {np.std(aucs, ddof=1):.4f} "
              f"(50 折)   pooled-OOF AUC = "
              f"{np.mean([roc_auc_score(y, o) for o in oof_rep]):.4f}   "
              f"[{C.timer(ts)}]")

    # ---------------- Table 2 ----------------
    C.section("Table 2  RF 的 CV 性能（含阈值指标，阈值取自推导集袋外）")
    t2_rows = []
    for m in names_by_model:
        v = np.array(fold_auc[m])
        oof_mean = np.mean(oof_by_repeat[m], axis=0)
        thr = C.youden_threshold(y, oof_mean)
        bm = C.binary_metrics(y, oof_mean, thr)
        cm = C.calibration_metrics(y, oof_mean)
        lo = np.percentile(v, 2.5); hi = np.percentile(v, 97.5)
        t2_rows.append({
            "Model": m, "Full name": C.MODEL_FULL[m],
            "AUC (50 folds)": f"{v.mean():.3f} ({v.std(ddof=1):.3f})",
            "AUC 2.5-97.5 pct": f"{lo:.3f}-{hi:.3f}",
            "AUC (pooled OOF)": f"{np.mean([roc_auc_score(y, o) for o in oof_by_repeat[m]]):.3f}",
            "Sensitivity": f"{bm['Sensitivity']:.3f}",
            "Specificity": f"{bm['Specificity']:.3f}",
            "PPV": f"{bm['PPV']:.3f}", "NPV": f"{bm['NPV']:.3f}",
            "Accuracy": f"{bm['Accuracy']:.3f}",
            "Brier": f"{cm['Brier']:.3f}",
            "Calibration slope": f"{cm['Calibration slope']:.3f}",
            "Calibration intercept": f"{cm['Calibration intercept']:.3f}",
            "ICI": f"{cm['ICI']:.3f}", "O:E": f"{cm['O:E']:.3f}",
            "Youden threshold": f"{thr:.3f}",
        })
    tb2 = pd.DataFrame(t2_rows)
    C.save_table(tb2, "Table2_derivation_model_performance")
    C.log(tb2[["Model", "AUC (50 folds)", "Sensitivity", "Specificity",
               "Accuracy"]].to_string(index=False))

    # ---------------- 主模型 = RF ----------------
    C.section("主模型 RF：在全部推导数据上拟合（供外部队列使用）")
    rf_final = C.build_estimator("RF", best_params["RF"] if "RF" in best_params
                                 else {"n_estimators": 500}, seed=C.SEED)
    rf_final.fit(Xv, y)
    oof_rf = np.mean(oof_by_repeat["RF"], axis=0)
    thr_rf = C.youden_threshold(y, oof_rf)
    C.save_pickle({"pipeline": rf_final, "features": C.FEATURES,
                   "threshold": thr_rf, "params": best_params.get("RF"),
                   "cohort": "original", "n": int(len(y))},
                  "m01_rf_original.pkl")
    C.save_json({"best_params": best_params, "threshold_rf_youden": thr_rf},
                "m01_spec.json")
    C.log(f"  RF 最终模型：阈值（推导集袋外 Youden）= {thr_rf:.4f}")

    # ---------------- Fig S1（原 Fig 2 内容，2026-09-20 迁为补充图）----------------
    # 正文 Figure 2 改为多队列 ROC，由 08_roc_shap_original.py 按旧 05 方法绘制。
    fig_discrimination(roc_by_model[0], roc_by_model[1], oof_rf, y,
                       "FigS1_derivation_discrimination")

    # ---------------- Fig 3 特征重要性 ----------------
    C.section("Fig 3  特征重要性")
    mdi_mat = np.array(mdi_by_model["RF"])
    mdi_df = pd.DataFrame({"var": C.FEATURES, "mean": mdi_mat.mean(axis=0),
                           "sd": mdi_mat.std(axis=0, ddof=1)})
    perm_mat = np.array(perm_by_model["RF"])
    # 置换重要性可能为负 → 截断到 0 仅用于显示区间下界
    perm_df = pd.DataFrame({"var": C.FEATURES, "mean": perm_mat.mean(axis=0),
                            "sd": perm_mat.std(axis=0, ddof=1)})
    perm_df["lo"] = perm_df["mean"] - 1.96 * perm_df["sd"] / np.sqrt(len(perm_mat))
    perm_df["hi"] = perm_df["mean"] + 1.96 * perm_df["sd"] / np.sqrt(len(perm_mat))
    mdi_df["lo"] = mdi_df["mean"] - 1.96 * mdi_df["sd"] / np.sqrt(len(mdi_mat))
    mdi_df["hi"] = mdi_df["mean"] + 1.96 * mdi_df["sd"] / np.sqrt(len(mdi_mat))
    mdi_df["rank"] = mdi_df["mean"].rank(ascending=False).astype(int)
    perm_df["rank"] = perm_df["mean"].rank(ascending=False).astype(int)
    fig_importance(mdi_df, perm_df, "Fig3_derivation_feature_importance")
    ti = mdi_df[["var", "mean", "sd", "rank"]].merge(
        perm_df[["var", "mean", "sd", "rank"]], on="var",
        suffixes=("_MDI", "_perm"))
    ti = ti.sort_values("rank_MDI")
    C.save_table(ti.round(4), "Table3_feature_importance")
    C.log(ti.head(10).round(4).to_string(index=False))

    # ---------------- Fig 4 SHAP ----------------
    C.section("Fig 4  SHAP（TreeExplainer on RF）")
    import shap
    explainer = shap.TreeExplainer(rf_final.named_steps["clf"])
    Ximp = rf_final.named_steps["imp"].transform(Xv)
    sv_raw = explainer.shap_values(Ximp)
    if isinstance(sv_raw, list):
        sv = np.asarray(sv_raw[1])
    elif np.ndim(sv_raw) == 3:
        sv = np.asarray(sv_raw)[:, :, 1]
    else:
        sv = np.asarray(sv_raw)
    C.log(f"  SHAP 矩阵 {sv.shape}")
    # ★ 出图职责已移交 08_roc_shap_original.py（按旧 05 plot_shap 方法出
    #   Fig4_derivation_shap / FigS2_shap_bar / FigS3_shap_dependence）。此处只保留
    #   与 Table 4 同源的 SHAP 重要性计算。
    imp_shap = pd.DataFrame({"var": C.FEATURES,
                             "mean_abs_shap": np.abs(sv).mean(axis=0)})
    imp_shap["rank"] = imp_shap["mean_abs_shap"].rank(ascending=False).astype(int)
    imp_shap = imp_shap.sort_values("rank")
    C.save_table(imp_shap.round(5), "Table4_shap_importance")
    C.log(imp_shap.round(4).to_string(index=False))

    # ---------------- Fig 1 流程 ----------------
    # 论文的主图 1（研究设计与受试者流程）由 34_figure1_design.py 单独绘制，
    # 输出 Figures/{pdf,tif,jpg}/Fig1_design.*。
    # 原示意图仍在本文件内（fig_study_flow），需要时取消下一行注释即可重出。
    # fig_study_flow(cohorts, len(X), "Fig1_study_flow")

    # ---------------- 汇总 ----------------
    summary = pd.DataFrame([{
        "cohort": "derivation (original)",
        "n": int(len(y)), "events": int(y.sum()),
        "model": "RF", "auc_cv50": float(np.mean(fold_auc["RF"])),
        "auc_cv50_sd": float(np.std(fold_auc["RF"], ddof=1)),
        "threshold_youden": float(thr_rf),
        "best_params": json.dumps(best_params.get("RF"), default=str),
    }])
    C.save_csv(summary, "s01_summary")
    C.save_csv(pd.DataFrame({m: fold_auc[m] for m in C.MODEL_NAMES}),
               "s01_fold_auc")
    C.save_csv(pd.DataFrame({"y": y, "oof_rf": oof_rf}), "s01_oof_rf")

    C.section(f"脚本 01 完成，总耗时 {C.timer(t0)}")


if __name__ == "__main__":
    main()
