# -*- coding: utf-8 -*-
"""
02 —— 地理外部队列验证
=============================================================================
训练集：original 442 例（脚本 01 得到的 RF，冻结全部超参与预处理）
验证集：Geographical external validation cohort（177 例，另一家医院）
模型：只建立随机森林（RF）一个模型（用户指令 2026-09-20）
★ 验证集绝不参与任何拟合/调参/选阈值
输出：
    Fig5  外部验证·判别力（分类指标条 / 五指标雷达）
          ※ 地理队列 ROC 已并入论文 Figure 2 的多队列 ROC（08 脚本按旧 05 方法绘制）
    Fig6  外部验证·校准度（校准曲线 / 校准指标 / 决策曲线 DCA）
    Table5  RF 在地理集上的外部验证性能
    TableS3 RF 在地理集上的混淆矩阵
=============================================================================
"""
import os
import sys
import json
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

FIVE = ["Sensitivity", "Specificity", "PPV", "NPV", "Accuracy"]


# =============================================================================
def paneltag(ax, tag, dx=-0.14, dy=1.07):
    ax.text(dx, dy, tag, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top", ha="left")


def fig_discrimination_geo(pg, yg, thr, params_best, path_out):
    """地理队列：分类指标（A）与五指标画像（B）。

    2026-09-20 变更：原 A 栏「地理队列 ROC」已由论文 Figure 2 的
    多队列 ROC（08_roc_shap_original.py，按旧 05 方法绘制）承担，
    本图相应改为 1 x 2。
    """
    fig = plt.figure(figsize=(7.4, 4.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.12, 1.0], wspace=0.32)

    # ---- A 五星图（条形 + Bootstrap CI）
    ax = fig.add_subplot(gs[0, 0])
    bm = C.binary_metrics(yg, pg, thr)
    yg_a, pg_a = np.asarray(yg), np.asarray(pg)
    vals, los, his = [], [], []
    for k in FIVE:
        v = bm[k]
        fn = lambda a, b, k=k: C.binary_metrics(a, b, thr)[k]
        l, h = C.boot_ci(yg_a, pg_a, fn, n_iter=2000)
        vals.append(v); los.append(l); his.append(h)
    x = np.arange(len(FIVE))
    ax.bar(x, vals, color=C.C_GEO, alpha=0.85, width=0.6)
    ax.errorbar(x, vals, yerr=[np.array(vals) - np.array(los),
                               np.array(his) - np.array(vals)],
                fmt="none", ecolor="#333333", lw=1.0, capsize=3)
    for xi, v in zip(x, vals):
        ax.text(xi, his[xi] + 0.045, f"{v:.2f}", ha="center", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["Sens.", "Spec.", "PPV", "NPV", "Accuracy"], fontsize=9)
    ax.set_ylim(0, 1.15); ax.set_ylabel("Value")
    ax.set_title(f"A  Classification metrics at threshold {thr:.3f}",
                 loc="left", fontweight="bold")
    C.style_axes(ax)

    # ---- B 雷达
    ax = fig.add_subplot(gs[0, 1], projection="polar")
    ang = np.linspace(0, 2 * np.pi, len(FIVE), endpoint=False).tolist()
    ang += ang[:1]
    vv = vals + vals[:1]
    vv_ref = [params_best[k] for k in FIVE] + [params_best[FIVE[0]]]
    ax.plot(ang, vv, color=C.C_GEO, lw=1.6, label="Geographical")
    ax.fill(ang, vv, color=C.C_GEO, alpha=0.18)
    ax.plot(ang, vv_ref, color=C.C_GRAY, lw=1.2, ls="--", label="Derivation (CV)")
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(["Sens.", "Spec.", "PPV", "NPV", "Acc."], fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8)
    ax.set_title("B  Five-metric profile", loc="left", fontweight="bold", pad=18)
    ax.grid(alpha=0.25, lw=0.6)
    C.style_legend(ax, loc="lower center", bbox_to_anchor=(0.5, -0.30),
                   fontsize=8.5, ncol=2)

    C.save_figure(fig, path_out)


def fig_calibration_geo(yg, pg, path_out):
    fig = plt.figure(figsize=(7.4, 4.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.35)

    # ---- A 校准曲线
    ax = fig.add_subplot(gs[0, 0])
    cc = C.calibration_curve_bins(yg, pg, n_bins=8)
    ax.errorbar(cc["pred"], cc["obs"], yerr=1.96 * cc["se"], fmt="o-",
                color=C.C_GEO, lw=1.6, ms=5, capsize=3, ecolor="#666666",
                label="Geographical cohort")
    C.ref_line(ax, diag=True, label="Ideal")
    cm = C.calibration_metrics(yg, pg)
    txt = (f"Calibration slope = {cm['Calibration slope']:.2f}\n"
           f"Calibration intercept = {cm['Calibration intercept']:.2f}\n"
           f"Brier score = {cm['Brier']:.3f}\n"
           f"ICI = {cm['ICI']:.3f}\n"
           f"O:E = {cm['O:E']:.2f}")
    C.stat_box(ax, txt, loc="upper left", fontsize=8.5)
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed proportion")
    ax.set_title("A  Calibration in the geographical cohort", loc="left",
                 fontweight="bold")
    C.style_legend(ax, loc="lower right", fontsize=9)
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.05, 1.05)

    # ---- B DCA
    ax = fig.add_subplot(gs[0, 1])
    d = C.dca(yg, pg)
    prev = np.mean(yg)
    ax.plot(d["threshold"], d["model"], color=C.C_GEO, lw=1.8, label="RF model")
    ax.plot(d["threshold"], d["treat_all"], color=C.C_GRAY, lw=1.2, ls="--",
            label="Treat all")
    C.ref_line(ax, y=0, color="#bbbbbb", ls=":", label="Treat none")
    ax.set_xlabel("Threshold probability"); ax.set_ylabel("Net benefit")
    ax.set_title("B  Decision curve analysis", loc="left",
                 fontweight="bold")
    C.style_legend(ax, loc="upper right", fontsize=9)
    ax.set_xlim(0, 0.7)
    ax.set_ylim(-0.10, max(0.35, prev + 0.1))
    C.save_figure(fig, path_out)


def main():
    C.log_init("02_geographical")
    t0 = time.time()

    cohorts, _ = C.load_cohorts()
    Xd, yd = C.derivation_label(cohorts["original"])
    Xg, yg = C.derivation_label(cohorts["geo"])

    spec = json.load(open(os.path.join(C.MODEL_DIR, "m01_spec.json"),
                          encoding="utf-8"))
    best_params = spec["best_params"]
    thr = spec["threshold_rf_youden"]
    C.section(f"外部验证：训练集 n={len(yd)}（original）→ 验证集 n={len(yg)}（geo）")
    C.log(f"  阈值沿用推导集袋外 Youden = {thr:.4f}（验证集不参与定阈值）")

    # ---------------- RF 在地理集上的表现 ----------------
    C.section("随机森林在地理集上的外部验证")
    rows = []
    pg_by_model = {}
    for m in C.MODEL_NAMES:
        pipe = C.build_estimator(m, best_params[m], seed=C.SEED)
        pipe.fit(Xd, yd)
        p = pipe.predict_proba(Xg)[:, 1]
        pg_by_model[m] = p
        a = roc_auc_score(yg, p)
        lo, hi = C.boot_ci(yg, p, lambda u, v: roc_auc_score(u, v), n_iter=2000)
        bm = C.binary_metrics(yg, p, thr)
        cm = C.calibration_metrics(yg, p)
        rows.append({
            "Model": m, "Full name": C.MODEL_FULL[m],
            "AUC": round(a, 3), "AUC 95% CI": f"{lo:.3f}-{hi:.3f}",
            "Sensitivity": round(bm["Sensitivity"], 3),
            "Specificity": round(bm["Specificity"], 3),
            "PPV": round(bm["PPV"], 3), "NPV": round(bm["NPV"], 3),
            "Accuracy": round(bm["Accuracy"], 3),
            "Brier": round(cm["Brier"], 3),
            "Calibration slope": round(cm["Calibration slope"], 3),
            "Calibration intercept": round(cm["Calibration intercept"], 3),
            "ICI": round(cm["ICI"], 3), "O:E": round(cm["O:E"], 3),
        })
        C.log(f"  {m:4s} AUC {a:.3f} ({lo:.3f}-{hi:.3f})  Sens {bm['Sensitivity']:.3f} "
              f" Spec {bm['Specificity']:.3f}  slope {cm['Calibration slope']:.2f}  "
              f"ICI {cm['ICI']:.3f}  O:E {cm['O:E']:.2f}")
    tb5 = pd.DataFrame(rows)
    C.save_table(tb5, "Table5_geographical_external_validation")

    pg = pg_by_model["RF"]
    cm_rf = C.calibration_metrics(yg, pg)
    bm_rf = C.binary_metrics(yg, pg, thr)
    auc_rf = roc_auc_score(yg, pg)

    # 推导集参照（用于雷达图与图题）
    oof = pd.read_csv(os.path.join(C.OUT_DIR, "s01_oof_rf.csv"))
    bm_dev = C.binary_metrics(oof["y"].values, oof["oof_rf"].values,
                              C.youden_threshold(oof["y"].values, oof["oof_rf"].values))

    # ---------------- Fig 5 / Fig 6 ----------------
    fig_discrimination_geo(pg, yg, thr, bm_dev,
                           "Fig5_geographical_discrimination")
    fig_calibration_geo(yg, pg, "FigS4_geographical_calibration")

    # ---------------- TableS3 混淆矩阵 ----------------
    cmat = pd.DataFrame(
        [[bm_rf["TP"], bm_rf["FN"]], [bm_rf["FP"], bm_rf["TN"]]],
        index=["Observed obstructive", "Observed non-obstructive"],
        columns=["Classified obstructive", "Classified non-obstructive"])
    C.save_table(cmat.reset_index().rename(columns={"index": ""}),
                 "TableS3_geo_confusion_matrix")

    # ---------------- 训练集内部参照 ----------------
    C.section("训练集（推导集）内部 10 折 CV 参照")
    fa = pd.read_csv(os.path.join(C.OUT_DIR, "s01_fold_auc.csv"))
    C.log(f"  RF CV AUC = {fa['RF'].mean():.3f} ± {fa['RF'].std(ddof=1):.3f}")
    C.log(f"  ΔAUC (geo - derivation CV) = {auc_rf - fa['RF'].mean():+.3f}")

    # ---------------- 汇总 ----------------
    a_lo, a_hi = C.boot_ci(yg, pg, lambda a, b: roc_auc_score(a, b), n_iter=2000)
    s = pd.DataFrame([{
        "train_cohort": "original", "train_n": len(yd),
        "test_cohort": "geo", "test_n": len(yg),
        "auc": auc_rf, "auc_lo": a_lo, "auc_hi": a_hi,
        "threshold": thr, **{k: bm_rf[k] for k in FIVE},
        "brier": cm_rf["Brier"], "slope": cm_rf["Calibration slope"],
        "intercept": cm_rf["Calibration intercept"], "ici": cm_rf["ICI"],
        "oe": cm_rf["O:E"],
        "derivation_cv_auc": float(fa["RF"].mean()),
    }])
    C.save_csv(s, "s02_geo_summary")
    C.save_csv(pd.DataFrame({"y": yg, "p_rf": pg}), "s02_geo_pred_rf")

    C.section(f"脚本 02 完成，总耗时 {C.timer(t0)}")


if __name__ == "__main__":
    main()
