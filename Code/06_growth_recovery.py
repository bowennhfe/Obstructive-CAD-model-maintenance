# -*- coding: utf-8 -*-
"""
06 —— 时间队列扩充的派生分析（不改动任何建模结果，只对 01-03 的产出做二次计算）
=============================================================================
产出
    Table9  内部 vs 外部性能随推导队列成熟度的分离轨迹（含阈值与操作点）
    Table10 各指标的「更新预算」：冻结值 / 最优步 / 残余差距 / 恢复比例
    TableS7 信息增益的剂量-反应（时间臂相对同样本量对照臂的净优势 vs 新增例数）
    TableS8 决策曲线轨迹（净获益、预测阳性率随扩充的变化）
    Fig11  (A) 时间臂内部-外部分离 (B) 对照臂同样分离缺失 (C) 剂量-反应
=============================================================================
设计要点
    - 所有数字来自 Table6/7/S4/8 与 s03_growth_full.pkl，不重新拟合模型
    - 地理队列自始至终只被评价
=============================================================================
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

PCTS = [0.0, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00]

# 展示名, Table7 列名, 理想目标, 目标口径
METRICS = [
    ("AUC",       "auc",       1.000, "internal"),
    ("slope",     "slope",     1.000, "nominal"),
    ("intercept", "intercept", 0.000, "nominal"),
    ("O:E",       "oe",        1.000, "nominal"),
    ("Brier",     "brier",     0.000, "nominal"),
    ("ICI",       "ici",       0.000, "nominal"),
]

METRIC_LABEL = {
    "AUC": "Discrimination (AUC)",
    "slope": "Calibration slope",
    "intercept": "Calibration intercept",
    "O:E": "Observed:expected ratio",
    "Brier": "Brier score",
    "ICI": "Integrated calibration index",
}


def goodness(key, v):
    """单调变换：数值越大越好。用于把不同量纲的指标放到同一杆秤上。"""
    if key == "AUC":
        return v
    if key in ("slope", "O:E"):
        return -abs(v - 1.0)
    return -abs(v)                      # intercept / Brier / ICI 理想为 0


def T(name):
    return pd.read_csv(os.path.join(C.TABLE_DIR, f"{name}.csv"))


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-8, 1 - 1e-8)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _fit_intercept(y, lo, max_iter=120):
    a = 0.0
    for _ in range(max_iter):
        mu = _sigmoid(a + lo)
        w = np.clip(mu * (1 - mu), 1e-9, None)
        h = -np.sum(w)
        step = np.sum(y - mu) / h if h != 0 else 0.0
        a_new = a - step
        if abs(a_new - a) < 1e-11:
            return a_new
        a = a_new
    return a


def crossfit_intercept(y, p, k=5, seed=C.SEED):
    """交叉拟合（仅截距）再校准，返回袋外重校准概率。与脚本 16 同一口径。"""
    from sklearn.model_selection import StratifiedKFold
    y = np.asarray(y, int)
    lo = _logit(p)
    out = np.full(len(y), np.nan)
    skf = StratifiedKFold(k, shuffle=True, random_state=seed)
    for tr, te in skf.split(lo.reshape(-1, 1), y):
        a = _fit_intercept(y[tr], lo[tr])
        out[te] = _sigmoid(a + lo[te])
    return out


def main():
    C.log_init("06_growth_recovery")
    t0 = C.timer()

    g = pd.read_pickle(os.path.join(C.OUT_DIR, "s03_growth_full.pkl"))
    t7 = T("Table7_growth_summary")
    ts4 = T("TableS4_growth_temporal_vs_control")
    t8 = T("Table8_growth_endpoint_tests")
    t2 = T("Table2_derivation_model_performance")

    dt = t7[t7["arm"] == "temporal"].sort_values("pct_temporal").reset_index(drop=True)
    dc = t7[t7["arm"] == "control"].sort_values("pct_temporal").reset_index(drop=True)

    # 内部 CV AUC（每臂、每步，重复实验间均值）
    cv = (g.groupby(["arm", "pct_temporal"])["cv_auc"].agg(["mean", "std"])
          .reset_index().rename(columns={"mean": "internal_auc", "std": "internal_auc_sd"}))
    cvt = cv[cv["arm"] == "temporal"].set_index("pct_temporal")
    cvc = cv[cv["arm"] == "control"].set_index("pct_temporal")

    # 新增的时间队列例数（时间臂）
    n_add = (g[g["arm"] == "temporal"].groupby("pct_temporal")["n_temporal"]
             .mean().round().astype(int))

    auc_dev = float(str(t2[t2["Model"] == "RF"]["AUC (50 folds)"].iloc[0]).split()[0])

    # 对照臂的内部交叉验证 AUC 与专用对照臂分析（16b → Table S15）同源：
    # 同一个量不应在两张表里出现两个值。
    _s16_path = os.path.join(C.OUT_DIR, "s16_control_arm.csv")
    if not os.path.exists(_s16_path):
        raise FileNotFoundError(
            "Table 9 的对照臂内部 AUC 依赖 Outputs/s16_control_arm.csv；"
            "请先运行 16b_control_arm.py")
    _s16all = pd.read_csv(_s16_path)
    _s16n = (_s16all.query("arm == 'control_naive'")
             .groupby("pct")["cv_auc"].mean())
    ctl_cv_naive = {round(float(k), 2): float(v) for k, v in _s16n.items()}
    # ★ 第十二轮 Major 10(vi)：对照臂内部 AUC 的主口径改为「按患者分组」的交叉验证。
    #   朴素分层划分会把同一患者的重采样副本分到折的两侧，产生 0.953→0.987 的伪上升。
    _s16g = (_s16all.query("arm == 'control_grouped'")
             .groupby("pct")["cv_auc"].mean())
    ctl_cv_grouped = {round(float(k), 2): float(v) for k, v in _s16g.items()}

    # =========================================================================
    # Table9 —— 内部 vs 外部
    # =========================================================================
    C.section("Table9 内部 vs 外部性能随推导队列成熟度")
    rows = []
    for _, r in dt.iterrows():
        p = float(r["pct_temporal"])
        pc = dc[dc["pct_temporal"] == p].iloc[0]
        rt = cvt.loc[p]
        rc = cvc.loc[p]
        rows.append({
            "pct_temporal": p,
            "n_derive": int(r["n_derive"]),
            "n_contemporary_added": int(n_add.loc[p]),
            "sensitivity": round(float(r["sens"]), 3),
            "specificity": round(float(r["spec"]), 3),
            "internal_auc": round(float(rt["internal_auc"]), 4),
            "internal_auc_sd": round(float(rt["internal_auc_sd"]), 4),
            "external_auc": round(float(r["auc"]), 4),
            "external_auc_sd": round(float(r["auc_sd"]), 4),
            "internal_minus_external": round(float(rt["internal_auc"] - r["auc"]), 4),
            "control_n_derive": int(pc["n_derive"]),
            "control_internal_auc": round(ctl_cv_grouped[round(p, 2)], 4),
            "control_internal_auc_naive": round(ctl_cv_naive[round(p, 2)], 4),
            "control_external_auc": round(float(pc["auc"]), 4),
        })
    t9 = pd.DataFrame(rows)
    # 阈值列（geo 阈值不存于 summary，取 detail 的 thr 均值）
    t9["youden_threshold"] = [round(float(
        g[(g["arm"] == "temporal") & (g["pct_temporal"] == p)]["thr"].mean()), 3)
        for p in t9["pct_temporal"]]
    C.save_table(t9, "Table9_internal_vs_external")
    C.log(t9.to_string(index=False))

    # =========================================================================
    # Table10 —— 更新预算：恢复度与饱和点
    # =========================================================================
    C.section("Table10 各指标的更新预算（恢复度 / 最优步 / 残余差距）")
    rows = []
    for key, col, target, basis in METRICS:
        tgt = auc_dev if basis == "internal" else target
        vals = dt[col].astype(float).values
        gaps = np.abs(vals - tgt)
        i_best = int(np.argmin(gaps))
        rows.append({
            "metric": key,
            "metric_label": METRIC_LABEL[key],
            "target": round(tgt, 3),
            "target_basis": ("internal cross-validated AUC of the "
                             "development cohort" if basis == "internal"
                             else "nominal ideal value"),
            "value_frozen": round(float(vals[0]), 3),
            "gap_frozen": round(float(gaps[0]), 3),
            "value_best": round(float(vals[i_best]), 3),
            "pct_best": float(dt["pct_temporal"].iloc[i_best]),
            "n_best": int(dt["n_derive"].iloc[i_best]),
            "added_best": int(n_add.iloc[i_best]),
            "gap_best": round(float(gaps[i_best]), 3),
            "value_final": round(float(vals[-1]), 3),
            "gap_final": round(float(gaps[-1]), 3),
            "recovery_best_pct": round(float((gaps[0] - gaps[i_best]) / gaps[0] * 100), 1),
            "recovery_final_pct": round(float((gaps[0] - gaps[-1]) / gaps[0] * 100), 1),
        })
    t10 = pd.DataFrame(rows)
    C.save_table(t10, "Table10_growth_recovery")
    C.log(t10.to_string(index=False))

    # =========================================================================
    # TableS7 —— 信息增益的剂量-反应
    # =========================================================================
    C.section("TableS7 时间臂相对同样本量对照臂的净优势 vs 新增例数")
    rows = []
    for key, col, _, _ in METRICS:
        adv = goodness(key, dt[col].astype(float).values) - \
            goodness(key, dc[col].astype(float).values)     # >0 表示时间臂更好
        n_x = n_add.values.astype(float)
        rho, pval = stats.spearmanr(n_x, adv)
        for i, p in enumerate(PCTS):
            rows.append({
                "metric": key,
                "metric_label": METRIC_LABEL[key],
                "pct_temporal": p,
                "n_contemporary_added": int(n_x[i]),
                "advantage_over_control": round(float(adv[i]), 4),
                "spearman_rho": round(float(rho), 3),
                "p_value": float(f"{pval:.4g}"),
                "n_steps": len(n_x),
            })
    ts7 = pd.DataFrame(rows)
    C.save_table(ts7, "TableS7_growth_dose_response")
    C.log(ts7[ts7["pct_temporal"] == 1.0][
        ["metric", "advantage_over_control", "spearman_rho", "p_value"]].to_string(index=False))

    # =========================================================================
    # TableS8 —— 决策曲线轨迹
    # =========================================================================
    C.section("TableS8 决策曲线轨迹（净获益 / 预测阳性率）")
    ths = np.round(np.arange(0.15, 0.56, 0.05), 2)

    def nb_mean(arm, pct, recal=False):
        sub = g[(g["arm"] == arm) & (np.isclose(g["pct_temporal"], pct))]
        mats = []
        for _, r in sub.iterrows():
            y = np.asarray(r["y_geo"], int)
            p = np.asarray(r["p_geo"], float)
            if recal:
                # 本地交叉拟合（仅截距）再校准，逐次实验袋外，不泄露
                p = crossfit_intercept(y, p)
            d = C.dca(y, p, thresholds=ths)
            d["predpos"] = (p[None, :] >= ths[:, None]).mean(axis=1)
            mats.append(d)
        m = pd.concat(mats).groupby("threshold", as_index=False).mean(numeric_only=True)
        return m.set_index("threshold")

    fz = nb_mean("temporal", 0.0)
    rz = nb_mean("temporal", 0.0, recal=True)
    up = nb_mean("temporal", 1.00)
    ct = nb_mean("control", 1.00)
    ts8 = pd.DataFrame({
        "threshold": ths,
        "net_benefit_frozen": fz["model"].round(4).values,
        "net_benefit_recal_intercept": rz["model"].round(4).values,
        "net_benefit_updated": up["model"].round(4).values,
        "net_benefit_control": ct["model"].round(4).values,
        "net_benefit_treat_all": fz["treat_all"].round(4).values,
        "net_benefit_treat_none": fz["treat_none"].round(4).values,
        "predicted_positive_rate_frozen": fz["predpos"].round(4).values,
        "predicted_positive_rate_recal_intercept": rz["predpos"].round(4).values,
        "predicted_positive_rate_updated": up["predpos"].round(4).values,
    })
    C.save_table(ts8, "TableS8_decision_curve")
    C.log(ts8.to_string(index=False))

    # =========================================================================
    # Fig11
    # =========================================================================
    C.section("Fig11 内部-外部分离 与 剂量-反应")
    fig = plt.figure(figsize=(7.4, 5.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0], hspace=0.58, wspace=0.34)

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, :])

    x = t9["n_derive"].values

    for ax, arm_key, title in ((axA, "temporal", "A  Contemporary data added"),
                               (axB, "control", "B  Row-count-matched resampling only")):
        if arm_key == "temporal":
            xi, internal, external = x, t9["internal_auc"], t9["external_auc"]
            ax.plot(xi, internal, "-o", color=C.C_ORIG, lw=1.8, ms=4.5,
                    label="Internal (cross-validated) AUC")
        else:
            xi = t9["control_n_derive"].values
            internal = t9["control_internal_auc"]
            external = t9["control_external_auc"]
            # 朴素分层划分（副本跨折）作为伪影对照，主口径为按患者分组的折
            ax.plot(xi, t9["control_internal_auc_naive"], "--s",
                    color="#999999", lw=1.5, ms=4.0,
                    label="Internal AUC, plain split (artefact)")
            ax.plot(xi, internal, "-o", color=C.C_ORIG, lw=1.8, ms=4.5,
                    label="Internal AUC, patient-grouped folds")
        ax.plot(xi, external, "-s", color=C.C_GEO, lw=1.8, ms=4.5,
                label="External (geographical) AUC")
        # 内部-外部差距（首末两步）
        for k, ha in ((0, "right"), (-1, "left")):
            gap = internal.iloc[k] - external.iloc[k]
            if abs(gap) >= 0.05:
                ax.annotate("", xy=(xi[k], internal.iloc[k]),
                            xytext=(xi[k], external.iloc[k]),
                            arrowprops=dict(arrowstyle="<->", color=C.C_RED, lw=1.0,
                                            shrinkA=1, shrinkB=1))
            else:
                # 跨距过短时双箭头两头会叠成一个红叉，改画细竖线
                ax.plot([xi[k], xi[k]], [external.iloc[k], internal.iloc[k]],
                        color=C.C_RED, lw=1.2, zorder=3)
            # 标注置于箭头下端之下：箭头右端紧贴图例框，
            # 置于上端会与图例文字重叠（_audit_fig_collisions 曾报 27%）
            ax.text(xi[k], external.iloc[k] - 0.028,
                    f"gap {gap:.3f}", fontsize=8.5, color=C.C_RED,
                    ha="center", va="top")
        ax.set_xlabel("Derivation sample size (n)")
        ax.set_title(title, fontsize=10.5, loc="left", fontweight="bold")
        ax.set_ylim(0.63, 1.06)
        ax.grid(alpha=0.25, lw=0.6)
        if arm_key == "control":
            # 对照面板：图例下移到内部线与外部线之间的空白带，避免遮住曲线
            C.style_legend(ax, loc="center right", fontsize=8.0,
                           bbox_to_anchor=(1.0, 0.594))
        else:
            C.style_legend(ax, loc="center right", fontsize=8.0)
    axA.set_ylabel("AUC")

    # ---- C 剂量-反应：各指标净优势（相对各自最大优势） ----
    palette = {"AUC": C.C_ORIG, "slope": C.C_RED, "intercept": C.C_PUR,
               "O:E": C.C_GEO, "Brier": C.C_TEMP, "ICI": C.C_BROWN}
    for key, col, _, _ in METRICS:
        d = ts7[ts7["metric"] == key].sort_values("pct_temporal")
        adv = d["advantage_over_control"].values
        rel = adv / np.max(adv)
        axC.plot(d["n_contemporary_added"], rel, "-o", color=palette[key],
                 lw=1.6, ms=3.6,
                 label=f"{METRIC_LABEL[key]}  (\u03c1={d['spearman_rho'].iloc[0]:.2f})")
    C.ref_line(axC, y=1.0, lw=0.9, zorder=0)
    axC.set_xlabel("Contemporary patients added to the derivation cohort (n)")
    axC.set_ylabel("Advantage over row-count-matched control\n(fraction of each metric's maximum)")
    axC.set_ylim(-0.42, 1.30)
    axC.set_yticks([0.00, 0.25, 0.50, 0.75, 1.00, 1.25])
    axC.set_title("C  Gain scales with data added - except the slope",
                  fontsize=10.5, loc="left", fontweight="bold")
    C.style_legend(axC, loc="lower center", fontsize=8, ncol=3,
                   handlelength=1.5, columnspacing=1.0)
    axC.grid(alpha=0.25, lw=0.6)

    C.save_figure(fig, "FigS8_growth_internal_external")

    C.section(f"脚本 06 完成，总耗时 {C.timer(t0)}")


if __name__ == "__main__":
    main()
