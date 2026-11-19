# -*- coding: utf-8 -*-
"""
17 —— 样本量与估计精度的量化说明（回应审稿意见 11）
=============================================================================
(a) 开发队列：按 Riley 2020 (BMJ 368:m441) / Riley 2019 (Stat Med 38:1276)
    的二分类结局四步法计算最小样本量，其中 R²_CS_adj 由本研究自己的
    交叉验证袋外预测反推（而不是外部借用）。
(b) 外部验证队列：按 Riley 2021 (Stat Med 40:4230) 与 Snell 2021
    (J Clin Epidemiol 135:79) 的口径，报告本研究实际达到的估计精度
    （Bootstrap SE 与 95% CI）——AUC、O:E、校准斜率、校准截距 in the large——
    并给出达到常用目标精度所需的例数/事件数。
    同时给出外部队列的最小可检测 AUC 差异。

输出
    Tables/TableS29_development_sample_size.csv
    Tables/TableS30_external_validation_precision.csv
=============================================================================
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

Z = 1.959963985


# =============================================================================
# (a) 开发样本量：Riley 四步法（二分类）
# =============================================================================
def max_r2_cs(p):
    """二分类下 Cox-Snell R² 的上限 = 1 - exp(2 lnL_null / n)。"""
    lnl_null_per_obs = p * np.log(p) + (1 - p) * np.log(1 - p)
    return 1 - np.exp(2 * lnl_null_per_obs)


def n_shrinkage(Q, r2_cs_adj, S_VH):
    """Riley 判据：n = Q / ((S_VH - 1) * ln(1 - R2_CS_adj / S_VH))。"""
    denom = (S_VH - 1.0) * np.log(1.0 - r2_cs_adj / S_VH)
    return Q / denom


def cox_snell_r2(y, p):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    n = len(y)
    pbar = y.mean()
    lnl_m = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    lnl_0 = n * (pbar * np.log(pbar) + (1 - pbar) * np.log(1 - pbar))
    return 1 - np.exp(2 * (lnl_0 - lnl_m) / n)


def sec_dev(Xd, yd):
    C.section("(a) 开发队列最小样本量（Riley 四步法，二分类）")
    oof = pd.read_csv(os.path.join(C.OUT_DIR, "s01_oof_rf.csv"))
    col = [c for c in oof.columns if c.lower() not in ("y", "label", "index")][0]
    y = oof["y"].astype(int).values if "y" in oof.columns else yd
    p = oof[col].values
    r2_cs_oof = cox_snell_r2(y, p)
    pbar = float(np.mean(y))
    mx = max_r2_cs(pbar)
    r2_nag = r2_cs_oof / mx
    C.log(f"  袋外 Cox-Snell R² = {r2_cs_oof:.3f}；上限 = {mx:.3f}；"
          f"Nagelkerke R² = {r2_nag:.3f}")

    Q = len(C.FEATURES)
    rows = []

    def add(label, n_req, note=""):
        rows.append({"Criterion": label, "Required n": int(np.ceil(n_req)),
                     "Required events": int(np.ceil(n_req * pbar)),
                     "Note": note})

    # 判据 1：总体风险估计足够精确（SE(总体风险) 对应的半宽 0.05）
    add("Precise estimation of overall risk (half-width 0.05)",
        (Z / 0.05) ** 2 * pbar * (1 - pbar))
    # 判据 2：目标收缩因子 S_VH ≥ 0.9（过拟合 ≤10%）
    add("Shrinkage S >= 0.9 (using OOF R2_CS_adj)",
        n_shrinkage(Q, r2_cs_oof, 0.9),
        f"Q={Q}, R2_CS_adj={r2_cs_oof:.3f}")
    # 判据 3：R²_CS 的表观值与调整值之差不超过 5%（相对）
    add("Small optimism in R2_CS (<=5% relative)",
        n_shrinkage(Q, r2_cs_oof, 1 / 1.05), "S_VH = 1/1.05")
    # 敏感性：把 R²_CS_adj 设为不同 Nagelkerke 目标
    for nag in (0.20, 0.30, 0.40, 0.50):
        r2a = nag * mx
        add(f"Shrinkage S >= 0.9 (anticipated Nagelkerke R2 = {nag:.2f})",
            n_shrinkage(Q, r2a, 0.9), f"R2_CS_adj={r2a:.3f}")
    t = pd.DataFrame(rows)
    t["Met by 442 (208 events)"] = np.where(t["Required n"] <= 442, "yes", "no")
    C.save_table(t, "TableS29_development_sample_size")
    C.log(t.to_string(index=False))
    C.log(f"  事件/参数比 (EPP) = {int(yd.sum())}/{Q} = {yd.sum()/Q:.1f}")
    return t, r2_cs_oof, r2_nag


# =============================================================================
# (b) 外部队列的估计精度
# =============================================================================
def sec_ext(yg, p_rf, p_algo=None):
    C.section("(b) 外部队列（n=177，事件 63）实际达到的估计精度")
    y = np.asarray(yg, int)
    n_ev, n_ne = int(y.sum()), int((1 - y).sum())

    def boot_se(p, fn, n_iter=2000, seed=C.SEED):
        rng = np.random.RandomState(seed)
        v = []
        for _ in range(n_iter):
            idx = rng.choice(len(y), len(y), replace=True)
            if np.unique(y[idx]).size < 2:
                continue
            try:
                v.append(fn(y[idx], p[idx]))
            except Exception:
                pass
        v = np.asarray(v)
        return float(v.std(ddof=1)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

    from sklearn.linear_model import LogisticRegression

    def slope_of(u, vv):
        lo = np.log(np.clip(vv, 1e-8, 1 - 1e-8) / (1 - np.clip(vv, 1e-8, 1 - 1e-8)))
        m = LogisticRegression(penalty=None, max_iter=1000).fit(lo.reshape(-1, 1), u)
        return float(m.coef_[0][0])

    def oe_log(u, vv):
        return float(np.log(u.sum() / np.clip(vv, 1e-8, None).sum()))

    def intercept_of(u, vv):
        return float(C.calibration_metrics(u, vv)["Calibration intercept"])

    se_sl, sl_lo, sl_hi = boot_se(p_rf, slope_of)
    se_ln, ln_lo, ln_hi = boot_se(p_rf, oe_log)
    se_au, au_lo, au_hi = boot_se(p_rf, lambda u, vv: roc_auc_score(u, vv))
    se_ic, ic_lo, ic_hi = boot_se(p_rf, intercept_of)
    ic_pt = float(C.calibration_metrics(y, p_rf)["Calibration intercept"])
    oe = y.sum() / p_rf.sum()

    rows = [
        {"Quantity": "Events / non-events", "Estimate": f"{n_ev} / {n_ne}",
         "Bootstrap SE": "", "95% CI": "",
         "Riley 2021 target": ">=100 events and >=100 non-events (rule of thumb)",
         "Achieved?": "yes" if (n_ev >= 100 and n_ne >= 100) else "no (events)"},
        {"Quantity": "O:E ratio", "Estimate": round(float(oe), 3),
         "Bootstrap SE": "",
         "95% CI": f"{np.exp(ln_lo):.3f}-{np.exp(ln_hi):.3f}",
         "Riley 2021 target": "SE(ln O:E) <= 0.10",
         "Achieved?": "yes" if se_ln <= 0.10 else "no"},
        {"Quantity": "ln(O:E)", "Estimate": round(float(np.log(oe)), 3),
         "Bootstrap SE": round(se_ln, 3), "95% CI": f"{ln_lo:.3f}-{ln_hi:.3f}",
         "Riley 2021 target": "SE <= 0.10",
         "Achieved?": "yes" if se_ln <= 0.10 else "no"},
        {"Quantity": "Calibration slope", "Estimate": round(slope_of(y, p_rf), 3),
         "Bootstrap SE": round(se_sl, 3), "95% CI": f"{sl_lo:.3f}-{sl_hi:.3f}",
         "Riley 2021 target": "SE <= 0.10",
         "Achieved?": "yes" if se_sl <= 0.10 else "no"},
        {"Quantity": "AUC", "Estimate": round(roc_auc_score(y, p_rf), 3),
         "Bootstrap SE": round(se_au, 3), "95% CI": f"{au_lo:.3f}-{au_hi:.3f}",
         "Riley 2021 target": "95% CI half-width <= 0.05 (SE <= 0.0255)",
         "Achieved?": "yes" if se_au <= 0.0255 else "no"},
        {"Quantity": "Calibration intercept in the large",
         "Estimate": round(ic_pt, 3),
         "Bootstrap SE": round(se_ic, 3), "95% CI": f"{ic_lo:.3f}-{ic_hi:.3f}",
         "Riley 2021 target": "no published target; reported for orientation",
         "Achieved?": "n/a"},
    ]
    # 达到目标精度所需的例数/事件数（按 SE ∝ 1/sqrt(E) 外推）
    e_needed = {
        "O:E (SE of ln O:E = 0.10)": int(np.ceil(n_ev * (se_ln / 0.10) ** 2)),
        "Calibration slope (SE = 0.10)": int(np.ceil(n_ev * (se_sl / 0.10) ** 2)),
        "AUC (target SE = 0.0255)": int(np.ceil(n_ev * (se_au / 0.0255) ** 2)),
    }
    for k, v in e_needed.items():
        rows.append({"Quantity": f"Events required to reach the target: {k}",
                     "Estimate": v, "Bootstrap SE": "", "95% CI": "",
                     "Riley 2021 target": "",
                     "Achieved?": ("already met (63 events)" if v <= n_ev
                                   else f"needs {v / n_ev:.1f}x the present cohort")})
    t = pd.DataFrame(rows)
    C.save_table(t, "TableS30_external_validation_precision")
    C.log(t.to_string(index=False))

    # 8 族的外部 AUC 精度（供正文说明比较功效）
    if p_algo is not None:
        s = []
        for k, pp in p_algo.items():
            a = roc_auc_score(y, pp)
            se, lo, hi = boot_se(pp, lambda u, vv: roc_auc_score(u, vv), n_iter=1000)
            s.append({"Model": k, "AUC": round(a, 3), "SE": round(se, 4),
                      "95% CI": f"{lo:.3f}-{hi:.3f}", "CI half-width": round((hi - lo) / 2, 3)})
        C.save_csv(pd.DataFrame(s), "s17_algo_auc_precision")
        C.log(pd.DataFrame(s).to_string(index=False))
    return t


def main():
    C.log_init("17_sample_size")
    t0 = time.time()
    cohorts, _ = C.load_cohorts()
    Xd, yd = C.derivation_label(cohorts["original"])
    Xg, yg = C.derivation_label(cohorts["geo"])
    sec_dev(Xd, yd)
    fz = pd.read_csv(os.path.join(C.OUT_DIR, "s02_geo_pred_rf.csv"))
    p_algo = None
    fp = os.path.join(C.OUT_DIR, "s16_algo_geo_pred.csv")
    if os.path.exists(fp):
        d = pd.read_csv(fp)
        p_algo = {c: d[c].values for c in d.columns if c != "y"}
    sec_ext(yg, fz["p_rf"].values, p_algo)
    C.section(f"脚本 17 完成，总耗时 {C.timer(t0)}")


if __name__ == "__main__":
    main()
