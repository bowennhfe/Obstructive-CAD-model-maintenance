# -*- coding: utf-8 -*-
"""
35 —— 冻结模型的交叉拟合再校准（第二轮复核硬问题 1）
=============================================================================
问题
    旧 Figure 3 / Table S5 的 "before" 曲线取的是阶梯分析 temporal 臂 pct=1.00
    的预测（即**全量更新后**的模型），不是冻结模型；"after" 又是 fold-pooled
    口径，而正文 Table 3 的主口径是 per-fold。两者都对不上 Section 3.4。

本脚本
    1) 用冻结模型（original 442 上按 m01_spec.json 拟合、不做任何更新）在
       地理队列 177 例上出预测；
    2) 交叉拟合做 100 次重复 5 折（分层），给出
         · per-fold 口径（每折 35 例单独算指标）的中位数与 2.5–97.5 百分位
         · fold-pooled 口径（5 折拼成 177 例）跨 100 次重复的中位数
         · 可用于部署的全队列拟合截距/斜率及其 Bootstrap 95% CI
    3) 顺带给出两院 O:E、截距、斜率差值的患者层面 Bootstrap 对比
       （复核轮：两院区间重叠，不能只说 "calibration was not the same"）；
    4) 阶梯两臂的 difference-in-differences（患者层面 Bootstrap）。

输出
    Figures/{pdf,tif,jpg}/Fig3_recal_frozen.*
    Tables/TableS5_recalibration_in_geo.csv     （覆盖旧口径）
    Tables/TableS54_hospital_calibration_contrast.csv
    Tables/TableS55_ladder_did.csv
    Outputs/s35_recal_frozen.json
=============================================================================
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

R_REPEAT = 100
K = 5
BOOT = 2000


# --------------------------------------------------------------------------
# 基础工具
# --------------------------------------------------------------------------
def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _fit_intercept_only(p_tr, y_tr):
    """真正的 offset 模型：logit(p) 的系数固定为 1，只估截距。

    注意不能用单变量 LogisticRegression——那会把斜率一起估出来，
    "仅截距"与"斜率+截距"就会得到完全相同的结果（第一版踩过）。
    """
    logit = _logit(p_tr)
    a = 0.0
    for _ in range(80):
        eta = a + logit
        mu = 1 / (1 + np.exp(-eta))
        w = np.clip(mu * (1 - mu), 1e-9, None)
        step = np.sum(y_tr - mu) / (-np.sum(w))
        a_new = a - step
        if abs(a_new - a) < 1e-10:
            return float(a_new)
        a = a_new
    return float(a)


def _apply_intercept(p, a):
    return 1 / (1 + np.exp(-(a + _logit(p))))


def _fit_slope_intercept(p_tr, y_tr):
    lr = LogisticRegression(penalty=None, max_iter=1000, solver="lbfgs")
    lr.fit(_logit(p_tr).reshape(-1, 1), y_tr)
    return lr


def _apply(lr, p):
    return lr.predict_proba(_logit(p).reshape(-1, 1))[:, 1]


def _metrics(y, p):
    m = C.calibration_metrics(y, p)
    m["AUC"] = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan
    return m


METRIC_KEYS = ["AUC", "O:E", "Calibration slope", "Calibration intercept",
               "Brier", "ICI"]
IDEAL = {"AUC": None, "O:E": 1.0, "Calibration slope": 1.0,
         "Calibration intercept": 0.0, "Brier": None, "ICI": None}


def _pct(a, q):
    a = np.asarray(a, float)
    a = a[~np.isnan(a)]
    return float(np.percentile(a, q)) if len(a) else np.nan


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main():
    C.log_init("35_recal_frozen")
    cohorts, _ = C.load_cohorts()
    Xd, yd = C.derivation_label(cohorts["original"])
    Xg, yg = C.derivation_label(cohorts["geo"])

    spec = json.load(open(os.path.join(C.MODEL_DIR, "m01_spec.json"),
                          encoding="utf-8"))
    pipe = C.build_estimator("RF", spec["best_params"]["RF"], seed=C.SEED)
    pipe.fit(Xd, yd)
    p0 = pipe.predict_proba(Xg)[:, 1]

    ids = (cohorts["geo"]["ID"].values
           if "ID" in cohorts["geo"].columns
           else np.arange(len(yg)))
    hosp = np.where(np.asarray(ids) < 1_000_000, "A", "B")
    m_frozen = _metrics(yg, p0)
    C.section(f"冻结模型：AUC {m_frozen['AUC']:.4f}  O:E {m_frozen['O:E']:.3f}  "
              f"slope {m_frozen['Calibration slope']:.3f}  "
              f"intercept {m_frozen['Calibration intercept']:.3f}")
    # 自检：与正文 Table 2 / Table 3 的冻结行对齐
    assert abs(m_frozen["O:E"] - 0.706) < 0.01, m_frozen["O:E"]
    assert abs(m_frozen["Calibration slope"] - 0.88) < 0.02

    # ---------------- 100 次重复 5 折交叉拟合 ----------------
    C.section(f"交叉拟合：{R_REPEAT} 次重复 × {K} 折（分层）")
    per_fold = {f: {k: [] for k in METRIC_KEYS} for f in ("intercept", "both")}
    per_fold_frozen = {k: [] for k in METRIC_KEYS}
    pooled = {f: {k: [] for k in METRIC_KEYS} for f in ("intercept", "both")}
    rec_stack = {f: [] for f in ("intercept", "both")}
    fold_size = None

    for r in range(R_REPEAT):
        skf = StratifiedKFold(n_splits=K, shuffle=True,
                              random_state=C.SEED + 1000 + r)
        rec = {f: np.full(len(yg), np.nan) for f in ("intercept", "both")}
        for tr, te in skf.split(np.zeros(len(yg)), yg):
            fold_size = len(te)
            a_f = _fit_intercept_only(p0[tr], yg[tr])
            lr_b = _fit_slope_intercept(p0[tr], yg[tr])
            rec["intercept"][te] = _apply_intercept(p0[te], a_f)
            rec["both"][te] = _apply(lr_b, p0[te])
            mm0 = _metrics(yg[te], p0[te])
            for k in METRIC_KEYS:
                per_fold_frozen[k].append(mm0[k])
            for form in ("intercept", "both"):
                mm = _metrics(yg[te], rec[form][te])
                for k in METRIC_KEYS:
                    per_fold[form][k].append(mm[k])
        for form in ("intercept", "both"):
            mm = _metrics(yg, rec[form])
            for k in METRIC_KEYS:
                pooled[form][k].append(mm[k])
            rec_stack[form].append(rec[form])

    rec_mean = {f: np.mean(np.vstack(rec_stack[f]), axis=0)
                for f in ("intercept", "both")}
    C.log(f"  每折 {fold_size} 例；per-fold 样本量 {len(per_fold['intercept']['AUC'])}")

    # ---------------- 可部署系数（全队列拟合） ----------------
    C.section("全队列拟合（可部署系数）")
    a_full = _fit_intercept_only(p0, yg)
    lr_b_full = _fit_slope_intercept(p0, yg)
    b1_full = float(lr_b_full.coef_[0, 0])
    b0_full = float(lr_b_full.intercept_[0])
    C.log(f"  intercept-only: a = {a_full:+.3f}")
    C.log(f"  slope+intercept: b1 = {b1_full:.3f}, b0 = {b0_full:+.3f}")

    rng = np.random.default_rng(C.SEED)
    boot = {"a": [], "b1": [], "b0": []}
    n = len(yg)
    for _ in range(BOOT):
        idx = rng.integers(0, n, n)
        if len(np.unique(yg[idx])) < 2:
            continue
        boot["a"].append(_fit_intercept_only(p0[idx], yg[idx]))
        lb = _fit_slope_intercept(p0[idx], yg[idx])
        boot["b1"].append(float(lb.coef_[0, 0]))
        boot["b0"].append(float(lb.intercept_[0]))
    ci_a = (_pct(boot["a"], 2.5), _pct(boot["a"], 97.5))
    ci_b1 = (_pct(boot["b1"], 2.5), _pct(boot["b1"], 97.5))
    ci_b0 = (_pct(boot["b0"], 2.5), _pct(boot["b0"], 97.5))
    C.log(f"  a 95% CI {ci_a[0]:+.3f} to {ci_a[1]:+.3f}；"
          f"b1 {ci_b1[0]:.3f} to {ci_b1[1]:.3f}")

    # ---------------- 两院校准对比 ----------------
    C.section("两院校准差值（患者层面 Bootstrap）")
    A, B = hosp == "A", hosp == "B"

    def _contrast(idx):
        out = {}
        ya, pa = yg[idx][A[idx]], p0[idx][A[idx]]
        yb, pb = yg[idx][B[idx]], p0[idx][B[idx]]
        if len(np.unique(ya)) < 2 or len(np.unique(yb)) < 2:
            return None
        ma, mb = _metrics(ya, pa), _metrics(yb, pb)
        out["dOE"] = mb["O:E"] - ma["O:E"]
        out["dSlope"] = mb["Calibration slope"] - ma["Calibration slope"]
        out["dIntercept"] = (mb["Calibration intercept"]
                             - ma["Calibration intercept"])
        out["dAUC"] = mb["AUC"] - ma["AUC"]
        return out

    obs = _contrast(np.arange(n))
    rng2 = np.random.default_rng(C.SEED + 7)
    draws = {k: [] for k in obs}
    for _ in range(BOOT):
        idx = rng2.integers(0, n, n)
        c = _contrast(idx)
        if c is None:
            continue
        for k, v in c.items():
            draws[k].append(v)
    rows_h = []
    lab = {"dOE": "O:E (B minus A)", "dSlope": "Calibration slope (B minus A)",
           "dIntercept": "Calibration intercept (B minus A)",
           "dAUC": "AUC (B minus A)"}
    for k in ("dAUC", "dOE", "dSlope", "dIntercept"):
        rows_h.append({"Contrast": lab[k], "Estimate": round(obs[k], 3),
                       "95% CI": f"{_pct(draws[k], 2.5):+.3f} to "
                                 f"{_pct(draws[k], 97.5):+.3f}"})
        C.log(f"  {lab[k]:34s} {obs[k]:+.3f} "
              f"({_pct(draws[k], 2.5):+.3f} to {_pct(draws[k], 97.5):+.3f})")
    C.save_table(pd.DataFrame(rows_h), "TableS54_hospital_calibration_contrast")

    # ---------------- 阶梯两臂的 difference-in-differences ----------------
    C.section("阶梯两臂 DiD（ΔAUC 与 ΔBrier，患者层面 Bootstrap）")
    g = pd.read_pickle(os.path.join(C.OUT_DIR, "s03_growth_full.pkl"))
    reps = sorted(g["repeat"].unique())
    arm_p = {}
    for arm in ("temporal", "control"):
        d0 = g[(g["arm"] == arm) & (g["pct_temporal"] == 0.0)].set_index("repeat")
        d1 = g[(g["arm"] == arm) & (g["pct_temporal"] == 1.0)].set_index("repeat")
        arm_p[arm] = {
            "p0": [np.asarray(d0.loc[r, "p_geo"], float) for r in reps],
            "p1": [np.asarray(d1.loc[r, "p_geo"], float) for r in reps],
        }
    y_lad = np.asarray(g.iloc[0]["y_geo"], float)

    def _did(idx, metric):
        vals = []
        for i, _ in enumerate(reps):
            a0 = metric(y_lad[idx], arm_p["temporal"]["p0"][i][idx])
            a1 = metric(y_lad[idx], arm_p["temporal"]["p1"][i][idx])
            c0 = metric(y_lad[idx], arm_p["control"]["p0"][i][idx])
            c1 = metric(y_lad[idx], arm_p["control"]["p1"][i][idx])
            vals.append((a1 - a0) - (c1 - c0))
        return float(np.mean(vals))

    mets = {"AUC": lambda y, p: roc_auc_score(y, p) if len(np.unique(y)) > 1
            else np.nan,
            "Brier": lambda y, p: float(np.mean((p - y) ** 2))}
    rng3 = np.random.default_rng(C.SEED + 11)
    rows_d = []
    for name, fn in mets.items():
        base = np.arange(len(y_lad))
        obs_d = _did(base, fn)
        draws_d = []
        for _ in range(1000):
            idx = rng3.integers(0, len(y_lad), len(y_lad))
            v = _did(idx, fn)
            if not np.isnan(v):
                draws_d.append(v)
        lo, hi = _pct(draws_d, 2.5), _pct(draws_d, 97.5)
        rows_d.append({"Metric": name, "DiD (temporal minus control)":
                       round(obs_d, 4),
                       "95% CI": f"{lo:+.4f} to {hi:+.4f}"})
        C.log(f"  {name:6s} DiD {obs_d:+.4f} ({lo:+.4f} to {hi:+.4f})")
    C.save_table(pd.DataFrame(rows_d), "TableS55_ladder_did")

    # ---------------- Table S5 ----------------
    def _row(form, accounting, n_eval, mdict, lo=None, hi=None,
             dep_oe=None, dep_ic=None):
        r = {"Form": form, "Accounting": accounting, "n per evaluation": n_eval}
        for k, lab_ in [("AUC", "AUC"), ("O:E", "O:E"),
                        ("Calibration slope", "Calibration slope"),
                        ("Calibration intercept", "Calibration intercept"),
                        ("Brier", "Brier"), ("ICI", "ICI")]:
            v = mdict[k]
            if lo is None:
                r[lab_] = "-" if v is None or np.isnan(v) else round(float(v), 3)
            else:
                r[lab_] = (f"{v:.3f} ({lo[k]:.3f} to {hi[k]:.3f})")
        # 距理想值的绝对偏离（正文 3.4 的主口径）：逐折行取 500 个逐折估计的
        # 均值；合并/冻结/部署行取该行自身的取值。 reviewer P1：显式列出，
        # 免得读者只能从 O:E 与截距两列反推。
        r["Mean |O:E - 1|"] = (round(float(np.nanmean(dep_oe)), 4)
                               if dep_oe is not None
                               else round(abs(mdict["O:E"] - 1.0), 4))
        r["Mean |intercept|"] = (round(float(np.nanmean(dep_ic)), 4)
                                 if dep_ic is not None
                                 else round(abs(mdict["Calibration intercept"]), 4))
        return r

    tb = []
    tb.append(_row("None (frozen)", "single evaluation of the whole cohort",
                   177, m_frozen))
    # ★ R21 复核（P1）：同一划分下的冻结模型逐折基线。正文"下降"的前后两项
    #   必须取自同一抽样单位——逐折均值对逐折均值，而不是全队列对逐折。
    med0 = {k: _pct(per_fold_frozen[k], 50) for k in METRIC_KEYS}
    lo0 = {k: _pct(per_fold_frozen[k], 2.5) for k in METRIC_KEYS}
    hi0 = {k: _pct(per_fold_frozen[k], 97.5) for k in METRIC_KEYS}
    tb.append(_row("None (frozen)", f"per fold (n = {fold_size})", fold_size,
                   med0, lo0, hi0,
                   dep_oe=np.abs(np.asarray(per_fold_frozen["O:E"], float) - 1.0),
                   dep_ic=np.abs(np.asarray(per_fold_frozen["Calibration intercept"],
                                            float))))
    for form, lab_form in (("intercept", "Intercept only"),
                           ("both", "Slope and intercept")):
        med = {k: _pct(per_fold[form][k], 50) for k in METRIC_KEYS}
        lo = {k: _pct(per_fold[form][k], 2.5) for k in METRIC_KEYS}
        hi = {k: _pct(per_fold[form][k], 97.5) for k in METRIC_KEYS}
        dep_oe = np.abs(np.asarray(per_fold[form]["O:E"], float) - 1.0)
        dep_ic = np.abs(np.asarray(per_fold[form]["Calibration intercept"],
                                   float))
        tb.append(_row(lab_form, f"per fold (n = {fold_size})", fold_size,
                       med, lo, hi, dep_oe=dep_oe, dep_ic=dep_ic))
        medp = {k: _pct(pooled[form][k], 50) for k in METRIC_KEYS}
        lop = {k: _pct(pooled[form][k], 2.5) for k in METRIC_KEYS}
        hip = {k: _pct(pooled[form][k], 97.5) for k in METRIC_KEYS}
        tb.append(_row(lab_form, "fold-pooled (n = 177)", 177, medp, lop, hip))

    # 可部署系数行
    p_dep_i = _apply_intercept(p0, a_full)
    p_dep_b = _apply(lr_b_full, p0)
    tb.append(_row("Intercept only, full-cohort fit "
                   "(apparent; not independently validated)",
                   f"applied intercept {a_full:+.3f} "
                   f"(95% CI {ci_a[0]:+.3f} to {ci_a[1]:+.3f})",
                   177, _metrics(yg, p_dep_i)))
    tb.append(_row("Slope and intercept, full-cohort fit "
                   "(apparent; not independently validated)",
                   f"applied slope {b1_full:.3f} "
                   f"({ci_b1[0]:.3f} to {ci_b1[1]:.3f}), "
                   f"intercept {b0_full:+.3f} "
                   f"({ci_b0[0]:+.3f} to {ci_b0[1]:+.3f})",
                   177, _metrics(yg, p_dep_b)))
    t5 = pd.DataFrame(tb)
    C.save_table(t5, "TableS5_recalibration_in_geo")
    print(t5.to_string(index=False))

    # ---------------- TableS28（与 Table S5 同源，保持旧表结构） ----------------
    s28_rows = []
    for form, lab in (("intercept", "intercept-only"),
                      ("both", "slope+intercept")):
        dep_oe = np.abs(np.asarray(per_fold[form]["O:E"], float) - 1.0)
        dep_ic = np.abs(np.asarray(per_fold[form]["Calibration intercept"],
                                   float))
        sl = np.asarray(per_fold[form]["Calibration slope"], float)
        s28_rows.append({
            "Recalibration": lab,
            "n evaluated per unit": "177 (pooled across folds) / 35 (per fold)",
            "pooled O:E distance from 1": round(abs(_pct(pooled[form]["O:E"], 50) - 1), 4),
            "pooled intercept distance from 0": round(abs(_pct(pooled[form]["Calibration intercept"], 50)), 4),
            "pooled slope": round(_pct(pooled[form]["Calibration slope"], 50), 3),
            "pooled AUC": round(_pct(pooled[form]["AUC"], 50), 3),
            "per-fold mean O:E distance from 1": round(float(np.nanmean(dep_oe)), 4),
            "per-fold mean intercept distance from 0": round(float(np.nanmean(dep_ic)), 4),
            "per-fold mean slope": round(float(np.nanmean(sl)), 3),
            "per-fold mean |slope - 1|": round(float(np.nanmean(np.abs(sl - 1))), 3),
            "per-fold SD of O:E distance from 1": round(float(np.nanstd(dep_oe)), 4),
            "per-fold SD of intercept distance from 0": round(float(np.nanstd(dep_ic)), 4),
            "per-fold AUC median": round(_pct(per_fold[form]["AUC"], 50), 3),
            "per-fold AUC 2.5th": round(_pct(per_fold[form]["AUC"], 2.5), 3),
            "per-fold AUC 97.5th": round(_pct(per_fold[form]["AUC"], 97.5), 3),
        })
    s28 = pd.DataFrame(s28_rows)
    C.save_table(s28, "TableS28_crossfit_pooled_vs_perfold")
    print(s28.to_string(index=False))

    # ---------------- 主表 3：全部数字来自本次运行 ----------------
    r19 = []
    r19.append({"Form": "None (frozen)", "Accounting": "-",
                "AUC": f"{m_frozen['AUC']:.3f}",
                "O:E distance from 1": f"{abs(m_frozen['O:E'] - 1):.3f}",
                "Intercept distance from 0": f"{abs(m_frozen['Calibration intercept']):.2f}"})
    for form, lab in (("intercept", "Intercept only"),
                      ("both", "Slope and intercept")):
        r = s28_rows[0 if form == "intercept" else 1]
        r19.append({"Form": lab, "Accounting": f"per fold (n = {fold_size} each)",
                    "AUC": f"{r['pooled AUC']:.3f}",
                    "O:E distance from 1": f"{r['per-fold mean O:E distance from 1']:.3f}",
                    "Intercept distance from 0": f"{r['per-fold mean intercept distance from 0']:.3f}"})
        r19.append({"Form": lab, "Accounting": "fold-pooled (n = 177)",
                    "AUC": f"{r['pooled AUC']:.3f}",
                    "O:E distance from 1": f"{r['pooled O:E distance from 1']:.4f}",
                    "Intercept distance from 0": f"{r['pooled intercept distance from 0']:.4f}"})
    t3 = pd.DataFrame(r19)
    t3.to_csv(os.path.join(C.TABLE_DIR, "R19_T3_recalibration.csv"), index=False,
              encoding="utf-8-sig")
    print(t3.to_string(index=False))

    # ---------------- Figure 3 ----------------
    fig_recal(yg, p0, rec_mean["intercept"], p_dep_i, m_frozen,
              per_fold["intercept"], per_fold_frozen,
              {k: _pct(pooled["intercept"][k], 50) for k in METRIC_KEYS})

    json.dump({
        "frozen": {k: round(float(v), 4) for k, v in m_frozen.items()
                   if isinstance(v, (int, float, np.floating))},
        "deployable_intercept_only": {"a": round(a_full, 4),
                                      "ci": [round(ci_a[0], 4),
                                             round(ci_a[1], 4)]},
        "deployable_slope_intercept": {"b0": round(b0_full, 4),
                                       "b1": round(b1_full, 4),
                                       "ci_b0": [round(ci_b0[0], 4),
                                                 round(ci_b0[1], 4)],
                                       "ci_b1": [round(ci_b1[0], 4),
                                                 round(ci_b1[1], 4)]},
        "per_fold_median": {k: round(_pct(per_fold["intercept"][k], 50), 4)
                            for k in METRIC_KEYS},
        "per_fold_2.5": {k: round(_pct(per_fold["intercept"][k], 2.5), 4)
                         for k in METRIC_KEYS},
        "per_fold_97.5": {k: round(_pct(per_fold["intercept"][k], 97.5), 4)
                          for k in METRIC_KEYS},
        "hospital_contrast": rows_h,
        "ladder_did": rows_d,
        "n_repeat": R_REPEAT, "k_fold": K, "fold_size": fold_size,
    }, open(os.path.join(C.OUT_DIR, "s35_recal_frozen.json"), "w",
            encoding="utf-8"), indent=1, ensure_ascii=False)


def fig_recal(yg, p0, p_rec, p_glob, m_frozen, per_fold, per_fold_frozen,
              pooled_med):
    import matplotlib.pyplot as plt
    n_bins = 8

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6))
    ax = axes[0]
    C.ref_line(ax, diag=True, label="Ideal")

    def _band(p_, n_boot=400, seed=17):
        rng = np.random.default_rng(seed)
        n = len(yg)
        acc = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            if len(np.unique(yg[idx])) < 2:
                continue
            acc.append(C.calibration_curve_bins(yg[idx], p_[idx],
                                                n_bins=n_bins)["obs"].values)
        if not acc:
            return None, None
        acc = np.vstack(acc)
        return np.nanpercentile(acc, 2.5, axis=0), np.nanpercentile(acc, 97.5,
                                                                   axis=0)

    for p_, lab, col, ls in [(p0, "Frozen model", C.C_RED, "-"),
                             (p_rec, "After cross-fitted recalibration", C.C_GEO, "-"),
                             (p_glob, "After global recalibration (apparent)",
                              C.C_GEO, "--")]:
        cc = C.calibration_curve_bins(yg, p_, n_bins=n_bins)
        if ls == "-":
            lo, hi = _band(p_)
            if lo is not None:
                ax.fill_between(cc["pred"].values, lo, hi, color=col,
                                alpha=0.14, lw=0, zorder=1)
        ax.plot(cc["pred"], cc["obs"], ls, marker="o" if ls == "-" else None,
                color=col, lw=1.5, ms=4, label=lab, zorder=3)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed proportion")
    ax.set_title("A  Cross-fitted recalibration", loc="left", fontweight="bold")
    C.style_legend(ax, fontsize=7.5)

    ax = axes[1]
    rows = ["Calibration slope", "Calibration intercept", "O:E", "Brier", "ICI"]
    ylab = ["Calibration slope\n(ideal = 1)", "Calibration intercept\n(ideal = 0)",
            "O:E ratio\n(ideal = 1)", "Brier score\n(ideal = 0)",
            "ICI\n(ideal = 0)"]
    for i, y in enumerate(np.arange(len(rows))[::-1].astype(float)):
        ax.plot([0, 0.98], [y, y], color="#ececec", lw=8.5,
                solid_capstyle="round", zorder=1)
    for i, (k, y) in enumerate(zip(rows, np.arange(len(rows))[::-1].astype(float))):
        ideal = IDEAL[k]
        # ★ R21 复核（P1b）：红色 before 点同样改为冻结模型逐折绝对偏离的均值，
        #   与绿色 after 点同一抽样单位（此前红=全队列、绿=逐折均值，Brier/ICI
        #   两行会出现虚假的"改善/恶化"）。全队列冻结值见表 S5 第一行。
        dep0_arr = (np.abs(np.asarray(per_fold_frozen[k], float) - ideal)
                    if ideal is not None
                    else np.abs(np.asarray(per_fold_frozen[k], float)))
        d0 = float(np.nanmean(dep0_arr))
        # ★ R21 复核（P1）：主口径 = 逐折绝对偏离的均值（与正文 0.081/0.148、
        #   表 S5 的 Mean 列同源）；横条 = 500 个逐折绝对偏离的 2.5-97.5 百分位
        #   （离散度展示，不是置信区间）；折合并估计只画空心对照点。
        dep = (np.abs(np.asarray(per_fold[k], float) - ideal)
               if ideal is not None else np.abs(np.asarray(per_fold[k], float)))
        d1 = float(np.nanmean(dep))
        d_lo = float(np.nanpercentile(dep, 2.5))
        d_hi = float(np.nanpercentile(dep, 97.5))
        d_pool = (abs(float(pooled_med[k]) - ideal) if ideal is not None
                  else abs(float(pooled_med[k])))
        ax.plot([d_lo, d_hi], [y, y], color=C.C_GEO,
                lw=1.1, alpha=0.55, solid_capstyle="butt", zorder=2)
        ax.plot(d0, y, "o", ms=11.0, mfc=C.C_RED, mec="white", mew=1.3,
                zorder=6, clip_on=False)
        ax.annotate(f"{d0:.2f}", (d0, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", va="bottom", fontsize=8,
                    color=C.C_RED, zorder=7,
                    bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none",
                              alpha=0.85))
        ax.plot(d1, y, "o", ms=8.2, mfc=C.C_GEO, mec="white", mew=1.3,
                zorder=6, clip_on=False)
        ax.annotate(f"{d1:.2f}", (d1, y), textcoords="offset points",
                    xytext=(0, -11), ha="center", va="top", fontsize=8,
                    color=C.C_GEO, zorder=7,
                    bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none",
                              alpha=0.85))
        ax.plot(d_pool, y, "o", ms=8.0, mfc="none", mec=C.C_GEO, mew=1.2,
                zorder=6, clip_on=False)
    ax.axvline(0, color="#9a9a9a", lw=0.9, ls=(0, (4, 3)), zorder=0)
    ax.set_yticks(np.arange(len(rows))[::-1].astype(float))
    ax.set_yticklabels(ylab, fontsize=8.0, linespacing=1.35)
    ax.set_ylim(-1.05, len(rows) - 0.32)
    ax.set_xlim(-0.025, 0.98)
    ax.set_xticks(np.arange(0.0, 0.91, 0.1))
    ax.set_xlabel("Absolute departure from the ideal value", fontsize=9)
    ax.set_title("B  Departure from ideal calibration", loc="left",
                 fontweight="bold")
    C.style_axes(ax, grid=True, axis="x")

    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", mfc=C.C_RED, mec="white",
                      mew=1.1, ms=8.5, label="Before (frozen, per-fold mean)"),
               Line2D([], [], marker="o", ls="", mfc=C.C_GEO, mec="white",
                      mew=1.1, ms=8.5,
                      label="After (per-fold mean)"),
               Line2D([], [], marker="o", ls="", mfc="none", mec=C.C_GEO,
                      mew=1.2, ms=8.0,
                      label="Fold-pooled (for comparison)"),
               ]
    # ★ 图例放中右：右下角会压住 ICI 行的红点标注（实测）。
    lg = ax.legend(handles=handles, loc="lower right", fontsize=7.0,
                   frameon=True, framealpha=0.9, edgecolor="#cccccc",
                   fancybox=False, borderpad=0.5, labelspacing=0.4)
    lg.get_frame().set_linewidth(0.6)
    fig.subplots_adjust(wspace=0.46)
    C.save_figure(fig, "Fig3_recal_frozen")


if __name__ == "__main__":
    main()
