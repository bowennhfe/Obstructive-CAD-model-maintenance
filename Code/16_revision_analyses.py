# -*- coding: utf-8 -*-
"""
16 —— 修订分析（回应同行评议的统计性质疑）
=============================================================================
本脚本不修改任何既有结果，只新增分析。逐项对应审稿意见：

(3) 斜率+截距再校准后 AUC 下降 0.0139 是否为交叉拟合伪影？
    → 在全体地理队列上拟合**单一全局**仿射再校准（理论上 AUC 严格不变），
      与交叉拟合版本并排报告；并给出各折斜率的离散度以说明伪影来源。

(4) 对照臂用有放回重采样产生重复患者，内部交叉验证是否泄露？
    → 重建对照臂：把每个重采样行映射回**原始患者 ID**，用 StratifiedGroupKFold
      做（调参 + 内部评估）交叉验证，同一患者的全部副本只出现在同一折；
      同时保留朴素 StratifiedKFold 版本以量化泄露幅度。

(5) "目标中心几十例即可修复尺度"是否成立？
    → 再校准样本量阶梯：在地理队列中取 m 例拟合再校准、在其余 177-m 例评价，
      m = 20…150，每档重复 400 次，报告与理想值的平均距离。

(2) Brier / ICI 不是纯校准指标。
    → Murphy 分解（reliability / resolution / uncertainty），再校准前后对比。

(6) 28 对算法比较：不能由"无显著差异"推出"等价"。
    → 每对的配对 Bootstrap ΔAUC（95% CI）、最小可检测差异（MDD, 80% power）、
      以及 ±0.05 等价界的 TOST 结论。

(7) 时间队列来自哪个中心？
    → 用推导集两中心的 19 变量二分类器（交叉拟合 AUC≈0.912）预测时间队列，
      给出混合比例的**估计**与其局限；并给出时间队列与两中心的 SMD。

输出
    Tables/TableS22_recalibration_global_vs_crossfit.csv
    Tables/TableS23_recalibration_sample_size.csv
    Tables/TableS24_control_arm_grouped.csv
    Tables/TableS25_brier_decomposition.csv
    Tables/TableS26_pairwise_equivalence.csv
    Tables/TableS27_temporal_centre_mixture.csv
    Outputs/s16_*.csv
=============================================================================
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import openpyxl

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (StratifiedKFold, StratifiedGroupKFold,
                                     GridSearchCV)
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import importlib
A14 = importlib.import_module("14_algo_comparison")   # 复用算法集与 make_pipe/tune

SEED = C.SEED
Z_POW = 1.959963985 + 0.841621234      # α=0.05 双侧、power=80% → MDD 系数 2.8016


# =============================================================================
# 基础工具
# =============================================================================
def logit(p):
    p = np.clip(np.asarray(p, float), 1e-8, 1 - 1e-8)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit_intercept(y, lo, max_iter=120):
    """仅截距（斜率固定 1）的 Newton 迭代，返回 a。"""
    a = 0.0
    for _ in range(max_iter):
        mu = sigmoid(a + lo)
        w = np.clip(mu * (1 - mu), 1e-9, None)
        h = -np.sum(w)
        step = np.sum(y - mu) / h if h != 0 else 0.0
        a_new = a - step
        if abs(a_new - a) < 1e-11:
            return a_new
        a = a_new
    return a


def fit_slope_intercept(y, lo):
    """y ~ sigmoid(b0 + b1*lo)，返回 (b0, b1)。"""
    b = C._irls(np.column_stack([np.ones_like(lo), lo]), np.asarray(y, float))
    return float(b[0]), float(b[1])


def apply_intercept(a, lo):
    return sigmoid(a + lo)


def apply_slope_intercept(b0, b1, lo):
    return sigmoid(b0 + b1 * lo)


def crossfit(y, p, mode, k=5, seed=SEED):
    """交叉拟合再校准：mode ∈ {'intercept','slope'}。返回袋外再校准概率与各折参数。"""
    y = np.asarray(y, int)
    lo = logit(p)
    out = np.full(len(y), np.nan)
    pars = []
    skf = StratifiedKFold(k, shuffle=True, random_state=seed)
    for tr, te in skf.split(lo.reshape(-1, 1), y):
        if mode == "intercept":
            a = fit_intercept(y[tr], lo[tr])
            out[te] = apply_intercept(a, lo[te])
            pars.append((a, 1.0))
        else:
            b0, b1 = fit_slope_intercept(y[tr], lo[tr])
            out[te] = apply_slope_intercept(b0, b1, lo[te])
            pars.append((b0, b1))
    return out, pars


def metrics(y, p):
    cm = C.calibration_metrics(y, p)
    return dict(AUC=float(roc_auc_score(y, p)),
                slope=cm["Calibration slope"],
                intercept=cm["Calibration intercept"],
                O_E=cm["O:E"], ICI=cm["ICI"], Brier=cm["Brier"])


def brier_decomp(y, p, n_bins=10):
    """Murphy 分解：Brier = uncertainty - resolution + reliability。

    分箱口径与 ICI 一致（按预测概率的等分位分箱）。
    """
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    obar = y.mean()
    order = np.argsort(p)
    bins = np.array_split(order, n_bins)
    rel = res = 0.0
    for b in bins:
        if len(b) == 0:
            continue
        w = len(b) / len(y)
        rel += w * (p[b].mean() - y[b].mean()) ** 2
        res += w * (y[b].mean() - obar) ** 2
    return dict(uncertainty=obar * (1 - obar), reliability=rel,
                resolution=res, brier_check=obar * (1 - obar) - res + rel)


# =============================================================================
# 1) 再校准的 AUC 不变性：全局 vs 交叉拟合
# =============================================================================
def sec1_global_vs_crossfit(Xd, yd, Xg, yg, geo_frozen):
    C.section("1) 再校准后的 AUC：全局仿射变换 vs 交叉拟合")
    tuned = {}
    for k in A14.ORDER:
        tuned[k] = A14.tune(k, Xd, yd)
    preds = {}
    for k in A14.ORDER:
        pipe = A14.make_pipe(k, tuned[k])
        pipe.fit(Xd, yd)
        preds[k] = pipe.predict_proba(Xg)[:, 1]
    # 一致性自检：与 Table12 的 AUC 对齐
    t12 = pd.read_csv(os.path.join(C.TABLE_DIR, "Table12_algo_external.csv"))
    chk = []
    for k in A14.ORDER:
        a_now = roc_auc_score(yg, preds[k])
        a_tab = float(t12.loc[t12["Model"] == k, "AUC"].iloc[0])
        chk.append(abs(round(a_now, 3) - a_tab))
    C.log(f"  与 Table12 的最大 AUC 偏差（3 位小数）= {max(chk):.3f}")
    # 冻结 RF（脚本 02 的模型，正文主模型）
    fz = pd.read_csv(os.path.join(C.OUT_DIR, "s02_geo_pred_rf.csv"))
    y_fz = fz["y"].astype(int).values
    preds["RF_frozen02"] = fz["p_rf"].values

    rows = []
    fold_b1 = {}
    for k, p in preds.items():
        y = y_fz if k.endswith("02") else np.asarray(yg, int)
        name = "RF (frozen, pipeline 02)" if k.endswith("02") else k
        m0 = metrics(y, p)
        lo = logit(p)
        b0g, b1g = fit_slope_intercept(y, lo)
        ag = fit_intercept(y, lo)
        p_glob_s = apply_slope_intercept(b0g, b1g, lo)
        p_glob_i = apply_intercept(ag, lo)
        p_cf_i, _ = crossfit(y, p, "intercept")
        p_cf_s, pars_s = crossfit(y, p, "slope")
        b1s = np.array([b for _, b in pars_s])
        fold_b1[name] = (b1s.min(), b1s.max())
        ms, mi = metrics(y, p_glob_s), metrics(y, p_glob_i)
        cfs, cfi = metrics(y, p_cf_s), metrics(y, p_cf_i)
        rows.append({
            "Model": name,
            "AUC raw": round(m0["AUC"], 4),
            "AUC after global intercept": round(mi["AUC"], 4),
            "AUC after global slope+intercept": round(ms["AUC"], 4),
            "ΔAUC global intercept": round(mi["AUC"] - m0["AUC"], 6),
            "ΔAUC global slope+intercept": round(ms["AUC"] - m0["AUC"], 6),
            "AUC after cross-fitted intercept": round(cfi["AUC"], 4),
            "AUC after cross-fitted slope+intercept": round(cfs["AUC"], 4),
            "ΔAUC cross-fitted intercept": round(cfi["AUC"] - m0["AUC"], 4),
            "ΔAUC cross-fitted slope+intercept": round(cfs["AUC"] - m0["AUC"], 4),
            "global slope (b1)": round(b1g, 3),
            "fold slope min": round(b1s.min(), 3),
            "fold slope max": round(b1s.max(), 3),
            "O:E global slope+intercept": round(ms["O_E"], 3),
            "slope after global slope+intercept": round(ms["slope"], 3),
            "intercept after global slope+intercept": round(ms["intercept"], 4),
        })
    t = pd.DataFrame(rows)
    C.save_table(t, "TableS22_recalibration_global_vs_crossfit")
    C.log(t.drop(columns=["fold slope min", "fold slope max"]).to_string(index=False))

    # ---- 交叉拟合指标：池化 vs 逐折平均（说明池化口径为何乐观）
    from sklearn.model_selection import StratifiedKFold as _SKF
    rows2 = []
    for mode, nm in [("intercept", "intercept-only"), ("slope", "slope+intercept")]:
        skf = _SKF(5, shuffle=True, random_state=SEED)
        lo = logit(preds["RF_frozen02"])
        pooled = np.full(len(y_fz), np.nan)
        per = []
        for tr, te in skf.split(lo.reshape(-1, 1), y_fz):
            if mode == "intercept":
                a = fit_intercept(y_fz[tr], lo[tr])
                pv = apply_intercept(a, lo[te])
            else:
                b0, b1 = fit_slope_intercept(y_fz[tr], lo[tr])
                pv = apply_slope_intercept(b0, b1, lo[te])
            pooled[te] = pv
            per.append(metrics(y_fz[te], pv))
        dper = pd.DataFrame(per)
        mp = metrics(y_fz, pooled)
        sd_oe = float((dper["O_E"] - 1).abs().std(ddof=1))
        sd_ic = float(dper["intercept"].abs().std(ddof=1))
        # 患者层面配对 Bootstrap：池化口径下 O:E 距 1、截距距 0 的 95% CI
        rng = np.random.default_rng(SEED + (0 if mode == "intercept" else 1))
        ok = ~np.isnan(pooled)
        yb, pb = y_fz[ok], pooled[ok]
        boots = []
        for _ in range(2000):
            idx = rng.integers(0, len(yb), len(yb))
            if yb[idx].min() == yb[idx].max():
                continue
            mb = metrics(yb[idx], pb[idx])
            boots.append((abs(mb["O_E"] - 1), abs(mb["intercept"])))
        ba = np.array(boots)
        rows2.append({
            "Recalibration": nm,
            "n evaluated per unit": "177 (pooled across folds) / 35 (per fold)",
            "pooled O:E distance from 1": round(abs(mp["O_E"] - 1), 4),
            "pooled intercept distance from 0": round(abs(mp["intercept"]), 4),
            "pooled slope": round(mp["slope"], 3),
            "per-fold mean O:E distance from 1": round((dper["O_E"] - 1).abs().mean(), 4),
            "per-fold mean intercept distance from 0": round(
                dper["intercept"].abs().mean(), 4),
            "per-fold mean slope": round(dper["slope"].mean(), 3),
            "per-fold mean |slope - 1|": round((dper["slope"] - 1).abs().mean(), 3),
            "per-fold SD of O:E distance from 1": round(sd_oe, 4),
            "per-fold SD of intercept distance from 0": round(sd_ic, 4),
            "boot 95% CI low, O:E distance from 1": round(float(np.percentile(ba[:, 0], 2.5)), 4),
            "boot 95% CI high, O:E distance from 1": round(float(np.percentile(ba[:, 0], 97.5)), 4),
            "boot 95% CI low, intercept distance from 0": round(float(np.percentile(ba[:, 1], 2.5)), 4),
            "boot 95% CI high, intercept distance from 0": round(float(np.percentile(ba[:, 1], 97.5)), 4),
        })
    t28 = pd.DataFrame(rows2)
    C.save_table(t28, "TableS28_crossfit_pooled_vs_perfold")
    C.log(t28.to_string(index=False))
    # 落盘 8 族的地理预测（供脚本 17 计算估计精度）
    C.save_csv(pd.DataFrame({"y": np.asarray(yg, int),
                             **{k: preds[k] for k in A14.ORDER}}),
               "s16_algo_geo_pred")
    return preds, y_fz, tuned


# =============================================================================
# 2) 再校准样本量阶梯
# =============================================================================
def sec2_recal_ladder(y, p, sizes=(20, 30, 40, 60, 80, 100, 118),
                      n_eval=59, R=600):
    """固定评价集规模，只改变再校准的拟合样本量 m。

    设计：每次重复把 177 例分层随机拆成「再校准池」(118) 与「评价集」(59)；
    在池中再分层抽取 m 例用于拟合再校准参数，然后在同一个评价集上评价。
    这样 m 的变化不会与评价集规模的变化混在一起。
    """
    C.section("2) 再校准样本量阶梯（固定评价集 n=59，只改变拟合样本量 m）")
    y = np.asarray(y, int)
    lo = logit(p)
    rng = np.random.RandomState(SEED)

    def _nb(yy, phat, t):
        pred = np.asarray(phat) >= t
        yy = np.asarray(yy, int)
        n = len(yy)
        tp = int(np.sum(pred & (yy == 1)))
        fp = int(np.sum(pred & (yy == 0)))
        return tp / n - fp / n * (t / (1 - t))

    idx1 = np.where(y == 1)[0]
    idx0 = np.where(y == 0)[0]
    n_pool = len(y) - n_eval
    n1_ev = int(round(n_eval * y.mean()))
    rows = []
    for m in sizes:
        acc = {k: [] for k in ["base_oe", "oe_i", "oe_s", "base_ic", "ic_i", "ic_s",
                               "base_sl", "sl_s", "brier_s", "au_raw", "au_s",
                               "base_nb50", "nb50_i", "base_nb40", "nb40_i",
                               "nb50_win", "nb40_win"]}
        for _ in range(R):
            e1 = rng.choice(idx1, n1_ev, replace=False)
            e0 = rng.choice(idx0, n_eval - n1_ev, replace=False)
            ev = np.concatenate([e1, e0])
            pool = np.setdiff1d(np.arange(len(y)), ev)
            if len(pool) < m:
                continue
            p1 = pool[y[pool] == 1]
            p0 = pool[y[pool] == 0]
            m1 = int(round(m * y.mean()))
            m0 = m - m1
            if m1 > len(p1) or m0 > len(p0) or m1 < 2 or m0 < 2:
                continue
            fit = np.concatenate([rng.choice(p1, m1, replace=False),
                                  rng.choice(p0, m0, replace=False)])
            a = fit_intercept(y[fit], lo[fit])
            b0, b1 = fit_slope_intercept(y[fit], lo[fit])
            if not np.isfinite(b1) or b1 <= 0:
                continue
            pe, ye = lo[ev], y[ev]
            pi = apply_intercept(a, pe)
            ps = apply_slope_intercept(b0, b1, pe)
            mi, ms, m0_ = metrics(ye, pi), metrics(ye, ps), metrics(ye, p[ev])
            acc["base_oe"].append(abs(m0_["O_E"] - 1))
            acc["oe_i"].append(abs(mi["O_E"] - 1))
            acc["oe_s"].append(abs(ms["O_E"] - 1))
            acc["base_ic"].append(abs(m0_["intercept"]))
            acc["ic_i"].append(abs(mi["intercept"]))
            acc["ic_s"].append(abs(ms["intercept"]))
            acc["base_sl"].append(abs(m0_["slope"] - 1))
            acc["sl_s"].append(abs(ms["slope"] - 1))
            acc["brier_s"].append(ms["Brier"])
            acc["au_raw"].append(m0_["AUC"])
            acc["au_s"].append(ms["AUC"])
            # 决策层级：同一评价集上的净获益（冻结 vs 仅截距再校准）
            nb50_b = _nb(ye, p[ev], 0.50)
            nb50_i = _nb(ye, pi, 0.50)
            nb40_b = _nb(ye, p[ev], 0.40)
            nb40_i = _nb(ye, pi, 0.40)
            acc["base_nb50"].append(nb50_b)
            acc["nb50_i"].append(nb50_i)
            acc["base_nb40"].append(nb40_b)
            acc["nb40_i"].append(nb40_i)
            acc["nb50_win"].append(1.0 if nb50_i > nb50_b + 1e-12 else 0.0)
            acc["nb40_win"].append(1.0 if nb40_i > nb40_b + 1e-12 else 0.0)
        rec = {"n_fit": m, "n_eval": n_eval, "n_pool": n_pool}
        for k, v in acc.items():
            v = np.asarray([x for x in v if np.isfinite(x)])
            if len(v) == 0:
                rec[k] = np.nan
                continue
            rec[k] = float(v.mean())
            rec[k + "_lo"] = float(np.percentile(v, 2.5))
            rec[k + "_hi"] = float(np.percentile(v, 97.5))
        # 距离闭合比例
        for a_, b_, nm in [("base_oe", "oe_i", "OE closed (intercept)"),
                           ("base_oe", "oe_s", "OE closed (slope+int)"),
                           ("base_ic", "ic_i", "intercept closed"),
                           ("base_sl", "sl_s", "slope closed")]:
            if rec[a_] and np.isfinite(rec[a_]):
                rec[nm] = round(1 - rec[b_] / rec[a_], 3)
            else:
                rec[nm] = np.nan
        rows.append(rec)
    t = pd.DataFrame(rows)
    keep = ["n_fit", "n_eval", "base_oe", "oe_i", "oe_s", "OE closed (intercept)",
            "OE closed (slope+int)", "base_ic", "ic_i", "ic_s", "intercept closed",
            "base_sl", "sl_s", "slope closed", "brier_s", "au_raw", "au_s",
            "base_nb50", "nb50_i", "nb50_win", "base_nb40", "nb40_i", "nb40_win"]
    ci = [c for c in t.columns if c.endswith(("_lo", "_hi"))
          and not c.startswith(("nb50_win", "nb40_win"))]
    t = t[keep + ci]
    C.save_table(t.round(4), "TableS23_recalibration_sample_size")
    C.log(t[keep].round(4).to_string(index=False))
    return t


# =============================================================================
# 3) 对照臂：去掉重复患者泄露
# =============================================================================
def strat_bootstrap_ids(y, n_target, rng):
    """分层有放回重采样，返回**原始行号**（可重复），用于分组交叉验证。"""
    out = []
    labs = np.asarray(y)
    for lab in np.unique(labs):
        sub = np.where(labs == lab)[0]
        k = int(round(n_target * (sub.size / labs.size)))
        out.append(rng.choice(sub, size=k, replace=True))
    return np.sort(np.concatenate(out))


def sec3_control_arm(Xd, yd, Xt, yt, Xg, yg,
                     pcts=(0.0, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00), R=10):
    C.section("3) 对照臂：泄露安全的重复患者分组交叉验证")
    grid = {"RF": {"n_estimators": [500], "max_depth": [None, 10],
                   "min_samples_leaf": [1, 2, 4]}}
    rows = []
    Tm = pd.concat([Xt, pd.Series(yt, name=C.TARGET, index=Xt.index)], axis=1)
    Dm = pd.concat([Xd, pd.Series(yd, name=C.TARGET, index=Xd.index)], axis=1)
    Xd_v, yd_v = Dm[C.FEATURES].reset_index(drop=True), Dm[C.TARGET].astype(int).values
    Xg_v = Xg.reset_index(drop=True)

    # 时间臂（无重复患者）：分组 CV 等价于普通 CV，重算一遍以保证代码路径一致
    for pct in pcts:
        for rep in range(R):
            rng = np.random.RandomState(2000 + rep + int(pct * 100) * 7)
            n_add = int(round(pct * len(Tm)))
            add = []
            for lab in np.unique(yt):
                sub = Tm.index[Tm[C.TARGET] == lab].values
                k = int(round(len(sub) * pct))
                add.append(rng.choice(sub, size=min(k, len(sub)), replace=False))
            add = np.sort(np.concatenate(add)) if len(np.concatenate(add)) else np.array([], int)
            pool = pd.concat([Dm, Tm.loc[add]], axis=0).reset_index(drop=True)
            Xp = pool[C.FEATURES]
            yp = pool[C.TARGET].astype(int).values
            gid = np.arange(len(yp))                       # 时间臂：患者唯一
            gs = GridSearchCV(C.build_estimator("RF", seed=SEED + rep),
                              C.prefix_params("RF", grid["RF"]), scoring="roc_auc",
                              cv=StratifiedGroupKFold(5, shuffle=True,
                                                      random_state=SEED + rep),
                              n_jobs=-1, refit=True)
            gs.fit(Xp, yp, groups=gid)
            best = gs.best_estimator_
            pg = best.predict_proba(Xg_v)[:, 1]
            rows.append({"arm": "temporal", "pct": pct, "repeat": rep,
                         "n_derive": len(yp), "cv_auc": float(gs.best_score_),
                         "geo_auc": float(roc_auc_score(yg, pg))})

    # 对照臂：有放回重采样到同样本量；重复患者用原始行号分组
    for pct in pcts:
        for rep in range(R):
            rng = np.random.RandomState(3000 + rep + int(pct * 100) * 7)
            n_add = int(round(pct * len(Tm)))
            n_target = len(Dm) + n_add
            ids = strat_bootstrap_ids(yd_v, n_target, rng)
            Xp = Xd_v.iloc[ids].reset_index(drop=True)
            yp = yd_v[ids]
            gid = ids                                     # 关键：分组变量 = 原患者行号
            for scheme in ("grouped", "naive"):
                cv = (StratifiedGroupKFold(5, shuffle=True, random_state=SEED + rep)
                      if scheme == "grouped"
                      else StratifiedKFold(5, shuffle=True, random_state=SEED + rep))
                gs = GridSearchCV(C.build_estimator("RF", seed=SEED + rep),
                                  C.prefix_params("RF", grid["RF"]),
                                  scoring="roc_auc", cv=cv, n_jobs=-1, refit=True)
                if scheme == "grouped":
                    gs.fit(Xp, yp, groups=gid)
                else:
                    gs.fit(Xp, yp)
                best = gs.best_estimator_
                pg = best.predict_proba(Xg_v)[:, 1]
                rows.append({"arm": f"control_{scheme}", "pct": pct, "repeat": rep,
                             "n_derive": len(yp), "cv_auc": float(gs.best_score_),
                             "geo_auc": float(roc_auc_score(yg, pg))})
    d = pd.DataFrame(rows)
    C.save_csv(d, "s16_control_arm")
    agg = (d.groupby(["arm", "pct"])
             .agg(n_derive=("n_derive", "mean"), cv_auc=("cv_auc", "mean"),
                  cv_auc_sd=("cv_auc", "std"), geo_auc=("geo_auc", "mean"),
                  geo_auc_sd=("geo_auc", "std"))
             .reset_index())
    agg["n_derive"] = agg["n_derive"].round(0).astype(int)
    piv = agg.pivot(index="pct", columns="arm",
                    values=["cv_auc", "geo_auc"]).round(4)
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    piv = piv.reset_index()
    piv["n_derive"] = agg[agg["arm"] == "temporal"].set_index("pct")["n_derive"].values
    C.save_table(piv, "TableS24_control_arm_grouped")
    C.log(piv.to_string(index=False))
    return d


# =============================================================================
# 4) Brier 的 Murphy 分解
# =============================================================================
def sec4_brier(preds, y_fz, yg):
    C.section("4) Brier 的 Murphy 分解（再校准前后）")
    rows = []
    for k, p in preds.items():
        y = y_fz if k.endswith("02") else np.asarray(yg, int)
        name = "RF (frozen, pipeline 02)" if k.endswith("02") else k
        p_cf_s, _ = crossfit(y, p, "slope")
        p_cf_i, _ = crossfit(y, p, "intercept")
        for stage, pp in [("before", p), ("after intercept-only", p_cf_i),
                          ("after slope+intercept", p_cf_s)]:
            d = brier_decomp(y, pp)
            rows.append({"Model": name, "stage": stage,
                         "Brier": round(C.calibration_metrics(y, pp)["Brier"], 4),
                         "uncertainty": round(d["uncertainty"], 4),
                         "resolution": round(d["resolution"], 4),
                         "reliability": round(d["reliability"], 4),
                         "decomposition residual": round(
                             d["brier_check"] - C.calibration_metrics(y, pp)["Brier"], 4)})
    t = pd.DataFrame(rows)
    C.save_table(t, "TableS25_brier_decomposition")
    C.log(t.to_string(index=False))
    return t


# =============================================================================
# 5) 28 对：效应量、95% CI、最小可检测差异、等价性
# =============================================================================
def sec5_equivalence(preds, yg, n_boot=2000, bound=0.05):
    C.section("5) 算法两两比较：效应量与最小可检测差异")
    y = np.asarray(yg, int)
    names = [k for k in A14.ORDER]
    P = {k: preds[k] for k in names}

    def paired_boot(p1, p2, n_iter=n_boot, seed=SEED):
        """与 common.paired_boot_diff 完全同种子同算法，额外返回 Bootstrap SE。"""
        n = len(y)
        obs = roc_auc_score(y, p1) - roc_auc_score(y, p2)
        rng = np.random.RandomState(seed)
        vals = np.full(n_iter, np.nan)
        for i in range(n_iter):
            idx = rng.choice(n, n, replace=True)
            if np.unique(y[idx]).size < 2:
                continue
            try:
                vals[i] = roc_auc_score(y[idx], p1[idx]) - roc_auc_score(y[idx], p2[idx])
            except Exception:
                vals[i] = np.nan
        vals = vals[~np.isnan(vals)]
        lo, hi = np.percentile(vals, [2.5, 97.5])
        p_val = 2 * min((vals <= 0).mean(), (vals >= 0).mean())
        return float(obs), float(lo), float(hi), float(p_val), float(vals.std(ddof=1))

    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            obs, lo, hi, p_val, se = paired_boot(P[a], P[b])
            rows.append({
                "Model A": a, "Model B": b,
                "AUC A": round(roc_auc_score(y, P[a]), 3),
                "AUC B": round(roc_auc_score(y, P[b]), 3),
                "Delta AUC (A-B)": round(obs, 3),
                "95% CI": f"{lo:.3f} to {hi:.3f}",
                "p (paired bootstrap)": round(p_val, 4),
                "paired SE": round(se, 4),
                "MDD (80% power)": round(Z_POW * se, 3),
                "equivalent within ±0.05": "yes" if (lo > -bound and hi < bound) else "no",
            })
    t = pd.DataFrame(rows)
    # Holm（与脚本 14 同一算法）
    m = len(t)
    order = np.argsort(t["p (paired bootstrap)"].values)
    adj = np.empty(m, dtype=bool)
    prev = False
    for rank, idx in enumerate(order):
        thresh = 0.05 / (m - rank)
        prev = prev or (t["p (paired bootstrap)"].values[idx] < thresh)
        adj[idx] = prev
    t["Holm significant (0.05)"] = np.where(adj, "yes", "no")
    C.save_table(t, "TableS26_pairwise_equivalence")
    # 同一口径、CI 写清楚的 Table13（替代原表，数字与脚本 14 一致）
    C.save_table(t[["Model A", "Model B", "AUC A", "AUC B", "Delta AUC (A-B)",
                    "95% CI", "p (paired bootstrap)",
                    "Holm significant (0.05)"]].copy(),
                 "Table13_algo_pairwise_auc_diff")

    mdd = t["MDD (80% power)"]
    C.log(f"  28 对的 MDD：中位数 {mdd.median():.3f}，"
          f"范围 {mdd.min():.3f}-{mdd.max():.3f}")
    C.log(f"  名义显著 p<0.05 的对数 = "
          f"{(t['p (paired bootstrap)'] < 0.05).sum()}；"
          f"Holm 后显著 = {(t['Holm significant (0.05)'] == 'yes').sum()}")
    C.log(f"  在 ±{bound} 等价界内可判定等价的配对数 = "
          f"{(t['equivalent within ±0.05'] == 'yes').sum()} / {len(t)}")
    return t


# =============================================================================
# 6) 时间队列的中心构成（估计）
# =============================================================================
MAP = {'age': 'age', 'gender': 'gender', 'systolic blood pressure': 'sbp',
       'diastolic blood pressure': 'dbp', 'mean arterial pressure': 'map',
       'pulse pressure index': 'ppi', 'pulse pressure': 'pp',
       'hypertension': 'htn', 'diabetes': 'dm', 'siabetes': 'dm',
       'smoking': 'smk', 'dyslipidemia': 'dys', 'total cholesterol': 'tc',
       'triglycerides': 'tg', 'LDL-C': 'ldl', 'HDL-C': 'hdl',
       'uric acid': 'ua', 'plasma fibrinogen': 'fib', 'creatinine': 'cr',
       'urea': 'urea'}
KEYS = ['age', 'gender', 'sbp', 'dbp', 'map', 'ppi', 'pp', 'htn', 'dm', 'smk',
        'dys', 'tc', 'tg', 'ldl', 'hdl', 'ua', 'fib', 'cr', 'urea']


def _read_any(p):
    for enc in ['gbk', 'utf-8-sig', 'utf-8']:
        try:
            return pd.read_csv(p, encoding=enc)
        except Exception:
            pass
    raise RuntimeError(p)


def _tidy_keys(df, lc):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={c: MAP[c] for c in df.columns if c in MAP})
    o = df[KEYS].apply(pd.to_numeric, errors='coerce')
    ys = pd.to_numeric(df[lc], errors='coerce')
    if set(ys.dropna().unique()) <= {1, 2}:
        ys = ys.map({1: 0, 2: 1})
    o['y'] = ys.astype(int)
    return o


def _fp(df):
    return (df[KEYS + ['y']].round(2).astype(object).fillna('NA').astype(str)
            .agg('|'.join, axis=1))


def sec6_temporal_centre(cohorts, Xt, yt, Xg):
    C.section("6) 时间队列的中心构成（估计，探索性）")
    o = cohorts["original"].copy()
    wb = openpyxl.load_workbook(os.path.join(C.DATA_DIR, "original data.xlsx"),
                                read_only=True, data_only=True)
    rows = list(wb["data20221010"].iter_rows(values_only=True))
    raw = pd.DataFrame(rows[1:], columns=[str(c).strip() for c in rows[0]]).dropna(how="all")
    fr = _fp(_tidy_keys(raw, "label"))
    se = set(_fp(_tidy_keys(_read_any(r"E:\CAD-RADS\旧数据\NHFE2017-2022.csv"), "label")))
    sf = set(_fp(_tidy_keys(_read_any(r"E:\CAD-RADS\旧数据\NHFY.csv"), "label")))
    lab = np.where(fr.isin(se).values, 0, np.where(fr.isin(sf).values, 1, -1))
    C.log(f"  中心归属：中心 1 = {(lab == 0).sum()}，中心 2 = {(lab == 1).sum()}，"
          f"无法归属 = {(lab == -1).sum()}")

    Xo = o[C.FEATURES].reset_index(drop=True)
    ok = lab >= 0
    lr = Pipeline([("imp", SimpleImputer(strategy="median")),
                   ("scl", StandardScaler()),
                   ("clf", LogisticRegression(max_iter=2000))])
    cvp = np.full(ok.sum(), np.nan)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    yo = lab[ok]
    Xok = Xo.loc[ok].reset_index(drop=True)
    for tr, te in skf.split(Xok, yo):
        m = Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("scl", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=2000))])
        m.fit(Xok.iloc[tr], yo[tr])
        cvp[te] = m.predict_proba(Xok.iloc[te])[:, 1]
    sep_auc = roc_auc_score(yo, cvp)

    # 中心 1（推导期）与时间队列（中心 1 后期）的逐变量对比 —— 同中心的时间漂移描述
    C1 = o[lab == 0].reset_index(drop=True)
    Tf = cohorts["temporal"]          # 时间队列完整框（含 label）
    rows = []
    for v in C.FEATURES:
        a = pd.to_numeric(C1[v], errors="coerce").astype(float)
        b = pd.to_numeric(Tf[v], errors="coerce").astype(float)
        med_a, med_b = a.median(), b.median()
        sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        smd = float((a.mean() - b.mean()) / sd) if sd > 0 else np.nan
        rows.append({
            "Variable": v,
            "Variable (full name)": C.LABEL_EN.get(v, v),
            "Centre 1, 2017-2022, median (IQR)": f"{med_a:.2f} "
                f"({a.quantile(.25):.2f}-{a.quantile(.75):.2f})",
            "Temporal, 2023-2026, median (IQR)": f"{med_b:.2f} "
                f"({b.quantile(.25):.2f}-{b.quantile(.75):.2f})",
            "SMD (Centre 1 vs Temporal)": round(smd, 3) if np.isfinite(smd) else np.nan,
        })
    # 结局（事件率）
    pa = float(pd.to_numeric(C1[C.TARGET], errors="coerce").mean())
    pb = float(pd.to_numeric(Tf[C.TARGET], errors="coerce").mean())
    rows.insert(0, {
        "Variable": "obstructive (outcome)",
        "Variable (full name)": "Obstructive CAD (CAD-RADS 3-5)",
        "Centre 1, 2017-2022, median (IQR)": f"{pa*100:.1f}%",
        "Temporal, 2023-2026, median (IQR)": f"{pb*100:.1f}%",
        "SMD (Centre 1 vs Temporal)": round(C.smd_bin(pa, pb), 3),
    })
    C.save_table(pd.DataFrame(rows), "TableS27_temporal_centre_mixture")

    # 时间队列与两中心的 SMD（保留，供描述性对比）
    smd_rows = []
    for nm, d in [("centre 1", o[lab == 0]), ("centre 2", o[lab == 1])]:
        for v in C.FEATURES:
            a = pd.to_numeric(Xt[v], errors="coerce").astype(float)
            b = pd.to_numeric(d[v], errors="coerce").astype(float)
            sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
            smd_rows.append({"comparison": f"temporal vs {nm}", "variable": v,
                             "SMD": round(float((a.mean() - b.mean()) / sd), 3)
                             if sd > 0 else np.nan})
    smd = pd.DataFrame(smd_rows)
    C.save_csv(smd, "s16_temporal_vs_centre_smd")
    C.log("  时间队列与两中心差异最大的变量（|SMD| 前 5）：")
    C.log(smd.reindex(smd["SMD"].abs().sort_values(ascending=False).index)
          .head(5).to_string(index=False))
    C.log("  已改：表 S27 现为「中心 1 推导期 vs 中心 1 后期（时间队列）」的基线对比，"
          "用于描述同一中心内部的时间漂移；中心 1/2 可分性 AUC = %.3f（仅作参照）。"
          % sep_auc)
    return np.array([]), np.array([])


# =============================================================================
def main():
    C.log_init("16_revision")
    t0 = time.time()
    cohorts, _ = C.load_cohorts()
    Xd, yd = C.derivation_label(cohorts["original"])
    Xt, yt = C.derivation_label(cohorts["temporal"])
    Xg, yg = C.derivation_label(cohorts["geo"])
    C.log(f"  推导 n={len(yd)}（事件 {int(yd.sum())}）| 时间 n={len(yt)} | "
          f"地理 n={len(yg)}（事件 {int(yg.sum())}）")

    preds, y_fz, tuned = sec1_global_vs_crossfit(Xd, yd, Xg, yg, None)
    sec2_recal_ladder(y_fz, preds["RF_frozen02"])
    # 对照臂（审稿意见 4）改由 16b_control_arm.py 单独产出 TableS24。
    # 原因：本文件的 sec3 每格都跑 GridSearchCV，7 档 × 10 重复 × 3 臂 ≈ 6500 次
    # RF 拟合（实测 27 分钟），且 naive 与 grouped 两臂混入了不同超参，
    # 无法把差异干净地归因于 CV 的折叠方式。16b 固定主线超参、只改折叠方式，
    # 既快又更干净。sec3_control_arm 保留在本文件中以便复现与审计，
    # 但不再由 main 调用。
    # sec3_control_arm(Xd, yd, Xt, yt, Xg, yg)
    sec4_brier(preds, y_fz, yg)
    sec5_equivalence(preds, yg)
    sec6_temporal_centre(cohorts, Xt, yt, Xg)

    C.section(f"脚本 16 完成，总耗时 {C.timer(t0)}")


if __name__ == "__main__":
    main()
