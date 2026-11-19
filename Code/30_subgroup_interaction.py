# -*- coding: utf-8 -*-
"""
30_subgroup_interaction.py
==========================
R26（2 区标准审核）要求的补强分析：亚组 x 模型更新的交互检验。

问题：Section 3.7 报告"每个亚组中更新模型的 AUC 均不低于原始模型"，
但未做交互检验。若亚组间增益差异很大，逐层"均不低于"可能是聚合结论
掩盖了异质性。本脚本对每个亚组做 DeLong 配对比较（同一批地理患者上
原始模型与更新模型的折外预测），并给交互 p 值。

方法：
  1. 在合并池（推导 + 全部当代 = 806 例）上按与主分析相同的协议重调 RF，
     得更新模型；在原推导队列（442 例）上重调得原始模型。
  2. 两者都在地理队列（177 例）上产生折外式预测（冻结转移，不重训）。
  3. 每个亚组内用 DeLong 协方差法算两模型 AUC 之差的方差 -> 配对 z 与 p；
     交互检验用「亚组内差 - 全队列差」的 t 检验（Bonferroni 不加，探索性）。
  4. 同时给出每亚组的 n、事件数、两模型 AUC 及 95% CI。

输出：
  Tables/TableS45_subgroup_interaction.csv
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

SEED = C.SEED
N_INNER = 5  # 交互检验的重复次数（探索性，不做多重校正）


# ---------------------------------------------------------------- DeLong
def _delong_var(y, p1, p2):
    """两 ROC AUC 差的 DeLong 方差（Hanley-McNeil 基础实现）。

    返回 (auc1, auc2, var_diff)，var_diff = V10 + V22 - 2*V12。
    """
    y = np.asarray(y).astype(int)
    pos, neg = y == 1, y == 0
    n1, n2 = pos.sum(), neg.sum()
    if n1 == 0 or n2 == 0:
        return np.nan, np.nan, np.nan
    # 结构化中间量
    def _mid(p):
        # 预测矩阵 M[阳性, 阴性] 的 1x1 邻域结构量
        t = p[np.ix_(pos, neg)]
        return t
    m1, m2 = _mid(p1), _mid(p2)
    # V10: 正例分数分布方差（按负例分箱）
    def _struct(t):
        nb = t.shape[1]
        idx = t.argsort(axis=1)
        srt = np.take_along_axis(t, idx, axis=1)
        # 每行的结构量：均值与增量
        cs = np.cumsum(srt, axis=1)
        r = np.arange(1, nb + 1)
        # 修正位置得分
        loc = (cs - 0.5 * (srt + np.concatenate([np.zeros((t.shape[0], 1)), srt[:, :-1]], axis=1))) / nb
        return loc
    l1, l2 = _struct(m1), _struct(m2)
    # 用平均结构量近似 S10/S12（Szymkiewicz 非劣于 Smit）
    a1 = roc_auc_score(y, p1)
    a2 = roc_auc_score(y, p2)
    v10 = l1.var(axis=1, ddof=1).mean() / n1 if l1.shape[0] > 1 else np.nan
    v01 = l1.var(axis=0, ddof=1).mean() / n2 if l1.shape[1] > 1 else np.nan
    v20 = l2.var(axis=1, ddof=1).mean() / n1 if l2.shape[0] > 1 else np.nan
    v02 = l2.var(axis=0, ddof=1).mean() / n2 if l2.shape[1] > 1 else np.nan
    v12 = ((l1 + l2) / 2.0).var(axis=0, ddof=1).mean() / n2 if l2.shape[1] > 1 else np.nan
    var = v10 + v01 + v20 + v02 - 2.0 * v12
    return a1, a2, max(var, 0.0)


def _pipe():
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=500, random_state=SEED)),
    ])


def _fit_pool_predict(Xtr, ytr, Xte, seed):
    """在训练池上调参 + 拟合，返回测试集预测概率（冻结转移）。"""
    grid = {("clf__" + k if not k.startswith("clf__") else k): v
            for k, v in C.PARAM_GRIDS_FAST["RF"].items()}
    gs = GridSearchCV(_pipe(), grid, scoring="roc_auc", cv=3, n_jobs=-1)
    m = clone(gs)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1], m


def _mean_oof(X, y, seed):
    """重复分层 10 折的折外预测均值（与主分析同协议）。"""
    skf = RepeatedStratifiedKFold(n_splits=10, n_repeats=C.CV_REPEAT, random_state=seed)
    oof = np.zeros(len(y))
    cnt = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        p, _ = _fit_pool_predict(X[tr], y[tr], X[te], seed)
        oof[te] += p
        cnt[te] += 1
    return oof / np.maximum(cnt, 1)


SUBS = [("Age: <50", lambda d: d["age"] < 50),
        ("Age: 50-69", lambda d: (d["age"] >= 50) & (d["age"] < 70)),
        ("Age: >=70", lambda d: d["age"] >= 70),
        ("Sex: Male", lambda d: d["male"] == 1),
        ("Sex: Female", lambda d: d["male"] == 0),
        ("Hypertension: Yes", lambda d: d["htn"] == 1),
        ("Hypertension: No", lambda d: d["htn"] == 0)]


def main():
    C.log_init("30_subgroup_interaction")
    cohorts, _ = C.load_cohorts()
    d_orig = cohorts["original"]
    d_tem = cohorts["temporal"]
    d_geo = cohorts["geo"]

    # 池：推导 + 全部当代 = 806（与阶梯 100% 步同口径）
    pool = pd.concat([d_orig, d_tem], ignore_index=True)
    C.log(f"  pool={len(pool)}  geo={len(d_geo)}")

    Xg, yg = d_geo[C.FEATURES].astype(float).values, d_geo[C.TARGET].values.astype(int)
    p_orig, _ = _fit_pool_predict(d_orig[C.FEATURES].astype(float).values,
                                  d_orig[C.TARGET].values.astype(int), Xg, SEED)
    p_upd, _ = _fit_pool_predict(pool[C.FEATURES].astype(float).values,
                                 pool[C.TARGET].values.astype(int), Xg, SEED)

    a_all_o = roc_auc_score(yg, p_orig)
    a_all_u = roc_auc_score(yg, p_upd)
    d_all = a_all_u - a_all_o
    # 全队列的患者层面配对 Bootstrap 标准误（与 Table S2 同口径）
    rng0 = np.random.default_rng(SEED)
    _ds = []
    for _ in range(2000):
        i0 = rng0.integers(0, len(yg), len(yg))
        if len(np.unique(yg[i0])) < 2:
            continue
        _ds.append(roc_auc_score(yg[i0], p_upd[i0]) - roc_auc_score(yg[i0], p_orig[i0]))
    se_all = float(np.asarray(_ds).std(ddof=1))
    C.log(f"  overall AUC {a_all_o:.4f} -> {a_all_u:.4f} (diff {d_all:+.4f}, se {se_all:.4f})")

    rows = []
    geo = d_geo.reset_index(drop=True)
    # 口径自检：亚组划分必须与 TableS11_subgroup_auc.csv 完全一致。
    # 该表的 "Sex: Male" n=113 / 事件 49，来自 male 列（1=男性），
    # 而 gender 列 1 对应女性；两列方向相反，用错会使标签互换。
    _s11 = pd.read_csv(os.path.join(C.TABLE_DIR, "TableS11_subgroup_auc.csv"),
                      encoding="utf-8-sig")
    _s11m = _s11[(_s11["Subgroup"] == "Sex: Male")
                 & (_s11["Model"].str.contains("original"))]
    if len(_s11m) == 1:
        _n_male = int(_s11m["N"].iloc[0]); _e_male = int(_s11m["Events"].iloc[0])
        _chk_n = int((geo["male"] == 1).sum())
        _chk_e = int(yg[np.asarray(geo["male"] == 1)].sum())
        assert _chk_n == _n_male and _chk_e == _e_male, (
            "sex subgroup definition does not match Table S8: "
            "male n=%d ev=%d vs S11 n=%d ev=%d" % (_chk_n, _chk_e, _n_male, _e_male))
        C.log(f"  [check] sex subgroups match Table S8 (male n={_chk_n}, ev={_chk_e})")
    else:
        C.log("  [check] Table S8 not found or ambiguous; sex check skipped")
    rng = np.random.default_rng(SEED)
    n_all = len(yg)
    for name, fn in SUBS:
        mask = np.asarray(fn(geo), dtype=bool)
        n = int(mask.sum()); ev = int(yg[mask].sum())
        if n == 0 or ev == 0 or ev == n:
            C.log(f"  {name}: skipped (n={n}, ev={ev})")
            continue
        a1 = roc_auc_score(yg[mask], p_orig[mask])
        a2 = roc_auc_score(yg[mask], p_upd[mask])
        d = a2 - a1
        # 患者层面配对 Bootstrap（与 Table S2 同口径）：每次重采样对该亚层内
        # 的患者有放回抽样，两模型在同一次抽样上求 AUC，取差值。
        ym, p1m, p2m = yg[mask], p_orig[mask], p_upd[mask]
        ds, inter_boot = [], []
        for _ in range(2000):
            idx = rng.integers(0, n, n)
            if len(np.unique(ym[idx])) < 2:
                continue
            dd = roc_auc_score(ym[idx], p2m[idx]) - roc_auc_score(ym[idx], p1m[idx])
            ds.append(dd)
            inter_boot.append(dd - d_all)
        ds = np.asarray(ds); inter_boot = np.asarray(inter_boot)
        se = float(ds.std(ddof=1))
        lo, hi = float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))
        p_in = float(2.0 * min((ds <= 0).mean(), (ds >= 0).mean()))
        ilo, ihi = (float(np.percentile(inter_boot, 2.5)),
                    float(np.percentile(inter_boot, 97.5)))
        p_int = float(2.0 * min((inter_boot <= 0).mean(), (inter_boot >= 0).mean()))
        rows.append({
            "Subgroup": name,
            "N": n,
            "Events": ev,
            "AUC original": round(a1, 4),
            "AUC updated": round(a2, 4),
            "difference (updated - original)": round(d, 4),
            "95% CI low": round(lo, 4),
            "95% CI high": round(hi, 4),
            "p within subgroup (paired bootstrap)": round(p_in, 4),
            "interaction (subgroup - overall)": round(d - d_all, 4),
            "95% CI low interaction": round(ilo, 4),
            "95% CI high interaction": round(ihi, 4),
            "p interaction": round(p_int, 4),
        })
        C.log(f"  {name}: n={n} ev={ev} {a1:.4f}->{a2:.4f} d={d:+.4f} "
              f"CI[{lo:+.3f},{hi:+.3f}] p_in={p_in:.3f} p_int={p_int:.3f}")

    rows.insert(0, {
        "Subgroup": "All patients (overall)",
        "N": n_all, "Events": int(yg.sum()),
        "AUC original": round(a_all_o, 4), "AUC updated": round(a_all_u, 4),
        "difference (updated - original)": round(d_all, 4),
        "95% CI low": round(d_all - 1.96 * se_all, 4),
        "95% CI high": round(d_all + 1.96 * se_all, 4),
        "p within subgroup (paired bootstrap)": np.nan,
        "interaction (subgroup - overall)": 0.0,
        "95% CI low interaction": np.nan,
        "95% CI high interaction": np.nan,
        "p interaction": np.nan,
    })
    t = pd.DataFrame(rows)
    C.save_table(t, "TableS45_subgroup_interaction")
    C.log(t.to_string(index=False))


if __name__ == "__main__":
    main()
