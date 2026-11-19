# -*- coding: utf-8 -*-
"""
23 —— 患者层面配对 Bootstrap（第十二轮审稿 Major 5 / Minor 8）
=============================================================================
目的
    阶梯分析把 10 次重复实验当作独立样本，p 值因此偏窄（伪重复）。本脚本改为
    在外部队列（n = 177）上做**患者层面配对 Bootstrap**，直接给出区间估计。

比较对象（每个对象 = 地理队列 177 例上的一个固定预测向量）
    L0    阶梯 0%  步（原队列 442，10 次重复的预测概率平均）
    L100  阶梯 100% 步（全并 806，10 次重复的预测概率平均）
    F     冻结模型（Models/m01_rf_original.pkl，预先指定的 10 折网格搜索）

配对方式
    同一次 Bootstrap 重抽同一批患者索引，分别对两个模型计算指标后取差；
    因此差值保留了患者层面的配对结构，重复实验之间的相关性被自动吸收
    （重复只在构造 L0 / L100 的预测时进入，不再是独立样本单位）。
    ★ 这里的 Bootstrap 仅用于**区间估计**，不用于任何稳定性论证。

指标
    AUC、校准斜率、整体校准截距、O:E、ICI、Brier
    净获益 @ 0.35 / 0.40 / 0.50 / 0.55

输出
    Tables/TableS37_paired_bootstrap.csv
    Outputs/s23_paired_boot_detail.csv
=============================================================================
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

B = 2000
NBS = (0.35, 0.40, 0.50, 0.55)


def net_benefit(y, p, thr):
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    n = len(y)
    pred = p >= thr
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    return tp / n - fp / n * (thr / (1 - thr))


def metrics(y, p):
    cm = C.calibration_metrics(y, p)
    out = {
        "AUC": float(roc_auc_score(y, p)),
        "slope": float(cm["Calibration slope"]),
        "intercept": float(cm["Calibration intercept"]),
        "O:E": float(cm["O:E"]),
        "ICI": float(cm["ICI"]),
        "Brier": float(cm["Brier"]),
    }
    for t in NBS:
        out[f"NB@{t:.2f}"] = float(net_benefit(y, p, t))
    return out


def paired_bootstrap(y, p_before, p_after, n_iter=B, seed=C.SEED):
    """患者层面配对 Bootstrap：返回 (obs_diff, lo, hi, p)。"""
    y = np.asarray(y, int)
    p_before = np.asarray(p_before, float)
    p_after = np.asarray(p_after, float)
    n = len(y)

    m0 = metrics(y, p_before)
    m1 = metrics(y, p_after)
    keys = list(m0.keys())
    obs = {k: m1[k] - m0[k] for k in keys}

    rng = np.random.RandomState(seed)
    draws = {k: np.full(n_iter, np.nan) for k in keys}
    for i in range(n_iter):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            a = metrics(y[idx], p_before[idx])
            b = metrics(y[idx], p_after[idx])
        except Exception:
            continue
        for k in keys:
            draws[k][i] = b[k] - a[k]

    res = {}
    for k in keys:
        v = draws[k]
        v = v[~np.isnan(v)]
        if len(v) < 50:
            res[k] = (obs[k], np.nan, np.nan, np.nan)
            continue
        lo, hi = np.percentile(v, [2.5, 97.5])
        p = 2 * min((v <= 0).mean(), (v >= 0).mean())
        res[k] = (float(obs[k]), float(lo), float(hi), float(min(p, 1.0)))
    return m0, m1, res


def main():
    C.log_init("23_paired_bootstrap")

    cohorts, _ = C.load_cohorts(verbose=False)
    Xg, yg = C.derivation_label(cohorts["geo"])
    yg = np.asarray(yg, int)

    # ---- 固定预测向量 ----
    g = pd.read_pickle(os.path.join(C.OUT_DIR, "s03_growth_full.pkl"))

    def ladder_avg(pct):
        sub = g[(g["arm"] == "temporal") & (np.isclose(g["pct_temporal"], pct))]
        P = np.array([np.asarray(p, float) for p in sub["p_geo"]])
        ys = [np.asarray(y, int) for y in sub["y_geo"]]
        assert all(np.array_equal(ys[0], y) for y in ys)
        return P.mean(axis=0), ys[0]

    p_l0, y_l0 = ladder_avg(0.00)
    p_l100, y_l100 = ladder_avg(1.00)
    assert np.array_equal(y_l0, yg) and np.array_equal(y_l100, yg)

    m01 = C.load_pickle("m01_rf_original.pkl")
    p_fro = m01["pipeline"].predict_proba(Xg)[:, 1]

    # 对照臂 100%（同样本量、无新信息）
    subc = g[(g["arm"] == "control") & (np.isclose(g["pct_temporal"], 1.00))]
    p_ctl = np.array([np.asarray(p, float) for p in subc["p_geo"]]).mean(axis=0)

    PAIRS = [
        ("ladder0_vs_ladder100", p_l0, p_l100,
         "Ladder 0% -> 100% (the comparison reported in Section 3.7)"),
        ("frozen_vs_ladder100", p_fro, p_l100,
         "Frozen model (Section 3.3) -> fully updated model"),
        ("frozen_vs_ladder0", p_fro, p_l0,
         "Frozen model (Section 3.3) -> ladder 0% step"),
    ]

    all_rows = []
    detail = {}
    for tag, pb, pa, label in PAIRS:
        m0, m1, res = paired_bootstrap(yg, pb, pa)
        C.section(f"{tag}  ({label})")
        C.log(f"{'metric':>10s} {'before':>9s} {'after':>9s} {'change':>9s} "
              f"{'95% CI':>19s} {'p':>8s}")
        for k in m0:
            d, lo, hi, p = res[k]
            C.log(f"{k:>10s} {m0[k]:9.4f} {m1[k]:9.4f} {d:+9.4f} "
                  f"[{lo:7.4f},{hi:7.4f}] {p:8.4f}")
            all_rows.append({
                "comparison": tag,
                "comparison_label": label,
                "metric": k,
                "before": round(m0[k], 4),
                "after": round(m1[k], 4),
                "change": round(d, 4),
                "ci_lo": round(lo, 4) if np.isfinite(lo) else np.nan,
                "ci_hi": round(hi, 4) if np.isfinite(hi) else np.nan,
                "p_boot": round(p, 4) if np.isfinite(p) else np.nan,
                "n_patients": int(len(yg)),
                "n_events": int(yg.sum()),
                "n_bootstrap": B,
            })
        detail[tag] = {"before": m0, "after": m1, "diff": res}

    ts37 = pd.DataFrame(all_rows)
    C.save_table(ts37, "TableS37_paired_bootstrap")

    # ------------------------------------------------------------------
    # 主表 8：把「重复实验内」的端点检验移入补充表 S39，
    #         主表改报患者层面配对 Bootstrap 的 95% 区间（Major 5）
    # ------------------------------------------------------------------
    t8_path = os.path.join(C.TABLE_DIR, "Table8_growth_endpoint_tests.csv")
    t8 = pd.read_csv(t8_path)
    pcols = [c for c in ("p_mannwhitney", "p_paired", "p_mixed")
             if c in t8.columns]
    if pcols:
        C.section("端点检验迁移：Table 8 -> 主表报区间，Table S13 保留重复实验内检验")
        C.save_table(t8[["metric"] + pcols].copy(), "TableS39_ladder_endpoint_tests")
        boot_l0 = detail["ladder0_vs_ladder100"]["diff"]
        rows8 = []
        for _, r in t8.iterrows():
            _obs, lo, hi, _p = boot_l0[r["metric"]]
            rows8.append({
                "metric": r["metric"],
                "baseline": round(float(r["baseline"]), 4),
                "final": round(float(r["final"]), 4),
                "change": round(float(r["change"]), 4),
                "ci_lo": round(lo, 4),
                "ci_hi": round(hi, 4),
            })
        C.save_table(pd.DataFrame(rows8), "Table8_growth_endpoint_tests")
        C.log(pd.DataFrame(rows8).to_string(index=False))
    else:
        C.log("Table 8 已为区间口径，跳过迁移（幂等）")

    # 供正文引用的关键数字
    key = {}
    for tag, _, _, _ in PAIRS:
        d = detail[tag]["diff"]
        for k, (obs, lo, hi, p) in d.items():
            key[f"{tag}|{k}"] = {
                "obs": round(obs, 4),
                "lo": round(lo, 4) if np.isfinite(lo) else None,
                "hi": round(hi, 4) if np.isfinite(hi) else None,
                "p": round(p, 4) if np.isfinite(p) else None,
            }
    key["_aucs"] = {
        "frozen": round(float(roc_auc_score(yg, p_fro)), 4),
        "ladder0": round(float(roc_auc_score(yg, p_l0)), 4),
        "ladder100": round(float(roc_auc_score(yg, p_l100)), 4),
        "control100": round(float(roc_auc_score(yg, p_ctl)), 4),
    }
    import json
    with open(os.path.join(C.OUT_DIR, "s23_paired_boot_key.json"), "w",
              encoding="utf-8") as f:
        json.dump(key, f, ensure_ascii=False, indent=2)
    C.log("\nkey -> Outputs/s23_paired_boot_key.json")
    C.log(json.dumps(key["_aucs"], ensure_ascii=False))


if __name__ == "__main__":
    main()
