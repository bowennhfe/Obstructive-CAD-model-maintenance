# -*- coding: utf-8 -*-
"""
18 —— 尺度分量内部的两级可修复性：水平便宜、离散程度昂贵
=============================================================================
回应审稿意见 3 与 5 的补充诊断。核心问句：
    "再多给目标中心一些病例，风险尺度的'离散程度'（校准斜率）能不能修好？"

设计（与脚本 16 的 Table S23 同一拆分口径）：
    重复 600 次：把地理队列(177)分层随机拆成再校准池(118) 与 评价集(59)；
    在池内再分层抽取 m 例拟合「仅截距」与「斜率+截距」两种再校准；
    在评价集上评价；记录
      · 拟合得到的斜率 b1 的分布（稳定性）
      · 评价集上校准斜率的**中位数**（是否去偏）与**平均绝对偏差**（是否更准）
      · O:E 距 1 的距离
m = 20, 40, 60, 80, 100, 118。

输出
    Tables/TableS31_recalibration_slope_spread.csv
=============================================================================
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import importlib
M16 = importlib.import_module("16_revision_analyses")

SEED = C.SEED


def main():
    C.log_init("18_recal_spread")
    t0 = time.time()
    fz = pd.read_csv(os.path.join(C.OUT_DIR, "s02_geo_pred_rf.csv"))
    y = fz["y"].astype(int).values
    p = fz["p_rf"].values
    lo = M16.logit(p)
    rng = np.random.RandomState(SEED)
    idx1, idx0 = np.where(y == 1)[0], np.where(y == 0)[0]
    n_eval, n_pool = 59, 118
    rows = []
    for m in (20, 40, 60, 80, 100, 118):
        b1s, sl_r, sl_0, oe_r, ae_r = [], [], [], [], []
        for _ in range(600):
            e1 = rng.choice(idx1, int(round(n_eval * y.mean())), replace=False)
            e0 = rng.choice(idx0, n_eval - int(round(n_eval * y.mean())), replace=False)
            ev = np.concatenate([e1, e0])
            pool = np.setdiff1d(np.arange(len(y)), ev)
            p1, p0 = pool[y[pool] == 1], pool[y[pool] == 0]
            m1, m0 = int(round(m * y.mean())), m - int(round(m * y.mean()))
            fit = np.concatenate([rng.choice(p1, m1, replace=False),
                                  rng.choice(p0, m0, replace=False)])
            a = M16.fit_intercept(y[fit], lo[fit])
            b0, b1 = M16.fit_slope_intercept(y[fit], lo[fit])
            if not np.isfinite(b1) or b1 <= 0:
                continue
            b1s.append(b1)
            mr = M16.metrics(y[ev], M16.apply_slope_intercept(b0, b1, lo[ev]))
            m0_ = M16.metrics(y[ev], M16.apply_intercept(a, lo[ev]))
            mb = M16.metrics(y[ev], p[ev])
            sl_r.append(mr["slope"]); sl_0.append(mb["slope"])
            oe_r.append(mr["O_E"])
            ae_r.append(abs(mr["AUC"] - mb["AUC"]))
        b1s = np.array(b1s); sl_r = np.array(sl_r); sl_0 = np.array(sl_0)
        rows.append({
            "n_fit": m, "n_eval": n_eval,
            "fitted slope, median": round(float(np.median(b1s)), 2),
            "fitted slope, 2.5th pct": round(float(np.percentile(b1s, 2.5)), 2),
            "fitted slope, 97.5th pct": round(float(np.percentile(b1s, 97.5)), 2),
            "eval slope, no correction (median)": round(float(np.median(sl_0)), 3),
            "eval slope, after slope+intercept (median)": round(float(np.median(sl_r)), 3),
            "mean |1 - slope|, no correction": round(float(np.abs(sl_0 - 1).mean()), 3),
            "mean |1 - slope|, after slope+intercept": round(float(np.abs(sl_r - 1).mean()), 3),
            "O:E distance, after slope+intercept": round(
                float(np.abs(np.array(oe_r) - 1).mean()), 4),
            "mean |ΔAUC| after recalibration": round(float(np.mean(ae_r)), 6),
        })
    t = pd.DataFrame(rows)
    C.save_table(t, "TableS31_recalibration_slope_spread")
    C.log(t.to_string(index=False))
    a = t.loc[t["n_fit"] == 40].iloc[0]
    b = t.loc[t["n_fit"] == 118].iloc[0]
    C.log(f"  m=40：拟合斜率 2.5-97.5 分位 {a['fitted slope, 2.5th pct']}-"
          f"{a['fitted slope, 97.5th pct']}；评价集斜率中位 "
          f"{a['eval slope, after slope+intercept (median)']}（不校正 "
          f"{a['eval slope, no correction (median)']}）；平均绝对偏差 "
          f"{a['mean |1 - slope|, after slope+intercept']} vs "
          f"{a['mean |1 - slope|, no correction']}")
    C.log(f"  m=118：拟合斜率 2.5-97.5 分位 {b['fitted slope, 2.5th pct']}-"
          f"{b['fitted slope, 97.5th pct']}；评价集斜率中位 "
          f"{b['eval slope, after slope+intercept (median)']}；平均绝对偏差 "
          f"{b['mean |1 - slope|, after slope+intercept']} vs "
          f"{b['mean |1 - slope|, no correction']}")
    C.section(f"脚本 18 完成，总耗时 {C.timer(t0)}")


if __name__ == "__main__":
    main()
