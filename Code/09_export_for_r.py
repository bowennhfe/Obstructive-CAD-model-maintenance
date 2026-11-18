# -*- coding: utf-8 -*-
"""
09_export_for_r.py — 为 R 临床图脚本导出输入数据
=============================================================================
用途
    「用 R 参照旧记录 07_Clinical_Plots_R.R 出关键临床图」所需的最小数据集。
    本脚本**不拟合任何模型**，只把既有产物（01/02/03）整理成 R 友好的长表：

        Outputs/r_input/analysis_set.csv    逐例预测概率 + 基线协变量（长表）
        Outputs/r_input/growth_steps.csv    数据积累阶梯的逐步指标
        Outputs/r_input/risk_cutoffs.csv    风险分层截断值（推导队列袋外三分位）

    队列/模型分组（R 侧 group 变量）
        Derivation      / CV (out-of-fold)   推导队列 10 折交叉验证袋外概率
        Geographical    / Original model     地理队列，模型只用 original 442 例拟合
        Geographical    / Updated model      地理队列，模型用 806 例（含同期数据）拟合
        Geographical    / Control arm        地理队列，原文队同规模重抽样对照臂

★ 说明：Updated / Control 均取第 1 次重复（repeat = 0）作为一次真实实现，
   与正文 Figure 2 的多队列 ROC 完全同源（同一批预测概率）。
=============================================================================
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

import common as C

ENDPOINT_PCT = 1.00
REPEAT = 0
R_IN = os.path.join(C.OUT_DIR, "r_input")


def _covariates(df):
    out = pd.DataFrame({
        "id": df["ID"].values if "ID" in df.columns else np.arange(len(df)),
        "y": df[C.TARGET].astype(int).values,
        "age": df["age"].values,
        "female": df["gender"].values,
        "htn": df["htn"].values,
        "dm": df["dm"].values,
        "smoke": df["smoke"].values,
        "dyslip": df["dyslip"].values,
    })
    out["male"] = 1 - out["female"]
    return out


def main():
    C.log_init("09_export_for_r")
    t0 = time.time()
    os.makedirs(R_IN, exist_ok=True)

    cohorts, _ = C.load_cohorts(verbose=False)
    g = pd.read_pickle(os.path.join(C.OUT_DIR, "s03_growth_full.pkl"))
    end = g[(g["pct_temporal"] == ENDPOINT_PCT) & (g["repeat"] == REPEAT)]
    row_t = end[end["arm"] == "temporal"].iloc[0]
    row_c = end[end["arm"] == "control"].iloc[0]

    # ---------------- 1. 长表：逐例概率 + 协变量 ----------------
    C.section("analysis_set.csv  逐例预测概率（长表）")
    blocks = []

    d = _covariates(cohorts["original"])
    d["p"] = C.read_csv_out("s01_oof_rf")["oof_rf"].values
    d["cohort"] = "Derivation"
    d["model"] = "CV (out-of-fold)"
    d["group"] = "Derivation (CV)"
    blocks.append(d)

    gp = C.read_csv_out("s02_geo_pred_rf")
    gg = _covariates(cohorts["geo"])
    assert np.array_equal(gg["y"].values, gp["y"].values), "地理队列结局行序不一致"
    for tag, arr in [("original model", gp["p_rf"].values),
                     ("updated model", np.asarray(row_t["p_geo"], float)),
                     ("control arm", np.asarray(row_c["p_geo"], float))]:
        b = gg.copy()
        b["p"] = arr
        b["cohort"] = "Geographical"
        b["model"] = tag
        b["group"] = "Geographical, " + tag
        blocks.append(b)

    long = pd.concat(blocks, ignore_index=True)
    assert np.array_equal(np.asarray(row_t["y_geo"]).astype(int),
                          gg["y"].values), "更新臂结局行序不一致"
    long = long[["group", "cohort", "model", "id", "y", "p", "age", "male",
                 "female", "htn", "dm", "smoke", "dyslip"]]
    long.to_csv(os.path.join(R_IN, "analysis_set.csv"), index=False,
                encoding="utf-8-sig")
    C.log(f"  {len(long)} 行  ->  {os.path.join(R_IN, 'analysis_set.csv')}")
    for (ch, md), sub in long.groupby(["cohort", "model"], sort=False):
        C.log(f"    {ch:13s} {md:18s} n={len(sub):3d}  事件={int(sub['y'].sum()):3d}")

    # ---------------- 2. 数据积累阶梯 ----------------
    C.section("growth_steps.csv  数据积累阶梯（7 档 x 2 臂 x 10 次）")
    det = pd.read_csv(os.path.join(C.OUT_DIR, "s03_growth_detail.csv"))
    keep = {"arm": "arm", "pct_temporal": "pct", "repeat": "repeat",
            "n_derive": "n_derive", "n_temporal": "n_temporal",
            "cv_auc": "cv_auc", "thr": "thr", "geo_auc": "geo_auc",
            "geo_slope": "geo_slope", "geo_intercept": "geo_intercept",
            "geo_brier": "geo_brier", "geo_ici": "geo_ici",
            "geo_oe": "geo_oe", "geo_Sensitivity": "geo_sens",
            "geo_Specificity": "geo_spec"}
    steps = det[list(keep)].rename(columns=keep)
    steps.to_csv(os.path.join(R_IN, "growth_steps.csv"), index=False,
                 encoding="utf-8-sig")
    C.log(f"  {len(steps)} 行  ->  {os.path.join(R_IN, 'growth_steps.csv')}")

    # ---------------- 3. 风险分层截断值（推导队列袋外三分位） ----------------
    C.section("risk_cutoffs.csv  风险分层截断值")
    p_oof = long[long["model"] == "CV (out-of-fold)"]["p"].values
    lo_c, hi_c = np.quantile(p_oof, [1 / 3, 2 / 3])
    cut = pd.DataFrame([{"low_cutoff": round(float(lo_c), 4),
                         "high_cutoff": round(float(hi_c), 4),
                         "source": "tertiles of derivation out-of-fold probabilities",
                         "n_derivation": int(len(p_oof))}])
    cut.to_csv(os.path.join(R_IN, "risk_cutoffs.csv"), index=False,
               encoding="utf-8-sig")
    C.log(f"  Low < {lo_c:.4f} | Intermediate {lo_c:.4f}-{hi_c:.4f} | "
          f"High >= {hi_c:.4f}")

    C.section(f"脚本 09 完成，总耗时 {C.timer(t0)}")


if __name__ == "__main__":
    main()
