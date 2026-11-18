# -*- coding: utf-8 -*-
"""
19_minor_revision.py
====================
补做审稿意见「次要意见」中真正需要新计算的两项：

  (1) 19 个候选变量的相关性矩阵（SHAP 归因的可解释性前提：
      若两个变量高度相关，SHAP 值会在两者之间分摊，不能被读成独立效应）
  (2) 三个队列的逐变量缺失值计数（原稿只报告了越界处置，未报告缺失）

输出：
  Tables/TableS32_variable_correlation.csv
  Tables/TableS33_missingness_by_cohort.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402


def sec1_correlation(cohorts):
    C.section("(1) 候选变量相关性矩阵（推导队列，Spearman）")
    d = cohorts["original"]
    X = d[C.FEATURES].astype(float)
    r = X.corr(method="spearman")
    r = r.round(3)
    # 表首列放变量名，便于排版
    out = r.reset_index().rename(columns={"index": "Variable"})
    C.save_table(out, "TableS32_variable_correlation")

    # 报告最强的一对（去掉自相关对角线）
    m = r.where(~np.eye(len(r), dtype=bool))
    mx = float(np.nanmax(m.values))
    pos = np.unravel_index(np.nanargmax(m.values), m.shape)
    mn = float(np.nanmin(m.values))
    neg = np.unravel_index(np.nanargmin(m.values), m.shape)
    C.log(f"  最强正相关：{r.index[pos[0]]} vs {r.columns[pos[1]]} = {mx:.3f}")
    C.log(f"  最强负相关：{r.index[neg[0]]} vs {r.columns[neg[1]]} = {mn:.3f}")
    # 与纤维蛋白原（SHAP 首位变量）相关的变量
    if "fib" in r.columns:
        s = r["fib"].drop("fib").sort_values(key=np.abs, ascending=False)
        C.log("  与纤维蛋白原 |rho| 最大的三个变量："
              + "、".join(f"{k}={v:.3f}" for k, v in s.head(3).items()))
    return r


def sec2_missingness():
    C.section("(2) 逐变量缺失值（三队列）")
    raw, _ = C.load_cohorts(apply_strategy_d=False, verbose=False)
    post, _ = C.load_cohorts(apply_strategy_d=True, verbose=False)
    rows = []
    for var in C.FEATURES:
        rec = {"Variable": var}
        for key, lab in (("original", "Derivation (n=442)"),
                         ("temporal", "Temporal (n=364)"),
                         ("geo", "Geographical (n=177)")):
            n = len(raw[key])
            miss_raw = int(raw[key][var].isna().sum())
            miss_post = int(post[key][var].isna().sum())
            rec[f"{lab}: missing before outlier rules"] = miss_raw
            rec[f"{lab}: missing after outlier rules"] = miss_post
            rec[f"{lab}: % after (%)"] = round(miss_post / n * 100, 2)
        rows.append(rec)
    t = pd.DataFrame(rows)
    C.save_table(t, "TableS33_missingness_by_cohort")
    tot = {lab: int(t[f"{lab}: missing after outlier rules"].sum())
           for lab in ("Derivation (n=442)", "Temporal (n=364)", "Geographical (n=177)")}
    C.log(f"  置缺失格总数（策略 D 前）："
          f"{sum(int(t[f'{k}: missing before outlier rules'].sum()) for k in tot)}")
    C.log(f"  置缺失格总数（策略 D 后）：{sum(tot.values())} -> {tot}")
    C.log("  说明：策略 D 只把越界值置为缺失，不删行、不改值；所有插补封在 "
          "Pipeline 内、只在训练折 fit。")
    return t


def main():
    C.log_init("19_minor_revision")
    cohorts, _ = C.load_cohorts(verbose=False)
    sec1_correlation(cohorts)
    sec2_missingness()
    C.section("脚本 19 完成")


if __name__ == "__main__":
    main()
