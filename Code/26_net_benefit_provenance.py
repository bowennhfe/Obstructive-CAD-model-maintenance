# -*- coding: utf-8 -*-
"""
26 —— 净获益数字的口径溯源表（第十六轮审稿 Major 6）
=============================================================================
背景
    正文与摘要中出现多个 0.50 阈值下的净获益数字（-0.001、0.000、0.006、
    0.012、0.035 等），它们来自不同的「模型 x 拟合方式 x 评估集 x 重复实验
    框架」组合，读者无法相互核对。本表把每一个数字逐行落到其确切口径上。

口径维度
    model          : 具体模型（冻结模型 / 阶梯 0% 步 / 完全更新模型 / 截距再校准）
    fitting        : 拟合方式（未拟合 / 在扩大推导池内网格搜索重调参 /
                     在合并池上重训 / 用目标中心结局拟合一个常数）
    evaluation_set : 评估集（地理队列 177 例，或按重复实验分别评估后取平均）
    repeat_frame   : 重复实验框架（逐重复分别评估后平均 / 合并预测向量上成对比较）
    source_table   : 数字出处

数据源
    Tables/TableS8_decision_curve.csv        逐重复平均口径
    Tables/TableS37_paired_bootstrap.csv     合并预测向量成对比较口径
    Tables/TableS4_growth_temporal_vs_control.csv（阶梯端点，供交叉核对）

输出
    Tables/TableS41_net_benefit_provenance.csv
    Outputs/s26_net_benefit_provenance_detail.csv
=============================================================================
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

THRESHOLDS = [0.35, 0.40, 0.50, 0.55]


def T(name):
    return pd.read_csv(os.path.join(C.TABLE_DIR, f"{name}.csv"))


def _nb(d8, thr, col):
    m = d8[np.isclose(d8["threshold"].astype(float), thr)]
    if len(m) == 0:
        return np.nan
    return float(m[col].iloc[0])


def _nb37(d37, comparison, thr, which):
    lab = "NB@%.2f" % thr
    m = d37[(d37["comparison"] == comparison) & (d37["metric"] == lab)]
    if len(m) == 0:
        return np.nan
    return float(m[which].iloc[0])


def main():
    C.log_init("26_net_benefit_provenance")
    d8 = T("TableS8_decision_curve")
    d37 = T("TableS37_paired_bootstrap")

    rows = []

    def add(thr, model, fitting, evalset, repframe, value, src):
        rows.append({
            "threshold": thr,
            "model": model,
            "fitting": fitting,
            "evaluation_set": evalset,
            "repeat_frame": repframe,
            "net_benefit": None if value is None or (
                isinstance(value, float) and np.isnan(value)) else round(float(value), 4),
            "source_table": src,
        })

    for thr in THRESHOLDS:
        # --- 表 S8：逐重复平均口径（= 图 8 版本）---
        add(thr, "Frozen model",
            "Not fitted (hyperparameters and coefficients fixed)",
            "Geographical cohort (n=177)",
            "Evaluated per repeated experiment and averaged over 10 repeats",
            _nb(d8, thr, "net_benefit_frozen"), "Table S8")
        add(thr, "Frozen model + intercept-only recalibration",
            "One constant fitted on the target-centre outcomes (cross-fitted, out-of-bag)",
            "Geographical cohort (n=177)",
            "Evaluated per repeated experiment and averaged over 10 repeats",
            _nb(d8, thr, "net_benefit_recal_intercept"), "Table S8")
        add(thr, "Fully updated model",
            "Re-tuned by 5-fold grid search within the enlarged derivation pool (100% step)",
            "Geographical cohort (n=177)",
            "Evaluated per repeated experiment and averaged over 10 repeats",
            _nb(d8, thr, "net_benefit_updated"), "Table S8")
        add(thr, "Row-count-matched control arm",
            "Re-tuned by 5-fold grid search within the enlarged derivation pool (control data)",
            "Geographical cohort (n=177)",
            "Evaluated per repeated experiment and averaged over 10 repeats",
            _nb(d8, thr, "net_benefit_control"), "Table S8")
        add(thr, "Treat-all strategy",
            "Not fitted", "Geographical cohort (n=177)",
            "Evaluated per repeated experiment and averaged over 10 repeats",
            _nb(d8, thr, "net_benefit_treat_all"), "Table S8")
        add(thr, "Treat-none strategy",
            "Not fitted", "Geographical cohort (n=177)",
            "Evaluated per repeated experiment and averaged over 10 repeats",
            _nb(d8, thr, "net_benefit_treat_none"), "Table S8")

        # --- 表 S37：合并预测向量成对比较口径（= §3.7 主口径）---
        add(thr, "Ladder 0% step",
            "Re-tuned by 5-fold grid search within the original derivation pool",
            "Geographical cohort, single pooled prediction vector (n=177)",
            "Paired patient-level bootstrap on pooled vectors (2000 resamples)",
            _nb37(d37, "ladder0_vs_ladder100", thr, "before"), "Table S37")
        add(thr, "Fully updated model (100% step)",
            "Re-tuned by 5-fold grid search within the enlarged derivation pool",
            "Geographical cohort, single pooled prediction vector (n=177)",
            "Paired patient-level bootstrap on pooled vectors (2000 resamples)",
            _nb37(d37, "ladder0_vs_ladder100", thr, "after"), "Table S37")
        add(thr, "Frozen model",
            "Not fitted (hyperparameters and coefficients fixed)",
            "Geographical cohort, single pooled prediction vector (n=177)",
            "Paired patient-level bootstrap on pooled vectors (2000 resamples)",
            _nb37(d37, "frozen_vs_ladder0", thr, "before"), "Table S37")

    out = pd.DataFrame(rows)
    C.save_table(out, "TableS41_net_benefit_provenance")
    C.save_csv(out, "s26_net_benefit_provenance_detail")
    C.log("\n" + out.to_string(index=False))
    C.log("\n-> Tables/TableS41_net_benefit_provenance.csv")


if __name__ == "__main__":
    main()
