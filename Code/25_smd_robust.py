# -*- coding: utf-8 -*-
"""
25 —— 基线 SMD 的口径敏感性：稳健 SMD（第十六轮审稿 Major 7）
=============================================================================
背景
    表 1 展示各变量的中位数（IQR），但 SMD 由均数与合并标准差算得。
    对肌酐这类右偏且含极端值的变量（推导队列 max = 1286 umol/L，
    属透析级取值），基于均数的 SMD 会被极端值稀释：肌酐 orig vs temporal
    的均数 SMD 仅 0.057，而其分布差异在临床上十分明显。

做法
    对每一个连续变量同时给出：
      ① mean-based SMD  = (mean_a - mean_b) / sqrt((sd_a^2 + sd_b^2)/2)
      ② robust SMD      = (median_a - median_b) /
                          sqrt((s_a^2 + s_b^2)/2)，
         其中 s = 1.4826 * MAD（中位绝对离差的正态一致估计）
    并额外报告去掉极端值（var > 500 umol/L 的肌酐类取值）后的 mean-based SMD。

输出
    Tables/TableS40_smd_robust.csv
    Outputs/s25_smd_robust_detail.csv
=============================================================================
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

CONT = ["age", "sbp", "dbp", "map", "pp", "ppi",
        "cr", "urea", "ua", "tc", "tg", "ldl", "hdl", "fib"]
CUT_EXTREME = 500.0          # umol/L，用于肌酐/尿酸的极端值敏感性


def robust_smd(a, b):
    ma, mb = float(np.median(a)), float(np.median(b))
    sa = 1.4826 * float(np.median(np.abs(a - ma)))
    sb = 1.4826 * float(np.median(np.abs(b - mb)))
    denom = np.sqrt((sa ** 2 + sb ** 2) / 2.0)
    if denom == 0:
        return np.nan, ma, mb, sa, sb
    return (ma - mb) / denom, ma, mb, sa, sb


def trimmed_smd(a, b, cut):
    a2 = a[a <= cut]
    b2 = b[b <= cut]
    if len(a2) < 3 or len(b2) < 3:
        return np.nan, 0, 0
    return C.smd_cont(a2, b2), len(a2), len(b2)


def main():
    C.log_init("25_smd_robust")
    cohorts, _ = C.load_cohorts(verbose=False)
    o, t, g = cohorts["original"], cohorts["temporal"], cohorts["geo"]

    rows = []
    det = []
    for v in CONT:
        if v not in o.columns:
            continue
        a = o[v].dropna().astype(float).values
        for name, other in (("temporal", t), ("geo", g)):
            if v not in other.columns:
                continue
            b = other[v].dropna().astype(float).values
            mean_smd = C.smd_cont(a, b)
            rob, ma, mb, sa, sb = robust_smd(a, b)
            row = {
                "variable": v,
                "comparison": "original vs %s" % name,
                "smd_mean_based": round(float(mean_smd), 3),
                "smd_robust": None if np.isnan(rob) else round(float(rob), 3),
                "median_original": round(ma, 2),
                "median_other": round(mb, 2),
                "mad_original": round(sa, 3),
                "mad_other": round(sb, 3),
                "n_original": int(len(a)),
                "n_other": int(len(b)),
            }
            if v in ("cr", "ua"):
                tr, n1, n2 = trimmed_smd(a, b, CUT_EXTREME)
                row["smd_mean_trimmed_le500"] = (
                    None if np.isnan(tr) else round(float(tr), 3))
                row["n_trimmed_original"] = n1
                row["n_trimmed_other"] = n2
            else:
                row["smd_mean_trimmed_le500"] = None
                row["n_trimmed_original"] = None
                row["n_trimmed_other"] = None
            rows.append(row)

            det.append({
                "variable": v, "comparison": name,
                "mean_original": float(a.mean()), "sd_original": float(a.std(ddof=1)),
                "max_original": float(a.max()),
                "mean_other": float(b.mean()), "sd_other": float(b.std(ddof=1)),
                "max_other": float(b.max()),
                "smd_mean_based": float(mean_smd), "smd_robust": float(rob),
            })
            C.log(f"{v:5s} orig-vs-{name:8s}  mean-SMD {mean_smd:+.3f}  "
                  f"robust-SMD {rob:+.3f}")

    t_csv = pd.DataFrame(rows)
    C.save_table(t_csv, "TableS40_smd_robust")
    C.save_csv(pd.DataFrame(det), "s25_smd_robust_detail")
    C.log("\n" + t_csv.to_string(index=False))
    C.log("\n-> Tables/TableS40_smd_robust.csv")


if __name__ == "__main__":
    main()
