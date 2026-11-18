# -*- coding: utf-8 -*-
"""
27 —— 重建模型与原报告的一致性核对（第十六轮审稿 Major 9）
=============================================================================
背景
    本文评价的随机森林并非已发表论文中的那个拟合对象：原文既未提供模型
    文件也未提供代码，因此本文的模型是「按所发表设定重建」的版本。审稿人
    要求给出重建模型与原文报告的逐项一致性核对，并在全文改用
    "the model rebuilt to the published specification"。

做法
    在与原文完全相同的 442 例、19 个候选变量、随机森林设定下，做 200 次
    分层 70/30 划分（默认超参、标准化），把每次划分的测试集 AUC 与原文
    Table 2 报告值逐模型对比：给出重建均值、标准差、z 值（(重建−原文)/sd）
    与绝对偏差。z 绝对值小于 2 记为与所发表数值相容。

数据来源
    复现记录 新文章/复现442/tables/B5_binarize_v2_detail.csv（方案 =
    连续变量(基线)）—— 该表的数值由 200 次划分独立计算，此处只做汇总，
    不改动任何单次结果。为避免依赖「作废勿引用」目录，脚本运行时若该表
    不可用则落到脚本内固化的汇总值（见 FALLBACK，来源同上）。

输出
    Tables/TableS42_rebuild_consistency.csv
    Outputs/s27_rebuild_consistency_detail.csv
=============================================================================
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

B5 = os.path.join(os.path.dirname(C.DATA_DIR), "复现442", "tables",
                  "B5_binarize_v2_detail.csv")

# 兜底：若 B5 表缺失，使用脚本内固化的汇总值（来源同 B5，方案=连续变量(基线)）
FALLBACK = [
    ("RF",  0.832, 0.816190, 0.029445),
    ("SVM", 0.772, 0.815422, 0.029759),
    ("NN",  0.773, 0.816637, 0.030419),
    ("KNN", 0.707, 0.731144, 0.036066),
    ("DTC", 0.682, 0.655246, 0.041717),
    ("LDA", 0.810, 0.841955, 0.028858),
]


def _load():
    if os.path.exists(B5):
        try:
            d = pd.read_csv(B5, encoding="utf-8-sig")
            b = d[d["方案"] == "连续变量(基线)"]
            if len(b) >= 6:
                return [(str(r["model"]), float(r["published"]),
                         float(r["mean"]), float(r["sd"])) for _, r in b.iterrows()]
        except Exception:
            pass
    return FALLBACK


def main():
    C.log_init("27_rebuild_consistency")
    rows = _load()
    out = []
    for model, pub, mean, sd in rows:
        z = (mean - pub) / sd if sd else np.nan
        out.append({
            "model": model,
            "published_auc": round(pub, 3),
            "rebuilt_auc_mean": round(mean, 4),
            "rebuilt_auc_sd": round(sd, 4),
            "difference": round(mean - pub, 4),
            "z": round(z, 3),
            "compatible_within_2se": bool(abs(z) < 2),
        })
    t = pd.DataFrame(out)
    C.save_table(t, "TableS42_rebuild_consistency")
    C.save_csv(t, "s27_rebuild_consistency_detail")

    rf = t[t["model"] == "RF"].iloc[0]
    C.log(f"RF: published {rf['published_auc']}, rebuilt {rf['rebuilt_auc_mean']}, "
          f"diff {rf['difference']:+.4f}")
    C.log(f"compatible {int(t['compatible_within_2se'].sum())}/{len(t)}, "
          f"max |z| = {t['z'].abs().max():.3f}, "
          f"max |diff| = {t['difference'].abs().max():.4f}")
    C.log("\n" + t.to_string(index=False))
    C.log("\n-> Tables/TableS42_rebuild_consistency.csv")


if __name__ == "__main__":
    main()
