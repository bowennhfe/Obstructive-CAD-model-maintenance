# -*- coding: utf-8 -*-
"""
31_tripod_flowdiagram.py
========================
R26（2 区标准审核）要求的补强：TRIPOD+AI 研究流程图（study flow diagram）。

TRIPOD+AI 与多数心血管影像/临床流行病学 2 区期刊要求正文给出
「筛选 -> 纳排 -> 分配 -> 随访/分析」的完整流程图。本图只列方法与
流程，不携带任何结果数值；人数一律取自 Tables/*.csv（唯一真相源）。

输出：
  Figures/{pdf,tif,jpg}/FigS14_tripod_flow.jpg
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

T = lambda name: pd.read_csv(os.path.join(C.TABLE_DIR, name + ".csv"),
                             encoding="utf-8-sig")


def _box(ax, x, y, w, h, text, fc, ec, fs=7.4, tc="#1a1a1a", lw=0.9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.006,rounding_size=0.008",
                                linewidth=lw, edgecolor=ec, facecolor=fc,
                                zorder=2))
    ax.text(x + w / 2.0, y + h / 2.0, text, ha="center", va="center",
            fontsize=fs, color=tc, zorder=3,
            fontweight="bold" if bold else "normal", linespacing=1.35)


def _arrow(ax, x1, y1, x2, y2, color="#444444", lw=0.9, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=7, linewidth=lw,
                                 color=color, zorder=1,
                                 shrinkA=0, shrinkB=0))


def _en():
    fig, ax = plt.subplots(figsize=(7.30, 8.10))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    hdr = dict(fc="#e8eef4", ec="#4a6785")
    dev = dict(fc="#fdeee6", ec="#b5764a")
    upd = dict(fc="#efe8f7", ec="#7a5b9e")
    geo = dict(fc="#e6f4ea", ec="#4a8459")
    out = dict(fc="#f2f2f2", ec="#777777")
    anc = dict(fc="#fff6d9", ec="#a08030")

    # ---------------- 阶段 1：来源与筛选 ----------------
    _box(ax, 0.05, 0.905, 0.90, 0.062,
         "Patients screened for eligibility at three participating centres\n"
         "(inpatients with suspected coronary artery disease referred for CCTA)",
         **hdr, fs=7.6, bold=True)
    _arrow(ax, 0.50, 0.905, 0.50, 0.868)

    _box(ax, 0.05, 0.792, 0.42, 0.076,
         "Derivation cohort\nCentres 1 and 2, Oct 2017 - Dec 2022\n"
         "n = 442, 208 obstructive (47.1%)\nretrospective, consecutive",
         **dev, fs=7.2)
    _box(ax, 0.53, 0.792, 0.42, 0.076,
         "Contemporary cohort\nCentre 1, Jan 2023 - Jul 2026\n"
         "n = 364, 172 obstructive (47.3%)\nretrospective, consecutive",
         **upd, fs=7.2)
    _box(ax, 0.29, 0.690, 0.42, 0.070,
         "Geographical external cohort\nTwo separate external centres, 2024 - 2026\n"
         "n = 177, 63 obstructive (35.6%)\nno patient contributed to development",
         **geo, fs=7.2)
    _arrow(ax, 0.26, 0.905, 0.26, 0.868)
    _arrow(ax, 0.74, 0.905, 0.74, 0.868)
    _arrow(ax, 0.74, 0.792, 0.60, 0.760)
    _arrow(ax, 0.26, 0.792, 0.40, 0.760)

    # ---------------- 阶段 2：分配 ----------------
    _box(ax, 0.05, 0.596, 0.90, 0.062,
         "No randomisation. Allocation was fixed by the clinical question: the derivation "
         "and\ncontemporary cohorts were developed on, the geographical cohort was predicted "
         "only and\nwas never used for cross-validation, hyperparameter selection or any "
         "threshold choice.",
         **anc, fs=7.0)
    _arrow(ax, 0.50, 0.690, 0.50, 0.658)

    # ---------------- 阶段 3：分析 ----------------
    _box(ax, 0.05, 0.508, 0.90, 0.040,
         "Analysis", fc="#ffffff", ec="#333333", fs=8.0, bold=True)
    _arrow(ax, 0.50, 0.508, 0.50, 0.490)

    _box(ax, 0.045, 0.366, 0.28, 0.130,
         "Development (derivation cohort)\n"
         "- 19 candidate variables, median imputation\n"
         "- 8 algorithm families, grids fixed\n"
         "- repeated stratified 10-fold CV\n"
         "  (5 repeats, 50 folds)\n"
         "- pipeline frozen after selection",
         **hdr, fs=6.6)
    _box(ax, 0.375, 0.366, 0.25, 0.130,
         "Temporal accumulation (specified\nbefore analysis)\n"
         "- 7 increments of the 364\n"
         "  contemporary patients\n"
         "- re-tuning at every step\n"
         "- row-count-matched control arm\n"
         "  (resampling adds no information)",
         **upd, fs=6.6)
    _box(ax, 0.675, 0.366, 0.28, 0.130,
         "External evaluation (geographical)\n"
         "- transferred frozen, predict only\n"
         "- discrimination and calibration\n"
         "- cross-fitted recalibration\n"
         "  (5 folds; fit on 4, evaluate on 1)\n"
         "- decision-curve analysis",
         **geo, fs=6.6)
    _arrow(ax, 0.185, 0.490, 0.185, 0.490)
    for x in (0.185, 0.500, 0.815):
        _arrow(ax, x, 0.366, x, 0.348)

    # ---------------- 阶段 4：产出 ----------------
    _box(ax, 0.045, 0.244, 0.28, 0.100,
         "Outputs, all traced to Tables/*.csv\n"
         "1. Derivation performance (Tables 2, 3, 9)\n"
         "2. External loss and its two components\n"
         "   (Tables 4, 5, 10, 11)\n"
         "3. Update trajectory and budgets\n"
         "   (Tables 6, 7, 8, 12)",
         **out, fs=6.6)
    _box(ax, 0.375, 0.244, 0.25, 0.100,
         "Sensitivity and robustness\n"
         "- Centre-aware and nested CV (S38)\n"
         "- Complete-case vs imputation (S44)\n"
         "- Rebuild consistency (S42)\n"
         "- Robust standardised differences (S40)\n"
         "- Subgroup interaction (S45)",
         **out, fs=6.6)
    _box(ax, 0.675, 0.244, 0.28, 0.100,
         "Excluded from every analysis\n"
         "- 5 records whose source centre\n"
         "  could not be assigned were kept\n"
         "  in pooled analyses but dropped\n"
         "  from centre-specific comparisons\n"
         "- No record was excluded for outcome",
         **out, fs=6.6)

    # ---------------- 阶段 5：声明 ----------------
    _box(ax, 0.045, 0.146, 0.91, 0.076,
         "Reporting follows TRIPOD+AI (development and external validation, incorporating the "
         "clustered-data\nitems of TRIPOD-Cluster). Risk of bias was assessed at the level of the four "
         "PROBAST+AI domains\nrather than item by item (Table S34). The design was specified before "
         "analysis and is not registered;\nno time-stamped analysis plan and no public repository "
         "entry exist. Full checklist provided as a supplementary file.",
         **hdr, fs=6.6)
    _arrow(ax, 0.50, 0.244, 0.50, 0.226)

    # ---------------- 阶段 6：适用性 ----------------
    _box(ax, 0.045, 0.050, 0.91, 0.082,
         "Scope of every result. All three cohorts consisted of patients who had already been "
         "referred for\nand had already undergone CCTA. The model is intended for pre-CCTA risk "
         "stratification, so the\nperformance reported here describes its behaviour inside the "
         "referred population and does not\naddress the decision it was designed to inform. One "
         "fully external cohort, 63 events; all\nexternal comparisons are exploratory and "
         "hypothesis-generating.",
         fc="#fdeaea", ec="#a24a4a", fs=6.6)

    C.save_figure(fig, "FigS14_tripod_flow")


def _zh():
    fig, ax = plt.subplots(figsize=(7.30, 8.10))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    hdr = dict(fc="#e8eef4", ec="#4a6785")
    dev = dict(fc="#fdeee6", ec="#b5764a")
    upd = dict(fc="#efe8f7", ec="#7a5b9e")
    geo = dict(fc="#e6f4ea", ec="#4a8459")
    out = dict(fc="#f2f2f2", ec="#777777")
    anc = dict(fc="#fff6d9", ec="#a08030")

    _box(ax, 0.05, 0.905, 0.90, 0.062,
         "四个中心筛选可纳入患者\n（因疑似冠心病接受 CCTA 的住院患者）",
         **hdr, fs=8.0, bold=True)
    _arrow(ax, 0.50, 0.905, 0.50, 0.868)

    _box(ax, 0.05, 0.792, 0.42, 0.076,
         "推导队列\n中心 1 与中心 2，2017 年 10 月至 2022 年 12 月\n"
         "n = 442，208 例阻塞性（47.1%）\n回顾性、连续纳入",
         **dev, fs=7.2)
    _box(ax, 0.53, 0.792, 0.42, 0.076,
         "当代队列\n中心 1，2023 年 1 月至 2026 年 7 月\n"
         "n = 364，172 例阻塞性（47.3%）\n回顾性、连续纳入",
         **upd, fs=7.2)
    _box(ax, 0.29, 0.690, 0.42, 0.070,
         "地理外部队列\n独立的两家外部中心，2024 至 2026 年\n"
         "n = 177，63 例阻塞性（35.6%）\n无任何患者进入模型开发",
         **geo, fs=7.2)
    _arrow(ax, 0.26, 0.905, 0.26, 0.868)
    _arrow(ax, 0.74, 0.905, 0.74, 0.868)
    _arrow(ax, 0.74, 0.792, 0.60, 0.760)
    _arrow(ax, 0.26, 0.792, 0.40, 0.760)

    _box(ax, 0.05, 0.596, 0.90, 0.062,
         "未作随机化。分配由临床问题固定：推导队列与当代队列用于开发，"
         "地理队列只作预测，\n从未用于交叉验证、超参数选择或任何阈值确定。",
         **anc, fs=7.2)
    _arrow(ax, 0.50, 0.690, 0.50, 0.658)

    _box(ax, 0.05, 0.508, 0.90, 0.040,
         "分析", fc="#ffffff", ec="#333333", fs=8.4, bold=True)
    _arrow(ax, 0.50, 0.508, 0.50, 0.490)

    _box(ax, 0.045, 0.366, 0.28, 0.130,
         "开发（推导队列）\n- 19 个候选变量、中位数插补\n"
         "- 8 个算法族、网格事先固定\n"
         "- 重复分层 10 折交叉验证\n  （5 次重复、50 折）\n"
         "- 选择后冻结整条流程",
         **hdr, fs=6.8)
    _box(ax, 0.375, 0.366, 0.25, 0.130,
         "时间累积（分析前指定）\n- 364 例当代患者分 7 个增量\n"
         "- 每步重新调参\n- 样本量匹配的对照臂\n"
         "  （重抽样不增加信息）",
         **upd, fs=6.8)
    _box(ax, 0.675, 0.366, 0.28, 0.130,
         "外部评价（地理队列）\n- 冻结转移、只作预测\n- 区分度与校准度\n"
         "- 交叉验证重校准\n  （5 折；4 折拟合、1 折评价）\n"
         "- 决策曲线分析",
         **geo, fs=6.8)
    for x in (0.185, 0.500, 0.815):
        _arrow(ax, x, 0.366, x, 0.348)

    _box(ax, 0.045, 0.244, 0.28, 0.100,
         "产出，均可溯至 Tables/*.csv\n"
         "1. 推导队列表现（表 2、3、9）\n"
         "2. 外部损失及其两个分量\n   （表 4、5、10、11）\n"
         "3. 更新轨迹与更新预算\n   （表 6、7、8、12）",
         **out, fs=6.8)
    _box(ax, 0.375, 0.244, 0.25, 0.100,
         "敏感性与稳健性\n- 中心感知与嵌套 CV（S38）\n"
         "- 完整案例与插补对照（S44）\n- 重建一致性（S42）\n"
         "- 稳健标准化差异（S40）\n- 亚组交互检验（S45）",
         **out, fs=6.8)
    _box(ax, 0.675, 0.244, 0.28, 0.100,
         "各分析中的排除\n- 5 例无法归属中心者保留在\n"
         "  合并分析中，但不进入\n  中心间比较\n"
         "- 无任何记录因结局被排除",
         **out, fs=6.8)

    _box(ax, 0.045, 0.146, 0.91, 0.076,
         "报告规范：TRIPOD+AI（开发与外部验证，并纳入 TRIPOD-Cluster 的"
         "聚集数据条目）。偏倚风险按 PROBAST+AI\n的四个域评估而非逐条评分（表 S34）。"
         "设计在分析前指定但未注册，不存在带时间戳的\n分析计划，也不存在公开仓库条目。"
         "完整清单作为补充文件提供。",
         **hdr, fs=6.8)
    _arrow(ax, 0.50, 0.244, 0.50, 0.226)

    _box(ax, 0.045, 0.050, 0.91, 0.082,
         "全部结果的适用范围。三个队列均为已被转诊并已完成 CCTA 的患者。"
         "该模型拟用于 CCTA 前的风险分层，\n因此本文所报性能描述的是它在"
         "已转诊人群内的行为，并不涉及它本应辅助的\n那个决策。仅有一个完全外部队列、"
         "63 例事件，全部外部比较均为探索性且\n属假设生成。",
         fc="#fdeaea", ec="#a24a4a", fs=6.8)

    C.save_figure(fig, "FigS14_tripod_flow_zh")


if __name__ == "__main__":
    C.log_init("31_tripod_flowdiagram")
    _en()
    _zh()
