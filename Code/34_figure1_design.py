# -*- coding: utf-8 -*-
"""
34_figure1_design.py — 第 19 轮：重做 Figure 1（研究设计与病例流程）

为什么重做
    旧 Figure 1（技术路线图）有三个问题：
      (1) 三个队列各写 "Excluded: 0"，与 579→442 排除 45/38/54 例的流程矛盾；
      (2) 有一个 "Two-Component Decomposition of the Loss" 面板，而审稿人指出
          本研究并没有做可加的损失分解，只是分别评价校准与判别；
      (3) 写 "Size-matched control arm"，但对照臂只匹配了行数（806 行、370 名
          不同患者），不是有效样本量。
    旧 Figure S14（流程图）另有两处口径问题：写 "three participating centres"
    （实为四家医院），同样保留 "size-matched" 与 "two components" 字样。

本图
    上：病例来源与筛选（含推导队列的真实排除数、外部队列的两家医院）
    下：分析序列（四步，标签按第 19 轮口径改写）
输出
    Figures/{pdf,tif,jpg}/Fig1_design.{fmt}      英文
    Figures/{pdf,tif,jpg}/Fig1_design_zh.{fmt}   中文
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import common as C


def _box(ax, x, y, w, h, text, fc, ec, fs=7.4, tc="#1a1a1a", lw=0.9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.006,rounding_size=0.008",
                                linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2.0, y + h / 2.0, text, ha="center", va="center",
            fontsize=fs, color=tc, zorder=3,
            fontweight="bold" if bold else "normal", linespacing=1.38)


def _arrow(ax, x1, y1, x2, y2, color="#444444", lw=0.9):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=7, linewidth=lw, color=color,
                                 zorder=1, shrinkA=0, shrinkB=0))


HDR = dict(fc="#e8eef4", ec="#4a6785")
DEV = dict(fc="#fdeee6", ec="#b5764a")
UPD = dict(fc="#efe8f7", ec="#7a5b9e")
GEO = dict(fc="#e6f4ea", ec="#4a8459")
OUT = dict(fc="#f2f2f2", ec="#777777")
ANC = dict(fc="#fff6d9", ec="#a08030")
WARN = dict(fc="#fdeaea", ec="#a24a4a")


EN = dict(
    head="Patients screened for eligibility at four hospitals, October 2017 to July 2026\n"
         "(inpatients with suspected coronary artery disease referred for CCTA)",
    dev="Derivation cohort\nCentres 1 and 2, Oct 2017 - Dec 2022\n"
        "n = 442, 208 obstructive (47.1%)",
    dev_exc="Excluded from 579 screened\nat these two centres\n"
            "- 45 unsatisfactory or incomplete CCTA\n"
            "- 38 incomplete clinical records\n"
            "- 54 previous revascularisation",
    temp="Contemporary cohort\nCentre 1, Jan 2023 - Jul 2026\n"
         "n = 364, 172 obstructive (47.3%)",
    geo="Geographical external cohort\nTwo further hospitals, 2024 - 2026\n"
        "n = 177, 63 obstructive (35.6%)",
    geoa="Hospital A\nn = 56\n24 obstructive\n(42.9%)",
    geob="Hospital B\nn = 121\n39 obstructive\n(32.2%)",
    seq_head="Analysis sequence",
    note="No randomisation. Allocation was fixed by the clinical question: models were developed on the derivation\n"
         "cohort, enlarged by contemporary patients in pre-specified increments; the geographical cohort was\n"
         "predicted only and never used for fitting, tuning, variable selection or threshold definition.",
    step1="1  Development\n- 19 candidate\n  variables\n- 8 algorithm families\n"
          "- repeated stratified\n  10-fold CV\n  (5 repeats, 50 folds)\n"
          "- random forest frozen\n  and carried forward",
    step2="2  External evaluation\n- frozen model,\n  prediction only\n- discrimination and\n  calibration\n"
          "- pooled and separately\n  for the two hospitals\n- cross-fitted recalibration\n"
          "- net benefit, hypothetical",
    step3="3  Temporal\n  accumulation\n- 7 increments of the\n  364 contemporary\n  patients\n"
          "- re-tuning at every step\n- resampling control matched\n  on rows (370 distinct\n"
          "  patients behind 806 rows)\n- no historical-period\n  control was available",
    step4="4  Reporting and\n  inference\n- TRIPOD+AI;\n  PROBAST+AI item by item\n  (Table S46)\n"
          "- one fully external\n  cohort, 63 events\n- exploratory; underpowered\n  for the primary endpoint",
    scope="Scope of every result. All patients had already been referred for and had undergone CCTA, so the performance\n"
          "reported here describes the model inside the referred population and does not address the decision it was\n"
          "designed to inform. The outcome is the CAD-RADS category recorded in the clinical report (3-5 versus 0-2),\n"
          "not haemodynamic significance.",
)

ZH = dict(
    head="四家医院 2017 年 10 月至 2026 年 7 月筛查的符合条件患者\n（疑诊冠心病、已转诊接受 CCTA 的住院患者）",
    dev="推导队列\n中心 1 与中心 2\n2017.10–2022.12\nn = 442，阻塞性 208 例\n（47.1%）",
    dev_exc="上述两中心筛查 579 例，排除：\n- CCTA 图像不合格或记录不全 45 例\n"
            "- 临床资料不完整 38 例\n- 既往血运重建 54 例",
    temp="当代队列\n中心 1\n2023.01–2026.07\nn = 364，阻塞性 172 例\n（47.3%）",
    geo="地理外部队列\n另两家医院\n2024–2026\nn = 177，阻塞性 63 例\n（35.6%）",
    geoa="医院 A\nn = 56\n阻塞性 24 例\n（42.9%）",
    geob="医院 B\nn = 121\n阻塞性 39 例\n（32.2%）",
    seq_head="分析序列",
    note="非随机分组。分组由临床问题决定：模型在推导队列上建立，按预设比例用当代患者扩充；\n"
         "地理队列只做预测，从未参与拟合、调参、变量选择或阈值确定。",
    step1="1  模型开发\n- 19 个候选变量\n- 8 个算法族\n- 重复分层 10 折\n  交叉验证（5 次，50 折）\n"
          "- 随机森林冻结后\n  进入迁移",
    step2="2  外部评价\n- 冻结模型，仅预测\n- 判别与校准\n- 合并报告，并按两家\n  医院分别报告\n"
          "- 交叉拟合再校准\n- 净获益为假设性分析",
    step3="3  时间累积\n- 364 例当代患者\n  分 7 档并入\n- 每一步重新调参\n"
          "- 重抽样对照臂仅匹配\n  行数（806 行对应\n  370 名患者）\n- 无法获得历史时段对照",
    step4="4  报告与推断\n- TRIPOD+AI；\n  PROBAST+AI 逐条评价\n  （表 S46）\n"
          "- 仅一个完整外部队列，\n  63 个事件\n- 探索性；对主终点\n  把握度不足",
    scope="全部结果的适用范围。所有患者均已转诊并完成 CCTA，故本文报告的是模型在已转诊人群内的表现，\n"
          "不涉及它本应辅助的决策。结局为临床报告中记录的 CAD-RADS 分级（3-5 对 0-2），\n"
          "不等同于血流动力学意义的阻塞。",
)


def draw(L, zh, out_name):
    if zh:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                           "Source Han Sans SC", "Arial"]
    else:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]

    fig = plt.figure(figsize=(7.30, 9.05))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fs = 6.6

    # ---------------- 上：病例来源 -------------------------------------
    _box(ax, 0.03, 0.928, 0.94, 0.055, L["head"], **HDR, fs=7.4, bold=True)
    for x in (0.18, 0.51, 0.83):
        _arrow(ax, x, 0.928, x, 0.880)

    _box(ax, 0.03, 0.800, 0.30, 0.080, L["dev"], **DEV, fs=fs)
    _box(ax, 0.36, 0.800, 0.30, 0.080, L["temp"], **UPD, fs=fs)
    _box(ax, 0.69, 0.800, 0.28, 0.080, L["geo"], **GEO, fs=fs)

    _box(ax, 0.03, 0.688, 0.30, 0.080, L["dev_exc"], **OUT, fs=6.2)
    _arrow(ax, 0.18, 0.800, 0.18, 0.768)

    _box(ax, 0.69, 0.688, 0.13, 0.080, L["geoa"], **GEO, fs=6.2)
    _box(ax, 0.84, 0.688, 0.13, 0.080, L["geob"], **GEO, fs=6.2)
    _arrow(ax, 0.83, 0.800, 0.83, 0.768)

    _box(ax, 0.03, 0.575, 0.94, 0.080, L["note"], **ANC, fs=6.4)
    _arrow(ax, 0.51, 0.688, 0.51, 0.655)

    # ---------------- 下：分析序列 -------------------------------------
    _box(ax, 0.03, 0.512, 0.94, 0.032, L["seq_head"], fc="#ffffff",
         ec="#333333", fs=7.4, bold=True)
    _arrow(ax, 0.51, 0.575, 0.51, 0.544)
    _arrow(ax, 0.51, 0.512, 0.51, 0.480)

    w, gap = 0.2225, 0.0125
    xs = [0.03 + i * (w + gap) for i in range(4)]
    for x, txt, sty in zip(xs, [L["step1"], L["step2"], L["step3"], L["step4"]],
                           [HDR, GEO, UPD, OUT]):
        _box(ax, x, 0.290, w, 0.190, txt, **sty, fs=6.1)
    for i in range(3):
        _arrow(ax, xs[i] + w, 0.385, xs[i + 1], 0.385)

    _box(ax, 0.03, 0.115, 0.94, 0.120, L["scope"], **WARN, fs=6.5)
    _arrow(ax, 0.51, 0.290, 0.51, 0.235)

    C.save_figure(fig, out_name)


if __name__ == "__main__":
    C.log_init("34_figure1_design")
    draw(EN, False, "Fig1_design")
    draw(ZH, True, "Fig1_design_zh")
    C.log("  done.")
