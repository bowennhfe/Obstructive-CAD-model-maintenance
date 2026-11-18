# -*- coding: utf-8 -*-
"""
14 —— 多算法横向对比：外部验证的"判别力损失"与"校准损失"是否可分离
=============================================================================
目的（服务于"外部验证后 AUC 与校准度双双下降"这一中心思想）
    把外部验证的性能损失拆成两个分量，并检验该拆分是否与算法无关：
      (a) 尺度分量（calibration）：风险尺度发生单调位移 → 可用少量目标域数据修复
      (b) 排序分量（discrimination）：患者排序本身不迁移 → 再校准无法修复

设计（严格沿用脚本 01/02 的既有协议，不引入新的泄露面）
    · 算法集预先指定 8 个：LR / LDA / SVM / KNN / DTC / MLP / RF / HGB
      —— 覆盖线性、生成式线性、核方法、实例法、单树、神经网络、装袋、提升 8 个族
    · 调参：在推导整池(442)上做 10 折网格搜索（与 01 同协议；偏乐观已在局限中披露）
    · 内部估计：RepeatedStratifiedKFold(10 × 5 = 50 折)，仅用固定超参
    · 外部验证：在推导整池上冻结，原样预测地理队列(177)，地理队列不参与任何拟合
    · 再校准：交叉拟合（地理队列内部 5 折；拟合折与评价折不重叠）
      - 仅截距再校准（校准 in the large）
      - 斜率+截距再校准（logistic 再校准）
    · 量纲敏感模型（LR/LDA/SVM/KNN/MLP）在 Pipeline 内加 StandardScaler，
      只在训练折 fit；树模型（RF/HGB/DTC）保持 imp+clf 与正文一致
    · 配对 Bootstrap 比较地理队列上的 AUC 差异，并做 Holm 校正

输出
    Tables/Table11_algo_internal.csv
    Tables/Table12_algo_external.csv
    Tables/Table13_algo_pairwise_auc_diff.csv
    Tables/Table14_algo_recalibration_decomposition.csv
    Tables/Table15_algo_rank_agreement.csv
    Figures/{pdf,tif,jpg}/FigS10_algo_discrimination.*
    Figures/{pdf,tif,jpg}/FigS11_algo_calibration.*
    Figures/{pdf,tif,jpg}/FigS12_algo_recalibration.*
    Outputs/s14_algo_*.csv
=============================================================================
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (RandomForestClassifier,
                              HistGradientBoostingClassifier)
from sklearn.model_selection import (StratifiedKFold, RepeatedStratifiedKFold,
                                     GridSearchCV)
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

# =============================================================================
# 预先指定的算法集（8 个族）
# -----------------------------------------------------------------------------
# 沿革：旧文（PeerJ 2023）比较了 LDA / SVM / NN / KNN / DTC / RF 六族。
# 本次在保留其可对话性的同时，(i) 补入逻辑回归作为临床默认基线（旧文缺失），
# (ii) 用梯度提升（HGB）代表现代表格数据的提升类方法，(iii) 保留单树 DTC
# 以使"树模型也需要足够样本"这一点可与装袋/提升并列观察。
# =============================================================================
ALGOS = {
    "LR": dict(
        full="Logistic regression (elastic net)",
        grid={"C": [0.01, 0.1, 1.0], "l1_ratio": [0.2, 0.5, 1.0]},
        scaled=True),
    "LDA": dict(
        full="Linear discriminant analysis",
        grid=None, scaled=True),
    "SVM": dict(
        full="Support vector machine (RBF)",
        grid={"C": [1.0, 10.0], "gamma": ["scale", 0.01]},
        scaled=True),
    "KNN": dict(
        full="k-nearest neighbours",
        grid={"n_neighbors": [5, 11, 21], "weights": ["uniform", "distance"]},
        scaled=True),
    "DTC": dict(
        full="Decision tree",
        grid={"max_depth": [3, 5], "min_samples_leaf": [4, 8]},
        scaled=False),
    "MLP": dict(
        full="Neural network (MLP)",
        grid={"hidden_layer_sizes": [(10,), (32,)], "alpha": [1e-4, 1e-2]},
        scaled=True),
    "RF": dict(
        full="Random forest",
        grid=C.PARAM_GRIDS["RF"],                 # 与脚本 01 完全一致
        scaled=False),
    "HGB": dict(
        full="Gradient boosting",
        grid={"learning_rate": [0.05, 0.1], "max_iter": [200],
              "max_leaf_nodes": [15, 31]},
        scaled=False),
}
ORDER = list(ALGOS.keys())

# 8 族配色（与 common 的调色板协调，彼此可区分）
AC = {"LR": "#1f77b4", "LDA": "#ff7f0e", "SVM": "#2ca02c", "KNN": "#d62728",
      "DTC": "#9467bd", "MLP": "#8c564b", "RF": "#17becf", "HGB": "#bcbd22"}


# =============================================================================
def make_pipe(key, params=None, seed=C.SEED):
    """按算法构建 Pipeline。插补（必要时标准化）全部在 Pipeline 内，只在训练折 fit。"""
    cfg = ALGOS[key]
    if key == "LR":
        est = LogisticRegression(penalty="elasticnet", solver="saga",
                                 max_iter=5000, random_state=seed)
    elif key == "LDA":
        est = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    elif key == "SVM":
        est = SVC(probability=True, random_state=seed, cache_size=1000)
    elif key == "KNN":
        est = KNeighborsClassifier()
    elif key == "DTC":
        est = DecisionTreeClassifier(random_state=seed)
    elif key == "MLP":
        est = MLPClassifier(random_state=seed, max_iter=3000)
    elif key == "RF":
        est = RandomForestClassifier(random_state=seed, n_jobs=1)
    elif key == "HGB":
        est = HistGradientBoostingClassifier(random_state=seed)
    else:
        raise KeyError(key)

    steps = [("imp", SimpleImputer(strategy="median"))]
    if cfg["scaled"]:
        steps.append(("scl", StandardScaler()))
    steps.append(("clf", est))
    pipe = Pipeline(steps)
    if params:
        pipe.set_params(**{f"clf__{k}": v for k, v in params.items()})
    return pipe


def tune(key, X, y, cv=C.CV_FOLDS, seed=C.SEED):
    """在推导整池上做网格搜索（与脚本 01 同协议）。无网格的算法直接返回空参。"""
    g = ALGOS[key]["grid"]
    if not g:
        return {}
    base = make_pipe(key, seed=seed)
    gs = GridSearchCV(base, {f"clf__{k}": v for k, v in g.items()},
                      scoring="roc_auc",
                      cv=StratifiedKFold(cv, shuffle=True, random_state=seed),
                      n_jobs=-1, refit=True)
    gs.fit(X, y)
    best = {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}
    return best


# =============================================================================
# 再校准（交叉拟合，仅在地理队列内部）
# =============================================================================
def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-8, 1 - 1e-8)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def crossfit_intercept_only(y, p, seed=C.SEED, k=5):
    """仅截距再校准（斜率固定 1）：交叉拟合，返回袋外再校准概率。"""
    y = np.asarray(y, int)
    lo = _logit(p)
    out = np.full(len(y), np.nan)
    skf = StratifiedKFold(k, shuffle=True, random_state=seed)
    for tr, te in skf.split(lo.reshape(-1, 1), y):
        a = 0.0
        for _ in range(80):
            mu = _sigmoid(a + lo[tr])
            w = np.clip(mu * (1 - mu), 1e-9, None)
            g = np.sum(y[tr] - mu)
            h = -np.sum(w)
            step = g / h if h != 0 else 0.0
            a_new = a - step
            if abs(a_new - a) < 1e-10:
                a = a_new
                break
            a = a_new
        out[te] = _sigmoid(a + lo[te])
    return out


def crossfit_slope_intercept(y, p, seed=C.SEED, k=5):
    """斜率+截距再校准（logistic）：交叉拟合，返回袋外再校准概率。"""
    y = np.asarray(y, int)
    lo = _logit(p)
    out = np.full(len(y), np.nan)
    skf = StratifiedKFold(k, shuffle=True, random_state=seed)
    for tr, te in skf.split(lo.reshape(-1, 1), y):
        b = C._irls(np.column_stack([np.ones_like(lo[tr]), lo[tr]]), y[tr])
        out[te] = _sigmoid(b[0] + b[1] * lo[te])
    return out


# =============================================================================
# 内部估计：50 折
# =============================================================================
def internal_cv(key, params, X, y, verbose=True):
    """50 折 = 10 折 × 5 次重复。

    与脚本 01 的折结构**完全一致**（每次重复新建 StratifiedKFold，
    random_state = SEED + rep；折内估计器种子 = SEED + rep*10 + k），
    以便 RF 的内部估计可与正文 Table 2 的 0.832 对齐核对。
    """
    Xv = X.reset_index(drop=True)
    fold_auc, oof_rep = [], []
    for rep in range(C.CV_REPEAT):
        skf = StratifiedKFold(C.CV_FOLDS, shuffle=True,
                              random_state=C.SEED + rep)
        oof = np.full(len(y), np.nan)
        for k, (tr, te) in enumerate(skf.split(Xv, y)):
            pipe = make_pipe(key, params, seed=C.SEED + rep * 10 + k)
            pipe.fit(Xv.iloc[tr], y[tr])
            p = pipe.predict_proba(Xv.iloc[te])[:, 1]
            oof[te] = p
            fold_auc.append(roc_auc_score(y[te], p))
        oof_rep.append(oof)
    pooled = np.mean([roc_auc_score(y, o) for o in oof_rep])
    oof_mean = np.mean(oof_rep, axis=0)
    thr = C.youden_threshold(y, oof_mean)
    bm = C.binary_metrics(y, oof_mean, thr)
    cm = C.calibration_metrics(y, oof_mean)
    if verbose:
        C.log(f"  {key:4s} CV AUC {np.mean(fold_auc):.3f} ± "
              f"{np.std(fold_auc, ddof=1):.3f} | pooled {pooled:.3f} | "
              f"slope {cm['Calibration slope']:.2f} | O:E {cm['O:E']:.2f}")
    return dict(fold_auc=np.array(fold_auc), oof_mean=oof_mean, pooled=pooled,
                thr=thr, bm=bm, cm=cm, params=params)


# =============================================================================
# 绘图
# =============================================================================
def paneltag(ax, tag, dx=-0.155, dy=1.075):
    ax.text(dx, dy, tag, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top", ha="left")


def fig_discrimination(ti, te, rows, path):
    """A 内部 vs 外部 AUC 哑铃图；B 逐算法 ΔAUC。

    上下两栏（而非左右）：两栏各占满 7.4 in 宽，8 个算法的标签与数值
    都不需要旋转即可排开，避免斜排标签互相压叠。
    """
    fig = plt.figure(figsize=(7.4, 6.0))
    gs = fig.add_gridspec(2, 1, hspace=0.42)

    # ---- A 哑铃图
    ax = fig.add_subplot(gs[0, 0])
    yy = np.arange(len(ORDER))[::-1]
    for i, k in enumerate(ORDER):
        a_in = ti.loc[k, "AUC (50 folds) mean"]
        a_ex = te.loc[k, "AUC"]
        ax.plot([a_in, a_ex], [yy[i], yy[i]], color="#cccccc", lw=1.4,
                zorder=1, solid_capstyle="round")
        ax.plot(a_in, yy[i], "o", ms=6.5, color=AC[k], zorder=3)
        ax.plot(a_ex, yy[i], "s", ms=5.5, color=AC[k], zorder=3,
                markerfacecolor="white", markeredgewidth=1.5)
        ax.text(max(a_in, a_ex) + 0.010, yy[i], f"{a_ex - a_in:+.3f}",
                va="center", fontsize=8.2, color="#444444")
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{k}" for k in ORDER])
    ax.set_xlabel("AUC")
    ax.set_xlim(0.55, 0.95)
    ax.set_title("A  Internal (50-fold CV, filled) vs external (open)",
                 loc="left", fontweight="bold", fontsize=10)
    C.style_axes(ax, axis="x")
    # 图例放在左下空白区：原先靠右下会压住 HGB 行的 ΔAUC 数值标签
    C.style_legend(ax, loc="lower left", fontsize=7.6, handles=[
        Line2D([], [], marker="o", ls="none", color="#555555",
               label="Internal CV"),
        Line2D([], [], marker="s", ls="none", mfc="white", mec="#555555",
               mew=1.5, label="External (geographical)"),
    ])

    # ---- B ΔAUC
    ax = fig.add_subplot(gs[1, 0])
    d = rows.copy()
    x = np.arange(len(d))
    ax.bar(x, d["delta_auc"], color=[AC[k] for k in d["model"]],
           alpha=0.9, width=0.62)
    C.ref_line(ax, y=0, color="#666666", lw=0.9, ls="--")
    for xi, v in zip(x, d["delta_auc"]):
        ax.text(xi, v - 0.006, f"{v:.3f}", ha="center", va="top",
                fontsize=8.2, color="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels(d["model"], fontsize=9)
    ax.set_ylabel("ΔAUC  (external − internal)")
    ax.set_ylim(min(-0.22, d["delta_auc"].min() - 0.07), 0.03)
    ax.set_title("B  Discrimination lost on transfer", loc="left",
                 fontweight="bold", fontsize=10)
    C.style_axes(ax, axis="y")

    C.save_figure(fig, path)


def fig_calibration(ti, te, path):
    """A 校准斜率；B O:E；C ICI；D 校准截距。内部 vs 外部。"""
    fig = plt.figure(figsize=(7.4, 5.8))
    gs = fig.add_gridspec(2, 2, wspace=0.34, hspace=0.55)

    specs = [("Calibration slope", "Calibration slope", 1.0, (0, 2.2)),
             ("O:E", "O:E ratio", 1.0, (0, 1.5)),
             ("ICI", "ICI", 0.0, (0, 0.30)),
             ("Calibration intercept", "Calibration intercept", 0.0,
              (-1.4, 0.6))]
    for j, (col_in, lab, ideal, ylim) in enumerate(specs):
        ax = fig.add_subplot(gs[j // 2, j % 2])
        x = np.arange(len(ORDER))
        vin = [ti.loc[k, col_in] for k in ORDER]
        vex = [te.loc[k, col_in] for k in ORDER]
        ax.bar(x - 0.19, vin, width=0.36, color="#bbbbbb",
               label="Internal (pooled OOF)")
        ax.bar(x + 0.19, vex, width=0.36,
               color=[AC[k] for k in ORDER], label="External")
        C.ref_line(ax, y=ideal, color="#d62728", lw=1.0, ls="--",
                   label="Ideal" if j == 0 else None)
        ax.set_xticks(x)
        ax.set_xticklabels(ORDER, fontsize=7.6)
        ax.set_ylabel(lab, fontsize=9)
        ax.set_ylim(*ylim)
        ax.set_title(f"{'ABCD'[j]}  {lab}", loc="left", fontweight="bold",
                     fontsize=9.5)
        C.style_axes(ax, axis="y")
        if j == 0:
            C.style_legend(ax, loc="upper left", fontsize=7.6, ncol=1)
    C.save_figure(fig, path)


def fig_recalibration(dec, path):
    """三联：A 判别力不随再校准移动；B O:E；C 校准斜率（后两栏展示尺度被修复）。

    三栏使用同一组"处理"图例：frozen / 仅截距 / 斜率+截距。
    这样"哪一分量被修复"在同一编码下可直接读出。
    """
    fig = plt.figure(figsize=(7.4, 3.7))
    gs = fig.add_gridspec(1, 3, wspace=0.46)
    fig.subplots_adjust(bottom=0.30)

    x = np.arange(len(dec))
    w = 0.26
    series = [("raw", "Frozen", "#bbbbbb"),
              ("intercept", "Intercept-only", "#ff7f0e"),
              ("slope", "Slope+intercept", "#2ca02c")]
    panels = [
        ("auc", "AUC", "A  AUC unchanged", None),
        ("oe", "O:E ratio", "B  O:E repaired", 1.0),
        ("slope", "Calibration slope", "C  Slope repaired", 1.0),
    ]
    for j, (stem, ylab, title, ideal) in enumerate(panels):
        ax = fig.add_subplot(gs[0, j])
        for si, (suf, lab, col) in enumerate(series):
            ax.bar(x + (si - 1) * w, dec[f"{stem}_{suf}"], width=w - 0.02,
                   color=col, alpha=0.92, label=lab)
        if ideal is not None:
            C.ref_line(ax, y=ideal, color="#d62728", lw=1.0, ls="--")
        vals = dec[[f"{stem}_{s}" for s, _, _ in series]]
        lo, hi = vals.min().min(), vals.max().max()
        if ideal is not None:
            lo, hi = min(lo, ideal - 0.06), max(hi, ideal + 0.06)
        pad = max(0.03, (hi - lo) * 0.20)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xticks(x)
        ax.set_xticklabels(dec["Model"], fontsize=7.2, rotation=90, ha="center")
        ax.set_ylabel(ylab, fontsize=9)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=9.5)
        C.style_axes(ax, axis="y")

    handles = [Patch(facecolor=c, label=l) for _, l, c in series]
    lg = fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.5,
                    frameon=True, framealpha=0.9, edgecolor="#cccccc",
                    bbox_to_anchor=(0.5, 0.005))
    lg.get_frame().set_linewidth(0.6)
    C.save_figure(fig, path)


# =============================================================================
def render_figures(t11, t12, t14):
    """由三张结果表出图（与计算解耦，便于单独重绘）。"""
    ti = t11.set_index("Model").loc[ORDER]
    te = t12.set_index("Model").loc[ORDER]
    rows = t12[["Model", "Delta AUC"]].rename(
        columns={"Model": "model", "Delta AUC": "delta_auc"})
    fig_discrimination(ti, te, rows, "FigS10_algo_discrimination")
    fig_calibration(ti, te, "FigS11_algo_calibration")
    fig_recalibration(t14, "FigS12_algo_recalibration")


def main():
    C.log_init("14_algo_comparison")

    # 仅重绘：复用已落盘的结果表，跳过全部计算（改图时用，约 340 s -> 数秒）
    if "--figs-only" in sys.argv:
        def _rt(n):
            return pd.read_csv(os.path.join(C.TABLE_DIR, f"{n}.csv"))
        C.section("仅重绘模式：复用 Tables/ 中已算好的结果表")
        render_figures(_rt("Table11_algo_internal"),
                       _rt("Table12_algo_external"),
                       _rt("Table14_algo_recalibration_decomposition"))
        C.section("脚本 14（仅重绘）完成")
        return

    t0 = time.time()
    C.section("多算法横向对比：外部验证损失的可分离性")

    cohorts, _ = C.load_cohorts()
    Xd, yd = C.derivation_label(cohorts["original"])
    Xg, yg = C.derivation_label(cohorts["geo"])
    C.log(f"  推导 n={len(yd)} 事件={int(yd.sum())} | "
          f"地理 n={len(yg)} 事件={int(yg.sum())}")

    spec = pd.read_csv(os.path.join(C.OUT_DIR, "s01_summary.csv"))
    auc_dev_rf = float(pd.read_csv(os.path.join(C.OUT_DIR, "s01_fold_auc.csv")
                                   )["RF"].mean())

    # ---------------- 逐算法：调参 → 内部 50 折 → 外部 ----------------
    C.section("逐算法：调参（推导整池 10 折网格）")
    tuned = {}
    for k in ORDER:
        tuned[k] = tune(k, Xd, yd)
        C.log(f"  {k:4s} best = {tuned[k] if tuned[k] else '(无超参)'}")

    C.section("逐算法：内部 50 折交叉验证")
    IN = {k: internal_cv(k, tuned[k], Xd, yd) for k in ORDER}

    C.section("逐算法：在地理队列上的外部验证（模型冻结，地理集不参与任何拟合）")
    EX = {}
    for k in ORDER:
        pipe = make_pipe(k, tuned[k])
        pipe.fit(Xd, yd)
        pg = pipe.predict_proba(Xg)[:, 1]
        a = roc_auc_score(yg, pg)
        lo, hi = C.boot_ci(yg, pg, lambda u, v: roc_auc_score(u, v), n_iter=2000)
        thr = IN[k]["thr"]                      # 阈值只来自推导集
        bm = C.binary_metrics(yg, pg, thr)
        cm = C.calibration_metrics(yg, pg)
        EX[k] = dict(p=pg, auc=a, lo=lo, hi=hi, bm=bm, cm=cm, thr=thr)
        C.log(f"  {k:4s} AUC {a:.3f} ({lo:.3f}-{hi:.3f}) | slope "
              f"{cm['Calibration slope']:.2f} | intercept "
              f"{cm['Calibration intercept']:+.2f} | O:E {cm['O:E']:.2f} | "
              f"ICI {cm['ICI']:.3f} | Brier {cm['Brier']:.3f}")

    # ---------------- Table11 内部 ----------------
    rows = []
    for k in ORDER:
        d = IN[k]
        rows.append({
            "Model": k, "Full name": ALGOS[k]["full"],
            "AUC (50 folds) mean": round(d["fold_auc"].mean(), 3),
            "AUC (50 folds) SD": round(d["fold_auc"].std(ddof=1), 3),
            "AUC (pooled OOF)": round(d["pooled"], 3),
            "Sensitivity": round(d["bm"]["Sensitivity"], 3),
            "Specificity": round(d["bm"]["Specificity"], 3),
            "Brier": round(d["cm"]["Brier"], 3),
            "Calibration slope": round(d["cm"]["Calibration slope"], 3),
            "Calibration intercept": round(d["cm"]["Calibration intercept"], 3),
            "ICI": round(d["cm"]["ICI"], 3), "O:E": round(d["cm"]["O:E"], 3),
            "Youden threshold": round(d["thr"], 3),
            "Best hyperparameters": str(d["params"]),
        })
    t11 = pd.DataFrame(rows).set_index("Model").loc[ORDER].reset_index()
    C.save_table(t11, "Table11_algo_internal")

    # ---------------- Table12 外部 ----------------
    rows = []
    for k in ORDER:
        d = EX[k]
        rows.append({
            "Model": k, "Full name": ALGOS[k]["full"],
            "AUC": round(d["auc"], 3),
            "AUC 95% CI": f"{d['lo']:.3f}-{d['hi']:.3f}",
            "AUC (internal, 50 folds)": round(IN[k]["fold_auc"].mean(), 3),
            "Delta AUC": round(d["auc"] - IN[k]["fold_auc"].mean(), 3),
            "Sensitivity": round(d["bm"]["Sensitivity"], 3),
            "Specificity": round(d["bm"]["Specificity"], 3),
            "PPV": round(d["bm"]["PPV"], 3), "NPV": round(d["bm"]["NPV"], 3),
            "Accuracy": round(d["bm"]["Accuracy"], 3),
            "Brier": round(d["cm"]["Brier"], 3),
            "Calibration slope": round(d["cm"]["Calibration slope"], 3),
            "Calibration intercept": round(d["cm"]["Calibration intercept"], 3),
            "ICI": round(d["cm"]["ICI"], 3), "O:E": round(d["cm"]["O:E"], 3),
            "Threshold (from derivation)": round(d["thr"], 3),
        })
    t12 = pd.DataFrame(rows).set_index("Model").loc[ORDER].reset_index()
    C.save_table(t12, "Table12_algo_external")

    C.section("内部排名 vs 外部排名")
    rank = pd.DataFrame({
        "model": ORDER,
        "auc_internal": [IN[k]["fold_auc"].mean() for k in ORDER],
        "auc_external": [EX[k]["auc"] for k in ORDER],
        "slope_err_internal": [abs(1 - IN[k]["cm"]["Calibration slope"]) for k in ORDER],
        "slope_err_external": [abs(1 - EX[k]["cm"]["Calibration slope"]) for k in ORDER],
        "oe_err_internal": [abs(1 - IN[k]["cm"]["O:E"]) for k in ORDER],
        "oe_err_external": [abs(1 - EX[k]["cm"]["O:E"]) for k in ORDER],
    })
    rank["rank_internal"] = rank["auc_internal"].rank(ascending=False).astype(int)
    rank["rank_external"] = rank["auc_external"].rank(ascending=False).astype(int)
    rank["rank_shift"] = rank["rank_external"] - rank["rank_internal"]
    sp = rank[["auc_internal", "auc_external"]].corr(method="spearman").iloc[0, 1]
    kd = rank[["auc_internal", "auc_external"]].corr(method="kendall").iloc[0, 1]
    sp_cal = rank[["slope_err_internal", "slope_err_external"]].corr(
        method="spearman").iloc[0, 1]
    C.log(f"  Spearman(AUC 内部, AUC 外部) = {sp:.3f}")
    C.log(f"  Kendall (AUC 内部, AUC 外部) = {kd:.3f}")
    C.log(f"  Spearman(校准斜率偏离 内部, 外部) = {sp_cal:.3f}")
    C.log("  内部最好的算法：" + rank.loc[rank['rank_internal'].idxmin(), 'model']
          + f"；外部最好的算法："
          + rank.loc[rank['rank_external'].idxmin(), 'model'])
    t15 = rank.copy()
    t15["spearman_auc_internal_external"] = round(sp, 3)
    t15["kendall_auc_internal_external"] = round(kd, 3)
    t15["spearman_slope_error"] = round(sp_cal, 3)
    for c in ["auc_internal", "auc_external", "slope_err_internal",
              "slope_err_external", "oe_err_internal", "oe_err_external"]:
        t15[c] = t15[c].round(3)
    C.save_table(t15, "Table15_algo_rank_agreement")

    # ---------------- Table13 配对 AUC 差异（地理队列） ----------------
    C.section("地理队列上的配对 Bootstrap AUC 差异（Holm 校正）")
    prs = []
    for i in range(len(ORDER)):
        for j in range(i + 1, len(ORDER)):
            a, b = ORDER[i], ORDER[j]
            obs, lo, hi, pv = C.paired_boot_diff(yg, EX[a]["p"], EX[b]["p"],
                                                 n_iter=2000)
            prs.append({"Model A": a, "Model B": b, "AUC A": round(EX[a]["auc"], 3),
                        "AUC B": round(EX[b]["auc"], 3),
                        "Delta AUC (A-B)": round(obs, 3),
                        "95% CI": f"{lo:.3f} to {hi:.3f}",
                        "p (paired bootstrap)": round(pv, 4)})
    t13 = pd.DataFrame(prs)
    # Holm
    m = len(t13)
    order_p = np.argsort(t13["p (paired bootstrap)"].values)
    adj = np.empty(m, dtype=bool)
    prev = 0
    for rank_i, idx in enumerate(order_p):
        thresh = 0.05 / (m - rank_i)
        rej = t13["p (paired bootstrap)"].values[idx] < thresh
        prev = prev or rej
        adj[idx] = prev
    t13["Holm significant (0.05)"] = np.where(adj, "yes", "no")
    C.save_table(t13, "Table13_algo_pairwise_auc_diff")
    n_sig = int((t13["Holm significant (0.05)"] == "yes").sum())
    C.log(f"  共 {m} 对比较，Holm 校正后显著 {n_sig} 对")

    # ---------------- Table14 再校准分量分解 ----------------
    C.section("再校准分量分解（地理队列内交叉拟合 5 折）")
    rows = []
    for k in ORDER:
        p = EX[k]["p"]
        p_i = crossfit_intercept_only(yg, p)
        p_s = crossfit_slope_intercept(yg, p)
        cm0 = EX[k]["cm"]
        cm_i = C.calibration_metrics(yg, p_i)
        cm_s = C.calibration_metrics(yg, p_s)
        rows.append({
            "Model": k,
            # 判别力（应几乎不动）
            "auc_raw": round(roc_auc_score(yg, p), 3),
            "auc_intercept": round(roc_auc_score(yg, p_i), 3),
            "auc_slope": round(roc_auc_score(yg, p_s), 3),
            # 校准（应被修复）
            "slope_raw": round(cm0["Calibration slope"], 3),
            "slope_intercept": round(cm_i["Calibration slope"], 3),
            "slope_slope": round(cm_s["Calibration slope"], 3),
            "intercept_raw": round(cm0["Calibration intercept"], 3),
            "intercept_intercept": round(cm_i["Calibration intercept"], 3),
            "intercept_slope": round(cm_s["Calibration intercept"], 3),
            "oe_raw": round(cm0["O:E"], 3),
            "oe_intercept": round(cm_i["O:E"], 3),
            "oe_slope": round(cm_s["O:E"], 3),
            "ici_raw": round(cm0["ICI"], 3),
            "ici_intercept": round(cm_i["ICI"], 3),
            "ici_slope": round(cm_s["ICI"], 3),
            "brier_raw": round(cm0["Brier"], 3),
            "brier_intercept": round(cm_i["Brier"], 3),
            "brier_slope": round(cm_s["Brier"], 3),
        })
    t14 = pd.DataFrame(rows)
    C.save_table(t14, "Table14_algo_recalibration_decomposition")
    d_max = (t14["auc_slope"] - t14["auc_raw"]).abs().max()
    C.log(f"  再校准后 |ΔAUC| 最大 = {d_max:.3f}（判别力不可被再校准修复）")
    C.log(f"  再校准后 O:E 距 1.00 的平均距离：raw "
          f"{(t14['oe_raw'] - 1).abs().mean():.3f} → 截距 "
          f"{(t14['oe_intercept'] - 1).abs().mean():.3f} → 斜率 "
          f"{(t14['oe_slope'] - 1).abs().mean():.3f}")

    # ---------------- 图 ----------------
    render_figures(t11, t12, t14)

    # ---------------- 落盘中间结果 ----------------
    C.save_csv(t12, "s14_algo_external")
    C.save_csv(t15, "s14_algo_rank")
    pred = pd.DataFrame({"y": yg})
    for k in ORDER:
        pred[f"p_{k}"] = EX[k]["p"]
    C.save_csv(pred, "s14_geo_pred_algo")
    C.save_csv(t14, "s14_algo_recal")

    C.section(f"脚本 14 完成，总耗时 {C.timer(t0)}")


if __name__ == "__main__":
    main()
