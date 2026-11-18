# -*- coding: utf-8 -*-
"""
公共模块 —— 论文流水线（文章及图表/Code）
=============================================================================
研究叙事
    ① 复现 PeerJ 2023 (Dai Y et al., 10.7717/peerj.15797) 的方法与最佳模型（RF）
    ② 用地理外部队列（另一家医院）做外部验证 → 判别力明显下降
    ③ 把时间扩展队列逐步并入推导序列并重新调参，再次在地理集上验证
       → 判别力与**校准度**持续改善 = 模型可持续改进

★ 全流程只建立**随机森林（RF）**一个模型（用户指令 2026-09-20）。
   RF 即旧文报告的最佳模型，因此本研究将其作为预先指定的唯一模型。

三个队列（统一口径）
    original  442 例   本院 2017-2022（旧文队列；label 1=无 / 2=有 → 0/1）
    temporal  364 例   本院后续时段（label 0/1；gender 0=男 / 1=女）
    geo       177 例   另一家医院（label 0/1；gender 0=男 / 1=女）

统一后的编码
    label  1 = 阻塞性（CAD-RADS 3-5，狭窄 ≥50%）；0 = 非阻塞性
    gender 0 = 男；1 = 女（原文件 original 为 1=男/2=女，已转换）

★ 抗数据泄露的核心约定（全流程强制）
    1) 缺失值插补封装在 Pipeline 内，**只在训练折上 fit**（RF 无需标准化）
    2) 阈值（Youden）只能来自推导集（或推导集的交叉验证袋外预测），绝不取自外部队列
    3) 地理集（geo）**不参与任何建模决策**（不调参、不选变量、不定阈值）
    4) 外部队列的预处理复用推导集拟合好的 Pipeline，不在外部集上重新 fit
=============================================================================
"""
from __future__ import annotations

import os
import json
import time
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (StratifiedKFold, RepeatedStratifiedKFold,
                                     GridSearchCV)
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss

warnings.filterwarnings("ignore")

# =============================================================================
# 路径
# =============================================================================
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 文章及图表
CODE_DIR  = os.path.join(ROOT, "Code")
FIG_DIR   = os.path.join(ROOT, "Figures")
TABLE_DIR = os.path.join(ROOT, "Tables")
MODEL_DIR = os.path.join(ROOT, "Models")
LOG_DIR   = os.path.join(ROOT, "Logs")
OUT_DIR   = os.path.join(ROOT, "Outputs")

DATA_DIR  = os.path.join(os.path.dirname(ROOT), "数据")   # 项目根/数据

for _d in (FIG_DIR, TABLE_DIR, MODEL_DIR, LOG_DIR, OUT_DIR):
    os.makedirs(_d, exist_ok=True)

FIG_FORMATS = ["pdf", "tif", "jpg"]
FIG_DPI     = {"pdf": 600, "tif": 600, "jpg": 600}

SEED      = 123
CV_FOLDS  = 10
CV_REPEAT = 5

# =============================================================================
# 变量定义（与旧文 19 个候选变量完全一致）
# =============================================================================
SHORT = {
    "label": "label", "age": "age", "gender": "gender",
    "systolic blood pressure": "sbp", "diastolic blood pressure": "dbp",
    "mean arterial pressure": "map", "pulse pressure index": "ppi",
    "pulse pressure": "pp", "hypertension": "htn", "diabetes": "dm",
    "smoking": "smoke", "dyslipidemia": "dyslip",
    "total cholesterol": "tc", "triglycerides": "tg",
    "LDL-C": "ldl", "HDL-C": "hdl", "uric acid": "ua",
    "plasma fibrinogen": "fib", "creatinine": "cr", "urea": "urea",
    "siabetes": "dm",
}

NUM = ["age", "sbp", "dbp", "map", "ppi", "pp",
       "tc", "tg", "ldl", "hdl", "ua", "fib", "cr", "urea"]
BIN = ["gender", "htn", "dm", "smoke", "dyslip"]
FEATURES = NUM + BIN                      # 19 个候选变量，顺序固定
TARGET = "label"

LABEL_EN = {
    "age": "Age (years)", "gender": "Female sex", "sbp": "Systolic BP (mmHg)",
    "dbp": "Diastolic BP (mmHg)", "map": "Mean arterial pressure (mmHg)",
    "ppi": "Pulse pressure index", "pp": "Pulse pressure (mmHg)",
    "htn": "Hypertension", "dm": "Diabetes mellitus", "smoke": "Smoking",
    "dyslip": "Dyslipidaemia", "tc": "Total cholesterol (mmol/L)",
    "tg": "Triglycerides (mmol/L)", "ldl": "LDL-C (mmol/L)",
    "hdl": "HDL-C (mmol/L)", "ua": "Uric acid (umol/L)",
    "fib": "Plasma fibrinogen (g/L)", "cr": "Creatinine (umol/L)",
    "urea": "Urea (mmol/L)",
}

# 越界值（疑似录入错误）→ 置缺失；age 不参与（用户明确保留 15 岁那例）
PLAUSIBLE_RANGE = {"sbp": (60, 300), "dbp": (30, 200), "map": (40, 220),
                   "ppi": (0.05, 0.95), "pp": (5, 180),
                   "tc": (1.0, 25.0), "tg": (0.2, 30.0), "ldl": (0.2, 20.0),
                   "hdl": (0.1, 6.0), "ua": (80, 1200), "fib": (0.3, 15.0),
                   "cr": (20, 1500), "urea": (0.5, 50.0)}

# =============================================================================
# 绘图规范
# =============================================================================
plt.rcParams.update({
    # ---- 字体：Arial + 嵌入 TrueType(Type 42)，符合 SCI 投稿系统要求 ----
    "font.family": "Arial",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.unicode_minus": False,
    # ---- 画布 ----
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "figure.dpi": 120,
    # ---- 坐标轴：仅保留左/下脊，细线，刻度朝外 ----
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.9,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    # ---- 网格：浅灰虚线、置于数据之下（旧脚本 ax.grid(ls="--", alpha=0.3)）----
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#b8b8b8",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.30,
    # ---- 线与误差棒 ----
    "lines.linewidth": 1.6,
    "lines.markersize": 4.5,
    "lines.solid_capstyle": "round",
    "errorbar.capsize": 3,
    # ---- 图例：白底半透明细边框 ----
    "legend.frameon": True,
    "legend.framealpha": 0.90,
    "legend.edgecolor": "#cccccc",
    "legend.fancybox": False,
})

C_ORIG = "#1f77b4"
C_TEMP = "#ff7f0e"
C_GEO  = "#2ca02c"
C_RED  = "#d62728"
C_PUR  = "#9467bd"
C_GRAY = "#7f7f7f"
C_BROWN = "#8c564b"
COHORT_C = {"original": C_ORIG, "temporal": C_TEMP, "geo": C_GEO,
            "derivation": C_PUR, "enlarged": C_PUR}


# ---------------------------------------------------------------------------
# 图细节助手（对齐「旧记录/Code」的出图规范）
# ---------------------------------------------------------------------------
def ref_line(ax, diag=False, y=None, x=None, color="#666666", lw=0.9, ls="--",
             alpha=None, label=None, zorder=0):
    """参考线。diag=True 画 45 度理想线（黑细虚线，alpha 0.5）；
    否则画 y=... / x=... 水平或垂直参考线（默认灰虚线）。"""
    if diag:
        return ax.plot([0, 1], [0, 1], ls=ls, lw=0.8 if lw is None else lw,
                       color="black", alpha=0.5 if alpha is None else alpha,
                       label=label, zorder=zorder)[0]
    if y is not None:
        return ax.axhline(y, ls=ls, lw=lw, color=color,
                          alpha=1.0 if alpha is None else alpha,
                          label=label, zorder=zorder)
    return ax.axvline(x, ls=ls, lw=lw, color=color,
                      alpha=1.0 if alpha is None else alpha,
                      label=label, zorder=zorder)


def stat_box(ax, text, loc="upper left", fontsize=8.5):
    """指标文本框：白底细边框，置于面板四角之一。"""
    corner = {"upper left": (0.035, 0.965, "left", "top"),
              "upper right": (0.965, 0.965, "right", "top"),
              "lower left": (0.035, 0.035, "left", "bottom"),
              "lower right": (0.965, 0.035, "right", "bottom")}[loc]
    return ax.text(corner[0], corner[1], text, transform=ax.transAxes,
                   va=corner[3], ha=corner[2], fontsize=fontsize,
                   linespacing=1.55,
                   bbox=dict(boxstyle="round,pad=0.38", fc="white",
                             ec="#cccccc", lw=0.7, alpha=0.92))


def style_legend(ax, loc="best", ncol=1, fontsize=None, **kw):
    """图例样式：白底半透明细边框（旧脚本用 frameon=True / framealpha=0.9）。"""
    lg = ax.legend(loc=loc, ncol=ncol, frameon=True, framealpha=0.9,
                   edgecolor="#cccccc", fancybox=False,
                   fontsize=fontsize or plt.rcParams["legend.fontsize"], **kw)
    lg.get_frame().set_linewidth(0.6)
    return lg


def style_axes(ax, grid=True, axis="both", alpha=0.30):
    """结果类面板统一样式：关顶/右脊 + 浅虚线网格置于数据之下。"""
    for s in ("top", "right"):
        if s in ax.spines:
            ax.spines[s].set_visible(False)
    if grid:
        ax.grid(True, axis=axis, linestyle="--", linewidth=0.6,
                alpha=alpha, color="#b8b8b8")
        ax.set_axisbelow(True)
    return ax


# =============================================================================
# 日志
# =============================================================================
_LOG_FH = None


def log_init(tag):
    """开启本次运行的日志文件（同时回显到 stdout）。"""
    global _LOG_FH
    p = os.path.join(LOG_DIR, f"{tag}_{datetime.now():%Y%m%d_%H%M%S}.log")
    _LOG_FH = open(p, "w", encoding="utf-8")
    log(f"log file -> {p}")
    return p


def log(msg=""):
    print(msg, flush=True)
    if _LOG_FH is not None:
        _LOG_FH.write(str(msg) + "\n")
        _LOG_FH.flush()


def section(title, ch="="):
    log("\n" + ch * 78)
    log(title)
    log(ch * 78)


# =============================================================================
# 出图 / 出表
# =============================================================================
def save_figure(fig, name, formats=None, verbose=True):
    """三格式（pdf/tif/jpg）保存到 Figures/<fmt>/<name>.<fmt>。"""
    formats = formats or FIG_FORMATS
    paths = []
    for fmt in formats:
        d = os.path.join(FIG_DIR, fmt)
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, f"{name}.{fmt}")
        kw = {}
        if fmt == "tif":
            kw = {"pil_kwargs": {"compression": "tiff_lzw"}}
        elif fmt == "jpg":
            kw = {"pil_kwargs": {"quality": 95}}
        fig.savefig(p, format=fmt, dpi=FIG_DPI.get(fmt, 600),
                    bbox_inches="tight", transparent=False, **kw)
        paths.append(p)
    if verbose:
        log(f"  [figure] {name}  ->  pdf/tif/jpg")
    plt.close(fig)
    return paths


def save_table(df, name, index=False):
    p = os.path.join(TABLE_DIR, f"{name}.csv")
    df.to_csv(p, index=False, encoding="utf-8-sig")
    log(f"  [table ] {name}.csv  ({len(df)} rows)")
    return p


def save_csv(df, name):
    """中间结果（不进 Tables 主目录）。"""
    p = os.path.join(OUT_DIR, f"{name}.csv")
    df.to_csv(p, index=False, encoding="utf-8-sig")
    return p


def read_csv_out(name):
    return pd.read_csv(os.path.join(OUT_DIR, f"{name}.csv"))


def save_pickle(obj, name):
    p = os.path.join(MODEL_DIR, name)
    with open(p, "wb") as f:
        pickle.dump(obj, f)
    log(f"  [model ] {name}")
    return p


def load_pickle(name):
    with open(os.path.join(MODEL_DIR, name), "rb") as f:
        return pickle.load(f)


def save_json(obj, name):
    p = os.path.join(MODEL_DIR, name)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    return p


# =============================================================================
# 数据读取与统一口径
# =============================================================================
def _tidy(df, is_original=False):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns=SHORT)

    # label → 1 = 阻塞性
    lab = pd.to_numeric(df["label"], errors="coerce")
    if set(lab.dropna().unique()) <= {1, 2}:          # original: 1/2
        lab = lab.map({1: 0, 2: 1})
    df["label"] = lab.astype(float)

    # gender → 0 = 男 / 1 = 女
    g = pd.to_numeric(df["gender"], errors="coerce")
    if set(g.dropna().unique()) <= {1, 2}:            # original: 1=男 2=女
        g = g.map({1: 0, 2: 1})
    df["gender"] = g

    for c in NUM + BIN + ["label"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["male"] = (df["gender"] == 0).astype(int)
    return df.reset_index(drop=True)


def flag_outliers(df, cohort):
    """策略 D：越界单元格置缺失（不删行、不改值），返回置缺明细。"""
    marks = []
    for c, (lo, hi) in PLAUSIBLE_RANGE.items():
        if c not in df.columns:
            continue
        m = ((df[c] < lo) | (df[c] > hi)).fillna(False)
        if m.any():
            for i in df.index[m]:
                marks.append({"cohort": cohort, "row": int(i), "var": c,
                              "value": float(df.at[i, c]), "action": "set_missing"})
            df.loc[m, c] = np.nan
    return df, marks


def load_cohorts(apply_strategy_d=True, verbose=True):
    """读入三个队列，统一编码并做越界处置。

    返回 (cohorts: dict, outlier_marks: DataFrame)
    """
    import openpyxl

    wb = openpyxl.load_workbook(os.path.join(DATA_DIR, "original data.xlsx"),
                                read_only=True, data_only=True)
    rows = list(wb["data20221010"].iter_rows(values_only=True))
    o = pd.DataFrame(rows[1:], columns=[str(c).strip() for c in rows[0]])
    o = o.dropna(how="all")
    o = _tidy(o, is_original=True)
    o["ID"] = [f"ORI{i+1:04d}" for i in range(len(o))]

    t = _tidy(pd.read_csv(os.path.join(DATA_DIR,
                "Temporal external validation cohort.csv")))
    g = _tidy(pd.read_csv(os.path.join(DATA_DIR,
                "Geographical external validation cohort.csv")))

    cohorts = {"original": o, "temporal": t, "geo": g}
    marks = []
    if apply_strategy_d:
        for k in cohorts:
            cohorts[k], mk = flag_outliers(cohorts[k], k)
            marks += mk
    marks = pd.DataFrame(marks)

    if verbose:
        section("队列概览（统一口径）")
        for k, d in cohorts.items():
            log(f"  {k:9s} n={len(d):4d}  事件={int(d[TARGET].sum()):4d} "
                f"({d[TARGET].mean()*100:5.1f}%)  男性={d['male'].mean()*100:5.1f}%  "
                f"缺失格={int(d[FEATURES].isna().sum().sum())}")
        if len(marks):
            log(f"  策略 D 置缺失：{len(marks)} 格 -> "
                + ", ".join(f"{r.cohort}/{r['var']}" for _, r in marks.iterrows()))
    return cohorts, marks


def derivation_label(df):
    """返回 (X, y)。X 为 19 列 DataFrame。"""
    return df[FEATURES].copy(), df[TARGET].astype(int).values


# =============================================================================
# 模型与 Pipeline（插补 封装在 Pipeline 内，只在训练折 fit）
# -----------------------------------------------------------------------------
# ★ 单一模型设计（用户指令 2026-09-20）：全流程只建立随机森林（RF）一个模型。
#   不再建立/比较 SVM、NN、KNN、DTC、LDA。RF 为旧文报告的最佳模型，
#   本研究中作为预先指定的唯一模型贯穿推导、外验与成长分析。
# =============================================================================
MODEL_NAMES = ["RF"]
MODEL_FULL = {"RF": "Random forest"}
PRIMARY_MODEL = "RF"


def build_estimator(name=PRIMARY_MODEL, params=None, seed=SEED):
    if name != "RF":
        raise ValueError(
            f"本流水线仅建立随机森林（RF）一个模型，收到 name={name!r}")
    est = RandomForestClassifier(random_state=seed, n_jobs=1)

    # RF 对量纲不敏感，无需标准化；缺失值插补仍封在 Pipeline 内，只在训练折 fit
    steps = [("imp", SimpleImputer(strategy="median")), ("clf", est)]
    pipe = Pipeline(steps)
    if params:
        # 允许传不带前缀的原始参数名（自动补 clf__）
        fixed = {}
        for k, v in params.items():
            fixed[k if k.split("__")[0] in ("clf", "imp") else f"clf__{k}"] = v
        pipe.set_params(**fixed)
    return pipe


def prefix_params(name, grid):
    """按是否包 Pipeline 自动加 clf__ 前缀。"""
    if name != "RF":
        raise ValueError(f"本流水线仅支持 RF，收到 name={name!r}")
    grids = grid if isinstance(grid, list) else [grid]
    out = []
    for d in grids:
        out.append({f"clf__{k}": v for k, v in d.items()})
    return out if isinstance(grid, list) else out[0]


PARAM_GRIDS = {
    "RF": {"n_estimators": [200, 500], "max_depth": [None, 5, 10],
           "min_samples_leaf": [1, 2, 4]},
}
# 成长分析用的精简网格（控制运行时间）
PARAM_GRIDS_FAST = {
    "RF": {"n_estimators": [500], "max_depth": [None, 10],
           "min_samples_leaf": [1, 2, 4]},
}


def tune_model(X, y, name=PRIMARY_MODEL, grid=None, cv=CV_FOLDS, seed=SEED,
               scoring="roc_auc", n_jobs=-1, verbose=0):
    """在给定推导集上做网格搜索（交叉验证只在推导集内部）。"""
    grid = PARAM_GRIDS if grid is None else grid
    base = build_estimator(name, seed=seed)
    gs = GridSearchCV(base, prefix_params(name, grid[name]),
                      scoring=scoring,
                      cv=StratifiedKFold(cv, shuffle=True, random_state=seed),
                      n_jobs=n_jobs, refit=True, verbose=verbose)
    gs.fit(X, y)
    return gs


# =============================================================================
# 评价指标
# =============================================================================
def youden_threshold(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])


def binary_metrics(y, p, thr):
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p) >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    n = tp + fp + fn + tn
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    acc = (tp + tn) / n if n else np.nan
    f1 = (2 * ppv * sens / (ppv + sens)
          if (ppv + sens) and not np.isnan(ppv) and not np.isnan(sens) else np.nan)
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "n": n,
            "Sensitivity": sens, "Specificity": spec, "PPV": ppv,
            "NPV": npv, "Accuracy": acc, "F1": f1}


def _irls(X, y, max_iter=60):
    """极简 IRLS 逻辑回归（避免 statsmodels 依赖）。X 已含截距列。"""
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -30, 30)
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(mu * (1 - mu), 1e-9, None)
        z = eta + (y - mu) / w
        WX = X * w[:, None]
        try:
            beta_new = np.linalg.solve(X.T @ WX, WX.T @ z)
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(beta_new - beta)) < 1e-9:
            beta = beta_new
            break
        beta = beta_new
    return beta


def calibration_metrics(y, p, n_bins=10):
    """校准：斜率/截距（logistic）、Brier、ICI、O:E、Hosmer-Lemeshow。"""
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 1e-8, 1 - 1e-8)
    logit = np.log(p / (1 - p))
    out = {}

    out["Brier"] = float(brier_score_loss(y, p))
    out["O:E"] = float(y.sum() / p.sum()) if p.sum() > 0 else np.nan

    # 校准斜率（y ~ a + b*logit）
    X = np.column_stack([np.ones_like(logit), logit])
    b = _irls(X, y)
    out["Calibration slope"] = float(b[1])
    # 校准截距 in the large（固定斜率 = 1）：logit(mu) = a + logit(p)
    a = 0.0
    for _ in range(60):
        eta = a + logit
        mu = 1 / (1 + np.exp(-eta))
        w = np.clip(mu * (1 - mu), 1e-9, None)
        g = np.sum(y - mu)
        h = -np.sum(w)
        step = g / h if h != 0 else 0.0
        a_new = a - step
        if abs(a_new - a) < 1e-10:
            a = a_new
            break
        a = a_new
    out["Calibration intercept"] = float(a)

    # ICI（等分位十分位）
    idx = np.argsort(p)
    bins = np.array_split(idx, n_bins)
    diffs, ws = [], []
    for b_ in bins:
        if len(b_) == 0:
            continue
        diffs.append(abs(y[b_].mean() - p[b_].mean()))
        ws.append(len(b_))
    out["ICI"] = float(np.average(diffs, weights=ws))

    # Hosmer-Lemeshow
    grp = pd.qcut(pd.Series(p), n_bins, duplicates="drop")
    obs = pd.Series(y).groupby(grp, observed=True).sum().values
    exp = pd.Series(p).groupby(grp, observed=True).sum().values
    nn = pd.Series(p).groupby(grp, observed=True).size().values
    with np.errstate(divide="ignore", invalid="ignore"):
        hl = np.nansum((obs - exp) ** 2 / (exp * (1 - exp / nn)))
    out["HL chi2"] = float(hl)
    out["HL df"] = int(len(obs) - 2)
    out["HL p"] = float(1 - __import__("scipy.stats", fromlist=["chi2"]).chi2.cdf(
        hl, max(1, len(obs) - 2)))
    out["n"] = int(len(y))
    out["events"] = int(y.sum())
    return out


def calibration_curve_bins(y, p, n_bins=10):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    grp = pd.qcut(pd.Series(p), n_bins, duplicates="drop")
    d = pd.DataFrame({"y": y, "p": p, "g": grp})
    t = d.groupby("g", observed=True).agg(obs=("y", "mean"), pred=("p", "mean"),
                                          n=("y", "size")).reset_index(drop=True)
    # 二项标准误
    t["se"] = np.sqrt(t["obs"] * (1 - t["obs"]) / t["n"])
    return t


def dca(y, p, thresholds=None):
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    if thresholds is None:
        thresholds = np.arange(0.01, 0.71, 0.01)
    n = len(y)
    prev = y.mean()
    rows = []
    for t in thresholds:
        pred = p >= t
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        nb = tp / n - fp / n * (t / (1 - t))
        nb_all = prev - (1 - prev) * (t / (1 - t))
        rows.append({"threshold": t, "model": nb, "treat_all": nb_all,
                     "treat_none": 0.0,
                     "standardized_net_benefit": nb / prev if prev > 0 else np.nan})
    return pd.DataFrame(rows)


def bootstrap_ci_metric(fn, n_iter=1000, alpha=0.05, seed=SEED):
    """对指标函数做百分位 Bootstrap 95% CI。fn(idx) -> float"""
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_iter):
        vals.append(fn(rng))
    vals = np.array([v for v in vals if v is not None and not np.isnan(v)])
    if len(vals) < 30:
        return np.nan, np.nan
    return float(np.percentile(vals, 100 * alpha / 2)), \
        float(np.percentile(vals, 100 * (1 - alpha / 2)))


def boot_ci(y, p, metric_fn, n_iter=1000, seed=SEED):
    """对 (y,p) 做有放回 Bootstrap 得到 metric_fn 的 95% CI。

    metric_fn(y, p) -> float
    """
    y = np.asarray(y)
    p = np.asarray(p)
    n = len(y)
    rng = np.random.RandomState(seed)
    vals = np.full(n_iter, np.nan)
    for i in range(n_iter):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            vals[i] = metric_fn(y[idx], p[idx])
        except Exception:
            vals[i] = np.nan
    vals = vals[~np.isnan(vals)]
    if len(vals) < 30:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def mean_roc(curves, n_grid=200):
    """把多条 ROC 曲线插值到公共 FPR 网格后求均值与 SD。"""
    grid = np.linspace(0, 1, n_grid)
    tprs = []
    for fpr, tpr in curves:
        tprs.append(np.interp(grid, fpr, tpr))
    tprs = np.array(tprs)
    tprs[:, 0] = 0.0
    tprs[:, -1] = 1.0
    return grid, tprs.mean(axis=0), tprs.std(axis=0), tprs


def smd_cont(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    s = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return np.nan if s == 0 else float((a.mean() - b.mean()) / s)


def smd_bin(p1, p2):
    s = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
    return np.nan if s == 0 else float((p1 - p2) / s)


# =============================================================================
# 统计检验小工具
# =============================================================================
def paired_boot_diff(y, p1, p2, n_iter=2000, seed=SEED):
    """同一批样本上两个模型 AUC 差值的配对 Bootstrap（BCa 简化：百分位）。"""
    y = np.asarray(y); p1 = np.asarray(p1); p2 = np.asarray(p2)
    n = len(y)
    obs = roc_auc_score(y, p1) - roc_auc_score(y, p2)
    rng = np.random.RandomState(seed)
    vals = np.full(n_iter, np.nan)
    for i in range(n_iter):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            vals[i] = roc_auc_score(y[idx], p1[idx]) - roc_auc_score(y[idx], p2[idx])
        except Exception:
            vals[i] = np.nan
    vals = vals[~np.isnan(vals)]
    if len(vals) < 50:
        return obs, np.nan, np.nan, np.nan
    lo, hi = np.percentile(vals, [2.5, 97.5])
    p_val = 2 * min((vals <= 0).mean(), (vals >= 0).mean())
    return float(obs), float(lo), float(hi), float(p_val)


def fmt(v, nd=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "NA"
    return f"{v:.{nd}f}"


def fmt_ci(v, lo, hi, nd=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "NA"
    if lo is None or hi is None or np.isnan(lo) or np.isnan(hi):
        return f"{v:.{nd}f}"
    return f"{v:.{nd}f} ({lo:.{nd}f}-{hi:.{nd}f})"


def timer(t0=None):
    if t0 is None:
        return time.time()
    return f"{time.time() - t0:.1f}s"
