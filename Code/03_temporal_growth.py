# -*- coding: utf-8 -*-
"""
03 —— 时间队列逐步并入 + 地理集再验证（模型成长 / 校准改善）
=============================================================================
设计
    推导序列 D(p) = original(442) + 时间队列的前 p 比例（分层抽样）
    p ∈ {0, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00}，每个 p 重复 R 次随机抽取
    每一步在 D(p) 上重新调参（5 折 CV 网格搜索）→ 冻结 → 在地理集上评价
    对照臂（关键）：把 original 用分层 Bootstrap 重采样到同样的 n，
        以分解「样本量增加的效应」与「新数据带来的信息增益」
★ 地理集只被评价，绝不参与调参、选变量或定阈值
输出
    Fig7  成长曲线·判别力（AUC vs 推导样本量；时间臂 vs 重采样对照臂）
    Fig8  成长曲线·校准度（斜率 / 截距 / Brier / ICI / O:E）
    Fig9  校准改善对比（p=0 / 0.40 / 1.00 的校准曲线）
    Fig10 外部集内交叉拟合再校准（探索性）
    Table6  逐步扩充的性能汇总
    TableS4 时间臂 vs 对照臂（配对 Wilcoxon / 混合模型 / 独立样本三种 p 口径）
=============================================================================
"""
import os
import sys
import json
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import t as tdist, wilcoxon

try:
    import statsmodels.formula.api as smf
    _STATSMODELS_OK = True
except Exception:                                    # pragma: no cover
    smf = None
    _STATSMODELS_OK = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

PCTS = [0.0, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00]
N_REPEAT = 10
GRID_FAST = {"RF": {"n_estimators": [500], "max_depth": [None, 10],
                    "min_samples_leaf": [1, 2, 4]}}


def _fmt_p(p):
    """p 值字符串化：与正文口径一致，避免 1e-8 被压成 '.00000'。"""
    if p is None or pd.isna(p):
        return "NA"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# =============================================================================
def strat_sample(df, frac, rng):
    """按结局分层抽取 frac 比例的行索引。"""
    if frac <= 0:
        return np.array([], dtype=int)
    idx = []
    for lab, sub in df.groupby(C.TARGET):
        k = int(round(len(sub) * frac))
        k = min(k, len(sub))
        if k > 0:
            idx.append(rng.choice(sub.index.values, size=k, replace=False))
    return np.sort(np.concatenate(idx)) if idx else np.array([], dtype=int)


def strat_bootstrap(df, n_target, rng):
    """分层有放回重采样，得到 n_target 行（对照臂：只增加样本量，不增加信息）。"""
    out = []
    labs = df[C.TARGET].values
    prop = pd.Series(labs).value_counts(normalize=True)
    for lab in prop.index:
        sub = df.index[df[C.TARGET] == lab].values
        k = int(round(n_target * prop[lab]))
        out.append(rng.choice(sub, size=k, replace=True))
    return np.sort(np.concatenate(out))


def eval_on_geo(model, Xg, yg, thr):
    p = model.predict_proba(Xg)[:, 1]
    a = roc_auc_score(yg, p)
    bm = C.binary_metrics(yg, p, thr)
    cm = C.calibration_metrics(yg, p)
    return p, a, bm, cm


def run_arm(Xd, yd, Xtemp, ytemp, Xg, yg, arm, tag, log_rows):
    """arm: 'temporal' 或 'control'。返回明细 DataFrame。"""
    rows = []
    Tm = pd.concat([Xtemp, pd.Series(ytemp, name=C.TARGET, index=Xtemp.index)],
                   axis=1)
    Dm = pd.concat([Xd, pd.Series(yd, name=C.TARGET, index=Xd.index)], axis=1)

    for pct in PCTS:
        for rep in range(N_REPEAT):
            rng = np.random.RandomState(1000 * (1 if arm == "temporal" else 2)
                                        + rep + int(pct * 100) * 7)
            n_add = int(round(pct * len(Tm)))
            if arm == "temporal":
                add_idx = strat_sample(Tm, pct, rng) if pct > 0 else np.array([], int)
                pool = pd.concat([Dm, Tm.loc[add_idx]], axis=0)
            else:
                pool = Dm.iloc[strat_bootstrap(Dm, len(Dm) + n_add, rng)]
            pool = pool.reset_index(drop=True)
            Xp = pool[C.FEATURES]; yp = pool[C.TARGET].astype(int).values

            gs = C.tune_model(Xp, yp, "RF", grid=GRID_FAST, cv=5,
                              seed=C.SEED + rep, n_jobs=-1)
            best = gs.best_estimator_
            # 阈值只能来自推导池（5 折袋外 Youden）
            oof = cross_val_predict(
                C.build_estimator("RF", {k.replace("clf__", ""): v
                                         for k, v in gs.best_params_.items()},
                                  seed=C.SEED + rep),
                Xp, yp, cv=StratifiedKFold(5, shuffle=True, random_state=C.SEED + rep),
                method="predict_proba", n_jobs=-1)[:, 1]
            thr = C.youden_threshold(yp, oof)

            pg, a, bm, cm = eval_on_geo(best, Xg, yg, thr)
            rows.append({
                "arm": arm, "pct_temporal": pct, "repeat": rep,
                "n_derive": int(len(yp)), "n_temporal": int(n_add),
                "cv_auc": float(gs.best_score_), "thr": float(thr),
                "geo_auc": float(a),
                **{f"geo_{k}": bm[k] for k in
                   ("Sensitivity", "Specificity", "PPV", "NPV", "Accuracy")},
                "geo_brier": cm["Brier"], "geo_slope": cm["Calibration slope"],
                "geo_intercept": cm["Calibration intercept"], "geo_ici": cm["ICI"],
                "geo_oe": cm["O:E"],
                "params": json.dumps({k.replace("clf__", ""): v
                                      for k, v in gs.best_params_.items()},
                                     default=str),
                "y_geo": yg.tolist(), "p_geo": pg.tolist(),
            })
            if rep == 0:
                C.log(f"    [{tag}] p={pct:.2f} n={len(yp):4d}  thr={thr:.3f}  "
                      f"geo AUC={a:.3f}  slope={cm['Calibration slope']:.2f}  "
                      f"intercept={cm['Calibration intercept']:+.2f}  "
                      f"Brier={cm['Brier']:.3f}  ICI={cm['ICI']:.3f}  "
                      f"O:E={cm['O:E']:.2f}")
    return pd.DataFrame(rows)


# =============================================================================
def fig_growth_discrimination(g, path_out):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6))

    def agg(df, col):
        a = df.groupby("pct_temporal").agg(
            n=("n_derive", "mean"), m=(col, "mean"), s=(col, "std"))
        a["se"] = a["s"] / np.sqrt(df.groupby("pct_temporal").size())
        return a

    ax = axes[0]
    for arm, col, lab in [("temporal", C.C_TEMP, "Contemporary data added"),
                          ("control", C.C_GRAY, "Row-count-matched resampled original")]:
        a = agg(g[g["arm"] == arm], "geo_auc")
        ax.errorbar(a["n"], a["m"], yerr=1.96 * a["se"], fmt="o-", color=col,
                    lw=1.7, ms=5, capsize=3, label=lab)
    base = g[(g["arm"] == "temporal") & (g["pct_temporal"] == 0)]["geo_auc"].mean()
    C.ref_line(ax, y=base, color=C.C_RED, ls=":", lw=1.1,
               label=f"Baseline (original only) {base:.3f}")
    ax.set_xlabel("Derivation sample size (n)")
    ax.set_ylabel("AUC in the geographical cohort")
    ax.set_title("A  Discrimination with data added", loc="left",
                 fontweight="bold")
    C.style_legend(ax, loc="lower right", fontsize=9)

    ax = axes[1]
    a = agg(g[g["arm"] == "temporal"], "geo_auc")
    ax.errorbar(a["n"], a["m"] - base, yerr=1.96 * a["se"], fmt="o-",
                color=C.C_TEMP, lw=1.7, ms=5, capsize=3)
    ac = agg(g[g["arm"] == "control"], "geo_auc")
    ax.errorbar(ac["n"], ac["m"] - base, yerr=1.96 * ac["se"], fmt="s--",
                color=C.C_GRAY, lw=1.4, ms=4, capsize=3)
    C.ref_line(ax, y=0, color=C.C_RED, ls=":", lw=1.1)
    ax.set_xlabel("Derivation sample size (n)")
    ax.set_ylabel("Change in AUC vs baseline")
    ax.set_title("B  Gain over baseline", loc="left",
                 fontweight="bold")
    fig.subplots_adjust(wspace=0.32)
    C.save_figure(fig, path_out)


def fig_growth_calibration(g, path_out):
    specs = [("geo_slope", "Calibration slope", 1.0, "Ideal = 1", C.C_GEO),
             ("geo_intercept", "Intercept (in the large)", 0.0, "Ideal = 0", C.C_BROWN),
             ("geo_brier", "Brier score", None, "Lower is better", C.C_RED),
             ("geo_ici", "ICI", None, "Lower is better", C.C_PUR),
             ("geo_oe", "Observed : expected", 1.0, "Ideal = 1", C.C_ORIG)]
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 5.2))
    axes = np.asarray(axes).ravel()
    fig.subplots_adjust(wspace=0.44, hspace=0.62)

    for ax, (col, ylab, ideal, note, colr) in zip(axes, specs):
        d = g[g["arm"] == "temporal"]
        a = d.groupby("pct_temporal").agg(n=("n_derive", "mean"), m=(col, "mean"),
                                          s=(col, "std"))
        a["se"] = a["s"] / np.sqrt(d.groupby("pct_temporal").size())
        ax.errorbar(a["n"], a["m"], yerr=1.96 * a["se"], fmt="o-",
                    color=colr, lw=1.7, ms=4.5, capsize=3)
        dc = g[g["arm"] == "control"]
        ac = dc.groupby("pct_temporal").agg(n=("n_derive", "mean"), m=(col, "mean"))
        ax.plot(ac["n"], ac["m"], "s--", color=C.C_GRAY, lw=1.2, ms=3.5)
        if ideal is not None:
            C.ref_line(ax, y=ideal)
        ax.set_xlabel("Derivation n", fontsize=9)
        ax.set_ylabel(ylab, fontsize=9)
        ax.set_title(note, loc="left", fontsize=9.5, color="#555555")
        ax.tick_params(labelsize=8.5)
    axes[5].axis("off")
    axes[5].plot([], [], "o-", color=C.C_GEO, label="Contemporary data added")
    axes[5].plot([], [], "s--", color=C.C_GRAY,
                 label="Row-count-matched resampled original")
    C.style_legend(axes[5], loc="center", fontsize=9, handlelength=1.8)
    C.save_figure(fig, path_out)


def fig_calibration_improvement(g, yg, path_out, sel=(0.0, 0.40, 1.00)):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6))
    colors = {0.0: C.C_RED, 0.40: C.C_TEMP, 1.00: C.C_GEO}

    ax = axes[0]
    C.ref_line(ax, diag=True, label="Ideal")
    for pct in sel:
        d = g[(g["arm"] == "temporal") & (g["pct_temporal"] == pct)]
        P = np.concatenate([np.array(x) for x in d["p_geo"]])
        Y = np.concatenate([np.array(x) for x in d["y_geo"]])
        cc = C.calibration_curve_bins(Y, P, n_bins=10)
        ax.plot(cc["pred"], cc["obs"], "o-", color=colors[pct], lw=1.5, ms=4,
                label=f"n = {int(d['n_derive'].mean())}  ({pct:.0%} contemporary)")
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed proportion")
    ax.set_title("A  Calibration before vs after", loc="left",
                 fontweight="bold")
    C.style_legend(ax, fontsize=9)
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.05, 1.05)

    ax = axes[1]
    for pct in sel:
        d = g[(g["arm"] == "temporal") & (g["pct_temporal"] == pct)]
        P = np.concatenate([np.array(x) for x in d["p_geo"]])
        from scipy.stats import gaussian_kde
        try:
            kde = gaussian_kde(P)
            xs = np.linspace(0.001, 0.999, 300)
            ax.plot(xs, kde(xs), color=colors[pct], lw=1.5,
                    label=f"n = {int(d['n_derive'].mean())}  (mean {P.mean():.3f})")
            ax.axvline(P.mean(), color=colors[pct], lw=1.0, ls=":")
        except Exception:
            pass
    ax.axvline(float(np.mean(Y)), color="#444444", lw=1.2, ls="--")
    ax.text(float(np.mean(Y)) - 0.015, ax.get_ylim()[1] * 0.62,
            f"observed event\nrate {np.mean(Y):.3f}", fontsize=8,
            color="#444444", ha="right", va="top")
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Density")
    ax.set_title("B  Distribution of predicted risks", loc="left",
                 fontweight="bold")
    C.style_legend(ax, loc="upper right", fontsize=8.5)
    fig.subplots_adjust(wspace=0.34)
    C.save_figure(fig, path_out)


def fig_recalibration_in_geo(g, yg, path_out, pct=1.00, n_folds=5):
    """【已停用】pct 默认 1.00 = 全量更新后的模型，不是冻结模型。

    冻结模型的交叉拟合再校准见 35_recal_frozen.py。保留本函数仅为历史对照，
    切勿再用于正文 Figure 3 / Table S5。
    """
    d = g[(g["arm"] == "temporal") & (g["pct_temporal"] == pct)]
    P = np.concatenate([np.array(x) for x in d["p_geo"]])
    Y = np.concatenate([np.array(x) for x in d["y_geo"]])
    # 说明：为绘图用，这里把重复实验的预测拼接；交叉拟合本身在原始 177 例上做
    P0 = np.array(d.iloc[0]["p_geo"]); Y0 = np.array(d.iloc[0]["y_geo"])

    from sklearn.linear_model import LogisticRegression
    rec = np.full(len(Y0), np.nan)
    for k in range(n_folds):
        te = np.arange(len(Y0)) % n_folds == k
        tr = ~te
        logit_tr = np.log(np.clip(P0[tr], 1e-6, 1 - 1e-6) /
                          (1 - np.clip(P0[tr], 1e-6, 1 - 1e-6))).reshape(-1, 1)
        lr = LogisticRegression(penalty=None, max_iter=1000).fit(logit_tr, Y0[tr])
        logit_te = np.log(np.clip(P0[te], 1e-6, 1 - 1e-6) /
                          (1 - np.clip(P0[te], 1e-6, 1 - 1e-6))).reshape(-1, 1)
        rec[te] = lr.predict_proba(logit_te)[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6))
    ax = axes[0]
    C.ref_line(ax, diag=True, label="Ideal")
    # ★ R26（2 区标准）：给两条校准曲线加患者层面 bootstrap 95% 置信带。
    #   重采样在 177 例上按患者有放回抽取，每次重算等频分箱的（pred, obs）
    #   中位数轨迹，取 2.5% 与 97.5% 分位作为带边界。带较宽是这一样本量的
    #   真实反映，不作平滑。binning 与 C.calibration_curve_bins 一致。
    def _cal_band(p_, n_boot=400, n_bins=8, seed=17):
        rng = np.random.default_rng(seed)
        n = len(Y0)
        lo, hi = [], []
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            if len(np.unique(Y0[idx])) < 2:
                continue
            cc = C.calibration_curve_bins(Y0[idx], p_[idx], n_bins=n_bins)
            lo.append(cc["obs"].values)
            hi.append(cc["pred"].values)
        if not lo:
            return None, None, None
        lo = np.vstack(lo); hi = np.vstack(hi)
        return (np.nanpercentile(lo, 2.5, axis=0),
                np.nanpercentile(hi, 97.5, axis=0))

    for p_, lab, col in [(P0, "Before recalibration", C.C_RED),
                         (rec, "After cross-fitted recalibration", C.C_GEO)]:
        cc = C.calibration_curve_bins(Y0, p_, n_bins=8)
        blo, bhi = _cal_band(p_)
        if blo is not None:
            ax.fill_between(cc["pred"].values, blo, bhi, color=col,
                            alpha=0.14, lw=0, zorder=1)
        ax.plot(cc["pred"], cc["obs"], "o-", color=col, lw=1.5, ms=4, label=lab,
                zorder=3)
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed proportion")
    ax.set_title("A  Cross-fitted recalibration", loc="left",
                 fontweight="bold")
    # ★ A 面板三条曲线名较长，图例字号收到 7.5 以缩小占位、避免压住曲线。
    C.style_legend(ax, fontsize=7.5)

    ax = axes[1]
    m0 = C.calibration_metrics(Y0, P0)
    m1 = C.calibration_metrics(Y0, rec)
    # ★ 面板 B 把所有指标折到同一条可比尺度上：距理想值的绝对偏差。
    #   斜率与 O:E 的理想值为 1，截距 / Brier / ICI 的理想值为 0。
    #   这样五个量纲不同的指标可以并排比较，箭头方向统一为「向理想值靠拢」。
    IDEAL = {"Calibration slope": 1.0, "Calibration intercept": 0.0,
             "O:E": 1.0, "Brier": 0.0, "ICI": 0.0}
    rows = ["Calibration slope", "Calibration intercept", "O:E", "Brier", "ICI"]
    ylab = ["Calibration slope\n(ideal = 1)", "Calibration intercept\n(ideal = 0)",
            "O:E ratio\n(ideal = 1)", "Brier score\n(ideal = 0)",
            "ICI\n(ideal = 0)"]
    n_row = len(rows)
    ypos = np.arange(n_row)[::-1].astype(float)          # 自上而下 4→0
    dev0 = np.array([abs(m0[k] - IDEAL[k]) for k in rows])
    dev1 = np.array([abs(m1[k] - IDEAL[k]) for k in rows])
    xmax = 0.55

    for i, y in enumerate(ypos):                          # 每行一条淡色导轨
        track = max(dev0[i], dev1[i]) + 0.055             # 长度随该指标量级伸缩
        ax.plot([0, track], [y, y], color="#e6e6e6", lw=8.5,
                solid_capstyle="round", zorder=1)
    for i, y in enumerate(ypos):
        d0, d1 = float(dev0[i]), float(dev1[i])
        if d0 > d1:                                       # 已被消除的偏差区间
            ax.barh(y, d0 - d1, left=d1, height=0.30, color=C.C_GEO,
                    alpha=0.30, edgecolor="none", zorder=2)
            ax.annotate("", xy=(d1, y), xytext=(d0, y), zorder=3,
                        arrowprops=dict(arrowstyle="-|>", color="#8f8f8f",
                                        lw=1.5, shrinkA=6.5, shrinkB=6.5))
        for d, col, dy, va, msz in ((d0, C.C_RED, 10, "bottom", 11.0),
                                    (d1, C.C_GEO, -11, "top", 8.2)):
            ax.plot(d, y, "o", ms=msz, mfc=col, mec="white", mew=1.3,
                    zorder=6, clip_on=False)
            ax.annotate(f"{d:.2f}", (d, y), textcoords="offset points",
                        xytext=(0, dy), ha="center", va=va, fontsize=8,
                        color=col, zorder=7,
                        bbox=dict(boxstyle="round,pad=0.10", fc="white",
                                  ec="none", alpha=0.85))
    ax.axvline(0, color="#9a9a9a", lw=0.9, ls=(0, (4, 3)), zorder=0)
    ax.set_yticks(ypos)
    ax.set_yticklabels(ylab, fontsize=8.0, linespacing=1.35)
    ax.set_ylim(-0.68, n_row - 0.32)
    ax.set_xlim(-0.018, xmax)
    ax.set_xticks(np.arange(0.0, 0.51, 0.1))
    ax.set_xlabel("Absolute departure from the ideal value", fontsize=9)
    ax.set_title("B  Departure from ideal calibration", loc="left",
                 fontweight="bold")
    C.style_axes(ax, grid=True, axis="x")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Line2D([], [], marker="o", ls="", mfc=C.C_RED, mec="white",
                      mew=1.1, ms=8.5, label="Before recalibration"),
               Line2D([], [], marker="o", ls="", mfc=C.C_GEO, mec="white",
                      mew=1.1, ms=8.5, label="After recalibration"),
               Patch(facecolor=C.C_GEO, alpha=0.16, edgecolor="none",
                     label="Departure removed")]
    lg = ax.legend(handles=handles, loc="lower right", fontsize=7.5,
                   frameon=True, framealpha=0.9, edgecolor="#cccccc",
                   fancybox=False, borderpad=0.5, labelspacing=0.45)
    lg.get_frame().set_linewidth(0.6)
    fig.subplots_adjust(wspace=0.46)
    C.save_figure(fig, path_out)
    return m0, m1


def main():
    C.log_init("03_growth")
    t0 = time.time()

    cohorts, _ = C.load_cohorts()
    Xd, yd = C.derivation_label(cohorts["original"])
    Xt, yt = C.derivation_label(cohorts["temporal"])
    Xg, yg = C.derivation_label(cohorts["geo"])
    C.section(f"成长分析：original {len(yd)} + temporal {len(yt)} → geo {len(yg)}")

    # ----- 只重绘图模式： python 03_temporal_growth.py figs -----
    figs_only = len(sys.argv) > 1 and sys.argv[1] == "figs"
    if figs_only:
        g = pd.read_pickle(os.path.join(C.OUT_DIR, "s03_growth_full.pkl"))
        C.log(f"  从缓存载入成长明细 {g.shape}")
        fig_growth_discrimination(g, "FigS5_growth_discrimination")
        fig_growth_calibration(g, "FigS6_growth_calibration")
        fig_calibration_improvement(g, yg, "FigS7_calibration_improvement")
        # Figure 3 与 Table S5 已改由 35_recal_frozen.py 产出：那里用的是**冻结模型**
        # 的预测，而本脚本的 temporal 臂在 pct=1.00 处给的是全量更新后的模型。
        # 此前本处默认 pct=1.00 而图题仍写"冻结模型"，是上一轮复核认定的硬错误。
        C.section(f"仅重绘图完成，总耗时 {C.timer(t0)}")
        return

    all_rows = []
    C.section("臂 A：逐步并入时间队列（每步重新调参）")
    all_rows.append(run_arm(Xd, yd, Xt, yt, Xg, yg, "temporal", "A", all_rows))
    C.section("臂 B：对照（把 original 分层重采样到同一样本量）")
    all_rows.append(run_arm(Xd, yd, Xt, yt, Xg, yg, "control", "B", all_rows))

    g = pd.concat(all_rows, ignore_index=True)
    g_slim = g.drop(columns=["y_geo", "p_geo"])
    C.save_table(g_slim, "Table6_growth_detail")
    C.save_csv(g_slim, "s03_growth_detail")
    g.to_pickle(os.path.join(C.OUT_DIR, "s03_growth_full.pkl"))

    # ---------------- Table 6 汇总 ----------------
    agg = g.groupby(["arm", "pct_temporal"]).agg(
        n_derive=("n_derive", "mean"),
        auc=("geo_auc", "mean"), auc_sd=("geo_auc", "std"),
        slope=("geo_slope", "mean"), slope_sd=("geo_slope", "std"),
        intercept=("geo_intercept", "mean"), intercept_sd=("geo_intercept", "std"),
        brier=("geo_brier", "mean"), brier_sd=("geo_brier", "std"),
        ici=("geo_ici", "mean"), ici_sd=("geo_ici", "std"),
        oe=("geo_oe", "mean"), oe_sd=("geo_oe", "std"),
        sens=("geo_Sensitivity", "mean"), spec=("geo_Specificity", "mean"),
    ).reset_index()
    agg["n_derive"] = agg["n_derive"].round(0).astype(int)
    C.save_table(agg.round(4), "Table7_growth_summary")
    C.log(agg.round(3).to_string(index=False))

    # ---------------- 时间臂 vs 对照臂：各步差异（配对口径） ----------------
    # ★ 两臂按 (pct_temporal, repeat) 成对：同一重复序号、同一样本量、同一评价队列，
    #   并不是独立样本。故主口径为配对 Wilcoxon 符号秩；另附把重复间变异显式建进
    #   模型的线性混合模型 value ~ arm + (1|repeat)，以及旧口径独立样本 Welch t。
    # ★ 本段即 TableS4 的权威产出处——该表原先由此处的单列 p 值，再由后处理器
    #   Code/_patch_r6_arms.py 增列重写；现内置于此，重跑 03 后无需再补跑补丁脚本。
    C.section("时间臂 vs 对照臂（同 n 下的差异，配对口径）")
    if not _STATSMODELS_OK:
        C.log("  [warn] statsmodels 不可用：p mixed model 列将写 NA")
    diff_rows = []
    for pct in PCTS:
        a = (g[(g["arm"] == "temporal") & (g["pct_temporal"] == pct)]
             .set_index("repeat"))
        b = (g[(g["arm"] == "control") & (g["pct_temporal"] == pct)]
             .set_index("repeat"))
        idx = a.index.intersection(b.index)
        for col, name in [("geo_auc", "AUC"), ("geo_slope", "slope"),
                          ("geo_intercept", "intercept"), ("geo_brier", "Brier"),
                          ("geo_ici", "ICI"), ("geo_oe", "O:E")]:
            va, vb = a.loc[idx, col].values, b.loc[idx, col].values
            # (1) 配对 Wilcoxon 符号秩——同一重复序号下的 10 对值（主口径）
            try:
                p_pair = float(wilcoxon(va, vb).pvalue)
            except Exception:
                p_pair = np.nan
            # (2) 线性混合模型：value ~ arm，重复作为随机截距
            p_mix = np.nan
            if _STATSMODELS_OK:
                long = pd.DataFrame({
                    "value": np.concatenate([va, vb]),
                    "arm": ["temporal"] * len(va) + ["control"] * len(vb),
                    "repeat": np.concatenate([idx.values, idx.values]),
                })
                try:
                    m = smf.mixedlm("value ~ arm", long,
                                    groups=long["repeat"]).fit(reml=False)
                    p_mix = float(m.pvalues["arm[T.temporal]"])
                except Exception:
                    p_mix = np.nan
            # (3) 独立样本 Welch t（旧口径，保留供对照）
            se = np.sqrt(va.var(ddof=1) / len(va) + vb.var(ddof=1) / len(vb))
            t = (va.mean() - vb.mean()) / se if se > 0 else np.nan
            p_welch = (2 * (1 - tdist.cdf(abs(t), len(va) + len(vb) - 2))
                       if se > 0 else np.nan)
            diff_rows.append({
                "pct_temporal": pct,
                "n_derive": int(a.loc[idx, "n_derive"].mean()),
                "metric": name,
                "temporal_mean": va.mean(),
                "control_mean": vb.mean(),
                "diff": va.mean() - vb.mean(),
                # p 值以字符串写出（与正文口径相同），避免 1e-8 被压成 ".00000"
                "p paired Wilcoxon": _fmt_p(p_pair),
                "p mixed model": _fmt_p(p_mix),
                "p independent": _fmt_p(p_welch),
            })
    # 均值列刻意不 round：与 _patch_r6_arms.py 的历史输出保持字节级一致；
    # 补充材料渲染端本就统一格式化为 4 位小数，显示不受影响。
    dd = pd.DataFrame(diff_rows)
    C.save_table(dd, "TableS4_growth_temporal_vs_control")
    C.log(dd[dd["metric"].isin(["AUC", "slope", "intercept", "Brier"])]
          .to_string(index=False))
    n_pair_sig = sum(1 for r in diff_rows
                     if r["p paired Wilcoxon"] != "NA"
                     and float(r["p paired Wilcoxon"].lstrip("<")) < 0.05)
    n_welch_sig = sum(1 for r in diff_rows
                      if r["p independent"] != "NA"
                      and float(r["p independent"].lstrip("<")) < 0.05)
    C.log(f"  显著项数（{len(diff_rows)} 个格）：paired {n_pair_sig} / "
          f"welch {n_welch_sig}")

    # ---------------- 图 ----------------
    fig_growth_discrimination(g, "FigS5_growth_discrimination")
    fig_growth_calibration(g, "FigS6_growth_calibration")
    fig_calibration_improvement(g, yg, "FigS7_calibration_improvement")
    # 再校准图/表见 35_recal_frozen.py（冻结模型口径）。

    # ---------------- Endpoint 显著性（BCa over repeats） ----------------
    C.section("端点比较（p=0 vs p=1.00，重复实验间）")
    from scipy.stats import mannwhitneyu
    ep = []
    for col, name in [("geo_auc", "AUC"), ("geo_slope", "slope"),
                      ("geo_intercept", "intercept"), ("geo_brier", "Brier"),
                      ("geo_ici", "ICI"), ("geo_oe", "O:E")]:
        a = g[(g["arm"] == "temporal") & (g["pct_temporal"] == 0.0)][col].values
        b = g[(g["arm"] == "temporal") & (g["pct_temporal"] == 1.0)][col].values
        u, p = mannwhitneyu(a, b)
        ep.append({"metric": name, "baseline": a.mean(), "final": b.mean(),
                   "change": b.mean() - a.mean(), "p_mannwhitney": p})
        C.log(f"  {name:10s} {a.mean():.4f} -> {b.mean():.4f}  "
              f"(Δ {b.mean()-a.mean():+.4f})  p = {p:.4f}")
    C.save_table(pd.DataFrame(ep).round(4), "Table8_growth_endpoint_tests")

    C.section(f"脚本 03 完成，总耗时 {C.timer(t0)}")


if __name__ == "__main__":
    main()
