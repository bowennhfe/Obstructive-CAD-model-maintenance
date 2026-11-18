# -*- coding: utf-8 -*-
"""
07 —— 第二个外部验证目标：同一中心「未来时段留出集」的当代外部验证
=============================================================================
动机
    03 只用地理队列（另一家医院）作为唯一外部验证目标。为检验「随数据积累而改善
    外部效度」这一结论是否只对该队列成立，这里预先指定第二个外部验证目标：
    把时间队列按 60/40 分层拆成
        - 当代更新池 (update pool)  ：只用于并入推导队列
        - 当代留出集 (holdout)      ：只用于评价，绝不参与拟合/调参/定阈值
    阶梯：推导 = original(442) + p × 更新池，p ∈ {0, .25, .50, .75, 1.00}
          每一步在推导池内重新网格搜索调参，随后同时在两个外部目标上评价
          （geo 177 与 当代留出集），阈值只来自推导池的 5 折袋外 Youden。
★ 这是预设的次要分析（secondary analysis），不是独立前瞻验证。
输出
    TableS9_second_target.csv
    FigS9_second_target
=============================================================================
"""
import os
import sys
import time
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

SPLIT_SEED = 20260920
FRAC_POOL = 0.60
PCTS = [0.0, 0.25, 0.50, 0.75, 1.00]
N_REPEAT = 5
GRID_FAST = {"RF": {"n_estimators": [500], "max_depth": [None, 10],
                    "min_samples_leaf": [1, 2, 4]}}


def strat_split(df, frac_pool, seed):
    rng = np.random.RandomState(seed)
    pool, hold = [], []
    for lab, sub in df.groupby(C.TARGET):
        ids = sub.index.values.copy()
        rng.shuffle(ids)
        k = int(round(len(ids) * frac_pool))
        pool.append(ids[:k])
        hold.append(ids[k:])
    return np.sort(np.concatenate(pool)), np.sort(np.concatenate(hold))


def strat_sample(df, frac, rng):
    if frac <= 0:
        return np.array([], dtype=int)
    idx = []
    for lab, sub in df.groupby(C.TARGET):
        k = min(int(round(len(sub) * frac)), len(sub))
        if k > 0:
            idx.append(rng.choice(sub.index.values, size=k, replace=False))
    return np.sort(np.concatenate(idx)) if idx else np.array([], dtype=int)


def main():
    # 仅重绘图（不重新拟合模型）：python 07_second_target.py figs
    if len(sys.argv) > 1 and sys.argv[1] == "figs":
        agg = pd.read_csv(os.path.join(C.TABLE_DIR, "TableS9_second_target.csv"))
        fig_second_target(agg, int(agg["holdout_n"].iloc[0]))
        print("仅重绘图完成。")
        return

    C.log_init("07_second_target")
    t0 = time.time()

    cohorts, _ = C.load_cohorts()
    Xd, yd = C.derivation_label(cohorts["original"])
    Xt, yt = C.derivation_label(cohorts["temporal"])
    Xg, yg = C.derivation_label(cohorts["geo"])

    Tm = pd.concat([Xt, pd.Series(yt, name=C.TARGET, index=Xt.index)], axis=1)

    i_pool, i_hold = strat_split(Tm, FRAC_POOL, SPLIT_SEED)
    Tm_pool = Tm.loc[i_pool]
    Tm_hold = Tm.loc[i_hold]
    Xh = Tm_hold[C.FEATURES]
    yh = Tm_hold[C.TARGET].astype(int).values

    C.section(f"第二外部目标：当代留出集 n={len(yh)}（事件 {int(yh.sum())}，"
              f"{yh.mean()*100:.1f}%）；更新池 n={len(Tm_pool)}")
    C.log(f"  geo  n={len(yg)}（事件 {int(yg.sum())}，{yg.mean()*100:.1f}%）")

    Dm = pd.concat([Xd, pd.Series(yd, name=C.TARGET, index=Xd.index)], axis=1)

    rows = []
    for pct in PCTS:
        for rep in range(N_REPEAT):
            rng = np.random.RandomState(4242 + rep + int(pct * 100) * 11)
            add = strat_sample(Tm_pool, pct, rng) if pct > 0 else np.array([], int)
            pool = pd.concat([Dm, Tm_pool.loc[add]], axis=0).reset_index(drop=True)
            Xp = pool[C.FEATURES]
            yp = pool[C.TARGET].astype(int).values

            gs = C.tune_model(Xp, yp, "RF", grid=GRID_FAST, cv=5,
                              seed=C.SEED + rep, n_jobs=-1)
            best = gs.best_estimator_
            oof = cross_val_predict(
                C.build_estimator("RF", {k.replace("clf__", ""): v
                                         for k, v in gs.best_params_.items()},
                                  seed=C.SEED + rep),
                Xp, yp, cv=StratifiedKFold(5, shuffle=True, random_state=C.SEED + rep),
                method="predict_proba", n_jobs=-1)[:, 1]
            thr = C.youden_threshold(yp, oof)

            rec = {"pct_pool": pct, "repeat": rep, "n_derive": int(len(yp)),
                   "n_added": int(len(add)), "internal_auc": float(gs.best_score_),
                   "thr": float(thr)}
            for tag, Xe, ye in (("geo", Xg, yg), ("holdout", Xh, yh)):
                p = best.predict_proba(Xe)[:, 1]
                bm = C.binary_metrics(ye, p, thr)
                cm = C.calibration_metrics(ye, p)
                rec[f"{tag}_auc"] = float(roc_auc_score(ye, p))
                rec[f"{tag}_sens"] = bm["Sensitivity"]
                rec[f"{tag}_spec"] = bm["Specificity"]
                rec[f"{tag}_slope"] = cm["Calibration slope"]
                rec[f"{tag}_intercept"] = cm["Calibration intercept"]
                rec[f"{tag}_brier"] = cm["Brier"]
                rec[f"{tag}_ici"] = cm["ICI"]
                rec[f"{tag}_oe"] = cm["O:E"]
            rows.append(rec)
        C.log(f"  p={pct:.2f} n_derive≈{int(np.mean([r['n_derive'] for r in rows if r['pct_pool'] == pct]))} "
              f"geo AUC={np.mean([r['geo_auc'] for r in rows if r['pct_pool'] == pct]):.3f} "
              f"holdout AUC={np.mean([r['holdout_auc'] for r in rows if r['pct_pool'] == pct]):.3f}")

    d = pd.DataFrame(rows)
    d.drop(columns=["repeat"]).to_csv(
        os.path.join(C.OUT_DIR, "s07_second_target_detail.csv"),
        index=False, encoding="utf-8-sig")

    agg = d.groupby("pct_pool").agg(
        n_derive=("n_derive", "mean"), n_added=("n_added", "mean"),
        internal_auc=("internal_auc", "mean"),
        **{f"geo_{k}": (f"geo_{k}", "mean") for k in
           ("auc", "slope", "intercept", "brier", "ici", "oe", "sens", "spec")},
        **{f"holdout_{k}": (f"holdout_{k}", "mean") for k in
           ("auc", "slope", "intercept", "brier", "ici", "oe", "sens", "spec")},
    ).reset_index()
    agg["holdout_n"] = len(yh)
    agg["holdout_events"] = int(yh.sum())
    agg["holdout_prev_pct"] = round(float(yh.mean() * 100), 1)
    agg["update_pool_n"] = len(Tm_pool)
    agg["n_repeats"] = N_REPEAT
    C.save_table(agg.round(4), "TableS9_second_target")
    C.log(agg.round(3).to_string(index=False))

    fig_second_target(agg, int(agg["holdout_n"].iloc[0]))

    C.section(f"脚本 07 完成，总耗时 {C.timer(t0)}")


def fig_second_target(agg, n_holdout, path_out="FigS9_second_target"):
    """Fig12：地理队列与当代留出集两个外部目标的对照。

    可单独调用（`python 07_second_target.py figs`），只需 TableS9 的结果表。
    """
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.6))
    x = agg["n_derive"].values
    spec = [("auc", "A  Discrimination", "AUC", None),
            ("slope", "B  Calibration slope", "Calibration slope", 1.0),
            ("oe", "C  Observed:expected ratio", "O:E ratio", 1.0)]
    for ax, (key, title, ylab, ideal) in zip(axes, spec):
        ax.plot(x, agg[f"geo_{key}"], "-o", color=C.C_GEO, lw=1.8, ms=4.5,
                label="Geographical cohort (other hospital, n=177)")
        ax.plot(x, agg[f"holdout_{key}"], "-s", color=C.C_TEMP, lw=1.8, ms=4.5,
                label="Contemporary hold-out (same hospital, n=%d)" % n_holdout)
        if ideal is not None:
            C.ref_line(ax, y=ideal)
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, ideal + 0.12 * (hi - lo))
            ax.text(0.04, ideal, "ideal", transform=ax.get_yaxis_transform(),
                    va="top", ha="left", fontsize=8, color="#666666")
        ax.set_xlabel("Derivation sample size (n)")
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=10.5)
        ax.grid(alpha=0.25, lw=0.6)
    # 三联窄栏放不下各自图例 → 共用图例置于图下
    h, l = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(wspace=0.38, bottom=0.30)
    lg = fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, -0.01),
                    ncol=2, frameon=True, framealpha=0.9, edgecolor="#cccccc",
                    fontsize=9, handlelength=1.8, columnspacing=1.4)
    lg.get_frame().set_linewidth(0.6)
    C.save_figure(fig, path_out)


if __name__ == "__main__":
    main()
