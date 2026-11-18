# -*- coding: utf-8 -*-
"""
15 —— 算法可更新性：判别力损失能否被"新信息"修复
=============================================================================
问题（承接脚本 14 的两分量分解）
    脚本 14 已证：在地理队列内做再校准，可以把 O:E 从 0.64–0.88 全部修到
    约 1.00，而判别力几乎不动（平均 ΔAUC ≈ −0.001）。也就是说"尺度分量"
    可用目标域数据直接修复，而"排序分量"不能。
    那排序分量能不能被**新信息**修复？本脚本对每个算法回答一次：
        冻结模型（推导 442）  →  更新模型（推导 + 同期 364 = 806，重新调参）
    在地理队列上比较二者的判别力与校准度。

设计要点
    · 与脚本 01/14 同协议：插补（必要时标准化）封装在 Pipeline 内，只在训练折 fit
    · 超参在每个训练池上重新做 10 折网格搜索（"每步重调参"）
    · 地理队列全程只执行 predict，不参与任何拟合/调参/选阈值
    · 阈值一律取自该训练池的袋外 Youden，绝不取自地理队列

输出
    Tables/Table16_algo_updatability.csv
    Figures/{pdf,tif,jpg}/FigS13_algo_updatability.*
    Outputs/s15_algo_updatability.csv
=============================================================================
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from importlib import import_module
M14 = import_module("14_algo_comparison")
ORDER = M14.ORDER
ALGOS = M14.ALGOS
AC = M14.AC


def tune_on(key, X, y, seed=C.SEED):
    """在给定训练池上重新做 10 折网格搜索（与 14 同协议）。"""
    g = ALGOS[key]["grid"]
    if not g:
        return {}
    gs = GridSearchCV(M14.make_pipe(key, seed=seed),
                      {f"clf__{k}": v for k, v in g.items()},
                      scoring="roc_auc",
                      cv=StratifiedKFold(C.CV_FOLDS, shuffle=True,
                                         random_state=seed),
                      n_jobs=-1, refit=True)
    gs.fit(X, y)
    return {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}


def oof_threshold(key, params, X, y, seed=C.SEED):
    """该训练池的袋外 Youden 阈值（阈值只能来自训练数据）。"""
    Xv = X.reset_index(drop=True)
    oof = np.full(len(y), np.nan)
    skf = StratifiedKFold(C.CV_FOLDS, shuffle=True, random_state=seed)
    for k, (tr, te) in enumerate(skf.split(Xv, y)):
        pipe = M14.make_pipe(key, params, seed=seed + k)
        pipe.fit(Xv.iloc[tr], y[tr])
        oof[te] = pipe.predict_proba(Xv.iloc[te])[:, 1]
    return C.youden_threshold(y, oof)


def evaluate(key, params, Xtr, ytr, Xte, yte, tag):
    pipe = M14.make_pipe(key, params)
    pipe.fit(Xtr, ytr)
    p = pipe.predict_proba(Xte)[:, 1]
    cm = C.calibration_metrics(yte, p)
    return {
        "model": key, "arm": tag,
        "auc": round(roc_auc_score(yte, p), 3),
        "brier": round(cm["Brier"], 3),
        "slope": round(cm["Calibration slope"], 3),
        "intercept": round(cm["Calibration intercept"], 3),
        "ici": round(cm["ICI"], 3),
        "oe": round(cm["O:E"], 3),
    }, p


# =============================================================================
def main():
    C.log_init("15_algo_updatability")
    t0 = time.time()
    C.section("算法可更新性：冻结模型 vs 并入全部同期数据后重新调参")

    cohorts, _ = C.load_cohorts()
    Xd, yd = C.derivation_label(cohorts["original"])
    Xt, yt = C.derivation_label(cohorts["temporal"])
    Xg, yg = C.derivation_label(cohorts["geo"])

    Xb = pd.concat([Xd, Xt], ignore_index=True)
    yb = np.concatenate([yd, yt])
    C.log(f"  冻结池 n={len(yd)}（事件 {int(yd.sum())}）")
    C.log(f"  更新池 n={len(yb)}（事件 {int(yb.sum())}，并入同期 {len(yt)} 例）")
    C.log(f"  地理评价队列 n={len(yg)}（事件 {int(yg.sum())}）")

    rows, preds = [], {"y": yg}
    for k in ORDER:
        ts = time.time()
        # 冻结：只在推导 442 上调参
        p_frozen = tune_on(k, Xd, yd)
        r0, p0 = evaluate(k, p_frozen, Xd, yd, Xg, yg, "Frozen")
        # 更新：在 806 上重新调参（"每步重调参"）
        p_upd = tune_on(k, Xb, yb)
        r1, p1 = evaluate(k, p_upd, Xb, yb, Xg, yg, "Updated")
        rows += [r0, r1]
        preds[f"frozen_{k}"] = p0
        preds[f"updated_{k}"] = p1
        C.log(f"  {k:4s} AUC {r0['auc']:.3f} → {r1['auc']:.3f} "
              f"({r1['auc']-r0['auc']:+.3f}) | slope {r0['slope']:.2f} → "
              f"{r1['slope']:.2f} | O:E {r0['oe']:.2f} → {r1['oe']:.2f} "
              f"[{C.timer(ts)}]")

    df = pd.DataFrame(rows)
    piv = df.pivot(index="model", columns="arm",
                   values=["auc", "brier", "slope", "intercept", "ici", "oe"])
    out = pd.DataFrame({"Model": ORDER,
                        "Full name": [ALGOS[k]["full"] for k in ORDER]})
    for m in ["auc", "brier", "slope", "intercept", "ici", "oe"]:
        out[f"{m} (frozen)"] = [piv.loc[k, (m, "Frozen")] for k in ORDER]
        out[f"{m} (updated)"] = [piv.loc[k, (m, "Updated")] for k in ORDER]
        out[f"delta {m}"] = (out[f"{m} (updated)"]
                             - out[f"{m} (frozen)"]).round(3)
    C.save_table(out, "Table16_algo_updatability")
    C.save_csv(out, "s15_algo_updatability")
    C.save_csv(pd.DataFrame(preds), "s15_geo_pred_updatability")

    C.section("汇总")
    d_auc = out["delta auc"]
    d_oe = (out["oe (updated)"] - 1).abs()
    C.log(f"  判别力改善：{d_auc.min():+.3f} 至 {d_auc.max():+.3f}"
          f"（平均 {d_auc.mean():+.3f}）")
    C.log(f"  更新后 O:E 距 1.00：平均 {d_oe.mean():.3f}"
          f"（冻结时平均 {(out['oe (frozen)'] - 1).abs().mean():.3f}）")
    C.log(f"  判别力仍未达内部水平（内部 50 折均值 − 更新后外部 AUC）的平均差："
          f"{(pd.read_csv(os.path.join(C.TABLE_DIR, 'Table11_algo_internal.csv'))
              .set_index('Model')['AUC (50 folds) mean'] - out.set_index('Model')
              ['auc (updated)']).mean():.3f}")

    # ---------------- 图 ----------------
    fig = plt.figure(figsize=(7.4, 5.6))
    gs = fig.add_gridspec(2, 1, hspace=0.45)
    x = np.arange(len(ORDER))
    w = 0.36

    ax = fig.add_subplot(gs[0, 0])
    ax.bar(x - w / 2, out["auc (frozen)"], width=w, color="#bbbbbb",
           label="Frozen (derivation n = 442)")
    ax.bar(x + w / 2, out["auc (updated)"], width=w,
           color=[AC[k] for k in ORDER], alpha=0.92,
           label="Updated (derivation + contemporary, n = 806)")
    for xi, (a, b) in enumerate(zip(out["auc (frozen)"], out["auc (updated)"])):
        ax.text(xi + w / 2, b + 0.004, f"{b - a:+.3f}", ha="center",
                fontsize=7.8, color="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels(ORDER, fontsize=9)
    ax.set_ylabel("External AUC")
    ax.set_ylim(0.60, max(0.80, out["auc (updated)"].max() + 0.05))
    ax.set_title("A  Adding information moves discrimination only a little",
                 loc="left", fontweight="bold", fontsize=10)
    C.style_axes(ax, axis="y")
    C.style_legend(ax, loc="lower right", fontsize=8.0)

    ax = fig.add_subplot(gs[1, 0])
    ax.bar(x - w / 2, out["oe (frozen)"], width=w, color="#bbbbbb")
    ax.bar(x + w / 2, out["oe (updated)"], width=w,
           color=[AC[k] for k in ORDER], alpha=0.92)
    C.ref_line(ax, y=1.0, color="#d62728", lw=1.0, ls="--", label="Ideal")
    for xi, (a, b) in enumerate(zip(out["oe (frozen)"], out["oe (updated)"])):
        ax.text(xi + w / 2, b + 0.015, f"{b - a:+.3f}", ha="center",
                fontsize=7.8, color="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels(ORDER, fontsize=9)
    ax.set_ylabel("O:E ratio in the geographical cohort")
    ax.set_ylim(0.55, max(1.15, out["oe (updated)"].max() + 0.12))
    ax.set_title("B  The scale is repaired but stays short of ideal",
                 loc="left", fontweight="bold", fontsize=10)
    C.style_axes(ax, axis="y")
    C.style_legend(ax, loc="lower right", fontsize=8.0)
    C.save_figure(fig, "FigS13_algo_updatability")

    C.section(f"脚本 15 完成，总耗时 {C.timer(t0)}")


if __name__ == "__main__":
    main()
