# -*- coding: utf-8 -*-
"""
33_external_by_centre.py — 地理外部队列**逐中心**外部验证（第 19 轮审稿 M4）

背景
    地理外部队列 177 例来自两家医院，但数据文件没有中心字段。两家的住院号
    号段完全不相交：190717–416348（6 位，n=56）为衡阳市中心医院，
    1068223–1449091（7 位，n=121）为联勤保障部队第九二二医院（作者确认）。
    原有分析只报告 pooled 结果，审稿人要求逐中心报告 n、事件数、AUC、校准
    斜率/截距、O:E、ICI、Brier 与净获益，并给出中心间差异。

分析内容
    1. 冻结模型（m01_rf_original.pkl）在中心 A / 中心 B / pooled 上的完整指标
    2. 100% 更新模型（原始 442 + 当代 364 重新调参）在同样三个集合上的指标
    3. 中心间 AUC 差异（各中心内独立 bootstrap）与校准指标差异
    4. 派生图表：逐中心 ROC 与平滑校准曲线

输出
    Tables/Table17_external_by_centre.csv
    Outputs/s33_external_by_centre.json
    Figures/{pdf,tif,jpg}/Fig12_external_by_centre.*

注意
    中心 A/B 在正文中以 "Hengyang Central Hospital" 与 "922nd Hospital" 表述；
    pooled 结果必须与 Table 5 的冻结模型数值一致（AUC 0.719），作为自检。
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import common as C
from sklearn.metrics import roc_auc_score, roc_curve

ID_CUT = 1_000_000          # 6 位号段 < 1e6 <= 7 位号段
N_REPEAT = 10               # 与 03_temporal_growth 一致
CENTRE_NAME = {0: "Hengyang Central Hospital", 1: "922nd Hospital"}
SHORT = {0: "Hospital A", 1: "Hospital B"}
THRESHOLDS = [0.35, 0.40, 0.50]


# ---------------------------------------------------------------------------
def strat_sample(df, frac, rng):
    """按结局分层抽样（与 03_temporal_growth 同实现，保证 100% 步可复现）。"""
    if frac <= 0:
        return np.array([], dtype=int)
    idx = []
    for lab, sub in df.groupby(C.TARGET):
        k = min(int(round(len(sub) * frac)), len(sub))
        if k > 0:
            idx.append(rng.choice(sub.index.values, size=k, replace=False))
    return np.sort(np.concatenate(idx)) if idx else np.array([], dtype=int)


def auc_ci(y, p, n=2000, seed=C.SEED):
    lo, hi = C.boot_ci(y, p, lambda a, b: roc_auc_score(a, b), n_iter=n, seed=seed)
    return roc_auc_score(y, p), lo, hi


def metrics_block(y, p, thr, n_boot=2000, seed=C.SEED):
    """一个队列的完整指标（含 bootstrap CI）。"""
    y = np.asarray(y, int); p = np.asarray(p, float)
    m = C.calibration_metrics(y, p)
    out = {"n": int(len(y)), "events": int(y.sum()),
           "prevalence": float(y.mean())}
    out["auc"], out["auc_lo"], out["auc_hi"] = auc_ci(y, p, n_boot, seed)

    # 校准斜率 / 截距 / O:E 的 bootstrap CI
    rng = np.random.RandomState(seed)
    sl, ic, oe, ici, br = [], [], [], [], []
    for _ in range(n_boot):
        idx = rng.choice(len(y), len(y), replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        mm = C.calibration_metrics(y[idx], p[idx])
        sl.append(mm["Calibration slope"]); ic.append(mm["Calibration intercept"])
        oe.append(mm["O:E"]); ici.append(mm["ICI"]); br.append(mm["Brier"])
    for k, v in [("slope", sl), ("intercept", ic), ("oe", oe), ("ici", ici), ("brier", br)]:
        v = np.array(v, float)
        out[f"{k}_lo"], out[f"{k}_hi"] = (float(np.percentile(v, 2.5)),
                                          float(np.percentile(v, 97.5)))

    out["slope"] = m["Calibration slope"]
    out["intercept"] = m["Calibration intercept"]
    out["oe"] = m["O:E"]
    out["ici"] = m["ICI"]
    out["brier"] = m["Brier"]
    out["hl_p"] = m["HL p"]

    # 分类表现（推导集阈值）
    bm = C.binary_metrics(y, p, thr)
    out["thr"] = float(thr)
    out["sens"] = bm["Sensitivity"]; out["spec"] = bm["Specificity"]
    out["pos_rate"] = float((p >= thr).mean())

    # 净获益
    d = C.dca(y, p, thresholds=THRESHOLDS)
    for _, r in d.iterrows():
        out[f"nb_{r.threshold:.2f}"] = float(r.model)
        out[f"nball_{r.threshold:.2f}"] = float(r.treat_all)
    return out


def auc_diff(yA, pA, yB, pB, n_iter=2000, seed=C.SEED):
    """两中心 AUC 差异：各中心内独立 bootstrap（不重采样跨中心）。"""
    yA, yB = np.asarray(yA, int), np.asarray(yB, int)
    pA, pB = np.asarray(pA, float), np.asarray(pB, float)
    obs = roc_auc_score(yA, pA) - roc_auc_score(yB, pB)
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_iter):
        iA = rng.choice(len(yA), len(yA), replace=True)
        iB = rng.choice(len(yB), len(yB), replace=True)
        if len(np.unique(yA[iA])) < 2 or len(np.unique(yB[iB])) < 2:
            continue
        vals.append(roc_auc_score(yA[iA], pA[iA]) - roc_auc_score(yB[iB], pB[iB]))
    vals = np.array(vals)
    return float(obs), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def smooth_calibration(ax, y, p, color, label, n_knots=None):
    """loess 平滑校准曲线（低样本量下比等频分箱稳定）。"""
    from statsmodels.nonparametric.smoothers_lowess import lowess
    y = np.asarray(y, float); p = np.asarray(p, float)
    frac = 0.75 if len(y) < 120 else 0.6
    sm = lowess(y, p, frac=frac, return_sorted=True)
    ax.plot(sm[:, 0], sm[:, 1], color=color, lw=1.8, label=label, zorder=3)


# ---------------------------------------------------------------------------
def main():
    C.log_init("33_external_by_centre")
    cohorts, _ = C.load_cohorts()
    o, t, g = cohorts["original"], cohorts["temporal"], cohorts["geo"]

    # --- 中心标签 -----------------------------------------------------------
    gid = pd.to_numeric(g["ID"], errors="coerce").values
    g["centre"] = (gid >= ID_CUT).astype(int)
    yd = g[C.TARGET].astype(int).values
    Xg, _ = C.derivation_label(g)
    C.section("地理外部队列的中心构成")
    for k in (0, 1):
        s = g["centre"] == k
        C.log(f"  {SHORT[k]:9s} ({CENTRE_NAME[k]:28s}) n={int(s.sum()):4d} "
              f"events={int(g.loc[s, C.TARGET].sum()):3d} "
              f"prev={g.loc[s, C.TARGET].mean()*100:5.1f}%  "
              f"ID {int(gid[s].min())}-{int(gid[s].max())}")

    # --- 冻结模型 -----------------------------------------------------------
    frozen = C.load_pickle("m01_rf_original.pkl")
    pipe, thr = frozen["pipeline"], frozen["threshold"]
    pg = pipe.predict_proba(Xg)[:, 1]

    # 自检：pooled AUC 应等于 Table 5 的 0.719
    pooled_auc = roc_auc_score(yd, pg)
    C.log(f"\n  [check] pooled frozen AUC = {pooled_auc:.4f} (Table 5: 0.719)")

    rows = []
    for tag, mask in [("Centre A (Hengyang Central)", g["centre"].values == 0),
                      ("Centre B (922nd)", g["centre"].values == 1),
                      ("Pooled", np.ones(len(yd), bool))]:
        r = metrics_block(yd[mask], pg[mask], thr)
        r.update({"model": "Frozen (rebuilt)", "set": tag})
        rows.append(r)
        C.log(f"  frozen | {tag:31s} AUC {r['auc']:.3f} ({r['auc_lo']:.3f}-{r['auc_hi']:.3f}) "
              f"slope {r['slope']:.2f} int {r['intercept']:.2f} O:E {r['oe']:.3f} "
              f"ICI {r['ici']:.3f} Brier {r['brier']:.3f}")

    dA, dB = (g["centre"].values == 0), (g["centre"].values == 1)
    obs, lo, hi = auc_diff(yd[dA], pg[dA], yd[dB], pg[dB])
    C.log(f"  frozen | AUC difference A-B = {obs:+.3f} (95% CI {lo:+.3f} to {hi:+.3f})")

    # --- 100% 更新模型：与阶梯 100% 步同一协议，10 次重复 ------------------
    C.section("100% 更新模型（与 03_temporal_growth 的 100% 步同协议，10 次重复）")
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    Xt, yt = C.derivation_label(t)
    Tm = pd.concat([Xt, pd.Series(yt, name=C.TARGET, index=Xt.index)], axis=1)
    Xo, yo = C.derivation_label(o)
    Dm = pd.concat([Xo, pd.Series(yo, name=C.TARGET, index=Xo.index)], axis=1)

    P = np.zeros((len(yd), N_REPEAT))
    thr_list = []
    for rep in range(N_REPEAT):
        rng = np.random.RandomState(1000 + rep + 100 * 7)
        add_idx = strat_sample(Tm, 1.0, rng)
        pool = pd.concat([Dm, Tm.loc[add_idx]], axis=0).reset_index(drop=True)
        Xp = pool[C.FEATURES]
        yp = pool[C.TARGET].astype(int).values
        gs = C.tune_model(Xp, yp, "RF", grid=C.PARAM_GRIDS_FAST, cv=5,
                          seed=C.SEED + rep, n_jobs=-1)
        P[:, rep] = gs.best_estimator_.predict_proba(Xg)[:, 1]
        oof = cross_val_predict(gs.best_estimator_, Xp, yp,
                                cv=StratifiedKFold(5, shuffle=True,
                                                   random_state=C.SEED + rep),
                                method="predict_proba", n_jobs=-1)[:, 1]
        thr_list.append(C.youden_threshold(yp, oof))
        C.log(f"  rep {rep:2d}: pooled AUC {roc_auc_score(yd, P[:, rep]):.4f}  "
              f"thr {thr_list[-1]:.3f}")
    thr_u = float(np.mean(thr_list))
    C.log(f"  repeat-averaged pooled external AUC = "
          f"{np.mean([roc_auc_score(yd, P[:, r]) for r in range(N_REPEAT)]):.4f} "
          f"(ladder Table 4: 0.742)")

    def metrics_mean(Pm, y, thr):
        """按 10 次重复取指标均值（与阶梯的重复平均口径一致）。"""
        y = np.asarray(y, int)
        R = Pm.shape[1]
        ms = [C.calibration_metrics(y, Pm[:, r]) for r in range(R)]
        out = {"n": int(len(y)), "events": int(y.sum()),
               "prevalence": float(y.mean()),
               "auc": float(np.mean([roc_auc_score(y, Pm[:, r]) for r in range(R)]))}
        for k, key in [("slope", "Calibration slope"), ("intercept", "Calibration intercept"),
                       ("oe", "O:E"), ("ici", "ICI"), ("brier", "Brier"),
                       ("hl_p", "HL p")]:
            out[k] = float(np.mean([m[key] for m in ms]))
        bm = C.binary_metrics(y, Pm.mean(axis=1), thr)
        out["thr"] = float(thr)
        out["sens"] = bm["Sensitivity"]; out["spec"] = bm["Specificity"]
        out["pos_rate"] = float((Pm.mean(axis=1) >= thr).mean())
        d = C.dca(y, Pm.mean(axis=1), thresholds=THRESHOLDS)
        for _, r in d.iterrows():
            out[f"nb_{r.threshold:.2f}"] = float(r.model)
            out[f"nball_{r.threshold:.2f}"] = float(r.treat_all)
        return out

    MASKS = [("Centre A (Hengyang Central)", dA), ("Centre B (922nd)", dB),
             ("Pooled", np.ones(len(yd), bool))]
    upd = {}
    for tag, mask in MASKS:
        r = metrics_mean(P[mask], yd[mask], thr_u)
        r.update({"model": "Updated (100%)", "set": tag})
        r["auc_lo"] = r["auc_hi"] = np.nan
        for k in ("slope", "intercept", "oe", "ici", "brier"):
            r[f"{k}_lo"] = r[f"{k}_hi"] = np.nan
        rows.append(r)
        upd[tag] = r
        C.log(f"  upd100 | {tag:31s} AUC {r['auc']:.3f} "
              f"slope {r['slope']:.2f} int {r['intercept']:.2f} O:E {r['oe']:.3f} "
              f"NB@0.40 {r['nb_0.40']:+.3f}")

    # 逐中心配对差异：每次重复内部取差，再对患者重抽样求区间
    C.section("逐中心的冻结 vs 更新配对差异（同中心内 bootstrap，含重复不确定性）")
    pair = {}
    rng = np.random.RandomState(C.SEED)
    for tag, mask in MASKS:
        yy, p1, Pm = yd[mask], pg[mask], P[mask]
        def dmean(idx):
            return float(np.mean([roc_auc_score(yy[idx], Pm[idx, r])
                                  for r in range(N_REPEAT)])) - roc_auc_score(yy[idx], p1[idx])
        obs_p = dmean(np.arange(len(yy)))
        vals = []
        for _ in range(2000):
            idx = rng.choice(len(yy), len(yy), replace=True)
            if len(np.unique(yy[idx])) < 2:
                continue
            vals.append(dmean(idx))
        vals = np.array(vals)
        lo_p, hi_p = np.percentile(vals, [2.5, 97.5])
        pair[tag] = dict(delta=float(obs_p), lo=float(lo_p), hi=float(hi_p))
        C.log(f"  {tag:31s} dAUC = {obs_p:+.3f} (95% CI {lo_p:+.3f} to {hi_p:+.3f})")

    # --- 汇总表 -------------------------------------------------------------
    cols = ["model", "set", "n", "events", "prevalence", "auc", "auc_lo", "auc_hi",
            "slope", "slope_lo", "slope_hi", "intercept", "intercept_lo", "intercept_hi",
            "oe", "oe_lo", "oe_hi", "ici", "brier", "hl_p",
            "thr", "sens", "spec", "pos_rate",
            "nb_0.35", "nb_0.40", "nb_0.50", "nball_0.35", "nball_0.40", "nball_0.50"]
    tab = pd.DataFrame(rows)[cols].round(4)
    C.save_table(tab, "Table17_external_by_centre")

    json.dump({"pooled_frozen_auc_check": float(pooled_auc),
               "centre_counts": {"A": int(dA.sum()), "B": int(dB.sum())},
               "frozen_centre_diff": {"delta": obs, "lo": lo, "hi": hi},
               "updated_centre_diff": auc_diff(yd[dA], P[dA].mean(axis=1),
                                               yd[dB], P[dB].mean(axis=1)),
               "paired_within_centre": pair,
               "updated_threshold": float(thr_u),
               "updated_threshold_repeats": thr_list,
               "pooled_updated_auc_repeat_mean": float(
                   np.mean([roc_auc_score(yd, P[:, r]) for r in range(N_REPEAT)])),
               "updated_by_centre": {k: {kk: vv for kk, vv in v.items()
                                         if kk not in ("model", "set")}
                                     for k, v in upd.items()}},
              open(os.path.join(C.OUT_DIR, "s33_external_by_centre.json"), "w",
                   encoding="utf-8"), ensure_ascii=False, indent=2, default=str)

    # --- 图：逐中心 ROC + 平滑校准 -----------------------------------------
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.9))
    cc = {0: "#1f77b4", 1: "#d62728"}

    ax = axes[0]
    for k in (0, 1):
        m = g["centre"].values == k
        fpr, tpr, _ = roc_curve(yd[m], pg[m])
        ax.plot(fpr, tpr, color=cc[k], lw=1.8,
                label=f"{SHORT[k]} (n={int(m.sum())}, AUC {roc_auc_score(yd[m], pg[m]):.2f})")
    fpr, tpr, _ = roc_curve(yd, pg)
    ax.plot(fpr, tpr, color="#333333", lw=1.4, ls="--",
            label=f"Pooled (n=177, AUC {pooled_auc:.2f})")
    C.ref_line(ax, diag=True)
    ax.set_xlabel("1 - Specificity"); ax.set_ylabel("Sensitivity")
    ax.set_title("A  Discrimination by hospital", loc="left", fontsize=10.5)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    C.style_legend(ax, loc="lower right", fontsize=7.8)
    C.style_axes(ax)

    ax = axes[1]
    ax.plot([0, 1], [0, 1], ls="--", lw=0.9, color="black", alpha=0.5)
    for k in (0, 1):
        m = g["centre"].values == k
        smooth_calibration(ax, yd[m], pg[m], cc[k], SHORT[k])
    smooth_calibration(ax, yd, pg, "#333333", "Pooled")
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed proportion")
    ax.set_title("B  Calibration by hospital (LOWESS)", loc="left", fontsize=10.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    C.style_legend(ax, loc="upper left", fontsize=7.8)
    C.style_axes(ax)

    ax = axes[2]
    for k in (0, 1):
        m = g["centre"].values == k
        d = C.dca(yd[m], pg[m])
        ax.plot(d.threshold, d.model, color=cc[k], lw=1.6, label=f"Model, {SHORT[k]}")
        ax.plot(d.threshold, d.treat_all, color=cc[k], lw=0.9, ls=":",
                label=f"Treat all, {SHORT[k]}")
    ax.axhline(0, color="#777777", lw=0.9, ls="--", label="Treat none")
    ax.set_xlim(0.05, 0.65); ax.set_ylim(-0.10, 0.45)
    ax.set_xlabel("Threshold probability"); ax.set_ylabel("Net benefit")
    ax.set_title("C  Net benefit by hospital", loc="left", fontsize=10.5)
    C.style_legend(ax, loc="upper right", fontsize=7.2, ncol=1)
    C.style_axes(ax)

    fig.tight_layout()
    C.save_figure(fig, "Fig12_external_by_centre")
    C.log("\n  done.")


if __name__ == "__main__":
    main()
