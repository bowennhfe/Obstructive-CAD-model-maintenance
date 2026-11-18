# -*- coding: utf-8 -*-
"""
16b_control_arm.py
==================
回复审稿意见 4：对照臂（row-count-matched resampled original）的内部交叉验证
因"有放回重采样 → 同一患者出现在训练折与验证折"而虚高。

本脚本把 16_revision_analyses.py 的 sec3 单独抽出来，并做两处提速：

  1. 跳过 GridSearchCV，固定使用推导主线超参
     (n_estimators=500, max_depth=10, min_samples_leaf=1)。
     两臂（naive / grouped）用**完全相同**的超参与重采样种子，
     因此二者之差**纯粹来自 CV 的折叠方式**，这比混入调参更干净。
  2. 时间臂无重复患者（分组退化 == 普通 CV），直接从脚本 03 的
     Outputs/s03_growth_detail.csv 读入，不再重算；对照臂在**地理队列**
     上的 AUC 同样取自 03 的对照臂，使本表与表 6 / 表 S4 / 表 S7 报告
     同一个对照臂（表间一致性）。

输出：
  Outputs/s16_control_arm.csv
  Tables/TableS24_control_arm_grouped.csv
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

from sklearn.model_selection import (StratifiedKFold, StratifiedGroupKFold,
                                     cross_val_score)
from sklearn.metrics import roc_auc_score

SEED = C.SEED                       # 123
SMOKE = bool(os.environ.get("SMOKE"))
PCTS = ([0.0, 1.00] if SMOKE else [0.0, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00])
R = 1 if SMOKE else 10
FIXED = {"n_estimators": 500, "max_depth": 10, "min_samples_leaf": 1}


def strat_bootstrap_ids(y, n_target, rng):
    """分层有放回重采样，返回**原始行号**（可重复），用于分组交叉验证。"""
    out = []
    labs = np.asarray(y)
    for lab in np.unique(labs):
        sub = np.where(labs == lab)[0]
        k = int(round(n_target * (sub.size / labs.size)))
        out.append(rng.choice(sub, size=k, replace=True))
    return np.sort(np.concatenate(out))


def main():
    C.section("对照臂：泄露安全的重复患者分组交叉验证（固定超参）")
    cohorts, _ = C.load_cohorts()
    Xd, yd = C.derivation_label(cohorts["original"])
    Xt, yt = C.derivation_label(cohorts["temporal"])
    Xg, yg = C.derivation_label(cohorts["geo"])
    Tm = pd.concat([Xt, pd.Series(yt, name=C.TARGET, index=Xt.index)], axis=1)
    Dm = pd.concat([Xd, pd.Series(yd, name=C.TARGET, index=Xd.index)], axis=1)
    Xd_v = Dm[C.FEATURES].reset_index(drop=True)
    yd_v = Dm[C.TARGET].astype(int).values
    Xg_v = Xg.reset_index(drop=True)
    yg = np.asarray(yg, int)

    rows = []

    # 对照臂在地理队列上的 AUC 直接取自积累阶梯自身的对照臂
    #（Outputs/s03_growth_detail.csv），以保证 Table S24 与 Table 6、
    # Table S4、Table S7 报告的是同一个对照臂。
    _g3 = pd.read_csv(os.path.join(C.OUT_DIR, "s03_growth_detail.csv"))
    _g3c = _g3[_g3["arm"] == "control"]
    ladder_geo = {(round(float(r["pct_temporal"]), 2), int(r["repeat"])): float(r["geo_auc"])
                  for _, r in _g3c.iterrows()}

    # ---- 对照臂：有放回重采样到同样本量，两种 CV 折叠方式 ----
    for pct in PCTS:
        for rep in range(R):
            # 与 03_temporal_growth.py 的 control 臂完全同种子
            rng = np.random.RandomState(1000 * 2 + rep + int(pct * 100) * 7)
            n_add = int(round(pct * len(Tm)))
            n_target = len(Dm) + n_add
            ids = strat_bootstrap_ids(yd_v, n_target, rng)
            Xp = Xd_v.iloc[ids].reset_index(drop=True)
            yp = yd_v[ids]
            gid = ids                                  # 分组变量 = 原患者行号
            n_unique = len(np.unique(ids))

            for scheme in ("naive", "grouped"):
                cv = (StratifiedGroupKFold(5, shuffle=True,
                                           random_state=SEED + rep)
                      if scheme == "grouped"
                      else StratifiedKFold(5, shuffle=True,
                                           random_state=SEED + rep))
                est = C.build_estimator("RF", FIXED, seed=SEED + rep)
                sc = cross_val_score(est, Xp, yp, cv=cv,
                                     groups=(gid if scheme == "grouped" else None),
                                     scoring="roc_auc", n_jobs=-1)
                rows.append({"arm": f"control_{scheme}", "pct": pct,
                             "repeat": rep, "n_derive": len(yp),
                             "n_unique_patients": n_unique,
                             "cv_auc": float(sc.mean()),
                             "cv_auc_sd_folds": float(sc.std()),
                             "geo_auc": ladder_geo[(round(pct, 2), rep)]})
            if rep == 0:
                C.log(f"    p={pct:.2f} n={len(yp):4d} "
                      f"(unique {n_unique})  "
                      f"naive CV={rows[-2]['cv_auc']:.4f}  "
                      f"grouped CV={rows[-1]['cv_auc']:.4f}")

    d = pd.DataFrame(rows)
    C.save_csv(d, "s16_control_arm")

    # ---- 时间臂：无重复患者，直接引用脚本 03 结果 ----
    g3 = pd.read_csv(os.path.join(C.OUT_DIR, "s03_growth_detail.csv"))
    t3 = (g3[g3["arm"] == "temporal"]
          .groupby("pct_temporal")
          .agg(n_derive=("n_derive", "mean"), cv_auc=("cv_auc", "mean"),
               cv_auc_sd=("cv_auc", "std"), geo_auc=("geo_auc", "mean"),
               geo_auc_sd=("geo_auc", "std"))
          .reset_index().rename(columns={"pct_temporal": "pct"}))
    t3 = t3[t3["pct"].isin(PCTS)].reset_index(drop=True)
    t3["arm"] = "temporal"
    t3["n_unique_patients"] = t3["n_derive"]

    agg = (d.groupby(["arm", "pct"])
             .agg(n_derive=("n_derive", "mean"),
                  n_unique_patients=("n_unique_patients", "mean"),
                  cv_auc=("cv_auc", "mean"), cv_auc_sd=("cv_auc", "std"),
                  geo_auc=("geo_auc", "mean"), geo_auc_sd=("geo_auc", "std"))
             .reset_index())
    alld = pd.concat([t3.rename(columns={"cv_auc": "cv_auc"})
                      [["arm", "pct", "n_derive", "n_unique_patients",
                        "cv_auc", "cv_auc_sd", "geo_auc", "geo_auc_sd"]],
                      agg], axis=0, ignore_index=True)

    piv = alld.pivot(index="pct", columns="arm",
                     values=["cv_auc", "geo_auc"]).round(4)
    piv.columns = [f"{b}_{a}" for a, b in piv.columns]
    piv = piv.reset_index()
    # geo_auc 来自积累阶梯的对照臂，与 CV 折叠方式无关 → 两臂完全相同，
    # 去掉冗余列
    piv = piv.drop(columns=[c for c in piv.columns
                            if c == "control_naive_geo_auc"])
    ref = (alld[alld["arm"] == "control_naive"].set_index("pct")
           .reindex(piv["pct"]))
    piv.insert(1, "n_derive",
               ref["n_derive"].round(0).astype(int).values)
    piv.insert(2, "n_unique_patients",
               ref["n_unique_patients"].round(0).astype(int).values)
    piv["cv_auc_inflation_naive_minus_grouped"] = (
        piv["control_naive_cv_auc"] - piv["control_grouped_cv_auc"]).round(4)
    C.save_table(piv, "TableS24_control_arm_grouped")
    C.log(piv.to_string(index=False))

    C.log("\n  解读：对照臂的 info 增量为 0（只是同一批患者的有放回重采样），"
          "因此 geo_auc 不随样本量上升；而 naive CV 的 cv_auc 单调升高，"
          "升高幅度即泄露造成的虚高。grouped CV 把同一患者的重复副本绑在"
          "同一折，虚高应被大幅压缩。")
    return d


if __name__ == "__main__":
    main()
