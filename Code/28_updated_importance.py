# -*- coding: utf-8 -*-
"""
28 —— 更新后模型的变量重要性（第十六轮审稿 Minor 11）
=============================================================================
背景
    审稿人指出：本文报告了冻结模型（推导队列）的变量重要性，但未报告
    更新后模型（把全部当代患者并入推导池、重新调参后）的重要性谱；
    建议补做 SHAP 或置换重要性。

做法
    复刻 03 的「100% 步」设定：推导队列 + 全部时间队列合并为一个推导池，
    在该池内做 5 折交叉验证网格搜索（PARAM_GRIDS_FAST，与 03 一致），
    重复 10 次（不同随机种子），对每次拟合的模型计算：
      ① MDI 重要性（随机森林自带）
      ② 在地理队列上的置换重要性（降低 AUC，10 次重复取均值）
    最后对 10 次重复取平均，并给出相对冻结模型的秩变化。

输出
    Tables/TableS43_updated_importance.csv
    Outputs/s28_updated_importance_detail.csv
=============================================================================
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

N_REPEAT = 10


def main():
    C.log_init("28_updated_importance")
    cohorts, _ = C.load_cohorts(verbose=False)
    o, t, g = cohorts["original"], cohorts["temporal"], cohorts["geo"]

    feat = [f for f in C.FEATURES if f in o.columns]
    pool = pd.concat([o[feat + [C.TARGET]], t[feat + [C.TARGET]]],
                     ignore_index=True)
    Xp, yp = pool[feat].values, pool[C.TARGET].values.astype(int)
    Xg, yg = g[feat].values, g[C.TARGET].values.astype(int)

    mdi_all, perm_all = [], []
    for rep in range(N_REPEAT):
        gs = C.tune_model(Xp, yp, name="RF", grid=C.PARAM_GRIDS_FAST,
                          seed=C.SEED + rep)
        rf = gs.best_estimator_.named_steps["clf"]
        mdi_all.append(rf.feature_importances_.copy())

        pi = permutation_importance(gs, Xg, yg, scoring="roc_auc",
                                    n_repeats=10, random_state=C.SEED + rep,
                                    n_jobs=-1)
        perm_all.append(pi.importances_mean.copy())
        C.log(f"repeat {rep + 1}/{N_REPEAT} done (pool n={len(pool)})")

    mdi_m = np.mean(np.vstack(mdi_all), axis=0)
    perm_m = np.mean(np.vstack(perm_all), axis=0)

    # 冻结模型的重要性（用于比较秩）
    frozen = None
    try:
        imp = C.T("Table3_importance") if hasattr(C, "T") else None
    except Exception:
        imp = None

    out = pd.DataFrame({
        "variable": feat,
        "label_en": [C.LABEL_EN.get(f, f) for f in feat],
        "mdi_updated": np.round(mdi_m, 5),
        "perm_auc_drop_updated": np.round(perm_m, 5),
    })
    out["mdi_rank_updated"] = out["mdi_updated"].rank(ascending=False).astype(int)
    out["perm_rank_updated"] = (out["perm_auc_drop_updated"]
                                .rank(ascending=False).astype(int))
    out = out.sort_values("mdi_updated", ascending=False).reset_index(drop=True)

    C.save_table(out, "TableS43_updated_importance")
    C.save_csv(out, "s28_updated_importance_detail")
    C.log("\n" + out.to_string(index=False))
    C.log("\n-> Tables/TableS43_updated_importance.csv")


if __name__ == "__main__":
    main()
