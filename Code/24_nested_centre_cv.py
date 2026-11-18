# -*- coding: utf-8 -*-
"""
24 —— 嵌套交叉验证与 centre-aware 交叉验证（第十二轮审稿 Major 6）
=============================================================================
目的
    原先报告的 0.832 来自「先在全部推导集上做网格搜索、再用同一批数据做
    交叉验证评价」，调参信息在评价中重复使用，因此偏乐观。本脚本给出两种
    不带该泄漏的内部口径：

    ① 嵌套交叉验证（repeated nested CV）
       外层 5 折分层，只用于评价；内层 5 折网格搜索只在训练折内进行；
       外层测试折的袋外预测汇合成一个 AUC。重复 R 次，报告均值与标准差。

    ② centre-aware 交叉验证（按中心分折，leave-one-centre-out）
       折 = 中心。每个折内先在该折的训练中心上做网格搜索，再评价另一中心；
       袋外预测汇合成一个 AUC。5 例中心归属未定的记录始终只进入训练。

    ③ 对照：非嵌套的内部 50 折 CV（原口径）与固定超参数的 LOCO（表 S14）

输出
    Tables/TableS38_nested_centre_cv.csv
    Outputs/s24_nested_centre_cv_detail.csv
=============================================================================
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import (StratifiedKFold, GridSearchCV,
                                     RepeatedStratifiedKFold)
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

R_NESTED = 3          # 嵌套 CV 的重复次数
OUTER = 5
INNER = 5
FIXED_LOCO = {"n_estimators": 500, "max_depth": 10, "min_samples_leaf": 1}


def _centre_labels():
    sys.path.insert(0, C.CODE_DIR)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "m11", os.path.join(C.CODE_DIR, "11_multicentre.py"))
    m11 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m11)
    return m11.build_centre_labels()


def nested_cv(X, y, seed, outer=OUTER, inner=INNER):
    """重复一次嵌套 CV，返回 (pooled_auc, fold_aucs, oof)."""
    X = X.reset_index(drop=True)
    oof = np.full(len(y), np.nan)
    fold_aucs = []
    skf = StratifiedKFold(outer, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        gs = GridSearchCV(
            C.build_estimator("RF", seed=seed),
            C.prefix_params("RF", C.PARAM_GRIDS["RF"]),
            scoring="roc_auc",
            cv=StratifiedKFold(inner, shuffle=True, random_state=seed),
            n_jobs=-1, refit=True)
        gs.fit(X.iloc[tr], y[tr])
        p = gs.best_estimator_.predict_proba(X.iloc[te])[:, 1]
        oof[te] = p
        fold_aucs.append(float(roc_auc_score(y[te], p)))
    return float(roc_auc_score(y, oof)), fold_aucs, oof


def centre_aware_cv(X, y, lab, seed):
    """按中心分折；折内网格搜索。返回 (pooled_auc, fold_rows, oof, mask)."""
    X = X.reset_index(drop=True)
    oof = np.full(len(y), np.nan)
    rows = []
    for c in ("E", "Y"):
        te = np.where(lab == c)[0]
        tr = np.where(lab != c)[0]                 # 另一中心 + 归属未定
        gs = GridSearchCV(
            C.build_estimator("RF", seed=seed),
            C.prefix_params("RF", C.PARAM_GRIDS["RF"]),
            scoring="roc_auc",
            cv=StratifiedKFold(INNER, shuffle=True, random_state=seed),
            n_jobs=-1, refit=True)
        gs.fit(X.iloc[tr], y[tr])
        p = gs.best_estimator_.predict_proba(X.iloc[te])[:, 1]
        oof[te] = p
        rows.append({
            "train_centre": "Centre 2 + unassigned" if c == "E" else
                            "Centre 1 + unassigned",
            "test_centre": "Centre 1" if c == "E" else "Centre 2",
            "n_train": int(len(tr)), "n_test": int(len(te)),
            "events_test": int(y[te].sum()),
            "auc": round(float(roc_auc_score(y[te], p)), 4),
            "best_params": str(gs.best_params_),
        })
    mask = ~np.isnan(oof)
    return float(roc_auc_score(y[mask], oof[mask])), rows, oof, mask


def loco_fixed(X, y, lab, seed):
    """固定超参数的 leave-one-centre-out（表 S14 的口径），用于对照。"""
    X = X.reset_index(drop=True)
    oof = np.full(len(y), np.nan)
    rows = []
    for c in ("E", "Y"):
        te = np.where(lab == c)[0]
        tr = np.where(lab != c)[0]
        pipe = C.build_estimator("RF", FIXED_LOCO, seed=seed)
        pipe.fit(X.iloc[tr], y[tr])
        p = pipe.predict_proba(X.iloc[te])[:, 1]
        oof[te] = p
        rows.append(float(roc_auc_score(y[te], p)))
    mask = ~np.isnan(oof)
    return float(roc_auc_score(y[mask], oof[mask])), rows, oof, mask


def main():
    C.log_init("24_nested_centre_cv")

    cohorts, _ = C.load_cohorts(verbose=False)
    o = cohorts["original"]
    X, y = C.derivation_label(o)
    y = np.asarray(y, int)
    lab = _centre_labels()
    assert len(lab) == len(y), (len(lab), len(y))
    C.log(f"推导队列 n={len(y)} 事件={y.sum()}")
    C.log(f"中心分布：Centre1={int((lab == 'E').sum())} "
          f"Centre2={int((lab == 'Y').sum())} 未定={int((lab == 'UNK').sum())}")

    out_rows = []

    # ---------- ③ 对照：原口径的 50 折 CV ----------
    C.section("对照：非嵌套 repeated 10x5 折内部 CV")
    rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=C.SEED)
    oof50 = np.full(len(y), np.nan)
    fold_auc = []
    for tr, te in rskf.split(X, y):
        pipe = C.build_estimator("RF", FIXED_LOCO, seed=C.SEED)
        pipe.fit(X.iloc[tr], y[tr])
        p = pipe.predict_proba(X.iloc[te])[:, 1]
        fold_auc.append(float(roc_auc_score(y[te], p)))
    auc50 = float(np.mean(fold_auc))
    C.log(f"内部 50 折 CV AUC（固定超参数，原口径）= {auc50:.4f} "
          f"(sd over folds {np.std(fold_auc):.4f})")
    out_rows.append({
        "analysis": "internal 50-fold CV (non-nested, reference)",
        "auc": round(auc50, 4), "sd": round(float(np.std(fold_auc)), 4),
        "n_outer_repeats": 5, "note": "hyperparameters fixed at the "
        "pre-specified grid optimum; the original reported figure"})

    # ---------- ① 嵌套 CV ----------
    C.section(f"嵌套交叉验证（外层 {OUTER} 折，内层 {INNER} 折，重复 {R_NESTED} 次）")
    nests, nests_sd, nests_fold = [], [], []
    for r in range(R_NESTED):
        a, fa, _ = nested_cv(X, y, seed=C.SEED + r)
        nests.append(a)
        nests_fold.append(fa)
        nests_sd.append(float(np.std(fa)))
        C.log(f"  repeat {r}: pooled AUC = {a:.4f}  "
              f"(fold AUCs {', '.join(f'{v:.3f}' for v in fa)})")
    C.log(f"嵌套 CV pooled AUC 均值 = {np.mean(nests):.4f} "
          f"(sd {np.std(nests):.4f})")
    out_rows.append({
        "analysis": "nested CV (outer 5-fold, inner 5-fold, grid searched "
                    "inside each outer training fold)",
        "auc": round(float(np.mean(nests)), 4),
        "sd": round(float(np.std(nests)), 4),
        "n_outer_repeats": R_NESTED,
        "note": "mean over repeats of the pooled out-of-fold AUC; "
                "inner-fold mean sd = %.4f" % float(np.mean(nests_sd))})

    # ---------- ② centre-aware CV ----------
    C.section("centre-aware 交叉验证（按中心分折，折内调参）")
    ca_auc, ca_rows, ca_oof, ca_mask = centre_aware_cv(X, y, lab, C.SEED)
    for r in ca_rows:
        C.log(f"  {r['train_centre']} -> {r['test_centre']}: AUC {r['auc']:.4f} "
              f"(n={r['n_test']}, events={r['events_test']})  {r['best_params']}")
    C.log(f"centre-aware pooled AUC = {ca_auc:.4f}  (n={int(ca_mask.sum())})")
    out_rows.append({
        "analysis": "centre-aware CV (leave-one-centre-out, inner grid search)",
        "auc": round(ca_auc, 4),
        "sd": round(float(np.std([r["auc"] for r in ca_rows])), 4),
        "n_outer_repeats": 2,
        "note": "folds are centres; the 5 records with no centre assignment "
                "enter training only"})

    # ---------- ③b 固定超参数的 LOCO（表 S14 口径）----------
    loco_auc, loco_a, _, lm = loco_fixed(X, y, lab, C.SEED)
    C.log(f"固定超参数 LOCO pooled AUC = {loco_auc:.4f} "
          f"(Centre1 {loco_a[0]:.3f} / Centre2 {loco_a[1]:.3f})")
    out_rows.append({
        "analysis": "leave-one-centre-out, fixed hyperparameters (Table S14)",
        "auc": round(loco_auc, 4),
        "sd": round(float(np.std(loco_a)), 4),
        "n_outer_repeats": 2,
        "note": "same centres as above but without inner tuning"})

    t = pd.DataFrame(out_rows)
    C.save_table(t, "TableS38_nested_centre_cv")
    C.log("\n" + t.to_string(index=False))

    det = pd.DataFrame({
        "centre": lab,
        "y": y,
        "oof_centre_aware": ca_oof,
    })
    C.save_csv(det, "s24_nested_centre_cv_detail")
    C.log("\n-> Tables/TableS38_nested_centre_cv.csv")


if __name__ == "__main__":
    main()
