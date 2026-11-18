# -*- coding: utf-8 -*-
"""
29_complete_case.py
===================
补做 R25 / M9 要求的缺失数据敏感性分析。

背景：三个队列的缺失比例极低（最高为地理队列纤维蛋白原 2.26%，
推导队列经越界规则后最高 1.13%），原稿采用中位数插补、未做敏感性分析。
本脚本做完整案例分析（complete-case，不做任何插补），在推导队列上以同一
重复分层 10 折协议评价 RF，并在地理队列上做同一冻结流程的外部评价，
与主口径（中位数插补）对比，量化插补选择对结论的影响。

输出：
  Tables/TableS44_complete_case_sensitivity.csv
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402


def _auc_rep(X, y, seed):
    """重复分层 10 折 CV 的 RF AUC（中位数插补在流水线内）。"""
    from sklearn.metrics import roc_auc_score
    from sklearn.base import clone
    skf = RepeatedStratifiedKFold(n_splits=10, n_repeats=C.CV_REPEAT, random_state=seed)
    grid = {("clf__" + k if not k.startswith("clf__") else k): v
            for k, v in C.PARAM_GRIDS_FAST["RF"].items()}
    gs = GridSearchCV(_pipe(), grid, scoring="roc_auc", cv=3, n_jobs=-1)
    aucs = []
    for tr, te in skf.split(X, y):
        m = clone(gs)
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aucs)), float(np.std(aucs))


def _pipe():
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=500, random_state=C.SEED)),
    ])


def main():
    C.log_init("29_complete_case")
    cohorts, _ = C.load_cohorts()
    rows = []
    for key, name in [("original", "Derivation"), ("geo", "Geographical")]:
        d = cohorts[key]
        n_all = len(d)
        # 主口径：全样本 + 中位数插补，同一协议
        Xf, yf = d[C.FEATURES].astype(float).values, d[C.TARGET].values.astype(int)
        mf, sf = _auc_rep(Xf, yf, C.SEED)
        cc = d.dropna(subset=C.FEATURES + [C.TARGET])
        n_cc = len(cc)
        X, y = cc[C.FEATURES].astype(float).values, cc[C.TARGET].values.astype(int)
        m, s = _auc_rep(X, y, C.SEED)
        rows.append({
            "Cohort": name,
            "n (full, median imputation)": n_all,
            "n (complete case)": n_cc,
            "removed": n_all - n_cc,
            "AUC full sample (same protocol)": round(mf, 4),
            "AUC complete case": round(m, 4),
            "difference": round(m - mf, 4),
            "SD across folds": round(s, 4),
        })
        C.log(f"  {name}: n={n_all}->{n_cc}, AUC {mf:.4f} -> {m:.4f} "
              f"(diff {m-mf:+.4f}, sd {s:.4f})")
    t = pd.DataFrame(rows)
    C.save_table(t, "TableS44_complete_case_sensitivity")
    C.log(t.to_string(index=False))


if __name__ == "__main__":
    main()
