# -*- coding: utf-8 -*-
"""
22 —— 中心分配与中心可分性的精度 / 稳健性补充
=============================================================================
回应外部方法学意见：
  (a) 中心分离 AUC 0.912 只给了点估计，应报告 95% CI；
  (b) 2.11 的中心分配用「19 个候选变量 + 结局」做精确指纹匹配，应做
      替代匹配策略的敏感性分析（不用结局）。

设计
  (a) 对每一对队列，用 19 个候选变量拟合 L2 逻辑回归、5 折交叉验证取袋外
      预测，得到交叉验证 AUC；再对「患者」做 2000 次有放回重抽样
      （分层保持两队列例数），重算 AUC，取 2.5/97.5 分位为 95% CI。
      与 17 号脚本对外部 AUC 的区间口径一致。
  (b) 把指纹匹配键从「19 变量 + 结局」换成「仅 19 变量」，重做分配，
      对比：各源例数、无法分配例数、两份分配完全一致的比例、以及
      在 19 变量键下同时命中两个来源登记表的记录数（歧义记录）。

输出
    Tables/TableS36_centre_assignment_sensitivity.csv
=============================================================================
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import openpyxl

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

MAP = {'age': 'age', 'gender': 'gender', 'systolic blood pressure': 'sbp',
       'diastolic blood pressure': 'dbp', 'mean arterial pressure': 'map',
       'pulse pressure index': 'ppi', 'pulse pressure': 'pp', 'hypertension': 'htn',
       'diabetes': 'dm', 'siabetes': 'dm', 'smoking': 'smk', 'dyslipidemia': 'dys',
       'total cholesterol': 'tc', 'triglycerides': 'tg', 'LDL-C': 'ldl', 'HDL-C': 'hdl',
       'uric acid': 'ua', 'plasma fibrinogen': 'fib', 'creatinine': 'cr', 'urea': 'urea'}
KEYS = ['age', 'gender', 'sbp', 'dbp', 'map', 'ppi', 'pp', 'htn', 'dm', 'smk', 'dys',
        'tc', 'tg', 'ldl', 'hdl', 'ua', 'fib', 'cr', 'urea']

N_BOOT = 2000


def read_any(p):
    for enc in ['gbk', 'utf-8-sig', 'utf-8']:
        try:
            return pd.read_csv(p, encoding=enc)
        except Exception:
            pass
    raise RuntimeError(p)


def tidy_keys(df, lc):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={c: MAP[c] for c in df.columns if c in MAP})
    o = df[KEYS].apply(pd.to_numeric, errors='coerce')
    ys = pd.to_numeric(df[lc], errors='coerce')
    if set(ys.dropna().unique()) <= {1, 2}:
        ys = ys.map({1: 0, 2: 1})
    o['y'] = ys.astype(int)
    return o


def fp(df, use_y=True):
    cols = KEYS + (['y'] if use_y else [])
    return df[cols].round(2).astype(object).fillna('NA').astype(str).agg('|'.join, axis=1)


def sec_separability(SETS):
    C.section("(a) 中心/队列可分性的交叉验证 AUC 与其 95% CI")
    names = list(SETS)
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            X = pd.concat([SETS[a][C.FEATURES], SETS[b][C.FEATURES]], ignore_index=True)
            y = np.r_[np.zeros(len(SETS[a])), np.ones(len(SETS[b]))]
            m = Pipeline([('imp', SimpleImputer(strategy='median')),
                          ('sc', StandardScaler()),
                          ('clf', LogisticRegression(max_iter=2000))])
            p = cross_val_predict(m, X, y, cv=StratifiedKFold(5, shuffle=True,
                                 random_state=C.SEED), method='predict_proba')[:, 1]
            auc = float(roc_auc_score(y, p))
            rng = np.random.RandomState(C.SEED)
            n1, n0 = int(y.sum()), int((1 - y).sum())
            i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]
            v = []
            for _ in range(N_BOOT):
                idx = np.r_[rng.choice(i1, n1, replace=True), rng.choice(i0, n0, replace=True)]
                v.append(roc_auc_score(y[idx], p[idx]))
            lo, hi = float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
            rows.append({"Analysis": "Cross-validated separability",
                         "Quantity": f"{a} vs {b}", "Estimate": round(auc, 3),
                         "95% CI": f"{lo:.3f}-{hi:.3f}"})
            C.log(f"  {a} vs {b}: CV AUC {auc:.3f} (95% CI {lo:.3f}-{hi:.3f})")
    return rows


def sec_assignment(original):
    C.section("(b) 中心分配对匹配键的敏感性（有/无结局）")
    wb = openpyxl.load_workbook(os.path.join(C.DATA_DIR, 'original data.xlsx'),
                                read_only=True, data_only=True)
    rows = list(wb['data20221010'].iter_rows(values_only=True))
    raw = pd.DataFrame(rows[1:], columns=[str(c).strip() for c in rows[0]]).dropna(how='all')
    raw2 = tidy_keys(raw, 'label')

    e = tidy_keys(read_any(r'E:\CAD-RADS\旧数据\NHFE2017-2022.csv'), 'label')
    y_ = tidy_keys(read_any(r'E:\CAD-RADS\旧数据\NHFY.csv'), 'label')
    se_y, sf_y = set(fp(e, True)), set(fp(y_, True))
    se_n, sf_n = set(fp(e, False)), set(fp(y_, False))

    f_y, f_n = fp(raw2, True), fp(raw2, False)
    a_y = np.where(f_y.isin(se_y), 'E', np.where(f_y.isin(sf_y), 'Y', 'UNK'))
    a_n = np.where(f_n.isin(se_n), 'E', np.where(f_n.isin(sf_n), 'Y', 'UNK'))
    n = len(a_y)
    same = int((a_y == a_n).sum())
    amb = int((f_n.isin(se_n) & f_n.isin(sf_n)).sum())
    C.log(f"  有结局键：E={int((a_y=='E').sum())} Y={int((a_y=='Y').sum())} UNK={int((a_y=='UNK').sum())}")
    C.log(f"  无结局键：E={int((a_n=='E').sum())} Y={int((a_n=='Y').sum())} UNK={int((a_n=='UNK').sum())}")
    C.log(f"  两键分配一致 {same}/{n}；19 变量键下同时命中两来源的歧义记录 {amb}")

    rows = [
        {"Analysis": "Centre assignment", "Quantity": "Records (of 442) assigned to the same source with and without the outcome in the matching key",
         "Estimate": f"{same} / {n}", "95% CI": "n/a"},
        {"Analysis": "Centre assignment", "Quantity": "Centre 1 / centre 2 / unassigned counts, outcome-free key",
         "Estimate": f"{int((a_n=='E').sum())} / {int((a_n=='Y').sum())} / {int((a_n=='UNK').sum())}", "95% CI": "n/a"},
        {"Analysis": "Centre assignment", "Quantity": "Records whose 19-variable fingerprint matched both source registries (ambiguous)",
         "Estimate": amb, "95% CI": "n/a"},
    ]
    return rows


def main():
    C.log_init("22_centre_assignment_sensitivity")
    t0 = time.time()
    cohorts, _ = C.load_cohorts(verbose=False)
    SETS = {'Centre 1': cohorts['original'].copy(), 'Centre 2': cohorts['original'].copy(),
            'Temporal': cohorts['temporal'].copy(), 'Geographical': cohorts['geo'].copy()}
    # 中心 1 / 中心 2 需先按指纹拆分（与 12 号脚本同口径）
    o = cohorts['original'].copy()
    wb = openpyxl.load_workbook(os.path.join(C.DATA_DIR, 'original data.xlsx'),
                                read_only=True, data_only=True)
    rws = list(wb['data20221010'].iter_rows(values_only=True))
    raw = pd.DataFrame(rws[1:], columns=[str(c).strip() for c in rws[0]]).dropna(how='all')
    fr = fp(tidy_keys(raw, 'label'), True)
    se = set(fp(tidy_keys(read_any(r'E:\CAD-RADS\旧数据\NHFE2017-2022.csv'), 'label'), True))
    sf = set(fp(tidy_keys(read_any(r'E:\CAD-RADS\旧数据\NHFY.csv'), 'label'), True))
    o['centre'] = np.where(fr.isin(se), 'E', np.where(fr.isin(sf), 'Y', 'UNK'))
    SETS['Centre 1'] = o[o.centre == 'E']
    SETS['Centre 2'] = o[o.centre == 'Y']

    rows = sec_separability(SETS) + sec_assignment(o)
    t = pd.DataFrame(rows)
    C.save_table(t, "TableS36_centre_assignment_sensitivity")
    C.log(t.to_string(index=False))
    C.section(f"脚本 22 完成，总耗时 {C.timer(t0)}")


if __name__ == "__main__":
    main()
