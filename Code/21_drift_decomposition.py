# -*- coding: utf-8 -*-
"""21 —— 变量判别力漂移的分解：跨中心分量 vs 同中心时间分量
=============================================================================
背景（作者 2026-09-27 确认）
    时间队列完全来自**中心 1**（NHFE），采集期为 2023-01 至 2026-07；
    NHFE 本体为 2017-2022。故：
        Centre 1 (2017-2022)   -> Temporal (Centre 1, 2023-01..2026-07)
        是同中心、跨时期的变化 = **时间漂移（temporal drift）**
        Centre 1 (2017-2022)   -> Centre 2 (2017-2022)
        是同时期、跨中心的变化 = **跨中心分量（between-centre component）**

目标
    对 19 个候选变量逐一算单变量 AUC（方向校正为 max(a, 1-a)），分解：
        |AUC(C1) - AUC(temporal)|   时间分量（同中心两时期）
        |AUC(C1) - AUC(C2)|         跨中心分量（同期两中心）
    并给出"其余变量落差 ≤0.067"这一说法的核查结果（该值原为混算）。

输出
    Tables/TableS18b_drift_decomposition.csv   逐变量三分量明细
    Outputs/drift_decomposition.json           摘要统计（供 04 生成器引用）
=============================================================================
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, openpyxl
import common as C
from sklearn.metrics import roc_auc_score as ra

MAP = {'age': 'age', 'gender': 'gender', 'systolic blood pressure': 'sbp',
       'diastolic blood pressure': 'dbp', 'mean arterial pressure': 'map',
       'pulse pressure index': 'ppi', 'pulse pressure': 'pp', 'hypertension': 'htn',
       'diabetes': 'dm', 'siabetes': 'dm', 'smoking': 'smk', 'dyslipidemia': 'dys',
       'total cholesterol': 'tc', 'triglycerides': 'tg', 'LDL-C': 'ldl', 'HDL-C': 'hdl',
       'uric acid': 'ua', 'plasma fibrinogen': 'fib', 'creatinine': 'cr', 'urea': 'urea'}
KEYS = ['age', 'gender', 'sbp', 'dbp', 'map', 'ppi', 'pp', 'htn', 'dm', 'smk',
        'dys', 'tc', 'tg', 'ldl', 'hdl', 'ua', 'fib', 'cr', 'urea']


def read_any(p):
    for enc in ['gbk', 'utf-8-sig', 'utf-8']:
        try:
            return pd.read_csv(p, encoding=enc)
        except Exception:
            pass
    raise RuntimeError(p)


def tidy_keys(df, lc):
    df = df.copy(); df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={c: MAP[c] for c in df.columns if c in MAP})
    o = df[KEYS].apply(pd.to_numeric, errors='coerce')
    ys = pd.to_numeric(df[lc], errors='coerce')
    if set(ys.dropna().unique()) <= {1, 2}:
        ys = ys.map({1: 0, 2: 1})
    o['y'] = ys.astype(int); return o


def fp(df):
    return df[KEYS + ['y']].round(2).astype(object).fillna('NA').astype(str).agg('|'.join, axis=1)


def uni_auc(d, v):
    x = pd.to_numeric(d[v], errors='coerce')
    y = pd.to_numeric(d['label'], errors='coerce') if 'label' in d else pd.to_numeric(d['y'], errors='coerce')
    m = x.notna() & y.notna()
    if m.sum() < 20 or y[m].nunique() < 2:
        return np.nan
    a = ra(y[m], x[m])
    return max(a, 1 - a)


# ---------------------------------------------------------------- 队列装配
cohorts, _ = C.load_cohorts(verbose=False)
o = cohorts['original'].copy(); t = cohorts['temporal'].copy(); g = cohorts['geo'].copy()

# original 的 xlsx 原始行（带 label 列）→ 与 NHFE/NHFY 指纹匹配，复原中心归属
wb = openpyxl.load_workbook(os.path.join(C.DATA_DIR, 'original data.xlsx'),
                            read_only=True, data_only=True)
rows = list(wb['data20221010'].iter_rows(values_only=True))
raw = pd.DataFrame(rows[1:], columns=[str(c).strip() for c in rows[0]]).dropna(how='all')

fr = fp(tidy_keys(raw, 'label'))
se = set(fp(tidy_keys(read_any(r'E:\CAD-RADS\旧数据\NHFE2017-2022.csv'), 'label')))
sf = set(fp(tidy_keys(read_any(r'E:\CAD-RADS\旧数据\NHFY.csv'), 'label')))
o['centre'] = np.where(fr.isin(se), 'E', np.where(fr.isin(sf), 'Y', 'UNK'))

C1 = o[o.centre == 'E'].copy()      # 中心 1，2017-2022
C2 = o[o.centre == 'Y'].copy()      # 中心 2，2017-2022
TMP = t.copy()                      # 中心 1，2023-01 至 2026-07
GEO = g.copy()                      # 另一家医院

print('=== 队列规模 ===')
for nm, d in [('Centre 1 (2017-2022)', C1), ('Centre 2 (2017-2022)', C2),
              ('Temporal (Centre 1, 2023.01-2026.05)', TMP),
              ('Geographical', GEO)]:
    print(f'  {nm:38s} n={len(d):4d}  事件={int(d.label.sum()):4d} '
          f'({d.label.mean()*100:5.1f}%)')

# ---------------------------------------------------------------- 逐变量分解
rec = []
for v in C.FEATURES:
    a1, a2, at, ag = (uni_auc(C1, v), uni_auc(C2, v), uni_auc(TMP, v), uni_auc(GEO, v))
    rec.append(dict(variable=v, label=C.LABEL_EN.get(v, v),
                    c1=a1, c2=a2, temporal=at, geo=ag,
                    temporal_drift=(a1 - at) if np.isfinite(a1) and np.isfinite(at) else np.nan,
                    between_centre=(a1 - a2) if np.isfinite(a1) and np.isfinite(a2) else np.nan))
D = pd.DataFrame(rec)

# 用于"其余变量"的判定：以时间漂移绝对值与跨中心绝对值分别取
D['abs_temporal'] = D.temporal_drift.abs()
D['abs_between'] = D.between_centre.abs()
D = D.sort_values('abs_temporal', ascending=False).reset_index(drop=True)

print()
print('=== 单变量 AUC 与两种漂移分解（按时间漂移降序）===')
show = D[['variable', 'c1', 'c2', 'temporal', 'geo',
          'temporal_drift', 'between_centre']].round(3)
print(show.to_string(index=False))

# ---------------------------------------------------------------- "≤0.067" 核查
fib = D[D.variable == 'fib']
oth = D[D.variable != 'fib']
t_max = float(oth.abs_temporal.max()); t_var = oth.loc[oth.abs_temporal.idxmax(), 'variable']
b_max = float(oth.abs_between.max());  b_var = oth.loc[oth.abs_between.idxmax(), 'variable']

print()
print('=== "其余变量落差 ≤0.067" 核查 ===')
print(f'  排除 fib 后，同中心时间漂移最大 = {t_max:.3f}  ({t_var})')
print(f'  排除 fib 后，跨中心漂移最大   = {b_max:.3f}  ({b_var})')
print(f'  原稿 0.067 系跨中心/混算口径 —— 见 TableS18 列 "Highest minus max(other cohorts)"')

# 可变方向的变量数统计
print()
n_t_down = int((D[D.variable != 'fib'].temporal_drift > 0).sum())
n_t_up = int((D[D.variable != 'fib'].temporal_drift < 0).sum())
print(f'=== 排除 fib 的 18 个变量：时间漂移方向 ===')
print(f'  AUC 下降（中心1 > 时间队列）= {n_t_down} 个；上升 = {n_t_up} 个')
print(f'  全部 19 个变量中下降 = {int((D.temporal_drift > 0).sum())} 个')

# fib 的两种情况
print()
print('=== fib 分解 ===')
print(f'  Centre 1 AUC = {fib.c1.iloc[0]:.3f} -> Temporal {fib.temporal.iloc[0]:.3f} '
      f'(时间漂移 {fib.temporal_drift.iloc[0]:+.3f})')
print(f'  Centre 1 AUC = {fib.c1.iloc[0]:.3f} -> Centre 2 {fib.c2.iloc[0]:.3f} '
      f'(跨中心 {fib.between_centre.iloc[0]:+.3f})')

# ---------------------------------------------------------------- 落盘（表）
out = D[['variable', 'label', 'c1', 'c2', 'temporal', 'geo',
         'temporal_drift', 'between_centre']].copy()
out = out.rename(columns={
    'variable': 'Variable', 'label': 'Variable (full name)',
    'c1': 'Centre 1 (2017-2022)', 'c2': 'Centre 2 (2017-2022)',
    'temporal': 'Temporal (Centre 1, 2023-2026)',
    'geo': 'Geographical (other hospital)',
    'temporal_drift': 'Temporal drift (C1 - Temporal)',
    'between_centre': 'Between-centre (C1 - C2)'})
for c in out.columns:
    if out[c].dtype.kind == 'f':
        out[c] = out[c].round(4)
C.save_table(out, 'TableS18b_drift_decomposition')

# 摘要统计（供 04 生成器引用）
summary = dict(
    var_temporal_max=t_var, temporal_max=round(t_max, 3),
    var_between_max=b_var, between_max=round(b_max, 3),
    fib_c1=round(float(fib.c1.iloc[0]), 3),
    fib_temporal=round(float(fib.temporal.iloc[0]), 3),
    fib_temporal_drift=round(float(fib.temporal_drift.iloc[0]), 3),
    fib_between=round(float(fib.between_centre.iloc[0]), 3),
    n_vars_19=19,
    n_auc_declined=int((D.temporal_drift > 0).sum()),
    n_others_declined=n_t_down,
    n_others_rose=n_t_up,
    n_c1=int(len(C1)), n_c2=int(len(C2)), n_temporal=int(len(TMP)), n_geo=int(len(GEO)),
    ev_c1=int(C1.label.sum()), ev_c2=int(C2.label.sum()),
    ev_temporal=int(TMP.label.sum()), ev_geo=int(GEO.label.sum()),
)
with open(os.path.join(C.OUT_DIR, 'drift_decomposition.json'), 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print()
print('已保存 TableS18b_drift_decomposition.csv / Outputs/drift_decomposition.json')
