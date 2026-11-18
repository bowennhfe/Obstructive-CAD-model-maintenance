# -*- coding: utf-8 -*-
"""
12_centre_structure.py — 推导集中心结构图（新增分析的可视化）

Panel A 队列两两可分性热图
Panel B 纤维蛋白原单变量判别力在四个队列间的崩塌
Panel C 推导集内部的跨中心迁移（leave-one-centre-out）
Panel D 同样本量下中心多样性的贡献（配对实验）
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, openpyxl
import matplotlib.pyplot as plt
import common as C
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

MAP = {'age':'age','gender':'gender','systolic blood pressure':'sbp',
       'diastolic blood pressure':'dbp','mean arterial pressure':'map',
       'pulse pressure index':'ppi','pulse pressure':'pp','hypertension':'htn',
       'diabetes':'dm','siabetes':'dm','smoking':'smk','dyslipidemia':'dys',
       'total cholesterol':'tc','triglycerides':'tg','LDL-C':'ldl','HDL-C':'hdl',
       'uric acid':'ua','plasma fibrinogen':'fib','creatinine':'cr','urea':'urea'}
KEYS = ['age','gender','sbp','dbp','map','ppi','pp','htn','dm','smk','dys',
        'tc','tg','ldl','hdl','ua','fib','cr','urea']


def read_any(p):
    for enc in ['gbk','utf-8-sig','utf-8']:
        try: return pd.read_csv(p, encoding=enc)
        except Exception: pass
    raise RuntimeError(p)


def tidy_keys(df, lc):
    df = df.copy(); df.columns=[str(c).strip() for c in df.columns]
    df = df.rename(columns={c:MAP[c] for c in df.columns if c in MAP})
    o = df[KEYS].apply(pd.to_numeric, errors='coerce')
    ys = pd.to_numeric(df[lc], errors='coerce')
    if set(ys.dropna().unique()) <= {1,2}: ys = ys.map({1:0,2:1})
    o['y'] = ys.astype(int); return o


def fp(df):
    return df[KEYS+['y']].round(2).astype(object).fillna('NA').astype(str).agg('|'.join,axis=1)


cohorts, _ = C.load_cohorts(verbose=False)
o, t, g = cohorts['original'].copy(), cohorts['temporal'].copy(), cohorts['geo'].copy()
wb = openpyxl.load_workbook(os.path.join(C.DATA_DIR,'original data.xlsx'),
                            read_only=True, data_only=True)
rows = list(wb['data20221010'].iter_rows(values_only=True))
raw = pd.DataFrame(rows[1:], columns=[str(c).strip() for c in rows[0]]).dropna(how='all')
fr = fp(tidy_keys(raw,'label'))
se = set(fp(tidy_keys(read_any(r'E:\CAD-RADS\旧数据\NHFE2017-2022.csv'),'label')))
sf = set(fp(tidy_keys(read_any(r'E:\CAD-RADS\旧数据\NHFY.csv'),'label')))
o['centre'] = np.where(fr.isin(se),'E',np.where(fr.isin(sf),'Y','UNK'))
E, Y = o[o.centre=='E'], o[o.centre=='Y']

# ---------------------------------------------------------------- 数据准备
SETS = {'Centre 1': E, 'Centre 2': Y, 'Temporal': t, 'Geographical': g}
names = list(SETS)

SEP = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
for i, a in enumerate(names):
    for b in names[i+1:]:
        X = pd.concat([SETS[a][C.FEATURES], SETS[b][C.FEATURES]], ignore_index=True)
        y = np.r_[np.zeros(len(SETS[a])), np.ones(len(SETS[b]))]
        m = Pipeline([('imp', SimpleImputer(strategy='median')),
                      ('sc', StandardScaler()),
                      ('clf', LogisticRegression(max_iter=2000))])
        p = cross_val_predict(m, X, y, cv=StratifiedKFold(5, shuffle=True,
                             random_state=C.SEED), method='predict_proba')[:, 1]
        v = roc_auc_score(y, p)
        SEP.loc[a, b] = SEP.loc[b, a] = v

FIB_AUC = {}
for nm, d in SETS.items():
    x = d['fib']; yv = d['label']; mk = x.notna()
    a = roc_auc_score(yv[mk], x[mk]); FIB_AUC[nm] = max(a, 1-a)

TRANSPORT = [('Centre 1 → Centre 2', 0.728, 'cross-centre'),
             ('Centre 2 → Centre 1', 0.780, 'cross-centre'),
             ('Centre 1 → geo', 0.705, 'to geo'),
             ('Centre 2 → geo', 0.652, 'to geo'),
             ('Both centres → geo', 0.691, 'to geo')]
REF_INTERNAL = 0.832

exp = C.read_csv_out('s11_diversity_experiment_raw')
CFG = ['Centre 1 only', 'Centre 2 only', 'Both centres (mixed)']
CFG_C = ['#1f77b4', '#ff7f0e', '#9467bd']

# ---------------------------------------------------------------- 绘图
fig = plt.figure(figsize=(7.4, 7.0))
gs = fig.add_gridspec(2, 2, hspace=0.60, wspace=0.42)

# A 可分性热图
axA = fig.add_subplot(gs[0, 0])
im = axA.imshow(SEP.values, cmap='Oranges', vmin=0.5, vmax=1.0)
axA.set_xticks(range(len(names))); axA.set_yticks(range(len(names)))
axA.set_xticklabels(names, rotation=34, ha='right', fontsize=7.4)
axA.set_yticklabels(names, fontsize=8)
for i in range(len(names)):
    for j in range(len(names)):
        v = SEP.values[i, j]
        axA.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7.6,
                 color='white' if v > 0.82 else '#222222')
axA.set_title('Cohort separability\n(LR CV AUC; 0.5 = indistinguishable)',
              fontsize=8.8, linespacing=1.35)
for sp in axA.spines.values():
    sp.set_visible(False)
axA.tick_params(length=0)
axA.set_xlabel('')
axA.set_ylabel('')

# B fib 单变量判别力
# 时间队列取自中心 1 的后期时段 → 用中心 1 的红色系（浅色）标记，与「另一中心」区分
axB = fig.add_subplot(gs[0, 1])
ks = list(FIB_AUC)
_COL_B = {'Centre 1': '#d62728', 'Centre 2': '#7f7f7f',
          'Temporal': '#f4a3a0', 'Geographical': '#2ca02c'}
cols = [_COL_B.get(k, '#7f7f7f') for k in ks]
bars = axB.bar(range(len(ks)), [FIB_AUC[k] for k in ks], color=cols, width=0.62,
               edgecolor='black', linewidth=0.5)
C.ref_line(axB, y=0.5, color='#666666', lw=0.9, ls='--')
axB.set_xticks(range(len(ks)))
_xt = ['Centre 1\n(2017-2022)', 'Centre 2\n(2017-2022)',
       'Centre 1\n(2023-2026)', 'Other hospital']
axB.set_xticklabels(_xt, fontsize=7.2, ha='center')
axB.set_ylim(0.45, 0.86)
axB.set_ylabel('Univariable AUC of fibrinogen', fontsize=8.5)
axB.set_title('Fibrinogen signal confined\nto a single centre-period cell',
              fontsize=8.8, linespacing=1.35)
for b, k in zip(bars, ks):
    axB.text(b.get_x()+b.get_width()/2, b.get_height()+0.008, f'{FIB_AUC[k]:.3f}',
             ha='center', fontsize=7.8)
C.style_axes(axB)

# C 跨中心迁移
axC = fig.add_subplot(gs[1, 0])
lab = [x[0] for x in TRANSPORT]; val = [x[1] for x in TRANSPORT]
cols = ['#8c564b' if x[2] == 'cross-centre' else '#2ca02c' for x in TRANSPORT]
axC.barh(range(len(lab))[::-1], val, color=cols, height=0.6,
         edgecolor='black', linewidth=0.5)
axC.axvline(REF_INTERNAL, color='#d62728', lw=1.2, ls='--')
axC.text(REF_INTERNAL-0.005, 0.03, f'internal CV {REF_INTERNAL:.3f}',
         transform=axC.get_xaxis_transform(), color='#d62728', fontsize=7.2,
         ha='right', va='bottom')
axC.set_yticks(range(len(lab))[::-1]); axC.set_yticklabels(lab, fontsize=8)
axC.set_xlim(0.55, 0.90)
axC.set_xlabel('External AUC', fontsize=8.5)
axC.set_title('Transport across sources', fontsize=8.8)
for i, v in enumerate(val):
    axC.text(v+0.004, len(lab)-1-i, f'{v:.3f}', va='center', fontsize=7.8)
C.style_axes(axC, axis='x')

# D 同样本量中心多样性
axD = fig.add_subplot(gs[1, 1])
n_show = 176
sub = exp[exp.n_train == n_show]
for i, (cfg, col) in enumerate(zip(CFG, CFG_C)):
    v = sub[sub.config == cfg]['AUC'].values
    xj = np.random.default_rng(7).normal(i, 0.055, len(v))
    axD.scatter(xj, v, s=9, color=col, alpha=0.55, linewidths=0)
    axD.plot([i-0.22, i+0.22], [v.mean()]*2, color=col, lw=2.0)
    axD.text(i, v.mean() + 0.024, f'{v.mean():.3f}', ha='center',
             fontsize=7.8, color=col, fontweight='bold')
# 使均值标签落在坐标区内，不与面板标题争位
_lo = min(sub['AUC'].min(), 0.60)
_hi = max(sub['AUC'].max(), 0.74)
_pad = 0.055 * (_hi - _lo)
axD.set_ylim(_lo - _pad, _hi + 2.9 * _pad)
axD.set_xticks(range(3))
axD.set_xticklabels(['Centre 1\nonly', 'Centre 2\nonly', 'Both centres\n(mixed)'],
                    fontsize=8)
axD.set_ylabel('AUC on geographical cohort', fontsize=8.5)
axD.set_title(f'Matched sample size\n(n={n_show} x {len(sub)//3} repeats)',
              fontsize=8.8, linespacing=1.35)
C.style_axes(axD)

fig.suptitle('Internal structure of the multicentre derivation cohort',
             fontsize=11, y=1.005)

C.save_figure(fig, 'Fig11_centre_structure')
plt.close(fig)

pd.DataFrame({'cohort': ks, 'fib_univariable_AUC': [FIB_AUC[k] for k in ks]}).to_csv(
    os.path.join(C.TABLE_DIR, 'TableS16_fib_by_cohort.csv'), index=False)
SEP.round(3).to_csv(os.path.join(C.TABLE_DIR, 'TableS17_separability.csv'))
C.log('Fig11 + TableS16 + TableS17 已生成')
