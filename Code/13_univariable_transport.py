# -*- coding: utf-8 -*-
"""探针：E 中心纤维蛋白原异常强判别力的来源核查 + 可分性-性能汇总。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, openpyxl
import common as C
from sklearn.metrics import roc_auc_score as ra

MAP={'age':'age','gender':'gender','systolic blood pressure':'sbp',
     'diastolic blood pressure':'dbp','mean arterial pressure':'map',
     'pulse pressure index':'ppi','pulse pressure':'pp','hypertension':'htn',
     'diabetes':'dm','siabetes':'dm','smoking':'smk','dyslipidemia':'dys',
     'total cholesterol':'tc','triglycerides':'tg','LDL-C':'ldl','HDL-C':'hdl',
     'uric acid':'ua','plasma fibrinogen':'fib','creatinine':'cr','urea':'urea'}
KEYS=['age','gender','sbp','dbp','map','ppi','pp','htn','dm','smk','dys','tc','tg','ldl','hdl','ua','fib','cr','urea']

def read_any(p):
    for enc in ['gbk','utf-8-sig','utf-8']:
        try: return pd.read_csv(p, encoding=enc)
        except Exception: pass
    raise RuntimeError(p)

def tidy_keys(df, lc):
    df=df.copy(); df.columns=[str(c).strip() for c in df.columns]
    df=df.rename(columns={c:MAP[c] for c in df.columns if c in MAP})
    o=df[KEYS].apply(pd.to_numeric, errors='coerce')
    ys=pd.to_numeric(df[lc], errors='coerce')
    if set(ys.dropna().unique())<={1,2}: ys=ys.map({1:0,2:1})
    o['y']=ys.astype(int); return o

def fp(df):
    return df[KEYS+['y']].round(2).astype(object).fillna('NA').astype(str).agg('|'.join,axis=1)

cohorts,_=C.load_cohorts(verbose=False)
o=cohorts['original'].copy(); t=cohorts['temporal'].copy(); g=cohorts['geo'].copy()
wb=openpyxl.load_workbook(os.path.join(C.DATA_DIR,'original data.xlsx'),read_only=True,data_only=True)
rows=list(wb['data20221010'].iter_rows(values_only=True))
raw=pd.DataFrame(rows[1:],columns=[str(c).strip() for c in rows[0]]).dropna(how='all')
fr=fp(tidy_keys(raw,'label'))
se=set(fp(tidy_keys(read_any(r'E:\CAD-RADS\旧数据\NHFE2017-2022.csv'),'label')))
sf=set(fp(tidy_keys(read_any(r'E:\CAD-RADS\旧数据\NHFY.csv'),'label')))
o['centre']=np.where(fr.isin(se),'E',np.where(fr.isin(sf),'Y','UNK'))
E=o[o.centre=='E'].copy(); Y=o[o.centre=='Y'].copy()

print('=== 纤维蛋白原按结局分组（事件 vs 非事件）===')
for nm,d in [('E',E),('Y',Y),('E+Y',o[o.centre!='UNK']),('temporal',t),('geo',g)]:
    a=d.loc[d.label==1,'fib'].dropna(); b=d.loc[d.label==0,'fib'].dropna()
    auc=ra(d.label.dropna(), d.fib.fillna(d.fib.median()))
    print(f'  {nm:9s} 事件组 fib 中位={a.median():.2f} (n={len(a):3d})  '
          f'非事件组={b.median():.2f} (n={len(b):3d})  差={a.median()-b.median():+.2f}  '
          f'AUC={auc:.3f}  重叠区[事件min={a.min():.2f}, 非事件max={b.max():.2f}]')

print()
print('=== E 中心：fib 在两组的区间分布（看是否人为分离）===')
for q in [0.05,0.25,0.5,0.75,0.95]:
    print(f"  q{q:.2f}: 事件={E.loc[E.label==1,'fib'].quantile(q):.2f}  "
          f"非事件={E.loc[E.label==0,'fib'].quantile(q):.2f}")

print()
print('=== 各队列全部变量的单变量 AUC（找 E 特异的变量）===')
tab={}
for nm,d in [('E',E),('Y',Y),('original442',o),('temporal',t),('geo',g)]:
    row={}
    for v in C.FEATURES:
        x=pd.to_numeric(d[v],errors='coerce'); yv=pd.to_numeric(d['label'],errors='coerce')
        m=x.notna()&yv.notna()
        if m.sum()<20 or yv[m].nunique()<2: row[v]=np.nan; continue
        a=ra(yv[m],x[m]); row[v]=max(a,1-a)
    tab[nm]=row
T=pd.DataFrame(tab)
T['E_minus_max_other']=T['E']-T[['Y','temporal','geo']].max(axis=1)
T=T.sort_values('E',ascending=False)
print(T.round(3).to_string())

print()
print('=== 训练-目标可分性 vs 外部表现（汇总）===')
rows=[
 ('Centre 1 -> Centre 2',0.912,0.728,0.95),
 ('Centre 2 -> Centre 1',0.912,0.780,1.30),
 ('Centre 1 -> geo',0.733,0.705,0.68),
 ('Centre 2 -> geo',0.916,0.652,0.89),
 ('Both centres -> geo',0.739,0.691,0.70),
 ('temporal -> geo (from main pipeline)',0.746,0.742,0.79),
 ('original442 -> geo (main pipeline)',0.739,0.719,0.70),
]
D=pd.DataFrame(rows,columns=['pair','separability','ext_AUC','ext_OE'])
print(D.round(3).to_string(index=False))
print()
print('  auc(separability vs ext_AUC)  =', round(float(np.corrcoef(D.separability[:5],D.ext_AUC[:5])[0,1]),3))

# ---------------------------------------------------------------- 落盘
T2 = T[['E','Y','original442','temporal','geo','E_minus_max_other']].round(4)
T2 = T2.rename(columns={'E': 'Centre 1', 'Y': 'Centre 2',
                        'original442': 'Derivation (pooled)',
                        'temporal': 'Temporal', 'geo': 'Geographical',
                        'E_minus_max_other': 'Highest minus max(other cohorts)'})
T2.index.name = 'Variable'
T2.to_csv(os.path.join(C.TABLE_DIR, 'TableS18_univariable_auc_by_cohort.csv'))
D.to_csv(os.path.join(C.TABLE_DIR, 'TableS19_separability_vs_transport.csv'),
         index=False)
print()
print('已保存 TableS18_univariable_auc_by_cohort.csv / TableS19_separability_vs_transport.csv')
