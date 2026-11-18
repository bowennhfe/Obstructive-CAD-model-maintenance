# -*- coding: utf-8 -*-
"""
11_multicentre.py — 推导队列的中心结构分析（新增分析）

背景：推导队列 442 例实为两个来源中心合并（E = 255 例、Y = 182 例，另有 5 例无法
归入任一来源，本分析中排除）。本脚本量化"中心多样性"对**第三方医院**（geo 177）
外部效度的贡献，并与"样本量"效应解耦。

核心问题：在**训练样本量完全相同**的条件下，来自 2 个中心的数据是否优于来自
1 个中心的数据？

输出：s11_*.csv（表）与 FigS10/s11 图（由本脚本直接出图）。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, openpyxl
import common as C
from sklearn.metrics import roc_auc_score

# -----------------------------------------------------------------------------
# 中心标签（依据 NHFE2017-2022.csv / NHFY.csv 与推导集的逐行指纹匹配）
# -----------------------------------------------------------------------------
MAP = {'age': 'age', 'gender': 'gender', 'systolic blood pressure': 'sbp',
       'diastolic blood pressure': 'dbp', 'mean arterial pressure': 'map',
       'pulse pressure index': 'ppi', 'pulse pressure': 'pp',
       'hypertension': 'htn', 'diabetes': 'dm', 'siabetes': 'dm',
       'smoking': 'smk', 'dyslipidemia': 'dys', 'total cholesterol': 'tc',
       'triglycerides': 'tg', 'LDL-C': 'ldl', 'HDL-C': 'hdl',
       'uric acid': 'ua', 'plasma fibrinogen': 'fib', 'creatinine': 'cr',
       'urea': 'urea'}
KEYS = ['age', 'gender', 'sbp', 'dbp', 'map', 'ppi', 'pp', 'htn', 'dm', 'smk',
        'dys', 'tc', 'tg', 'ldl', 'hdl', 'ua', 'fib', 'cr', 'urea']
CENTRE_LABEL = {'E': 'Centre 1', 'Y': 'Centre 2'}
R_REPEAT = 20
N_GRID = [100, 140, 176]
FIXED_PARAMS = {'n_estimators': 500, 'max_depth': 10, 'min_samples_leaf': 1}


def read_any(p):
    for enc in ['gbk', 'utf-8-sig', 'utf-8']:
        try:
            return pd.read_csv(p, encoding=enc)
        except Exception:
            pass
    raise RuntimeError(p)


def tidy_keys(df, label_col):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={c: MAP[c] for c in df.columns if c in MAP})
    o = df[KEYS].apply(pd.to_numeric, errors='coerce')
    ys = pd.to_numeric(df[label_col], errors='coerce')
    if set(ys.dropna().unique()) <= {1, 2}:
        ys = ys.map({1: 0, 2: 1})
    o['y'] = ys.astype(int)
    return o


def fingerprint(df):
    return (df[KEYS + ['y']].round(2).astype(object).fillna('NA')
            .astype(str).agg('|'.join, axis=1))


def build_centre_labels():
    """返回与 common.load_cohorts()['original'] 同序的中心标签数组。"""
    wb = openpyxl.load_workbook(os.path.join(C.DATA_DIR, 'original data.xlsx'),
                                read_only=True, data_only=True)
    rows = list(wb['data20221010'].iter_rows(values_only=True))
    raw = pd.DataFrame(rows[1:],
                       columns=[str(c).strip() for c in rows[0]]).dropna(how='all')
    nk = tidy_keys(raw, 'label')
    nhfe = tidy_keys(read_any(r'E:\CAD-RADS\旧数据\NHFE2017-2022.csv'), 'label')
    nhfy = tidy_keys(read_any(r'E:\CAD-RADS\旧数据\NHFY.csv'), 'label')
    fr = fingerprint(nk)
    se, sf = set(fingerprint(nhfe)), set(fingerprint(nhfy))
    lab = np.where(fr.isin(se), 'E', np.where(fr.isin(sf), 'Y', 'UNK'))
    return lab


def fit_eval(Xtr, ytr, Xte, yte, seed):
    pipe = C.build_estimator('RF', params=dict(FIXED_PARAMS), seed=seed)
    pipe.fit(Xtr, ytr)
    p = pipe.predict_proba(Xte)[:, 1]
    cm = C.calibration_metrics(yte, p)
    return {'AUC': float(roc_auc_score(yte, p)),
            'slope': cm['Calibration slope'],
            'intercept': cm['Calibration intercept'],
            'O:E': cm['O:E'], 'Brier': cm['Brier'], 'ICI': cm['ICI']}, p


def stratified_sample(df, n, rng):
    """按结局分层抽 n 例。"""
    if n >= len(df):
        return df
    pos = df[df[C.TARGET] == 1]
    neg = df[df[C.TARGET] == 0]
    n_pos = int(round(n * len(pos) / len(df)))
    n_neg = n - n_pos
    take_p = rng.choice(len(pos), size=min(n_pos, len(pos)), replace=False)
    take_n = rng.choice(len(neg), size=min(n_neg, len(neg)), replace=False)
    return pd.concat([pos.iloc[take_p], neg.iloc[take_n]])


def main():
    C.log_init('11_multicentre')
    C.section('推导队列的中心结构（新增分析）')

    cohorts, _ = C.load_cohorts(apply_strategy_d=True, verbose=True)
    o, geo = cohorts['original'].copy(), cohorts['geo'].copy()

    lab = build_centre_labels()
    assert len(lab) == len(o), (len(lab), len(o))
    o['centre'] = lab
    C.log(f'  中心归属：' + str(pd.Series(lab).value_counts().to_dict()))

    # ---------------------------------------------------------------- 表 S13
    C.section('表 S13：推导队列内部两来源中心的基线对比')
    rows = []
    for v in C.FEATURES:
        a, b = o.loc[o.centre == 'E', v], o.loc[o.centre == 'Y', v]
        if v in C.BIN:
            rows.append({'Variable': C.LABEL_EN.get(v, v),
                         'Centre 1 (n=%d)' % (o.centre == 'E').sum():
                             f'{int(a.sum())} ({a.mean()*100:.1f})',
                         'Centre 2 (n=%d)' % (o.centre == 'Y').sum():
                             f'{int(b.sum())} ({b.mean()*100:.1f})',
                         'SMD': C.smd_bin(float(a.mean()), float(b.mean()))})
        else:
            rows.append({'Variable': C.LABEL_EN.get(v, v),
                         'Centre 1 (n=%d)' % (o.centre == 'E').sum():
                             f'{a.median():.2f} ({a.quantile(.25):.2f}-{a.quantile(.75):.2f})',
                         'Centre 2 (n=%d)' % (o.centre == 'Y').sum():
                             f'{b.median():.2f} ({b.quantile(.25):.2f}-{b.quantile(.75):.2f})',
                         'SMD': C.smd_cont(a, b)})
    t13 = pd.DataFrame(rows)
    C.save_table(t13, 'TableS13_centre_baseline')
    C.log('  ' + f"|SMD| 最大 = {t13['SMD'].abs().max():.3f}  ("
          f"{t13.loc[t13['SMD'].abs().idxmax(), 'Variable']})")

    # ------------------------------------------------------------ 表 S14 LOCO
    C.section('表 S14：推导集内部的跨中心迁移（leave-one-centre-out）')
    E, Y = o[o.centre == 'E'], o[o.centre == 'Y']
    loco_rows, store = [], {}
    for tag, tr, te in [('Centre 1 -> Centre 2', E, Y),
                        ('Centre 2 -> Centre 1', Y, E),
                        ('Centre 1 -> geo', E, geo),
                        ('Centre 2 -> geo', Y, geo),
                        ('Both centres -> geo', o[o.centre != 'UNK'], geo)]:
        Xtr, ytr = C.derivation_label(tr)
        Xte, yte = C.derivation_label(te)
        m, _ = fit_eval(Xtr, ytr, Xte, yte, C.SEED)
        row = {'Training -> Test': tag, 'n train': len(tr), 'n test': len(te),
               'events (test)': int(yte.sum())}
        row.update({k: round(v, 3) for k, v in m.items()})
        loco_rows.append(row)
        store[tag] = m
        C.log(f"  {tag:24s} n_train={len(tr):3d} n_test={len(te):3d} "
              f"AUC={m['AUC']:.3f} O:E={m['O:E']:.2f} "
              f"slope={m['slope']:.2f} ICI={m['ICI']:.3f}")
    C.save_table(pd.DataFrame(loco_rows), 'TableS14_loco_transport')

    # -------------------------------------------- 表 S15 中心多样性（核心实验）
    C.section('表 S15：同样本量下中心多样性的贡献（核心实验）')
    Xte, yte = C.derivation_label(geo)
    recs = []
    for n in N_GRID:
        rng = np.random.default_rng(C.SEED + n)
        for r in range(R_REPEAT):
            # 1) 单中心 E
            s = stratified_sample(E, n, rng)
            m, _ = fit_eval(*C.derivation_label(s), Xte, yte, C.SEED + r)
            recs.append({'config': 'Centre 1 only', 'n_train': n, 'rep': r, **m})
            # 2) 单中心 Y
            s = stratified_sample(Y, n, rng)
            m, _ = fit_eval(*C.derivation_label(s), Xte, yte, C.SEED + r)
            recs.append({'config': 'Centre 2 only', 'n_train': n, 'rep': r, **m})
            # 3) 双中心混合（各半）
            se = stratified_sample(E, n // 2, rng)
            sy = stratified_sample(Y, n - n // 2, rng)
            m, _ = fit_eval(*C.derivation_label(pd.concat([se, sy])), Xte, yte,
                            C.SEED + r)
            recs.append({'config': 'Both centres (mixed)', 'n_train': n,
                         'rep': r, **m})
    exp = pd.DataFrame(recs)
    C.save_csv(exp, 's11_diversity_experiment_raw')

    agg = (exp.groupby(['n_train', 'config'])[['AUC', 'slope', 'intercept',
                                               'O:E', 'Brier', 'ICI']]
           .agg(['mean', 'std']).round(4))
    agg.columns = ['%s %s' % (a, b) for a, b in agg.columns]
    agg = agg.reset_index().rename(
        columns={'n_train': 'n training', 'config': 'Training configuration'})
    C.save_table(agg, 'TableS15_diversity_matched')
    C.log('')
    for n in N_GRID:
        sub = exp[exp.n_train == n]
        C.log(f'  n_train = {n}')
        for cfg in ['Centre 1 only', 'Centre 2 only', 'Both centres (mixed)']:
            s = sub[sub.config == cfg]
            C.log(f"    {cfg:24s} AUC={s.AUC.mean():.3f}±{s.AUC.std():.3f}  "
                  f"O:E={s['O:E'].mean():.2f}  Brier={s.Brier.mean():.3f}  "
                  f"ICI={s.ICI.mean():.3f}")
        # 配对差（同一次重复内）
        w = sub.pivot(index='rep', columns='config', values='AUC')
        d1 = (w['Both centres (mixed)'] - w['Centre 1 only'])
        d2 = (w['Both centres (mixed)'] - w['Centre 2 only'])
        C.log(f'    ΔAUC(mixed - E only) = {d1.mean():+.4f} '
              f'(SD {d1.std():.4f}, 胜率 {(d1 > 0).mean()*100:.0f}%)')
        C.log(f'    ΔAUC(mixed - Y only) = {d2.mean():+.4f} '
              f'(SD {d2.std():.4f}, 胜率 {(d2 > 0).mean()*100:.0f}%)')

    C.log('')
    C.log('完成。')
    return exp


if __name__ == '__main__':
    main()
