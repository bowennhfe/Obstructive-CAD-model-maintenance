# Variable dictionary

Nineteen candidate variables and one outcome, identical in all three cohorts. The
order below is the order used by `common.FEATURES` and is the order in which the
variables enter every model in the study.

## Outcome

| Name | Definition |
|---|---|
| `label` | Obstructive coronary artery disease on CCTA: **1** = CAD-RADS 3-5 (≥50% luminal stenosis), **0** = CAD-RADS 0-2. Taken from the CAD-RADS category recorded in the clinical report of the examination; not centrally adjudicated. |

## Continuous variables (14)

| Code | Variable | Unit | Range accepted before analysis |
|---|---|---|---|
| `age` | Age | years | not truncated |
| `sbp` | Systolic blood pressure | mmHg | 60-300 |
| `dbp` | Diastolic blood pressure | mmHg | 30-200 |
| `map` | Mean arterial pressure | mmHg | 40-220 |
| `ppi` | Pulse pressure index (pulse pressure / systolic) | ratio | 0.05-0.95 |
| `pp` | Pulse pressure | mmHg | 5-180 |
| `tc` | Total cholesterol | mmol/L | 1.0-25.0 |
| `tg` | Triglycerides | mmol/L | 0.2-30.0 |
| `ldl` | LDL cholesterol | mmol/L | 0.2-20.0 |
| `hdl` | HDL cholesterol | mmol/L | 0.1-6.0 |
| `ua` | Uric acid | µmol/L | 80-1200 |
| `fib` | Plasma fibrinogen | g/L | 0.3-15.0 |
| `cr` | Creatinine | µmol/L | 20-1500 |
| `urea` | Urea | mmol/L | 0.5-50.0 |

Values outside the accepted range are treated as data-entry errors and set to
missing before analysis (14 cells in total across the three cohorts; Supplementary
Table S1). Age is not truncated. Missing values are imputed with the median, inside
the modelling pipeline, so that the imputation is fitted on training folds only and
is carried frozen into the external cohorts. Missingness is at most 2.26% for any
variable in any cohort (Supplementary Table S33).

## Binary variables (5)

| Code | Variable | Coding |
|---|---|---|
| `gender` | Sex | **0 = male, 1 = female**. Recoded to this convention in all three files; the derivation file uses 1 = male, 2 = female as originally recorded. |
| `htn` | Hypertension | 0 = no, 1 = yes (history or diagnosis at admission) |
| `dm` | Diabetes mellitus | 0 = no, 1 = yes |
| `smoke` | Smoking | 0 = never or former, 1 = current |
| `dyslip` | Dyslipidaemia | 0 = no, 1 = yes |

## Derived quantity

`map` and `pp` may be supplied directly or derived from `sbp` and `dbp`:

* `map = dbp + (sbp - dbp) / 3`
* `pp  = sbp - dbp`
* `ppi = pp / sbp`

The published analysis used the values recorded in the source files; the formulas
are given here so that the variables can be reconstructed.

## Hospital labels

Neither cohort file carries an explicit hospital column. Hospital membership is
reconstructed in `33_external_by_centre.py` (external cohort) and `11_multicentre.py`
(development cohort) by exact matching on the 19 variables plus the outcome, and, for
the external cohort, by the record-identifier series of the two hospitals. The
derivation cohort resolves to 255 and 182 patients with 5 unassignable; the external
cohort to 56 and 121 patients. Repeating the match without the outcome term leaves
every assignment unchanged (Supplementary Table S36).

## What is not released

The analysis data set itself, and any table that would allow a patient to be
re-identified. See the *Data availability* section of `README.md`.
