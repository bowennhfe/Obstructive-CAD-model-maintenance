# Calibration and discrimination drift of a CAD model on CT angiography — analysis code

This repository contains the analysis code behind the manuscript

> **Calibration and discrimination drift of a diagnostic model for obstructive
> coronary artery disease on CT angiography: an exploratory four-hospital study of
> external validation and updating**

It reproduces, from three cohort files, every number, figure and table in the
manuscript and in the supplementary material. No patient-level data are included
(see *Data required* and *Data availability* below); the aggregated tables from
which the manuscript's numerical statements are taken are in `Derived_tables/`.
`Derived_tables/TABLE_INDEX.csv` maps every table file to its display number in the
manuscript's supplementary material (the supplement reproduces the 39 numbered tables that
the text cites, plus the companion Table S13b; the remaining process/provenance
tables stay in this repository only).

---

## 1. What the code does

The study has four stages, and the scripts are numbered in the order in which the
published results were produced.

| Stage | Scripts | What is computed |
|---|---|---|
| **1. Derivation** | `01_derivation_model.py`, `08_roc_shap_original.py`, `27_rebuild_consistency.py` | The random forest rebuilt to the published specification; repeated stratified 10-fold cross-validation (5 repeats, 50 folds); cross-validated discrimination, calibration, variable importance and SHAP values; an item-by-item check that the rebuilt pipeline reproduces the published performance |
| **2. Frozen external validation** | `02_geographical_validation.py`, `33_external_by_centre.py`, `14_algo_comparison.py` | The frozen pipeline applied to the two external hospitals, pooled and separately; AUC, calibration slope, intercept in the large, O:E ratio, ICI, Brier score, net benefit, with 2000-resample bootstrap intervals; the eight algorithm families |
| **3. Recalibration and updating** | `35_recal_frozen.py`, `18_recal_spread.py`, `03_temporal_growth.py`, `16b_control_arm.py`, `15_algo_updatability.py`, `06_growth_recovery.py` | Cross-fitted logistic recalibration of the **frozen** model inside the external cohort, intercept-only and slope-plus-intercept, over 100 repeated stratified 5-fold partitions, with per-fold dispersion and the deployable intercept; the temporal accumulation ladder at 0/10/20/40/60/80/100% with a row-count-matched resampling control arm and a patient-level difference-in-differences; updatability of the eight families |
| **4. Structure, precision and reporting** | `05_leakage_audit.py`, `11_multicentre.py`, `12_centre_structure.py`, `13_univariable_transport.py`, `21_drift_decomposition.py`, `22_centre_assignment_sensitivity.py`, `23_paired_bootstrap.py`, `24_nested_centre_cv.py`, `17_sample_size.py`, `25_smd_robust.py`, `26_net_benefit_provenance.py`, `29_complete_case.py`, `30_subgroup_interaction.py`, `07_second_target.py`, `19_minor_revision.py`, `28_updated_importance.py`, `16_revision_analyses.py`, `31_tripod_flowdiagram.py`, `34_figure1_design.py` | Data-leakage self-audit; centre structure and leave-one-hospital-out transport; drift decomposition; paired bootstrap contrasts with Holm adjustment; nested and centre-aware cross-validation; sample size against the Riley criteria; baseline SMD sensitivity; net-benefit provenance; complete-case and subgroup analyses; the flow diagram and Figure 1 |

Two R scripts produce the clinical figures and the `pmsampsize` calculation:
`10_clinical_plots_R.R` and `32_sample_size_pmsampsize.R`.

`common.py` is the shared module: paths, the 19-variable definition, the estimation
grids, the pipeline constructor, the calibration and decision-curve functions, the
bootstrap helper, and the figure and table writers.

In-code comments and docstrings are in Chinese, the working language of the group.

---

## 2. Data required

The three cohort files are **not** distributed with this code. To run it, create a
`data/` directory in the repository root (a sibling of `Code/`) and place the files
there under exactly these names (they are also the names used in the manuscript):

| File | Cohort | Rows |
|---|---|---|
| `original data.xlsx` (sheet `data20221010`) | Derivation cohort, two hospitals, 2017-2022 | 442 |
| `Temporal external validation cohort.csv` | Contemporary cohort, development hospital 1, 2023-2026 | 364 |
| `Geographical external validation cohort.csv` | External cohort, two further hospitals, 2024-2026 | 177 |

Each file needs one row per patient with the 19 candidate variables and the outcome
column. The variable names, units, coding and the outcome definition are given in
[`VARIABLES.md`](VARIABLES.md); the preprocessing that the code applies to them
(out-of-range cells set to missing, median imputation fitted on training folds only)
is described in `Code/common.py`.

`common.py` resolves the data directory as `<repository root>/data`. If your layout
differs, change the single line

```python
DATA_DIR = os.path.join(ROOT, "data")
```

The copy of `common.py` shipped here was edited for this release to read `data/`;
the working copy in the authors' project read a differently named directory.

### The frozen model

The release does include the frozen primary model, under `Models/`:

| File | Contents |
|---|---|
| `m01_rf_original.pkl` | A dictionary holding the fitted pipeline (median imputer + random forest, 500 trees, `max_depth` 10, `random_state` 123), the 19 feature names in the order the pipeline expects, the derivation Youden threshold (0.5312), the selected hyperparameters and the size of the cohort it was fitted on |
| `m01_spec.json` | The same specification in plain text |

It contains no patient records — a fitted forest is a set of split rules plus leaf
proportions — which is why it can be released while the cohort files cannot.
`08_roc_shap_original.py`, `23_paired_bootstrap.py` and `33_external_by_centre.py`
load it directly; `02_geographical_validation.py` re-fits from `m01_spec.json` instead,
which reproduces the same forest exactly from the fixed seed. To score a new patient:

```python
import pickle, pandas as pd
m = pickle.load(open("Models/m01_rf_original.pkl", "rb"))
x = pd.DataFrame([{...}])[m["features"]]      # the 19 columns, see VARIABLES.md
p = m["pipeline"].predict_proba(x)[0, 1]      # probability of CAD-RADS 3-5
positive = p >= m["threshold"]
```

---

## 3. Running it

```
pip install -r requirements.txt

python Code/01_derivation_model.py          # ~15 min
python Code/02_geographical_validation.py
python Code/03_temporal_growth.py           # the ladder; the slowest step (~60 min)
python Code/33_external_by_centre.py        # per-hospital results, Tables 2 and 6
python Code/34_figure1_design.py            # Figure 1
...
```

Scripts write to `Figures/{pdf,tif,jpg}` (600 dpi), `Tables/` (CSV, UTF-8 with BOM),
`Outputs/` (JSON) and `Logs/`. Directories are created on import of `common.py`.

Every script is independent given the cohort files: none reads another script's
intermediate output except `06_growth_recovery.py`, `19_minor_revision.py` and
`28_updated_importance.py`, which post-process the JSON written by stage 3 and say
so in their docstrings.

---

## 4. Randomness and reproducibility

| Quantity | Value |
|---|---|
| Global seed (`common.SEED`) | 123 |
| Cross-validation | repeated stratified 10-fold, 5 repeats, 50 folds; folds grouped on the patient identifier wherever resampled rows are involved |
| Temporal accumulation ladder | 7 increments (0, 10, 20, 40, 60, 80, 100%), 10 repeated stratified draws, seed `SEED + repeat` |
| Bootstrap intervals | 2000 resamples, patient level |
| Recalibration | 5-fold cross-fitting over 100 repeated stratified partitions, so no patient is used to fit and to evaluate the correction and no estimate rests on one split |
| Hyperparameters | chosen by grid search inside the cross-validation; the grids are in `common.PARAM_GRIDS` and are reproduced in Supplementary Table S2 |

Two consecutive runs on the same machine reproduce the reported values exactly.
The Python version and package versions that produced the manuscript are pinned in
`requirements.txt`.

---

## 5. Data availability

The cohorts are retrospective clinical records. The approvals and consent waivers
granted by the review boards of the four participating hospitals cover the analysis
and reporting of those records but do not extend to an open release of patient-level
data. The de-identified, analysis-ready dataset is therefore not included here; it is
available from the corresponding author on reasonable request, subject to confirmation
by the requesting institution's review board and a signed data use agreement
restricting use to academic reproduction of the published analyses.

**Raw data requests:** contact the corresponding author, Yueli Dai
(Neihanzidyl@163.com), Department of Radiology, Second Affiliated Hospital of
University of South China.

What *is* released, in addition to this code: the variable dictionary
(`VARIABLES.md`), the final model specifications, the hyperparameter grids and seeds,
and the summary-level inputs from which every figure and table was generated.

---

## 6. Licence and citation

Released under the MIT licence (`LICENSE`). If you use this code, please cite the
manuscript. A versioned archival DOI has not been assigned for this release; if one
is added later it will be recorded in `CITATION.cff`.
