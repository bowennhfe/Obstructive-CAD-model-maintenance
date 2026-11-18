# Archiving the release and obtaining a DOI

The manuscript states that the analysis code is deposited in a public repository and
archived with a DOI at publication. This file is the checklist for producing that DOI.
Nothing here has been done for you — the repository and the archive have to be created
under the corresponding author's account.

## 1. Choose one repository

| Option | When to use it | Notes |
|---|---|---|
| **GitHub + Zenodo** (recommended) | The code is public | Zenodo archives each GitHub release and mints a DOI automatically; the DOI is versioned, and a "concept DOI" covers all versions |
| **Zenodo only** | You do not want a GitHub account | Upload a zip of this directory; Zenodo mints a DOI directly |
| **OSF** | You also want to register the analysis plan | Supports private projects with an embargo, then public release |

Whichever you choose, the licence is MIT (see `LICENSE`) and the release must not
contain patient-level data. The frozen model object and the aggregated tables are
part of the release (see the checklist in section 4).

## 2. GitHub + Zenodo, step by step

1. Create a repository, e.g. `obstructive-cad-maintenance`. Public, no data files.
2. From this directory:

   ```
   git init
   git add .
   git commit -m "Analysis code for the four-centre external validation and updating study"
   git branch -M main
   git remote add origin https://github.com/<account>/<repo>.git
   git push -u origin main
   ```

3. Log in to <https://zenodo.org> with GitHub, open **Settings → GitHub**, and switch
   the repository on.
4. On GitHub, draft a release: tag `v1.0.0`, title "Manuscript submission version".
   Publishing the release triggers Zenodo, which mints a DOI within a minute.
5. Copy the DOI badge. Two DOIs appear: the **version DOI** (`10.5281/zenodo.xxxxxxx`,
   specific to v1.0.0) and the **concept DOI** (`10.5281/zenodo.yyyyyyy`, always
   resolving to the latest version). Cite the version DOI in the manuscript.

## 3. Fill in the places that are waiting for it

| Where | What to put |
|---|---|
| Manuscript, Declarations → *Code availability* | DONE — repository URL filled in on 2026-11-18: `https://github.com/bowennhfe/Obstructive-CAD-model-maintenance`. Remaining: the Zenodo DOI. |
| `CITATION.cff`, field `repository-code` | DONE — set to the URL above. Remaining: `doi` (currently `10.5281/zenodo.xxxxxxx`). |
| Cover letter | the cover letter no longer carries an open-items section; check that the repository sentence matches the URL above once filled |
| TRIPOD+AI checklist, item 18f | the Zenodo DOI |

After the DOI exists, replace every occurrence of
`[Repository URL and DOI to be inserted by the authors.]` in the manuscript with the
sentence:

> The complete analysis pipeline is deposited at https://github.com/<account>/<repo>
> and archived at https://doi.org/10.5281/zenodo.xxxxxxx.

## 4. Before you upload

- [ ] No `.xlsx` or `.csv` cohort files in the directory (`git status` should show
      only code, `README.md`, `VARIABLES.md`, `requirements.txt`, `LICENSE`,
      `CITATION.cff`, `ZENODO.md`)
- [ ] No absolute paths containing a user name or an internal drive letter
- [ ] No identifiers and no patient-level rows. Two things are deliberately included:/n      `Models/m01_rf_original.pkl` plus `m01_spec.json`, which hold fitted tree
      parameters and the selected settings but no records, and `Derived_tables/`,
      which holds the aggregated tables behind the manuscript. Neither carries a
      patient identifier.
- [ ] `requirements.txt` matches the environment that produced the results
- [x] `CITATION.cff` carries the personal author list in the published order
      (Bowen Li, first author; Yueli Dai, corresponding author — filled in
      2026-11-18)
- [ ] A dry run of at least `01_derivation_model.py` and `33_external_by_centre.py`
      on a machine that is not the one used for the analysis, if time allows
