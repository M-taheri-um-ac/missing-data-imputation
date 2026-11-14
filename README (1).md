# Missing Data Imputation on the Bike Sharing Dataset

A comparison of missing-data imputation methods on the daily UCI Bike Sharing dataset. Artificial missingness is injected under three mechanisms (MCAR, MAR, MNAR), filled with several methods, and scored against the known true values.

## Overview

Real datasets rarely come with ground truth for their missing values, so imputation methods are hard to evaluate. This project avoids that problem by starting from a (nearly) complete dataset, deleting values on purpose, imputing them, and measuring the error with MAE and RMSE on exactly the cells that were deleted.

**Question:** which imputation method recovers the weather variables best, and does the answer change with the missingness mechanism?

## Dataset

- **Source:** UCI *Bike Sharing Dataset*, file `day.csv` (daily data, 2011-2012).
- **Size:** 731 rows, 16 columns, no duplicate or missing dates.
- **Variables used for imputation:** `temp`, `atemp`, `hum`, `windspeed` (all normalized to [0, 1]).
- **Target of the original dataset:** `cnt` (rentals) = `casual` + `registered`. It is not used for imputation.

Credit: Fanaee-T, H. and Gama, J. (2013), *Event labeling combining ensemble detectors and background knowledge*, Progress in Artificial Intelligence. Please check the dataset's license on the UCI page before redistributing `day.csv`; this repo does not include it (see *How to run*).

## Workflow

1. **EDA**: structure, data quality, distributions, outliers (IQR), correlations, time trend.
2. **Cleaning**: one invalid value found, `hum = 0` on 2011-03-10 (physically implausible), converted to `NaN`.
3. **Missingness simulation** on `temp`, `hum`, `windspeed` (see below).
4. **Imputation** with several methods.
5. **Evaluation** with MAE and RMSE on the deleted cells only, per mechanism and per column.

### EDA highlights

- `temp` and `atemp` are almost the same variable (correlation 0.992) and are bimodal (seasonal).
- `hum` is roughly symmetric; its correlation with the other variables is weak (strongest: `windspeed`, -0.25).
- `windspeed` is right-skewed (skewness 0.68) with 13 IQR outliers, which look like real windy days and were kept.
- `cnt` grows from 2011 to 2012 and is strongly seasonal; the lowest value (22, on 2012-10-29) coincides with severe weather.

## Missingness mechanisms

`make_missing()` masks values with probability depending on the mechanism:

| Mechanism | Rule |
|---|---|
| **MCAR** | every cell is deleted with probability 0.20 |
| **MAR** | probability depends on another, observed variable: 0.30 if `atemp` is below its median, 0.05 otherwise |
| **MNAR** | probability depends on the value itself: 0.30 if the value is above its column median, 0.05 otherwise |

Resulting share of complete rows (all three columns observed): MCAR 49%, MAR 60%, MNAR 53%. Realized missing rates per column are about 17-20%.

Seeds are fixed (`seed=k` per mechanism) so the masks are reproducible.

## Methods compared

| Method | Notes |
|---|---|
| **IterativeImputer** (scikit-learn) | MICE-style, `BayesianRidge` estimator, `max_iter=10`, mean initialization |
| **MissForest** | random-forest imputation (`missforest` package), 100 trees |
| **Linear interpolation** | uses neighboring days in time order |

During exploration on `temp` (MCAR) I also tried mean/median imputation, a linear regression on `atemp`, and KNN with several values of *k* (best *k* = 5). Mean imputation was far worse (RMSE 0.194 vs 0.013 for KNN), which is why it was not carried into the final comparison.

## Results

RMSE on the deleted cells (lower is better). Bold = best per row. For scale, the standard deviations of the original columns are `temp` 0.183, `hum` 0.142, `windspeed` 0.077.

| Mechanism | Column | IterativeImputer | MissForest | Linear interpolation |
|---|---|---|---|---|
| MCAR | temp | **0.0162** | 0.0600 | 0.0519 |
| MCAR | hum | 0.1356 | 0.1376 | **0.1139** |
| MCAR | windspeed | 0.0967 | 0.0778 | **0.0768** |
| MAR | temp | **0.0128** | 0.0626 | 0.0619 |
| MAR | hum | 0.1342 | 0.1426 | **0.1304** |
| MAR | windspeed | **0.0733** | 0.0761 | 0.0921 |
| MNAR | temp | **0.0181** | 0.0676 | 0.0395 |
| MNAR | hum | 0.1589 | 0.1548 | **0.1274** |
| MNAR | windspeed | **0.0807** | 0.0814 | 0.0841 |

MAE results are in the notebook (27 experiments in total).

### Key findings

- **`temp` is easy for IterativeImputer** (RMSE 0.013-0.018) because the fully observed `atemp` predicts it almost perfectly (r = 0.99).
- **`hum` is hard for every method.** RMSE is 0.11-0.16 against a standard deviation of 0.142, so no method is much better than filling with the mean. `hum` has weak correlation with all other columns, so there is little to learn from. Linear interpolation wins on `hum` in all three mechanisms, likely because humidity is autocorrelated from day to day.
- **`windspeed` has no clear winner.** In MCAR, IterativeImputer (0.0967) is worse than the column's own standard deviation (0.077), i.e. worse than mean imputation; differences between MissForest and interpolation (0.0778 vs 0.0768) are too small to call from one run.
- **MissForest was never the best method** on this dataset.
- **MNAR is the hardest case**: errors for `hum` are the highest there, as expected when missingness depends on the value itself.

## Limitations

- **One run per mechanism** (one mask, one seed). Small differences between methods are probably within noise; repeating with many seeds and reporting mean ± std would be more reliable.
- Only **three** methods are in the final comparison; mean, median, regression and KNN were explored on a single column only.
- Linear interpolation exploits the **time order** of the rows, while IterativeImputer and MissForest use only the other columns, so it is not a like-for-like comparison.
- Only four numeric columns are used; there are no categorical columns with missing values, so categorical imputation is not evaluated.
- Missing rates differ slightly between mechanisms (about 17.5% for MAR vs 20% for MCAR), so cross-mechanism comparisons are approximate.

## Deep-learning methods (not implemented)

HyperImpute provides four deep-learning imputers: **Sinkhorn** (optimal transport), **GAIN** (generative adversarial imputation), **MIWAE** (importance-weighted autoencoder) and **MIRACLE** (causally-aware imputation). They exist and could be run through the `hyperimpute` library, but they are **not included in this project**: the dataset has only 731 rows and 4 numeric columns, which is too small and low-dimensional to train these neural models reliably, so they were not evaluated here. They would be more appropriate for larger, higher-dimensional tables.

## How to run

```bash
git clone <this-repo-url>
cd <repo-folder>
pip install -r requirements.txt
```

1. Download `day.csv` from the UCI Bike Sharing Dataset and place it in `data/day.csv`.
2. Update the path in the first notebook cell (`pd.read_csv("data/day.csv", parse_dates=["dteday"])`).
3. Open the notebook and use *Restart & Run All*.

`requirements.txt`:

```
numpy
pandas
matplotlib
seaborn
scikit-learn
missforest
jupyter
```

```

## Tools

Python, pandas, NumPy, scikit-learn, missforest, matplotlib, seaborn.
