# Ariel Data Challenge 2025 Data Description

## Overview

The competition asks participants to recover exoplanet transmission spectra from noisy simulated Ariel observations. For each `planet_id`, the model must predict a spectrum mean (`mu`) and a positive uncertainty (`sigma`) over 283 wavelength bins. The scoring metric is Gaussian Log-Likelihood, so both spectral accuracy and uncertainty calibration matter.

## Top-Level Files

Expected dataset files include:

```text
adc_info.csv
train.csv
train_star_info.csv
test_star_info.csv
wavelengths.csv
sample_submission.csv
train/
test/
```

`train.csv` contains the supervised targets. It is keyed by `planet_id` and has 283 wavelength columns:

```text
planet_id, wl_1, wl_2, ..., wl_283
```

`train_star_info.csv` and `test_star_info.csv` provide physical metadata for each planetary system:

```text
planet_id, Rs, Ms, Ts, Mp, e, P, sma, i
```

These columns represent stellar radius, stellar mass, stellar temperature, planet mass, eccentricity, orbital period, semi-major axis, and inclination. They can be joined to targets or observations by `planet_id`.

`wavelengths.csv` maps the 283 spectral bins to wavelength positions. Use it for plotting, smoothing, wavelength-aware feature engineering, and submission sanity checks.

`adc_info.csv` provides gain/offset information used to convert raw detector counts into a corrected signal scale during preprocessing.

`sample_submission.csv` defines the required output schema. Conceptually, each row contains:

```text
planet_id, mu_1, ..., mu_283, sigma_1, ..., sigma_283
```

All `sigma` values must be strictly positive.

## Raw Observation Layout

Raw observations are stored under `train/` and `test/`, grouped by `planet_id`. Each planet folder contains sensor signal parquet files and calibration products. A typical structure is:

```text
train/
  <planet_id>/
    AIRS-CH0_signal_0.parquet
    FGS1_signal_0.parquet
    AIRS-CH0_calibration_0/
      dark.parquet
      dead.parquet
      flat.parquet
      linear_corr.parquet
    FGS1_calibration_0/
      dark.parquet
      dead.parquet
      flat.parquet
      linear_corr.parquet
test/
  <planet_id>/
    ...
```

Some planets may have repeated observations, indicated by suffixes such as `_0`, `_1`, etc. Treat repeated observations as belonging to the same `planet_id`; validation splits should be grouped by planet to avoid leakage.

## Instruments

`AIRS-CH0` is the main spectroscopic sensor. It carries wavelength-dependent time-series information and is the primary source for reconstructing the 283-bin transmission spectrum.

`FGS1` is a photometric/guide sensor. It is useful for estimating transit timing, baseline drift, ingress/egress boundaries, and global transit depth. It should usually be processed alongside AIRS rather than ignored.

## Practical Modeling Implications

A baseline preprocessing path should be:

```text
raw signal
-> ADC correction
-> dead/hot pixel masking
-> dark subtraction
-> linearity correction
-> flat-field correction
-> correlated double sampling
-> temporal binning
-> light curve extraction
-> transit window detection
-> depth/features
-> spectrum + sigma model
```

The most important joins are by `planet_id`. Keep target spectra, stellar metadata, extracted AIRS features, extracted FGS1 features, and repeated-observation summaries aligned explicitly.

For validation, split by `planet_id`, not by row or wavelength, because rows from the same planet share physical parameters and observations.
