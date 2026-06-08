# Forecasting Serious Adverse Drug Events with Deep LSTM
Quarterly time-series forecasting of FDA-reported serious adverse drug events (2015–2025) using a stacked LSTM. Trained on 239 drugs with per-drug normalization, evaluated against naive and moving-average baselines, and
extended to multi-output forecasting across reaction categories (cardiac, infection, neurological, gastrointestinal, other).

## Overview

This project predicts the number of **serious adverse events** reported quarterly to the FDA Adverse Event Reporting System (FAERS) per drug. A deep LSTM model is trained on 239 drugs that have at least 40 quarters of
historical data, using sliding windows of 8 quarters to predict the next quarter.

A second, extended model performs **multi-output forecasting** across five reaction categories (cardiac, infection, neurological, gastrointestinal, other), capturing the structure of the adverse-event signal beyond a single
aggregate count.

## Dataset

- **Source:** FDA FAERS (cleaned)
- **Period:** 2015 Q1 – 2025 Q4 (44 quarters)
- **Initial size:** 528,000 records × 30 columns
- **After cleaning:** 528,000 × 28 columns (dropped `patient_weight_kg` and `serious_flags` due to >57% missing)
- **Final modeling set:** 239 drugs × 44 quarters

## Methodology

**Preprocessing**
- Aggregate serious events per `(drug, quarter)`
- Filter to drugs with ≥ 40 quarters of history (239 drugs)
- Per-drug `MinMaxScaler` — each drug normalized independently
- Sliding windows: 8 quarters → predict next quarter
- Per-drug 80/20 train/test split (preserves temporal order)

**Model architecture (Deep LSTM)**
```
LSTM(128, return_sequences=True) → Dropout(0.2)
LSTM(64,  return_sequences=True) → Dropout(0.2)
LSTM(32,  return_sequences=False) → Dropout(0.2)
Dense(16, relu) → Dense(1)
```
Total params: 128,929 · Optimizer: Adam · Loss: MSE · EarlyStopping (patience=10)

**Training set:** (6,692, 8, 1) · **Test set:** (1,912, 8, 1)

## Results

### Base model — single-target forecast

| Model | MSE | MAE | Improvement vs Naive |
|---|---|---|---|
| Naive Forecast | 6.52% | 16.63% | — |
| Moving Average (4Q) | 4.98% | 15.16% | 23.6% MSE |
| **Deep LSTM** | **4.59%** | **14.61%** | **29.6% MSE / 12.1% MAE** |
