# Groundwater Level Prediction (XGBoost) — Implementation Plan for Codex

## 1) Project Target (What we are building)

### Goal
Build an end-to-end **machine learning regression project** that predicts **Groundwater Level (GW_LEVEL)** using:
- **Monthly rainfall**
- **Season / month information**
- **Lagged groundwater signals** (GW_LAG_1, GW_LAG_3, GW_LAG_6)

### Problem Type
**Supervised Regression**
- Input (X): RAINFALL, MONTH, SEASON, GW_LAG_1, GW_LAG_3, GW_LAG_6 (and any other provided feature columns)
- Output (y): GW_LEVEL

### Model
Use **XGBoost Regressor** (as required: no reuse of research models)

### Minimum Deliverables (Semester-ready)
- Trained model file saved locally (e.g., `models/xgb_model.joblib`)
- Metrics report (MAE, RMSE)
- Plot: Actual vs Predicted
- Clean README + reproducible run commands

---

## 2) What we have done already (Current status)

### Dataset Preparation Completed (by teammate)
We have a **preprocessed, ML-ready dataset**:

**File:**
- `final_processed_groundwater_dataset.csv`

**Verified columns:**
- `DATE` (monthly timestamp)
- `GW_LEVEL` (target variable)
- `RAINFALL` (feature)
- `MONTH` (1–12)
- `SEASON` (encoded season category)
- `GW_LAG_1` (lag-1 groundwater)
- `GW_LAG_3` (lag-3 groundwater)
- `GW_LAG_6` (lag-6 groundwater)

**Dataset is ready for training**
- No need to redo preprocessing unless asked.
- We will proceed directly to modeling + evaluation + packaging.

---

## 3) What we need to do now (Step-by-step implementation)

### Step 0 — Create project structure
Create folders:
- `data/`
- `src/`
- `models/`
- `outputs/`

Place dataset:
- `data/final_processed_groundwater_dataset.csv`

Suggested structure:
---

### Step 1 — Environment setup
Create `requirements.txt`:
- pandas
- numpy
- scikit-learn
- xgboost
- joblib
- matplotlib

Install:
```bash
pip install -r requirements.txt

Step 2 — Load dataset and validate schema

In code:
	•	Read CSV
	•	Print shape + column list
	•	Ensure target GW_LEVEL exists
	•	Ensure DATE is parseable

Validation checks:
	•	Missing values count
	•	Basic stats (min/max)

Output:
	•	Save a outputs/data_profile.txt

⸻

Step 3 — Feature/Target split

Define:
	•	Target: y = df["GW_LEVEL"]
	•	Features: X = df.drop(["GW_LEVEL", "DATE"], axis=1)

Ensure:
	•	All feature columns are numeric
	•	If SEASON is categorical string, encode it
	•	If it’s already numeric, keep as-is

⸻

Step 4 — Train/Test split (time-aware)

Because data is time-series-like:
	•	Use a time-based split, not random.

Example:
	•	Train: first 80%
	•	Test: last 20%

This prevents leakage.

⸻

Step 5 — Train XGBoost Regressor

Train:
	•	XGBRegressor(...)
	•	Fit on train data

Save model:
	•	models/xgb_model.joblib

Also save feature names for reproducibility.

⸻

Step 6 — Evaluate model

Compute:
	•	MAE
	•	RMSE

Save results:
	•	outputs/metrics.txt

Also generate:
	•	Actual vs Predicted plot
Save plot:
	•	outputs/actual_vs_pred.png

⸻

Step 7 — Inference script (demo prediction)

Create predict.py that:
	•	Loads model
	•	Takes last row (or user input)
	•	Prints predicted groundwater level

Optional:
	•	Accept a CSV row input as CLI argument

⸻

Step 8 — README for submission

Write README.md containing:
	•	Project goal
	•	Dataset description (columns)
	•	How to run:
	•	python src/train_xgb.py
	•	python src/evaluate.py
	•	python src/predict.py
	•	Output files list
	•	Metrics interpretation

