# Groundwater Level Prediction using XGBoost

A machine learning regression project to predict **Groundwater Level (GW_LEVEL)** using monthly rainfall, seasonal information, and lagged groundwater signals.

---

## 📋 Project Overview

### Problem Type
**Supervised Regression**

### Model
**XGBoost Regressor**

### Input Features
| Feature | Description |
|---------|-------------|
| `RAINFALL` | Monthly rainfall data |
| `MONTH` | Month of the year (1-12) |
| `SEASON` | Encoded season category (1-4) |
| `GW_LAG_1` | Groundwater level lag-1 (previous observation) |
| `GW_LAG_3` | Groundwater level lag-3 |
| `GW_LAG_6` | Groundwater level lag-6 |

### Target Variable
- `GW_LEVEL` - Groundwater Level

---

## 📁 Project Structure

```
ccp4thsem/
├── data/
│   └── final_processed_groundwater_dataset.csv
├── src/
│   ├── train_xgb.py      # Training pipeline
│   ├── evaluate.py       # Model evaluation
│   └── predict.py        # Inference script
├── models/
│   ├── xgb_model.joblib  # Trained model (after running train)
│   └── feature_names.joblib
├── outputs/
│   ├── data_profile.txt  # Dataset profile
│   ├── metrics.txt       # MAE, RMSE metrics
│   ├── actual_vs_pred.png # Visualization
│   ├── X_test.npy        # Test features
│   └── y_test.npy        # Test targets
├── requirements.txt
├── goal.md
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python src/train_xgb.py
```

This will:
- Load and validate the dataset
- Create a data profile (`outputs/data_profile.txt`)
- Perform time-based train/test split (80/20)
- Train XGBoost Regressor
- Save the model to `models/xgb_model.joblib`

### 3. Evaluate the Model

```bash
python src/evaluate.py
```

This will:
- Load the trained model
- Compute MAE and RMSE metrics
- Generate Actual vs Predicted plot
- Save results to `outputs/`

### 4. Make Predictions

**Using last row of dataset:**
```bash
python src/predict.py
```

**Using custom input:**
```bash
python src/predict.py --rainfall 5.0 --month 8 --season 3 --gw_lag_1 2.5 --gw_lag_3 3.0 --gw_lag_6 2.8
```

---

## 📊 Output Files

| File | Description |
|------|-------------|
| `outputs/data_profile.txt` | Dataset statistics and validation |
| `outputs/metrics.txt` | MAE and RMSE evaluation metrics |
| `outputs/actual_vs_pred.png` | Visualization comparing actual vs predicted values |
| `models/xgb_model.joblib` | Trained XGBoost model |

---

## 📈 Metrics Interpretation

- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual values. Lower is better.
- **RMSE (Root Mean Squared Error)**: Square root of average squared differences. Penalizes larger errors more heavily.

---

## 📝 Dataset Information

- **Source**: Preprocessed groundwater monitoring data
- **Time Range**: 2002-2023
- **Samples**: 79 observations
- **Frequency**: Quarterly (Jan, May, Aug, Nov measurements)

---

## 🛠️ Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- xgboost
- joblib
- matplotlib

---

## 👥 Team

Semester Project - CCP 4th Semester
