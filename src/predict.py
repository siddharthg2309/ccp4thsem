"""
predict.py - Inference Script for Groundwater Level Prediction

This script:
1. Loads the trained model
2. Takes input features (last row or user input)
3. Predicts groundwater level
4. Displays the result
"""

import numpy as np
import pandas as pd
import joblib
import argparse
import os

# Paths
MODEL_PATH = "models/xgb_model.joblib"
FEATURES_PATH = "models/feature_names.joblib"
DATA_PATH = "data/final_processed_groundwater_dataset.csv"

def load_model():
    """Load trained model and feature names."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train_xgb.py first!")
    
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    
    return model, feature_names

def predict_from_last_row():
    """Predict using the last row of the dataset."""
    print("\n" + "=" * 50)
    print("PREDICTION FROM LAST ROW OF DATASET")
    print("=" * 50)
    
    # Load model
    model, feature_names = load_model()
    print(f"\n✓ Model loaded")
    print(f"✓ Feature names: {feature_names}")
    
    # Load data
    df = pd.read_csv(DATA_PATH)
    last_row = df.iloc[-1]
    
    print(f"\n✓ Using last row from dataset:")
    print(f"  DATE: {last_row['DATE']}")
    print(f"  Actual GW_LEVEL: {last_row['GW_LEVEL']}")
    
    # Prepare features
    X = last_row[feature_names].values.reshape(1, -1)
    
    print(f"\n  Input features:")
    for i, fname in enumerate(feature_names):
        print(f"    {fname}: {X[0][i]}")
    
    # Predict
    prediction = model.predict(X)[0]
    
    print("\n" + "-" * 50)
    print(f"PREDICTED GW_LEVEL: {prediction:.2f}")
    print(f"ACTUAL GW_LEVEL: {last_row['GW_LEVEL']:.2f}")
    print(f"DIFFERENCE: {abs(prediction - last_row['GW_LEVEL']):.2f}")
    print("-" * 50)
    
    return prediction

def predict_from_input(rainfall, month, season, gw_lag_1, gw_lag_3, gw_lag_6):
    """Predict using user-provided input values."""
    print("\n" + "=" * 50)
    print("PREDICTION FROM USER INPUT")
    print("=" * 50)
    
    # Load model
    model, feature_names = load_model()
    print(f"\n✓ Model loaded")
    
    # Create input array matching feature order
    input_dict = {
        'RAINFALL': rainfall,
        'MONTH': month,
        'SEASON': season,
        'GW_LAG_1': gw_lag_1,
        'GW_LAG_3': gw_lag_3,
        'GW_LAG_6': gw_lag_6
    }
    
    print(f"\n  Input features:")
    X = []
    for fname in feature_names:
        if fname in input_dict:
            X.append(input_dict[fname])
            print(f"    {fname}: {input_dict[fname]}")
        else:
            raise ValueError(f"Missing feature: {fname}")
    
    X = np.array(X).reshape(1, -1)
    
    # Predict
    prediction = model.predict(X)[0]
    
    print("\n" + "-" * 50)
    print(f"PREDICTED GW_LEVEL: {prediction:.2f}")
    print("-" * 50)
    
    return prediction

def main():
    """Main prediction interface."""
    parser = argparse.ArgumentParser(description='Predict Groundwater Level')
    parser.add_argument('--rainfall', type=float, help='Monthly rainfall')
    parser.add_argument('--month', type=int, help='Month (1-12)')
    parser.add_argument('--season', type=int, help='Season (1-4)')
    parser.add_argument('--gw_lag_1', type=float, help='Groundwater level lag-1')
    parser.add_argument('--gw_lag_3', type=float, help='Groundwater level lag-3')
    parser.add_argument('--gw_lag_6', type=float, help='Groundwater level lag-6')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("GROUNDWATER LEVEL PREDICTION - INFERENCE")
    print("=" * 60)
    
    # Check if user provided all inputs
    if all([args.rainfall is not None, args.month is not None, 
            args.season is not None, args.gw_lag_1 is not None,
            args.gw_lag_3 is not None, args.gw_lag_6 is not None]):
        # Use user input
        predict_from_input(
            args.rainfall, args.month, args.season,
            args.gw_lag_1, args.gw_lag_3, args.gw_lag_6
        )
    else:
        # Use last row from dataset
        print("\nNo input provided. Using last row from dataset...")
        predict_from_last_row()
    
    print("\n" + "=" * 60)
    print("PREDICTION COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()
