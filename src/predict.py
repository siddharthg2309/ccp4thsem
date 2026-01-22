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
import matplotlib.pyplot as plt

# Paths
MODEL_PATH = "models/xgb_model.joblib"
FEATURES_PATH = "models/feature_names.joblib"
DATA_PATH = "data/final_processed_groundwater_dataset.csv"
CUSTOM_PLOT_PATH = "outputs/custom_prediction.png"

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

def create_prediction_plot(input_dict, prediction):
    """Create visualization for custom prediction."""
    os.makedirs(os.path.dirname(CUSTOM_PLOT_PATH), exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Input Features Bar Chart
    ax1 = axes[0]
    features = list(input_dict.keys())
    values = list(input_dict.values())
    colors = ['#3498db', '#2ecc71', '#9b59b6', '#e74c3c', '#f39c12', '#1abc9c']
    
    bars = ax1.bar(features, values, color=colors, edgecolor='black', linewidth=1.2)
    ax1.set_xlabel('Features', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax1.set_title('Input Features for Prediction', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 2: Prediction Result - Simple gauge style
    ax2 = axes[1]
    
    # Create a simple visual with the prediction
    ax2.barh(['GW_LEVEL'], [prediction], color='#27ae60', height=0.5, edgecolor='black', linewidth=2)
    ax2.set_xlim(0, max(20, prediction + 2))
    ax2.set_xlabel('Groundwater Level (meters)', fontsize=12, fontweight='bold')
    ax2.set_title(f'PREDICTED GW_LEVEL: {prediction:.2f} m', fontsize=16, fontweight='bold', color='#27ae60')
    ax2.grid(axis='x', alpha=0.3)
    
    # Add prediction value on bar
    ax2.text(prediction + 0.3, 0, f'{prediction:.2f}', va='center', fontsize=14, fontweight='bold', color='#2c3e50')
    
    # Add reference line for average
    ax2.axvline(x=3.77, color='#e74c3c', linestyle='--', linewidth=2, label=f'Dataset Avg: 3.77')
    ax2.legend(loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(CUSTOM_PLOT_PATH, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n✓ Prediction plot saved to: {CUSTOM_PLOT_PATH}")

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
    
    # Create visualization
    create_prediction_plot(input_dict, prediction)
    
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
