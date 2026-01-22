"""
evaluate.py - Evaluate XGBoost Model for Groundwater Level Prediction

This script:
1. Loads the trained model
2. Loads test data
3. Generates predictions
4. Computes MAE and RMSE
5. Creates Actual vs Predicted plot
6. Saves metrics and plot
"""

import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
from datetime import datetime

# Paths
MODEL_PATH = "models/xgb_model.joblib"
X_TEST_PATH = "outputs/X_test.npy"
Y_TEST_PATH = "outputs/y_test.npy"
METRICS_PATH = "outputs/metrics.txt"
PLOT_PATH = "outputs/actual_vs_pred.png"

def load_model_and_data():
    """Load trained model and test data."""
    print("=" * 50)
    print("STEP 1: Loading Model and Test Data")
    print("=" * 50)
    
    # Load model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train_xgb.py first!")
    
    model = joblib.load(MODEL_PATH)
    print(f"\n✓ Model loaded from: {MODEL_PATH}")
    
    # Load test data
    X_test = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)
    print(f"✓ Test data loaded")
    print(f"  X_test shape: {X_test.shape}")
    print(f"  y_test shape: {y_test.shape}")
    
    return model, X_test, y_test

def make_predictions(model, X_test):
    """Generate predictions on test data."""
    print("\n" + "=" * 50)
    print("STEP 2: Generating Predictions")
    print("=" * 50)
    
    y_pred = model.predict(X_test)
    print(f"\n✓ Predictions generated")
    print(f"  Prediction range: {y_pred.min():.2f} to {y_pred.max():.2f}")
    
    return y_pred

def compute_metrics(y_test, y_pred):
    """Compute MAE and RMSE."""
    print("\n" + "=" * 50)
    print("STEP 3: Computing Evaluation Metrics")
    print("=" * 50)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"\n✓ Metrics computed:")
    print(f"  MAE (Mean Absolute Error): {mae:.4f}")
    print(f"  RMSE (Root Mean Squared Error): {rmse:.4f}")
    
    return mae, rmse

def save_metrics(mae, rmse, y_test, y_pred, metrics_path):
    """Save metrics to file."""
    print("\n" + "=" * 50)
    print("STEP 4: Saving Metrics")
    print("=" * 50)
    
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    
    with open(metrics_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("GROUNDWATER LEVEL PREDICTION - MODEL EVALUATION METRICS\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("EVALUATION METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Mean Absolute Error (MAE): {mae:.4f}\n")
        f.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}\n\n")
        
        f.write("INTERPRETATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"- On average, predictions deviate by {mae:.2f} units from actual values\n")
        f.write(f"- RMSE penalizes larger errors more heavily\n\n")
        
        f.write("PREDICTION STATISTICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Test samples: {len(y_test)}\n")
        f.write(f"Actual GW_LEVEL - Min: {y_test.min():.2f}, Max: {y_test.max():.2f}, Mean: {y_test.mean():.2f}\n")
        f.write(f"Predicted GW_LEVEL - Min: {y_pred.min():.2f}, Max: {y_pred.max():.2f}, Mean: {y_pred.mean():.2f}\n")
    
    print(f"\n✓ Metrics saved to: {metrics_path}")

def create_plot(y_test, y_pred, plot_path):
    """Create Actual vs Predicted plot."""
    print("\n" + "=" * 50)
    print("STEP 5: Creating Actual vs Predicted Plot")
    print("=" * 50)
    
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Scatter plot - Actual vs Predicted
    ax1 = axes[0]
    ax1.scatter(y_test, y_pred, alpha=0.7, edgecolors='black', linewidth=0.5, c='steelblue', s=80)
    
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax1.set_xlabel('Actual GW_LEVEL', fontsize=12)
    ax1.set_ylabel('Predicted GW_LEVEL', fontsize=12)
    ax1.set_title('Actual vs Predicted Groundwater Level', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Time series comparison
    ax2 = axes[1]
    x_axis = range(len(y_test))
    ax2.plot(x_axis, y_test, 'b-o', label='Actual', markersize=6, linewidth=1.5)
    ax2.plot(x_axis, y_pred, 'r-s', label='Predicted', markersize=6, linewidth=1.5)
    
    ax2.set_xlabel('Test Sample Index', fontsize=12)
    ax2.set_ylabel('GW_LEVEL', fontsize=12)
    ax2.set_title('Actual vs Predicted (Test Set Timeline)', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Plot saved to: {plot_path}")

def main():
    """Main evaluation pipeline."""
    print("\n" + "=" * 60)
    print("GROUNDWATER LEVEL PREDICTION - EVALUATION PIPELINE")
    print("=" * 60)
    
    # Step 1: Load model and data
    model, X_test, y_test = load_model_and_data()
    
    # Step 2: Generate predictions
    y_pred = make_predictions(model, X_test)
    
    # Step 3: Compute metrics
    mae, rmse = compute_metrics(y_test, y_pred)
    
    # Step 4: Save metrics
    save_metrics(mae, rmse, y_test, y_pred, METRICS_PATH)
    
    # Step 5: Create plot
    create_plot(y_test, y_pred, PLOT_PATH)
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE!")
    print("=" * 60)
    print(f"\nOutputs:")
    print(f"  - Metrics: {METRICS_PATH}")
    print(f"  - Plot: {PLOT_PATH}")
    print(f"\nNext step:")
    print(f"  Run: python src/predict.py")

if __name__ == "__main__":
    main()
