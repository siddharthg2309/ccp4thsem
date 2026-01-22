"""
train_xgb.py - Train XGBoost Regressor for Groundwater Level Prediction

This script:
1. Loads the preprocessed dataset
2. Validates data schema
3. Splits features and target
4. Performs time-based train/test split
5. Trains XGBoost Regressor
6. Saves the model and feature names
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import joblib
import os
from datetime import datetime

# Paths
DATA_PATH = "data/final_processed_groundwater_dataset.csv"
MODEL_PATH = "models/xgb_model.joblib"
FEATURES_PATH = "models/feature_names.joblib"
PROFILE_PATH = "outputs/data_profile.txt"

def load_and_validate_data(data_path):
    """Load dataset and perform validation checks."""
    print("=" * 50)
    print("STEP 1: Loading and Validating Dataset")
    print("=" * 50)
    
    # Load data
    df = pd.read_csv(data_path)
    print(f"\n✓ Dataset loaded successfully")
    print(f"  Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"  Columns: {list(df.columns)}")
    
    # Validate required columns
    required_cols = ['DATE', 'GW_LEVEL', 'RAINFALL', 'MONTH', 'SEASON', 
                     'GW_LAG_1', 'GW_LAG_3', 'GW_LAG_6']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    print(f"✓ All required columns present")
    
    # Parse DATE
    df['DATE'] = pd.to_datetime(df['DATE'])
    print(f"✓ DATE parsed successfully")
    print(f"  Date range: {df['DATE'].min()} to {df['DATE'].max()}")
    
    # Check for missing values
    missing_counts = df.isnull().sum()
    print(f"\n✓ Missing values check:")
    for col in df.columns:
        print(f"    {col}: {missing_counts[col]}")
    
    # Basic stats
    print(f"\n✓ Target variable (GW_LEVEL) statistics:")
    print(f"    Min: {df['GW_LEVEL'].min():.2f}")
    print(f"    Max: {df['GW_LEVEL'].max():.2f}")
    print(f"    Mean: {df['GW_LEVEL'].mean():.2f}")
    print(f"    Std: {df['GW_LEVEL'].std():.2f}")
    
    return df

def save_data_profile(df, profile_path):
    """Save data profile to file."""
    os.makedirs(os.path.dirname(profile_path), exist_ok=True)
    
    with open(profile_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("GROUNDWATER LEVEL PREDICTION - DATA PROFILE\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns\n\n")
        
        f.write("Columns:\n")
        for col in df.columns:
            f.write(f"  - {col}\n")
        
        f.write(f"\nDate Range: {df['DATE'].min()} to {df['DATE'].max()}\n\n")
        
        f.write("Missing Values:\n")
        for col in df.columns:
            f.write(f"  {col}: {df[col].isnull().sum()}\n")
        
        f.write("\nNumeric Statistics:\n")
        f.write(df.describe().to_string())
    
    print(f"\n✓ Data profile saved to: {profile_path}")

def prepare_features_target(df):
    """Split data into features and target."""
    print("\n" + "=" * 50)
    print("STEP 2: Preparing Features and Target")
    print("=" * 50)
    
    # Target
    y = df['GW_LEVEL'].values
    
    # Features (drop target and DATE)
    feature_cols = [col for col in df.columns if col not in ['GW_LEVEL', 'DATE']]
    X = df[feature_cols].values
    
    print(f"\n✓ Feature columns: {feature_cols}")
    print(f"✓ X shape: {X.shape}")
    print(f"✓ y shape: {y.shape}")
    
    # Check all features are numeric
    for col in feature_cols:
        if not np.issubdtype(df[col].dtype, np.number):
            raise ValueError(f"Non-numeric feature found: {col}")
    print(f"✓ All features are numeric")
    
    return X, y, feature_cols

def time_based_split(X, y, train_ratio=0.8):
    """Perform time-based train/test split."""
    print("\n" + "=" * 50)
    print("STEP 3: Time-Based Train/Test Split")
    print("=" * 50)
    
    n_samples = len(y)
    split_idx = int(n_samples * train_ratio)
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\n✓ Total samples: {n_samples}")
    print(f"✓ Training samples: {len(y_train)} ({train_ratio*100:.0f}%)")
    print(f"✓ Test samples: {len(y_test)} ({(1-train_ratio)*100:.0f}%)")
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Train XGBoost Regressor."""
    print("\n" + "=" * 50)
    print("STEP 4: Training XGBoost Regressor")
    print("=" * 50)
    
    # Initialize model with reasonable hyperparameters
    model = XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='reg:squarederror'
    )
    
    print("\n✓ Model configuration:")
    print(f"    n_estimators: 100")
    print(f"    max_depth: 5")
    print(f"    learning_rate: 0.1")
    print(f"    subsample: 0.8")
    print(f"    colsample_bytree: 0.8")
    
    # Train
    print("\n  Training model...")
    model.fit(X_train, y_train)
    print("✓ Model training complete!")
    
    return model

def save_model(model, feature_names, model_path, features_path):
    """Save trained model and feature names."""
    print("\n" + "=" * 50)
    print("STEP 5: Saving Model")
    print("=" * 50)
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save model
    joblib.dump(model, model_path)
    print(f"\n✓ Model saved to: {model_path}")
    
    # Save feature names
    joblib.dump(feature_names, features_path)
    print(f"✓ Feature names saved to: {features_path}")

def main():
    """Main training pipeline."""
    print("\n" + "=" * 60)
    print("GROUNDWATER LEVEL PREDICTION - TRAINING PIPELINE")
    print("=" * 60)
    
    # Step 1: Load and validate data
    df = load_and_validate_data(DATA_PATH)
    
    # Save data profile
    save_data_profile(df, PROFILE_PATH)
    
    # Step 2: Prepare features and target
    X, y, feature_names = prepare_features_target(df)
    
    # Step 3: Time-based split
    X_train, X_test, y_train, y_test = time_based_split(X, y, train_ratio=0.8)
    
    # Save test data for evaluation script
    os.makedirs('outputs', exist_ok=True)
    np.save('outputs/X_test.npy', X_test)
    np.save('outputs/y_test.npy', y_test)
    print(f"\n✓ Test data saved for evaluation")
    
    # Step 4: Train model
    model = train_model(X_train, y_train)
    
    # Step 5: Save model
    save_model(model, feature_names, MODEL_PATH, FEATURES_PATH)
    
    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"  1. Run: python src/evaluate.py")
    print(f"  2. Run: python src/predict.py")

if __name__ == "__main__":
    main()
