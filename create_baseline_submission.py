#!/usr/bin/env python3
"""
Create a simple baseline submission for Kaggle
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

def main():
    print("Creating baseline submission...")
    
    # Load data
    train_df = pd.read_parquet('train.parquet')
    test_df = pd.read_parquet('test.parquet')
    
    print(f"Training data: {train_df.shape}")
    print(f"Test data: {test_df.shape}")
    
    # Simple feature engineering
    def add_simple_features(df):
        df_new = df.copy()
        df_new['bid_ask_spread'] = df_new['ask_qty'] - df_new['bid_qty']
        df_new['buy_sell_ratio'] = df_new['buy_qty'] / (df_new['sell_qty'] + 1e-8)
        df_new['volume_intensity'] = df_new['volume'] / (df_new['buy_qty'] + df_new['sell_qty'] + 1e-8)
        return df_new
    
    train_enhanced = add_simple_features(train_df)
    test_enhanced = add_simple_features(test_df)
    
    # Prepare features
    target_col = 'label'
    feature_cols = [col for col in train_enhanced.columns if col != target_col]
    
    X = train_enhanced[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y = train_enhanced[target_col]
    
    # Simple train/validation split (last 20% for validation)
    split_idx = int(0.8 * len(X))
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Training: {len(X_train)}, Validation: {len(X_val)}")
    
    # Feature selection (top 100 features)
    selector = SelectKBest(score_func=f_regression, k=100)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_val_selected = selector.transform(X_val)
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_val_scaled = scaler.transform(X_val_selected)
    
    # Train Ridge regression
    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)
    
    # Validation performance
    val_pred = model.predict(X_val_scaled)
    val_rmse = np.sqrt(np.mean((y_val - val_pred) ** 2))
    print(f"Validation RMSE: {val_rmse:.6f}")
    
    # Prepare test data and make predictions
    X_test = test_enhanced[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    X_test_selected = selector.transform(X_test)
    X_test_scaled = scaler.transform(X_test_selected)
    
    test_predictions = model.predict(X_test_scaled)
    
    # Create submission
    submission = pd.DataFrame({
        'ID': range(1, len(test_predictions) + 1),
        'prediction': test_predictions
    })
    
    submission.to_csv('baseline_submission.csv', index=False)
    print(f"Baseline submission created with {len(submission)} predictions")
    print(f"Prediction stats: mean={test_predictions.mean():.6f}, std={test_predictions.std():.6f}")
    
    return submission

if __name__ == "__main__":
    submission = main()
