#!/usr/bin/env python3
"""
Feature Engineering and Baseline Models for Crypto Market Prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare the datasets"""
    print("Loading datasets...")
    train_df = pd.read_parquet('train.parquet')
    test_df = pd.read_parquet('test.parquet')
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    return train_df, test_df

def create_time_features(df):
    """Create time-based features"""
    print("Creating time-based features...")

    df_new = df.copy()

    # Check if index has datetime attributes
    if hasattr(df_new.index, 'hour'):
        # Extract time components
        df_new['hour'] = df_new.index.hour
        df_new['day_of_week'] = df_new.index.dayofweek
        df_new['month'] = df_new.index.month
        df_new['day_of_month'] = df_new.index.day

        # Cyclical encoding for time features
        df_new['hour_sin'] = np.sin(2 * np.pi * df_new['hour'] / 24)
        df_new['hour_cos'] = np.cos(2 * np.pi * df_new['hour'] / 24)
        df_new['dow_sin'] = np.sin(2 * np.pi * df_new['day_of_week'] / 7)
        df_new['dow_cos'] = np.cos(2 * np.pi * df_new['day_of_week'] / 7)

        # Drop original time features
        df_new = df_new.drop(['hour', 'day_of_week', 'month', 'day_of_month'], axis=1)
    else:
        # For test data without datetime index, create dummy time features
        print("No datetime index found, creating dummy time features...")
        df_new['hour_sin'] = 0.0
        df_new['hour_cos'] = 1.0
        df_new['dow_sin'] = 0.0
        df_new['dow_cos'] = 1.0

    return df_new

def create_market_features(df):
    """Create additional market-based features"""
    print("Creating market-based features...")
    
    df_new = df.copy()
    
    # Ratios and spreads
    df_new['bid_ask_spread'] = df_new['ask_qty'] - df_new['bid_qty']
    df_new['bid_ask_ratio'] = df_new['bid_qty'] / (df_new['ask_qty'] + 1e-8)
    df_new['buy_sell_ratio'] = df_new['buy_qty'] / (df_new['sell_qty'] + 1e-8)
    df_new['buy_sell_imbalance'] = df_new['buy_qty'] - df_new['sell_qty']
    
    # Volume features
    df_new['volume_per_trade'] = df_new['volume'] / (df_new['buy_qty'] + df_new['sell_qty'] + 1e-8)
    df_new['bid_volume_ratio'] = df_new['bid_qty'] / (df_new['volume'] + 1e-8)
    df_new['ask_volume_ratio'] = df_new['ask_qty'] / (df_new['volume'] + 1e-8)
    
    return df_new

def create_lag_features(df, target_col='label', lags=[1, 2, 3, 5, 10]):
    """Create lagged features (only for training data with target)"""
    print(f"Creating lag features for lags: {lags}...")
    
    df_new = df.copy()
    
    if target_col in df_new.columns:
        for lag in lags:
            df_new[f'{target_col}_lag_{lag}'] = df_new[target_col].shift(lag)
    
    return df_new

def create_rolling_features(df, windows=[5, 10, 30, 60]):
    """Create rolling window features"""
    print(f"Creating rolling features for windows: {windows}...")
    
    df_new = df.copy()
    
    # Rolling features for market data
    market_features = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']
    
    for feature in market_features:
        for window in windows:
            df_new[f'{feature}_roll_mean_{window}'] = df_new[feature].rolling(window=window).mean()
            df_new[f'{feature}_roll_std_{window}'] = df_new[feature].rolling(window=window).std()
    
    return df_new

def select_features(X_train, y_train, k=200):
    """Select top k features using univariate selection"""
    print(f"Selecting top {k} features...")
    
    # Remove any infinite or NaN values
    X_train_clean = X_train.fillna(0)
    X_train_clean = X_train_clean.replace([np.inf, -np.inf], 0)
    
    selector = SelectKBest(score_func=f_regression, k=k)
    X_selected = selector.fit_transform(X_train_clean, y_train)
    
    selected_features = X_train.columns[selector.get_support()].tolist()
    print(f"Selected features: {len(selected_features)}")
    
    return X_selected, selected_features, selector

def evaluate_model(model, X_train, y_train, X_val, y_val):
    """Evaluate a model and return metrics"""
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    train_mae = mean_absolute_error(y_train, train_pred)
    val_mae = mean_absolute_error(y_val, val_pred)
    
    return {
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
        'train_mae': train_mae,
        'val_mae': val_mae
    }

def run_baseline_models(X_train, y_train, X_val, y_val):
    """Run baseline models and compare performance"""
    print("\nRunning baseline models...")
    
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        metrics = evaluate_model(model, X_train, y_train, X_val, y_val)
        results[name] = metrics
        
        print(f"  Train RMSE: {metrics['train_rmse']:.6f}")
        print(f"  Val RMSE: {metrics['val_rmse']:.6f}")
        print(f"  Train MAE: {metrics['train_mae']:.6f}")
        print(f"  Val MAE: {metrics['val_mae']:.6f}")
    
    return results

def main():
    """Main feature engineering and baseline modeling pipeline"""
    print("CRYPTO MARKET PREDICTION - FEATURE ENGINEERING & BASELINE MODELS")
    print("="*70)
    
    # Load data
    train_df, test_df = load_and_prepare_data()
    
    # Feature engineering for training data
    print("\nFeature Engineering for Training Data:")
    train_enhanced = create_time_features(train_df)
    train_enhanced = create_market_features(train_enhanced)
    train_enhanced = create_lag_features(train_enhanced)
    train_enhanced = create_rolling_features(train_enhanced)
    
    # Feature engineering for test data (no target-based features)
    print("\nFeature Engineering for Test Data:")
    test_enhanced = create_time_features(test_df)
    test_enhanced = create_market_features(test_enhanced)
    test_enhanced = create_rolling_features(test_enhanced)
    
    print(f"Enhanced training data shape: {train_enhanced.shape}")
    print(f"Enhanced test data shape: {test_enhanced.shape}")
    
    # Prepare features and target
    target_col = 'label'
    feature_cols = [col for col in train_enhanced.columns if col != target_col]
    
    X = train_enhanced[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y = train_enhanced[target_col]
    
    # Time series split for validation
    print("\nSplitting data for time series validation...")
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Use the last split for final validation
    splits = list(tscv.split(X))
    train_idx, val_idx = splits[-1]
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    
    # Feature selection
    X_train_selected, selected_features, selector = select_features(X_train, y_train, k=200)
    X_val_selected = selector.transform(X_val.fillna(0).replace([np.inf, -np.inf], 0))
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_val_scaled = scaler.transform(X_val_selected)
    
    # Run baseline models
    results = run_baseline_models(X_train_scaled, y_train, X_val_scaled, y_val)
    
    print("\n" + "="*70)
    print("FEATURE ENGINEERING & BASELINE MODELING COMPLETED!")
    print("="*70)
    
    return {
        'train_enhanced': train_enhanced,
        'test_enhanced': test_enhanced,
        'selected_features': selected_features,
        'selector': selector,
        'scaler': scaler,
        'results': results
    }

if __name__ == "__main__":
    pipeline_results = main()
