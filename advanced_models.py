#!/usr/bin/env python3
"""
Advanced Models for Crypto Market Prediction
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

def load_processed_data():
    """Load and process data using the same pipeline as feature_engineering.py"""
    print("Loading and processing data...")
    
    # Load raw data
    train_df = pd.read_parquet('train.parquet')
    test_df = pd.read_parquet('test.parquet')
    
    # Simple feature engineering (key features only for speed)
    def create_features(df, is_train=True):
        df_new = df.copy()
        
        # Market features
        df_new['bid_ask_spread'] = df_new['ask_qty'] - df_new['bid_qty']
        df_new['bid_ask_ratio'] = df_new['bid_qty'] / (df_new['ask_qty'] + 1e-8)
        df_new['buy_sell_ratio'] = df_new['buy_qty'] / (df_new['sell_qty'] + 1e-8)
        df_new['buy_sell_imbalance'] = df_new['buy_qty'] - df_new['sell_qty']
        df_new['volume_per_trade'] = df_new['volume'] / (df_new['buy_qty'] + df_new['sell_qty'] + 1e-8)
        
        # Time features (only for training data with datetime index)
        if is_train and hasattr(df_new.index, 'hour'):
            df_new['hour'] = df_new.index.hour
            df_new['day_of_week'] = df_new.index.dayofweek
            df_new['hour_sin'] = np.sin(2 * np.pi * df_new['hour'] / 24)
            df_new['hour_cos'] = np.cos(2 * np.pi * df_new['hour'] / 24)
            df_new['dow_sin'] = np.sin(2 * np.pi * df_new['day_of_week'] / 7)
            df_new['dow_cos'] = np.cos(2 * np.pi * df_new['day_of_week'] / 7)
            df_new = df_new.drop(['hour', 'day_of_week'], axis=1)
        elif not is_train:
            # Dummy time features for test data
            df_new['hour_sin'] = 0.0
            df_new['hour_cos'] = 1.0
            df_new['dow_sin'] = 0.0
            df_new['dow_cos'] = 1.0
        
        # Simple rolling features (smaller windows for speed)
        for feature in ['volume', 'buy_qty', 'sell_qty']:
            df_new[f'{feature}_roll_mean_5'] = df_new[feature].rolling(window=5).mean()
            df_new[f'{feature}_roll_std_5'] = df_new[feature].rolling(window=5).std()
        
        return df_new
    
    train_enhanced = create_features(train_df, is_train=True)
    test_enhanced = create_features(test_df, is_train=False)
    
    return train_enhanced, test_enhanced

def prepare_data_for_modeling(train_df, test_df, n_features=150):
    """Prepare data for modeling with feature selection"""
    print("Preparing data for modeling...")
    
    # Prepare features and target
    target_col = 'label'
    feature_cols = [col for col in train_df.columns if col != target_col]
    
    X = train_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y = train_df[target_col]
    
    # Time series split
    tscv = TimeSeriesSplit(n_splits=3)
    splits = list(tscv.split(X))
    train_idx, val_idx = splits[-1]
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Feature selection
    print(f"Selecting top {n_features} features...")
    selector = SelectKBest(score_func=f_regression, k=n_features)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_val_selected = selector.transform(X_val)
    
    selected_features = X.columns[selector.get_support()].tolist()
    
    # Prepare test data
    X_test = test_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    X_test_selected = selector.transform(X_test)
    
    return {
        'X_train': X_train_selected,
        'X_val': X_val_selected,
        'X_test': X_test_selected,
        'y_train': y_train,
        'y_val': y_val,
        'selected_features': selected_features,
        'selector': selector
    }

def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost model"""
    print("Training XGBoost...")
    
    # XGBoost parameters optimized for time series
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Create DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Train model
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=100
    )
    
    # Predictions
    train_pred = model.predict(dtrain)
    val_pred = model.predict(dval)
    
    # Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    train_mae = mean_absolute_error(y_train, train_pred)
    val_mae = mean_absolute_error(y_val, val_pred)
    
    print(f"XGBoost - Train RMSE: {train_rmse:.6f}, Val RMSE: {val_rmse:.6f}")
    print(f"XGBoost - Train MAE: {train_mae:.6f}, Val MAE: {val_mae:.6f}")
    
    return model, {
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
        'train_mae': train_mae,
        'val_mae': val_mae
    }

def train_lightgbm(X_train, y_train, X_val, y_val):
    """Train LightGBM model"""
    print("Training LightGBM...")
    
    # LightGBM parameters
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Train model
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )
    
    # Predictions
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    # Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    train_mae = mean_absolute_error(y_train, train_pred)
    val_mae = mean_absolute_error(y_val, val_pred)
    
    print(f"LightGBM - Train RMSE: {train_rmse:.6f}, Val RMSE: {val_rmse:.6f}")
    print(f"LightGBM - Train MAE: {train_mae:.6f}, Val MAE: {val_mae:.6f}")
    
    return model, {
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
        'train_mae': train_mae,
        'val_mae': val_mae
    }

def create_ensemble_predictions(models, X_test):
    """Create ensemble predictions from multiple models"""
    print("Creating ensemble predictions...")
    
    predictions = []
    
    # XGBoost prediction
    if 'xgb' in models:
        dtest = xgb.DMatrix(X_test)
        xgb_pred = models['xgb'].predict(dtest)
        predictions.append(xgb_pred)
    
    # LightGBM prediction
    if 'lgb' in models:
        lgb_pred = models['lgb'].predict(X_test)
        predictions.append(lgb_pred)
    
    # Simple average ensemble
    ensemble_pred = np.mean(predictions, axis=0)
    
    return ensemble_pred

def main():
    """Main advanced modeling pipeline"""
    print("CRYPTO MARKET PREDICTION - ADVANCED MODELS")
    print("="*50)
    
    # Load and process data
    train_df, test_df = load_processed_data()
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    # Prepare data for modeling
    data = prepare_data_for_modeling(train_df, test_df)
    
    print(f"Training set size: {len(data['y_train'])}")
    print(f"Validation set size: {len(data['y_val'])}")
    print(f"Selected features: {len(data['selected_features'])}")
    
    # Train models
    models = {}
    results = {}
    
    # XGBoost
    xgb_model, xgb_results = train_xgboost(
        data['X_train'], data['y_train'], 
        data['X_val'], data['y_val']
    )
    models['xgb'] = xgb_model
    results['xgb'] = xgb_results
    
    # LightGBM
    lgb_model, lgb_results = train_lightgbm(
        data['X_train'], data['y_train'], 
        data['X_val'], data['y_val']
    )
    models['lgb'] = lgb_model
    results['lgb'] = lgb_results
    
    # Create ensemble predictions
    ensemble_pred = create_ensemble_predictions(models, data['X_test'])
    
    # Create submission
    submission = pd.DataFrame({
        'ID': range(1, len(ensemble_pred) + 1),
        'prediction': ensemble_pred
    })
    
    submission.to_csv('submission.csv', index=False)
    print(f"Submission saved with {len(submission)} predictions")
    
    print("\n" + "="*50)
    print("ADVANCED MODELING COMPLETED!")
    print("="*50)
    
    return models, results, submission

if __name__ == "__main__":
    models, results, submission = main()
