#!/usr/bin/env python3
"""
Simple Advanced Model for Crypto Market Prediction
"""

import sys
import os
sys.path.append(os.getcwd())

try:
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest, f_regression
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    import warnings
    warnings.filterwarnings('ignore')
    
    def load_and_process_data():
        """Load and process data with simple feature engineering"""
        print("Loading data...")
        train_df = pd.read_parquet('train.parquet')
        test_df = pd.read_parquet('test.parquet')
        
        print(f"Training data shape: {train_df.shape}")
        print(f"Test data shape: {test_df.shape}")
        
        # Simple feature engineering
        def add_features(df):
            df_new = df.copy()
            
            # Market features
            df_new['bid_ask_spread'] = df_new['ask_qty'] - df_new['bid_qty']
            df_new['bid_ask_ratio'] = df_new['bid_qty'] / (df_new['ask_qty'] + 1e-8)
            df_new['buy_sell_ratio'] = df_new['buy_qty'] / (df_new['sell_qty'] + 1e-8)
            df_new['buy_sell_imbalance'] = df_new['buy_qty'] - df_new['sell_qty']
            df_new['volume_intensity'] = df_new['volume'] / (df_new['buy_qty'] + df_new['sell_qty'] + 1e-8)
            
            # Simple rolling features
            for col in ['volume', 'buy_qty', 'sell_qty']:
                df_new[f'{col}_ma3'] = df_new[col].rolling(3).mean()
                df_new[f'{col}_ma5'] = df_new[col].rolling(5).mean()
            
            return df_new
        
        train_enhanced = add_features(train_df)
        test_enhanced = add_features(test_df)
        
        return train_enhanced, test_enhanced
    
    def prepare_modeling_data(train_df, n_features=100):
        """Prepare data for modeling"""
        print("Preparing modeling data...")
        
        # Features and target
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
        
        print(f"Training set: {len(X_train)}, Validation set: {len(X_val)}")
        
        # Feature selection
        print(f"Selecting top {n_features} features...")
        selector = SelectKBest(score_func=f_regression, k=min(n_features, X_train.shape[1]))
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_val_selected = selector.transform(X_val)
        
        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_val_scaled = scaler.transform(X_val_selected)
        
        return {
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'y_train': y_train,
            'y_val': y_val,
            'selector': selector,
            'scaler': scaler,
            'feature_cols': feature_cols
        }
    
    def train_gradient_boosting(X_train, y_train, X_val, y_val):
        """Train Gradient Boosting model"""
        print("Training Gradient Boosting Regressor...")
        
        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Predictions
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        # Metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        train_mae = mean_absolute_error(y_train, train_pred)
        val_mae = mean_absolute_error(y_val, val_pred)
        
        print(f"Gradient Boosting Results:")
        print(f"  Train RMSE: {train_rmse:.6f}, Val RMSE: {val_rmse:.6f}")
        print(f"  Train MAE: {train_mae:.6f}, Val MAE: {val_mae:.6f}")
        
        return model, {
            'train_rmse': train_rmse,
            'val_rmse': val_rmse,
            'train_mae': train_mae,
            'val_mae': val_mae
        }
    
    def create_submission(model, test_df, selector, scaler, feature_cols):
        """Create submission file"""
        print("Creating submission...")
        
        # Prepare test data
        X_test = test_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        X_test_selected = selector.transform(X_test)
        X_test_scaled = scaler.transform(X_test_selected)
        
        # Predictions
        predictions = model.predict(X_test_scaled)
        
        # Create submission
        submission = pd.DataFrame({
            'ID': range(1, len(predictions) + 1),
            'prediction': predictions
        })
        
        submission.to_csv('submission_advanced.csv', index=False)
        print(f"Submission saved with {len(submission)} predictions")
        
        return submission
    
    def main():
        """Main pipeline"""
        print("CRYPTO MARKET PREDICTION - ADVANCED MODEL")
        print("="*50)
        
        # Load and process data
        train_df, test_df = load_and_process_data()
        
        # Prepare modeling data
        data = prepare_modeling_data(train_df)
        
        # Train advanced model
        model, results = train_gradient_boosting(
            data['X_train'], data['y_train'],
            data['X_val'], data['y_val']
        )
        
        # Create submission
        submission = create_submission(
            model, test_df, data['selector'], 
            data['scaler'], data['feature_cols']
        )
        
        print("\n" + "="*50)
        print("ADVANCED MODEL COMPLETED!")
        print(f"Best validation RMSE: {results['val_rmse']:.6f}")
        print("="*50)
        
        return model, results, submission
    
    if __name__ == "__main__":
        model, results, submission = main()

except ImportError as e:
    print(f"Import error: {e}")
    print("Please install required packages: pip install pandas pyarrow scikit-learn")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
