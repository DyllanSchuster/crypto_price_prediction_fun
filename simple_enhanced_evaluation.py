#!/usr/bin/env python3
"""
Simple Enhanced Model Evaluation with Pearson Correlation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

def comprehensive_evaluation(y_true, y_pred, model_name="Model"):
    """
    Comprehensive evaluation including Pearson correlation coefficient
    """
    # Basic regression metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Correlation metrics
    pearson_corr, pearson_p = pearsonr(y_true, y_pred)
    spearman_corr, spearman_p = spearmanr(y_true, y_pred)
    
    # Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    # Directional accuracy (for time series)
    if len(y_true) > 1:
        true_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred))
        directional_accuracy = np.mean(true_direction == pred_direction) * 100
    else:
        directional_accuracy = np.nan
    
    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'Pearson_Correlation': pearson_corr,
        'Pearson_p_value': pearson_p,
        'Spearman_Correlation': spearman_corr,
        'Spearman_p_value': spearman_p,
        'MAPE': mape,
        'Directional_Accuracy': directional_accuracy
    }
    
    print(f"\n{model_name} - Comprehensive Evaluation:")
    print("="*50)
    print(f"RMSE:                    {rmse:.6f}")
    print(f"MAE:                     {mae:.6f}")
    print(f"R² Score:                {r2:.6f}")
    print(f"Pearson Correlation:     {pearson_corr:.6f} (p={pearson_p:.6f})")
    print(f"Spearman Correlation:    {spearman_corr:.6f} (p={spearman_p:.6f})")
    print(f"MAPE:                    {mape:.2f}%")
    if not np.isnan(directional_accuracy):
        print(f"Directional Accuracy:    {directional_accuracy:.2f}%")
    
    return metrics

def interpret_correlation(corr_value):
    """Interpret correlation coefficient strength"""
    abs_corr = abs(corr_value)
    if abs_corr >= 0.9:
        return "Very strong"
    elif abs_corr >= 0.7:
        return "Strong"
    elif abs_corr >= 0.5:
        return "Moderate"
    elif abs_corr >= 0.3:
        return "Weak"
    elif abs_corr >= 0.1:
        return "Very weak"
    else:
        return "Negligible"

def main():
    """
    Main function to demonstrate enhanced evaluation with Pearson correlation
    """
    print("ENHANCED MODEL EVALUATION WITH PEARSON CORRELATION")
    print("="*60)
    
    # Load data
    print("Loading data...")
    train_df = pd.read_parquet('train.parquet')
    
    # Simple feature engineering
    def add_features(df):
        df_new = df.copy()
        df_new['bid_ask_spread'] = df_new['ask_qty'] - df_new['bid_qty']
        df_new['buy_sell_ratio'] = df_new['buy_qty'] / (df_new['sell_qty'] + 1e-8)
        df_new['volume_intensity'] = df_new['volume'] / (df_new['buy_qty'] + df_new['sell_qty'] + 1e-8)
        return df_new
    
    train_enhanced = add_features(train_df)
    print(f"Data shape: {train_enhanced.shape}")
    
    # Prepare data
    target_col = 'label'
    feature_cols = [col for col in train_enhanced.columns if col != target_col]
    
    X = train_enhanced[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y = train_enhanced[target_col]
    
    # Train/validation split (time series aware)
    split_idx = int(0.8 * len(X))
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Training set: {len(X_train)}, Validation set: {len(X_val)}")
    
    # Feature selection
    print("Selecting features...")
    selector = SelectKBest(score_func=f_regression, k=100)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_val_selected = selector.transform(X_val)
    
    # Scaling
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_val_scaled = scaler.transform(X_val_selected)
    
    # Train Ridge model
    print("Training Ridge Regression model...")
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    train_pred = model.predict(X_train_scaled)
    val_pred = model.predict(X_val_scaled)
    
    # Comprehensive evaluation
    print(f"\n{'='*60}")
    print("COMPREHENSIVE MODEL EVALUATION")
    print(f"{'='*60}")
    
    train_metrics = comprehensive_evaluation(y_train, train_pred, "Training Set")
    val_metrics = comprehensive_evaluation(y_val, val_pred, "Validation Set")
    
    # Summary and interpretation
    print(f"\n{'='*60}")
    print("SUMMARY & INTERPRETATION")
    print(f"{'='*60}")
    
    val_pearson = val_metrics['Pearson_Correlation']
    val_r2 = val_metrics['R²']
    val_rmse = val_metrics['RMSE']
    val_directional = val_metrics['Directional_Accuracy']
    
    print(f"\nKey Validation Metrics:")
    print(f"  • RMSE: {val_rmse:.6f}")
    print(f"  • Pearson Correlation: {val_pearson:.6f}")
    print(f"  • R² Score: {val_r2:.6f}")
    print(f"  • Directional Accuracy: {val_directional:.2f}%")
    
    print(f"\nInterpretation:")
    corr_strength = interpret_correlation(val_pearson)
    print(f"  • Correlation Strength: {corr_strength} ({'positive' if val_pearson > 0 else 'negative'} relationship)")
    print(f"  • Variance Explained: {val_r2*100:.2f}% of target variance")
    print(f"  • Prediction Quality: {'Good' if val_pearson > 0.3 else 'Moderate' if val_pearson > 0.1 else 'Poor'}")
    print(f"  • Direction Prediction: {'Good' if val_directional > 60 else 'Moderate' if val_directional > 50 else 'Poor'}")
    
    # Statistical significance
    if val_metrics['Pearson_p_value'] < 0.001:
        significance = "Highly significant (p < 0.001)"
    elif val_metrics['Pearson_p_value'] < 0.01:
        significance = "Very significant (p < 0.01)"
    elif val_metrics['Pearson_p_value'] < 0.05:
        significance = "Significant (p < 0.05)"
    else:
        significance = "Not significant (p >= 0.05)"
    
    print(f"  • Statistical Significance: {significance}")
    
    # Comparison with baseline
    print(f"\n{'='*60}")
    print("COMPARISON WITH PREVIOUS RESULTS")
    print(f"{'='*60}")
    
    print("Previous baseline results:")
    print("  • Linear Regression: Val RMSE: 0.174")
    print("  • Ridge Regression: Val RMSE: 0.177")
    print("  • Simple baseline: Val RMSE: 1.079")
    
    print(f"\nCurrent enhanced evaluation:")
    print(f"  • Ridge Regression: Val RMSE: {val_rmse:.6f}")
    print(f"  • Pearson Correlation: {val_pearson:.6f} ({corr_strength})")
    
    if val_rmse < 0.2:
        performance = "Excellent"
    elif val_rmse < 0.5:
        performance = "Good"
    elif val_rmse < 1.0:
        performance = "Moderate"
    else:
        performance = "Poor"
    
    print(f"  • Overall Performance: {performance}")
    
    return {
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'model': model
    }

if __name__ == "__main__":
    results = main()
