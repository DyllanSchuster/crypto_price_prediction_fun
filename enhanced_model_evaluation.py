#!/usr/bin/env python3
"""
Enhanced Model Evaluation with Comprehensive Metrics
Including Pearson Correlation Coefficient
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def comprehensive_model_evaluation(y_true, y_pred, model_name="Model"):
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
    
    # Additional metrics
    mse = mean_squared_error(y_true, y_pred)
    
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
        'MSE': mse,
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

def plot_prediction_analysis(y_true, y_pred, model_name="Model"):
    """
    Create comprehensive prediction analysis plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Actual vs Predicted scatter plot
    axes[0,0].scatter(y_true, y_pred, alpha=0.6, s=1)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[0,0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[0,0].set_xlabel('Actual Values')
    axes[0,0].set_ylabel('Predicted Values')
    axes[0,0].set_title(f'{model_name} - Actual vs Predicted')
    
    # Add correlation coefficient to plot
    corr, _ = pearsonr(y_true, y_pred)
    axes[0,0].text(0.05, 0.95, f'Pearson r = {corr:.4f}', 
                   transform=axes[0,0].transAxes, fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. Residuals plot
    residuals = y_true - y_pred
    axes[0,1].scatter(y_pred, residuals, alpha=0.6, s=1)
    axes[0,1].axhline(y=0, color='r', linestyle='--')
    axes[0,1].set_xlabel('Predicted Values')
    axes[0,1].set_ylabel('Residuals')
    axes[0,1].set_title(f'{model_name} - Residuals Plot')
    
    # 3. Residuals histogram
    axes[1,0].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    axes[1,0].set_xlabel('Residuals')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].set_title(f'{model_name} - Residuals Distribution')
    axes[1,0].axvline(x=0, color='r', linestyle='--')
    
    # 4. Time series plot (sample)
    sample_size = min(1000, len(y_true))
    indices = np.random.choice(len(y_true), sample_size, replace=False)
    indices.sort()
    
    axes[1,1].plot(indices, y_true.iloc[indices], label='Actual', alpha=0.7)
    axes[1,1].plot(indices, y_pred[indices], label='Predicted', alpha=0.7)
    axes[1,1].set_xlabel('Time Index')
    axes[1,1].set_ylabel('Values')
    axes[1,1].set_title(f'{model_name} - Time Series Comparison (Sample)')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig(f'{model_name.lower()}_evaluation_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def evaluate_model_with_enhanced_metrics(model, X_train, y_train, X_val, y_val, model_name):
    """
    Train model and evaluate with comprehensive metrics including Pearson correlation
    """
    # Train model
    model.fit(X_train, y_train)
    
    # Predictions
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    # Comprehensive evaluation
    print(f"\n{'='*60}")
    print(f"ENHANCED EVALUATION: {model_name}")
    print(f"{'='*60}")
    
    train_metrics = comprehensive_model_evaluation(y_train, train_pred, f"{model_name} (Training)")
    val_metrics = comprehensive_model_evaluation(y_val, val_pred, f"{model_name} (Validation)")
    
    # Create plots
    plot_prediction_analysis(y_val, val_pred, f"{model_name}_Validation")
    
    return {
        'model': model,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'train_predictions': train_pred,
        'val_predictions': val_pred
    }

def main():
    """
    Main function to demonstrate enhanced evaluation
    """
    print("ENHANCED MODEL EVALUATION WITH PEARSON CORRELATION")
    print("="*60)
    
    # Load data
    train_df = pd.read_parquet('train.parquet')
    
    # Simple feature engineering
    def add_features(df):
        df_new = df.copy()
        df_new['bid_ask_spread'] = df_new['ask_qty'] - df_new['bid_qty']
        df_new['buy_sell_ratio'] = df_new['buy_qty'] / (df_new['sell_qty'] + 1e-8)
        df_new['volume_intensity'] = df_new['volume'] / (df_new['buy_qty'] + df_new['sell_qty'] + 1e-8)
        return df_new
    
    train_enhanced = add_features(train_df)
    
    # Prepare data
    target_col = 'label'
    feature_cols = [col for col in train_enhanced.columns if col != target_col]
    
    X = train_enhanced[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y = train_enhanced[target_col]
    
    # Train/validation split
    split_idx = int(0.8 * len(X))
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Feature selection
    selector = SelectKBest(score_func=f_regression, k=100)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_val_selected = selector.transform(X_val)
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_val_scaled = scaler.transform(X_val_selected)
    
    # Evaluate Ridge model with enhanced metrics
    ridge_model = Ridge(alpha=1.0, random_state=42)
    results = evaluate_model_with_enhanced_metrics(
        ridge_model, X_train_scaled, y_train, X_val_scaled, y_val, "Ridge Regression"
    )
    
    # Summary comparison
    print(f"\n{'='*60}")
    print("METRICS COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    val_metrics = results['val_metrics']
    print(f"Validation Metrics:")
    print(f"  RMSE:                 {val_metrics['RMSE']:.6f}")
    print(f"  MAE:                  {val_metrics['MAE']:.6f}")
    print(f"  R² Score:             {val_metrics['R²']:.6f}")
    print(f"  Pearson Correlation:  {val_metrics['Pearson_Correlation']:.6f}")
    print(f"  Spearman Correlation: {val_metrics['Spearman_Correlation']:.6f}")
    print(f"  Directional Accuracy: {val_metrics['Directional_Accuracy']:.2f}%")
    
    # Interpretation
    print(f"\n{'='*60}")
    print("INTERPRETATION")
    print(f"{'='*60}")
    
    pearson_corr = val_metrics['Pearson_Correlation']
    if pearson_corr > 0.7:
        strength = "Strong positive"
    elif pearson_corr > 0.3:
        strength = "Moderate positive"
    elif pearson_corr > 0.1:
        strength = "Weak positive"
    elif pearson_corr > -0.1:
        strength = "Very weak"
    elif pearson_corr > -0.3:
        strength = "Weak negative"
    else:
        strength = "Moderate to strong negative"
    
    print(f"Pearson Correlation: {strength} linear relationship")
    print(f"R² Score: Model explains {val_metrics['R²']*100:.2f}% of variance")
    print(f"Directional Accuracy: Model predicts direction correctly {val_metrics['Directional_Accuracy']:.1f}% of time")
    
    return results

if __name__ == "__main__":
    results = main()
