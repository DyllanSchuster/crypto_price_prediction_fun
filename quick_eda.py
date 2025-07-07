#!/usr/bin/env python3
"""
Quick EDA for Crypto Market Prediction
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def main():
    print("CRYPTO MARKET PREDICTION - QUICK EDA")
    print("="*50)
    
    # Load data
    print("Loading datasets...")
    train_df = pd.read_parquet('train.parquet')
    test_df = pd.read_parquet('test.parquet')
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    # Target analysis
    print("\nTARGET ANALYSIS:")
    target = train_df['label']
    print(f"Mean: {target.mean():.6f}")
    print(f"Std: {target.std():.6f}")
    print(f"Min: {target.min():.6f}")
    print(f"Max: {target.max():.6f}")
    print(f"Skewness: {target.skew():.6f}")
    print(f"Kurtosis: {target.kurtosis():.6f}")
    
    # Market features analysis
    print("\nMARKET FEATURES ANALYSIS:")
    market_features = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']
    print(train_df[market_features].describe())
    
    # Quick correlation with target (sample for speed)
    print("\nCORRELATION WITH TARGET (sample):")
    sample_size = min(50000, len(train_df))
    sample_df = train_df.sample(n=sample_size, random_state=42)
    
    market_corr = sample_df[market_features + ['label']].corr()['label'].drop('label')
    print("Market features correlation:")
    print(market_corr.sort_values(ascending=False))
    
    # Sample X features correlation
    x_features = [col for col in train_df.columns if col.startswith('X')][:20]  # First 20 X features
    x_corr = sample_df[x_features + ['label']].corr()['label'].drop('label')
    print(f"\nTop 10 X features correlation (from first 20):")
    print(x_corr.abs().sort_values(ascending=False).head(10))
    
    # Time patterns
    print("\nTIME PATTERNS:")
    train_df_time = train_df.copy()
    train_df_time['hour'] = train_df_time.index.hour
    train_df_time['day_of_week'] = train_df_time.index.dayofweek
    
    hourly_avg = train_df_time.groupby('hour')['label'].mean()
    daily_avg = train_df_time.groupby('day_of_week')['label'].mean()
    
    print("Average target by hour (top 5):")
    print(hourly_avg.sort_values(ascending=False).head())
    
    print("\nAverage target by day of week:")
    print(daily_avg)
    
    # Feature statistics
    print("\nFEATURE STATISTICS:")
    print(f"Total features: {len(train_df.columns)}")
    print(f"Market features: {len(market_features)}")
    print(f"Engineered features: {len(x_features)}")
    
    # Missing values
    print(f"\nMissing values in train: {train_df.isnull().sum().sum()}")
    print(f"Missing values in test: {test_df.isnull().sum().sum()}")
    
    # Data types
    print(f"\nData types: {train_df.dtypes.value_counts()}")
    
    print("\n" + "="*50)
    print("QUICK EDA COMPLETED!")
    print("="*50)
    
    # Save key insights
    insights = {
        'target_mean': target.mean(),
        'target_std': target.std(),
        'target_range': (target.min(), target.max()),
        'market_correlations': market_corr.to_dict(),
        'top_x_correlations': x_corr.abs().sort_values(ascending=False).head(10).to_dict(),
        'hourly_patterns': hourly_avg.to_dict(),
        'daily_patterns': daily_avg.to_dict()
    }
    
    return train_df, test_df, insights

if __name__ == "__main__":
    train_df, test_df, insights = main()
