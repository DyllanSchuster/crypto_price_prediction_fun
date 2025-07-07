#!/usr/bin/env python3
"""
Crypto Market Prediction - Exploratory Data Analysis
====================================================

This script performs comprehensive exploratory data analysis on the crypto market prediction dataset.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data():
    """Load training and test datasets"""
    print("Loading datasets...")
    train_df = pd.read_parquet('train.parquet')
    test_df = pd.read_parquet('test.parquet')
    sample_sub = pd.read_csv('sample_submission.csv')
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    print(f"Sample submission shape: {sample_sub.shape}")
    
    return train_df, test_df, sample_sub

def analyze_target_distribution(train_df):
    """Analyze the distribution of the target variable"""
    print("\n" + "="*50)
    print("TARGET VARIABLE ANALYSIS")
    print("="*50)
    
    target = train_df['label']
    
    print(f"Target statistics:")
    print(f"Mean: {target.mean():.6f}")
    print(f"Median: {target.median():.6f}")
    print(f"Std: {target.std():.6f}")
    print(f"Skewness: {target.skew():.6f}")
    print(f"Kurtosis: {target.kurtosis():.6f}")
    print(f"Min: {target.min():.6f}")
    print(f"Max: {target.max():.6f}")
    
    # Check for outliers
    Q1 = target.quantile(0.25)
    Q3 = target.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = target[(target < lower_bound) | (target > upper_bound)]
    print(f"Outliers (IQR method): {len(outliers)} ({len(outliers)/len(target)*100:.2f}%)")
    
    # Plot distribution
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Histogram
    axes[0,0].hist(target, bins=100, alpha=0.7, edgecolor='black')
    axes[0,0].set_title('Target Distribution')
    axes[0,0].set_xlabel('Label Value')
    axes[0,0].set_ylabel('Frequency')
    
    # Box plot
    axes[0,1].boxplot(target)
    axes[0,1].set_title('Target Box Plot')
    axes[0,1].set_ylabel('Label Value')
    
    # Q-Q plot
    stats.probplot(target, dist="norm", plot=axes[1,0])
    axes[1,0].set_title('Q-Q Plot (Normal Distribution)')
    
    # Time series plot (sample)
    sample_indices = np.random.choice(len(target), min(10000, len(target)), replace=False)
    sample_indices.sort()
    axes[1,1].plot(sample_indices, target.iloc[sample_indices], alpha=0.6)
    axes[1,1].set_title('Target Time Series (Sample)')
    axes[1,1].set_xlabel('Time Index')
    axes[1,1].set_ylabel('Label Value')
    
    plt.tight_layout()
    plt.savefig('target_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close instead of show

def analyze_market_features(train_df):
    """Analyze the basic market features"""
    print("\n" + "="*50)
    print("MARKET FEATURES ANALYSIS")
    print("="*50)
    
    market_features = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']
    
    # Basic statistics
    print("Market features statistics:")
    print(train_df[market_features].describe())
    
    # Correlation with target
    print("\nCorrelation with target:")
    correlations = train_df[market_features + ['label']].corr()['label'].drop('label')
    print(correlations.sort_values(ascending=False))
    
    # Plot market features
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, feature in enumerate(market_features):
        # Distribution
        axes[i].hist(train_df[feature], bins=50, alpha=0.7, edgecolor='black')
        axes[i].set_title(f'{feature} Distribution')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Frequency')
        axes[i].set_yscale('log')  # Log scale due to potential skewness
    
    # Correlation heatmap
    corr_matrix = train_df[market_features].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[5])
    axes[5].set_title('Market Features Correlation')
    
    plt.tight_layout()
    plt.savefig('market_features_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close instead of show

def analyze_engineered_features(train_df, n_features=50):
    """Analyze a subset of engineered features"""
    print("\n" + "="*50)
    print("ENGINEERED FEATURES ANALYSIS")
    print("="*50)
    
    # Get X features
    x_features = [col for col in train_df.columns if col.startswith('X')]
    print(f"Total engineered features: {len(x_features)}")
    
    # Sample features for analysis
    sample_features = x_features[:n_features]
    
    # Basic statistics
    print(f"\nAnalyzing first {n_features} engineered features:")
    feature_stats = train_df[sample_features].describe()
    print("Feature statistics summary:")
    print(f"Mean range: {feature_stats.loc['mean'].min():.4f} to {feature_stats.loc['mean'].max():.4f}")
    print(f"Std range: {feature_stats.loc['std'].min():.4f} to {feature_stats.loc['std'].max():.4f}")
    
    # Correlation with target
    print("\nTop 10 features correlated with target:")
    correlations = train_df[x_features + ['label']].corr()['label'].drop('label')
    top_corr = correlations.abs().sort_values(ascending=False).head(10)
    print(top_corr)
    
    # Plot feature distributions
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.flatten()
    
    for i, feature in enumerate(sample_features[:16]):
        axes[i].hist(train_df[feature], bins=30, alpha=0.7, edgecolor='black')
        axes[i].set_title(f'{feature}')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('engineered_features_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close instead of show
    
    return top_corr

def analyze_time_patterns(train_df):
    """Analyze temporal patterns in the data"""
    print("\n" + "="*50)
    print("TEMPORAL PATTERNS ANALYSIS")
    print("="*50)
    
    # Create time-based features
    train_df_time = train_df.copy()
    train_df_time['hour'] = train_df_time.index.hour
    train_df_time['day_of_week'] = train_df_time.index.dayofweek
    train_df_time['month'] = train_df_time.index.month
    
    # Analyze patterns
    print("Average target by hour:")
    hourly_avg = train_df_time.groupby('hour')['label'].mean()
    print(hourly_avg)
    
    print("\nAverage target by day of week:")
    daily_avg = train_df_time.groupby('day_of_week')['label'].mean()
    print(daily_avg)
    
    # Plot temporal patterns
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Hourly pattern
    hourly_avg.plot(kind='bar', ax=axes[0,0])
    axes[0,0].set_title('Average Target by Hour')
    axes[0,0].set_xlabel('Hour')
    axes[0,0].set_ylabel('Average Label')
    
    # Daily pattern
    daily_avg.plot(kind='bar', ax=axes[0,1])
    axes[0,1].set_title('Average Target by Day of Week')
    axes[0,1].set_xlabel('Day of Week (0=Monday)')
    axes[0,1].set_ylabel('Average Label')
    
    # Monthly pattern
    monthly_avg = train_df_time.groupby('month')['label'].mean()
    monthly_avg.plot(kind='bar', ax=axes[1,0])
    axes[1,0].set_title('Average Target by Month')
    axes[1,0].set_xlabel('Month')
    axes[1,0].set_ylabel('Average Label')
    
    # Rolling average
    rolling_avg = train_df['label'].rolling(window=1440).mean()  # 24 hours
    sample_indices = np.arange(0, len(rolling_avg), 1440)  # Daily samples
    axes[1,1].plot(sample_indices, rolling_avg.iloc[sample_indices])
    axes[1,1].set_title('24-Hour Rolling Average of Target')
    axes[1,1].set_xlabel('Time Index')
    axes[1,1].set_ylabel('Rolling Average Label')
    
    plt.tight_layout()
    plt.savefig('temporal_patterns.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close instead of show

def main():
    """Main EDA function"""
    print("CRYPTO MARKET PREDICTION - EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    # Load data
    train_df, test_df, sample_sub = load_data()
    
    # Analyze target distribution
    analyze_target_distribution(train_df)
    
    # Analyze market features
    analyze_market_features(train_df)
    
    # Analyze engineered features
    top_features = analyze_engineered_features(train_df)
    
    # Analyze temporal patterns
    analyze_time_patterns(train_df)
    
    print("\n" + "="*60)
    print("EDA COMPLETED - Check generated plots for insights!")
    print("="*60)
    
    return train_df, test_df, top_features

if __name__ == "__main__":
    train_df, test_df, top_features = main()
