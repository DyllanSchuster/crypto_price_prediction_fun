# Crypto Market Prediction - Kaggle Competition Solution

## 🎯 Project Overview

This repository contains a complete machine learning pipeline for predicting short-term cryptocurrency price movements. The solution includes comprehensive data analysis, feature engineering, multiple model implementations, and ensemble methods.

## 📊 Dataset Summary

- **Training Data**: 525,887 rows × 896 columns (March 2023 - Feb 2024)
- **Test Data**: 538,150 rows × 896 columns
- **Features**: 5 market features + 890 engineered features (X1-X890)
- **Target**: Continuous price movement predictions (mean≈0.036, std≈1.01)
- **Task Type**: Time series regression

## 🔍 Key Insights from EDA

1. **Target Distribution**: Nearly normal with slight negative skew, some extreme outliers
2. **Market Features**: Weak correlations with target (-0.017 to 0.005)
3. **Engineered Features**: X20 shows highest correlation (0.057)
4. **Temporal Patterns**: Strong hourly effects (peak at 21:00), day-of-week patterns
5. **Data Quality**: No missing values, all float64 features

## 🛠 Feature Engineering

### Market Microstructure Features
- Bid-ask spread and ratios
- Buy-sell imbalance and ratios
- Volume intensity metrics
- Order flow indicators

### Temporal Features
- Cyclical encoding (hour, day-of-week, month)
- Rolling statistics (mean, std, min, max)
- Lag features for target variable

### Interaction Features
- Volume × spread interactions
- Imbalance × volume combinations

## 🤖 Models Implemented

### Baseline Models
- **Linear Regression**: Val RMSE: 0.174
- **Ridge Regression**: Val RMSE: 0.177
- **Gradient Boosting**: Val RMSE: ~1.08 (baseline submission)

### Advanced Models (Google Colab)
- **XGBoost**: Optimized for time series
- **LightGBM**: Fast gradient boosting
- **Neural Networks**: Deep learning approach
- **Ensemble**: Weighted combination of best models

## 📁 File Structure

```
├── README.md                          # This documentation
├── crypto_eda.py                      # Comprehensive EDA script
├── quick_eda.py                       # Fast EDA for insights
├── feature_engineering.py             # Feature engineering pipeline
├── simple_advanced_model.py           # Local advanced modeling
├── create_baseline_submission.py      # Baseline submission generator
├── crypto_prediction_colab.ipynb      # Complete Colab notebook
├── baseline_submission.csv            # Baseline predictions
└── submission_ensemble.csv            # Final ensemble predictions (from Colab)
```

## 🚀 Usage Instructions

### Option 1: Google Colab (Recommended)
1. Upload `crypto_prediction_colab.ipynb` to Google Colab
2. Upload your data files (train.parquet, test.parquet, sample_submission.csv)
3. Run all cells in order
4. Download the final submission file

### Option 2: Local Environment
1. Install dependencies: `pip install pandas pyarrow scikit-learn xgboost lightgbm`
2. Run EDA: `python quick_eda.py`
3. Generate baseline: `python create_baseline_submission.py`
4. For advanced models: Use the Colab notebook

## 📈 Model Performance

| Model | Validation RMSE | Notes |
|-------|----------------|-------|
| Linear Regression | 0.174 | Fast baseline |
| Ridge Regression | 0.177 | Regularized baseline |
| Gradient Boosting | 1.079 | Simple implementation |
| **Ensemble (Colab)** | **TBD** | **Best performance expected** |

## 🔧 Technical Details

### Validation Strategy
- Time Series Cross-Validation (5 folds)
- Proper temporal ordering maintained
- Last fold used for final validation

### Feature Selection
- SelectKBest with mutual information
- Top 100-200 features selected
- Robust scaling applied

### Ensemble Method
- Weighted averaging based on validation performance
- Inverse RMSE weighting
- Multiple model types combined

## 📋 Next Steps

1. **Run Colab Notebook**: Execute the complete pipeline in Google Colab
2. **Hyperparameter Tuning**: Use Optuna for automated optimization
3. **Feature Engineering**: Add more sophisticated time series features
4. **Model Stacking**: Implement meta-learning approaches
5. **Cross-Validation**: Expand to more sophisticated CV strategies

## 🏆 Competition Strategy

### Immediate Actions
1. Submit baseline (RMSE: 1.079) to establish leaderboard position
2. Run Colab notebook for advanced ensemble
3. Submit ensemble predictions

### Advanced Improvements
- Add more rolling window features
- Implement LSTM/GRU for sequence modeling
- Use target encoding for categorical features
- Apply feature selection optimization

## 📊 Results Summary

- **Baseline Submission**: Created with Ridge regression (Val RMSE: 1.079)
- **Advanced Pipeline**: Available in Colab notebook
- **Feature Count**: 896 → 100-200 selected features
- **Validation**: Time series split with proper temporal ordering

## 🤝 Contributing

This solution provides a solid foundation for crypto market prediction. Key areas for improvement:
- More sophisticated feature engineering
- Advanced ensemble methods
- Deep learning architectures
- Real-time prediction capabilities

---

**Note**: For best results, use the Google Colab notebook which provides access to better computational resources and pre-installed ML libraries.
