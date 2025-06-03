# Customer churn analysis and predictions using machine learning

This project implements and compares multiple tree-based machine learning models to predict customer churn for a telecommunications company. The analysis includes comprehensive exploratory data analysis, feature engineering, and model evaluation using various ensemble methods.

## Project Overview

Customer churn prediction is a critical business problem for telecommunications companies. This project analyzes customer data to identify patterns and build predictive models to determine which customers are likely to cancel their service.

## Dataset

The analysis uses the Telco Customer Churn dataset (`Telco-Customer-Churn.csv`) which contains customer information including:
- **Demographics**: Gender, Senior Citizen status, Partner, Dependents
- **Services**: Phone Service, Internet Service, Online Security, Tech Support, etc.
- **Account Information**: Contract type, Payment method, Monthly charges, Total charges
- **Target Variable**: Churn (Yes/No)

## Key Features

### 1. **Comprehensive Exploratory Data Analysis (EDA)**
- Data quality assessment and missing value analysis
- Class balance visualization
- Feature correlation analysis with churn
- Distribution analysis of key variables

### 2. **Advanced Cohort Analysis**
- Tenure-based customer segmentation
- Churn rate analysis by tenure cohorts:
  - 0-12 Months
  - 12-24 Months  
  - 24-48 Months
  - Over 48 Months
- Visualization of churn patterns across different customer segments

### 3. **Multiple Tree-Based Models**
- **Decision Tree**: Single tree with hyperparameter tuning
- **Random Forest**: Ensemble of decision trees with grid search optimization
- **AdaBoost**: Adaptive boosting classifier
- **Gradient Boosting**: Gradient boosting classifier

### 4. **Model Optimization**
- Grid search cross-validation for hyperparameter tuning
- Feature importance analysis
- Error analysis and model comparison
- Performance visualization

## Technologies Used

- **Python 3.x**
- **Data Analysis**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Machine Learning**: scikit-learn
- **Model Evaluation**: classification_report, confusion_matrix, ConfusionMatrixDisplay

## Key Findings

### Churn Patterns
1. **Tenure Impact**: Customers with longer tenure have significantly lower churn rates
2. **Contract Type**: Month-to-month contracts show highest churn rates
3. **Feature Correlations**: Key predictors include contract type, tenure, and payment methods

### Model Performance Summary

| Model | Accuracy |
|-------|----------|
| Decision Tree | ~80% |
| Random Forest | ~83% |
| AdaBoost | ~83% |
| Gradient Boosting | ~82% |



## Code Structure

```
├── Part 0: Data Import and Setup
├── Part 1: Data Quality Assessment
├── Part 2: Exploratory Data Analysis
│   ├── Feature correlation analysis
│   ├── Churn distribution visualization
│   └── Relationship analysis
├── Part 3: Cohort Analysis
│   ├── Tenure-based segmentation
│   ├── Churn rate calculation by cohort
│   └── Advanced visualization
└── Part 4: Predictive Modeling
    ├── Decision Tree with Grid Search
    ├── Random Forest Optimization
    ├── Boosting Methods (AdaBoost & Gradient Boosting)
    └── Model Comparison and Evaluation
```


## Prerequisites

- Python 3.7+
- Jupyter Notebook
- Required packages: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

## Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

## Running the Project

1. Clone or download the project files
2. Ensure the path for the dataset `Telco-Customer-Churn.csv` used correctly
3. Open the Jupyter notebook:
   ```bash
   jupyter notebook Tree-Based-Models-for-Churn-Analysis.ipynb
   ```
4. Run all cells sequentially to reproduce the analysis




