# Titanic Survival Prediction

## CS5785 Assignment 1 - Part II

This project implements logistic regression for predicting passenger survival on the Titanic using machine learning techniques.

## Features

- **Data Exploration**: Comprehensive analysis of survival rates by gender, class, and other factors
- **Feature Engineering**: Family size, title extraction, age groups, fare groups
- **Logistic Regression**: Implemented using scikit-learn
- **Model Evaluation**: Achieves 81% accuracy on test set
- **Written Exercises**: Complete solutions to conceptual questions and mathematical derivations

## Files

- `titanic_analysis.py` - Complete survival analysis script
- `titanic_feature_importance.png` - Feature importance visualization
- `titanic_submission.csv` - Kaggle submission file
- `titanic_train.csv` - Training data
- `written_exercises.py` - Solutions to written exercises

## Usage

```bash
# Run Titanic analysis
python3 titanic_analysis.py

# Run written exercises
python3 written_exercises.py
```

## Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

## Results

- **Model Performance**: 81% accuracy on test set
- **Key Features**: Sex (most important), Pclass, Age, HasCabin
- **Survival Rates**: Female (74%), Male (19%), Class 1 (63%), Class 3 (24%)
- **Feature Engineering**: 13 engineered features including family size and title extraction

## Key Insights

1. **Gender**: Female passengers had 74% survival rate vs 19% for males
2. **Class**: First class had 63% survival rate vs 24% for third class
3. **Age**: Children had higher survival rates
4. **Family Size**: Small families (2-3 people) had better survival rates
5. **Cabin**: Passengers with cabin information had higher survival rates

## Written Exercises

The `written_exercises.py` file contains complete solutions to:

- **Conceptual Questions**: Gradient descent vs normal equations, discretization, one-vs-all classification, polynomial regression complexity
- **Newton-Raphson**: Complete derivation for logistic regression with matrix form
- **Maximum Likelihood**: Ball game probability estimation with MLE

## Mathematical Derivation

The project includes complete mathematical derivations for:
- Log-likelihood function for logistic regression
- Newton-Raphson update formula
- Connection to iteratively reweighted least squares (IRLS)
- Maximum likelihood estimation for categorical data
