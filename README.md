# 🚢 Titanic Survival Prediction

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CS5785](https://img.shields.io/badge/CS5785-Assignment%201-orange.svg)](https://github.com/beyzasimsekoglu)

> **CS5785 Assignment 1 - Part II**: Complete machine learning solution for predicting passenger survival on the Titanic using logistic regression and comprehensive written exercises.

## 🎯 Overview

This project implements a comprehensive machine learning solution for the [Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic) Kaggle competition. The solution features logistic regression with advanced feature engineering and includes complete solutions to all written exercises for CS5785.

## ✨ Key Features

- 🔍 **Comprehensive Data Exploration**: Analysis of survival rates by gender, class, and demographics
- 🛠️ **Advanced Feature Engineering**: Family size, title extraction, age groups, fare categorization
- 🤖 **Logistic Regression**: Implemented using scikit-learn with feature importance analysis
- 📊 **Model Evaluation**: Achieves 81% accuracy on test set
- 📚 **Written Exercises**: Complete solutions to conceptual questions and mathematical derivations
- 🏆 **Kaggle Ready**: Complete submission file for competition entry

## 📁 Project Structure

```
titanic-survival-project/
├── 📄 titanic_analysis.py              # Main survival analysis script
├── 📄 written_exercises.py             # Complete written exercises solutions
├── 🖼️ titanic_feature_importance.png   # Feature importance visualization
├── 📄 titanic_submission.csv           # Kaggle submission file
├── 📄 titanic_train.csv                # Training dataset
└── 📄 README.md                        # This file
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

### Running the Analysis

```bash
# Clone the repository
git clone https://github.com/beyzasimsekoglu/titanic-survival-analysis.git
cd titanic-survival-analysis

# Run Titanic survival analysis
python3 titanic_analysis.py

# Run written exercises
python3 written_exercises.py
```

## 📊 Results & Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | 81.0% |
| **Training Samples** | 712 |
| **Test Samples** | 179 |
| **Features Used** | 13 (engineered) |
| **Key Predictor** | Sex (most important) |

## 🔬 Technical Implementation

### Feature Engineering
- **Family Size**: SibSp + Parch + 1
- **Title Extraction**: Mr, Mrs, Miss, Master, Rare from passenger names
- **Age Groups**: Child, Teen, Adult, Middle, Senior
- **Fare Groups**: Low, Medium, High, VeryHigh
- **Cabin Indicator**: Binary feature for cabin availability

### Logistic Regression
```python
# Logistic regression with feature scaling
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)
```

### Data Preprocessing
- **Missing Values**: Median imputation by class and gender for age
- **Categorical Encoding**: Label encoding for all categorical variables
- **Feature Scaling**: StandardScaler for numerical stability

## 📈 Key Insights & Survival Patterns

### Demographics
- **👩 Female Survival**: 74% (233/314 passengers)
- **👨 Male Survival**: 19% (109/577 passengers)
- **Gender Gap**: 55 percentage point difference

### Social Class
- **🥇 First Class**: 63% survival rate
- **🥈 Second Class**: 47% survival rate  
- **🥉 Third Class**: 24% survival rate

### Family Dynamics
- **👨‍👩‍👧‍👦 Small Families**: 2-3 people had best survival rates
- **👤 Solo Travelers**: Lower survival rates
- **👶 Children**: Higher survival rates than adults

## 📚 Written Exercises Solutions

The `written_exercises.py` file contains complete solutions to:

### 1. Conceptual Questions
- **Gradient Descent vs Normal Equations**: Advantages and disadvantages
- **Discretization for Regression**: Why it's insufficient for continuous prediction
- **One-vs-All vs Multi-class**: Trade-offs in classification approaches
- **Polynomial Regression Complexity**: O(d^p) feature dimension analysis

### 2. Newton-Raphson for Logistic Regression
- **Log-likelihood Derivation**: Complete mathematical proof
- **Newton-Raphson Update**: Step-by-step derivation
- **Matrix Form**: Connection to iteratively reweighted least squares (IRLS)
- **Weighted Least Squares**: Mathematical relationship demonstration

### 3. Maximum Likelihood Estimation
- **Ball Game Problem**: Complete MLE solution
- **Parameter Estimation**: θ_R=0.333, θ_B=1.000, θ_G=1.000, θ_Y=0.000
- **Likelihood Analysis**: Why MLE optimization is principled
- **Potential Issues**: Small sample size and overfitting concerns

## 🎓 Academic Context

This project was developed for **CS5785 - Machine Learning** at Cornell University, demonstrating:
- Understanding of logistic regression theory
- Feature engineering and data preprocessing
- Mathematical derivations and proofs
- Model evaluation and interpretation
- Professional software development practices

## 📝 Assignment Requirements Met

- ✅ **Data Preprocessing**: Missing values, feature engineering, encoding
- ✅ **Logistic Regression**: Implementation and evaluation
- ✅ **Feature Justification**: Analysis of feature importance
- ✅ **Kaggle Submission**: Complete competition entry
- ✅ **Written Exercises**: All conceptual questions and derivations
- ✅ **Mathematical Proofs**: Complete Newton-Raphson and MLE derivations

## 🔬 Mathematical Highlights

### Logistic Regression Derivation
```
P(Y=1|X) = σ(θX) where σ(x) = 1/(1 + exp(-x))
ℓ(θ) = Σ[y^(i)θx^(i) - log(1 + exp(θx^(i)))]
```

### Newton-Raphson Update
```
θ^(t+1) = (X^T W X)^(-1) X^T W z
where W = diag(p(x^(i))(1-p(x^(i))))
```

### Maximum Likelihood Estimation
```
L(θ) = θ_R^2(1-θ_R) × θ_B^3 × θ_G × (1-θ_Y)
```

## 🤝 Contributing

Feel free to fork this repository and submit pull requests for improvements!

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Beyza Simsekoglu**  
*CS5785 Student*  
[GitHub](https://github.com/beyzasimsekoglu) | [LinkedIn](https://linkedin.com/in/beyzasimsekoglu)

---

⭐ **Star this repository if you found it helpful!**
