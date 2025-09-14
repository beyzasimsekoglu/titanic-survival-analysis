#!/usr/bin/env python3
"""
CS5785 Assignment 1 - Titanic Survival Analysis
Part II: The Titanic Disaster - Machine Learning from Disaster

This script implements:
1. Data preprocessing for Titanic dataset
2. Logistic regression for survival prediction
3. Model evaluation and Kaggle submission
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class TitanicAnalysis:
    def __init__(self, train_path='titanic_train.csv'):
        """Initialize the Titanic analysis class."""
        self.train_df = pd.read_csv(train_path)
        print(f"Titanic dataset loaded: {self.train_df.shape}")
        print(f"Columns: {list(self.train_df.columns)}")
        
    def explore_data(self):
        """Explore the Titanic dataset."""
        print("\n" + "=" * 60)
        print("TITANIC DATASET EXPLORATION")
        print("=" * 60)
        
        print("Dataset Info:")
        print(f"Shape: {self.train_df.shape}")
        print(f"Columns: {list(self.train_df.columns)}")
        
        print("\nMissing values:")
        missing = self.train_df.isnull().sum()
        print(missing[missing > 0])
        
        print("\nSurvival rate:")
        survival_rate = self.train_df['Survived'].mean()
        print(f"Overall survival rate: {survival_rate:.3f}")
        
        print("\nSurvival by gender:")
        gender_survival = self.train_df.groupby('Sex')['Survived'].agg(['count', 'sum', 'mean'])
        print(gender_survival)
        
        print("\nSurvival by passenger class:")
        class_survival = self.train_df.groupby('Pclass')['Survived'].agg(['count', 'sum', 'mean'])
        print(class_survival)
        
        return self.train_df
    
    def preprocess_data(self):
        """Preprocess the Titanic data for modeling."""
        print("\n" + "=" * 60)
        print("TITANIC DATA PREPROCESSING")
        print("=" * 60)
        
        df = self.train_df.copy()
        
        print("Preprocessing steps:")
        print("1. Handling missing values...")
        
        # Handle missing values
        # Age: fill with median age by passenger class and gender
        df['Age'].fillna(df.groupby(['Pclass', 'Sex'])['Age'].transform('median'), inplace=True)
        
        # Embarked: fill with mode
        df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
        
        # Cabin: create a binary feature indicating if cabin is known
        df['HasCabin'] = df['Cabin'].notna().astype(int)
        df.drop('Cabin', axis=1, inplace=True)
        
        print("2. Feature engineering...")
        
        # Create family size feature
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        
        # Create is alone feature
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        
        # Extract title from name
        df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        df['Title'] = df['Title'].replace('Mlle', 'Miss')
        df['Title'] = df['Title'].replace('Ms', 'Miss')
        df['Title'] = df['Title'].replace('Mme', 'Mrs')
        
        # Age groups
        df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
        
        # Fare groups
        df['FareGroup'] = pd.cut(df['Fare'], bins=[0, 7.91, 14.45, 31, 1000], labels=['Low', 'Medium', 'High', 'VeryHigh'])
        
        print("3. Encoding categorical variables...")
        
        # Encode categorical variables
        categorical_features = ['Sex', 'Embarked', 'Title', 'AgeGroup', 'FareGroup']
        
        for feature in categorical_features:
            if feature in df.columns:
                le = LabelEncoder()
                df[feature] = le.fit_transform(df[feature].astype(str))
        
        # Select features for modeling
        feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 
                       'Embarked', 'HasCabin', 'FamilySize', 'IsAlone', 
                       'Title', 'AgeGroup', 'FareGroup']
        
        # Remove any features that don't exist
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        X = df[feature_cols]
        y = df['Survived']
        
        print(f"Selected features: {feature_cols}")
        print(f"Feature matrix shape: {X.shape}")
        
        self.X = X
        self.y = y
        self.feature_cols = feature_cols
        self.processed_df = df
        
        return X, y, feature_cols
    
    def train_logistic_regression(self):
        """Train logistic regression model."""
        print("\n" + "=" * 60)
        print("LOGISTIC REGRESSION TRAINING")
        print("=" * 60)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train logistic regression
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred_train = lr_model.predict(X_train_scaled)
        y_pred_test = lr_model.predict(X_test_scaled)
        
        # Evaluate model
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        print(f"Training accuracy: {train_accuracy:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_test))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred_test))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'coefficient': lr_model.coef_[0]
        }).sort_values('coefficient', key=abs, ascending=False)
        
        print("\nFeature Importance (by coefficient magnitude):")
        print(feature_importance)
        
        # Visualize feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance, x='coefficient', y='feature')
        plt.title('Logistic Regression Feature Importance')
        plt.xlabel('Coefficient Value')
        plt.tight_layout()
        plt.savefig('titanic_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        self.model = lr_model
        self.scaler = scaler
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        return lr_model, scaler
    
    def create_submission(self, test_path=None):
        """Create submission file for Kaggle."""
        print("\n" + "=" * 60)
        print("CREATING TITANIC SUBMISSION")
        print("=" * 60)
        
        if test_path is None:
            print("No test file provided. Creating sample predictions on training data...")
            # Use a subset of training data as "test" data
            test_df = self.processed_df.sample(n=100, random_state=42)
        else:
            test_df = pd.read_csv(test_path)
            # Apply same preprocessing
            test_df = self._preprocess_test_data(test_df)
        
        # Prepare features
        X_test = test_df[self.feature_cols]
        X_test_scaled = self.scaler.transform(X_test)
        
        # Make predictions
        predictions = self.model.predict(X_test_scaled)
        probabilities = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Create submission
        submission = pd.DataFrame({
            'PassengerId': test_df['PassengerId'] if 'PassengerId' in test_df.columns else range(len(predictions)),
            'Survived': predictions
        })
        
        submission.to_csv('titanic_submission.csv', index=False)
        
        print(f"Submission file created: titanic_submission.csv")
        print(f"Predictions: {predictions.sum()} survived out of {len(predictions)} passengers")
        print(f"Survival rate: {predictions.mean():.3f}")
        
        return submission
    
    def _preprocess_test_data(self, test_df):
        """Apply same preprocessing to test data."""
        df = test_df.copy()
        
        # Handle missing values (same as training)
        df['Age'].fillna(df.groupby(['Pclass', 'Sex'])['Age'].transform('median'), inplace=True)
        df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
        df['Fare'].fillna(df['Fare'].median(), inplace=True)
        
        # Feature engineering (same as training)
        df['HasCabin'] = df['Cabin'].notna().astype(int)
        df.drop('Cabin', axis=1, inplace=True)
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        
        # Extract title
        df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        df['Title'] = df['Title'].replace('Mlle', 'Miss')
        df['Title'] = df['Title'].replace('Ms', 'Miss')
        df['Title'] = df['Title'].replace('Mme', 'Mrs')
        
        # Age and fare groups
        df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
        df['FareGroup'] = pd.cut(df['Fare'], bins=[0, 7.91, 14.45, 31, 1000], labels=['Low', 'Medium', 'High', 'VeryHigh'])
        
        # Encode categorical variables
        categorical_features = ['Sex', 'Embarked', 'Title', 'AgeGroup', 'FareGroup']
        
        for feature in categorical_features:
            if feature in df.columns:
                le = LabelEncoder()
                df[feature] = le.fit_transform(df[feature].astype(str))
        
        return df
    
    def run_complete_analysis(self):
        """Run the complete Titanic analysis."""
        print("CS5785 Assignment 1 - Titanic Survival Analysis")
        print("=" * 60)
        
        # Step 1: Explore data
        self.explore_data()
        
        # Step 2: Preprocess data
        X, y, feature_cols = self.preprocess_data()
        
        # Step 3: Train logistic regression
        model, scaler = self.train_logistic_regression()
        
        # Step 4: Create submission
        submission = self.create_submission()
        
        print("\n" + "=" * 60)
        print("TITANIC ANALYSIS COMPLETE!")
        print("=" * 60)
        print("Files generated:")
        print("  - titanic_feature_importance.png")
        print("  - titanic_submission.csv")
        
        return {
            'model': model,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'submission': submission
        }

if __name__ == "__main__":
    # Run the complete analysis
    analyzer = TitanicAnalysis()
    results = analyzer.run_complete_analysis()
