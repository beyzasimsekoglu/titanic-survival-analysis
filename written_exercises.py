#!/usr/bin/env python3
"""
CS5785 Assignment 1 - Written Exercises Solutions
Part 2: Written Exercises

This script contains solutions to the written exercises including:
1. Conceptual questions
2. Newton-Raphson for logistic regression
3. Maximum Likelihood Estimation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas as pd

class WrittenExercises:
    def __init__(self):
        """Initialize the written exercises class."""
        pass
    
    def conceptual_questions(self):
        """Solutions to conceptual questions."""
        print("=" * 80)
        print("CONCEPTUAL QUESTIONS SOLUTIONS")
        print("=" * 80)
        
        print("\n1.(a) Gradient Descent vs Analytical Formula (Normal Equations)")
        print("-" * 60)
        print("ADVANTAGE of Gradient Descent:")
        print("• Scalability: Works efficiently with large datasets (n >> p)")
        print("• Memory efficiency: Processes data in batches, doesn't need to store X^T X")
        print("• Online learning: Can update parameters as new data arrives")
        print("• Works with non-invertible matrices (when X^T X is singular)")
        
        print("\nDISADVANTAGE of Gradient Descent:")
        print("• Convergence: May not reach exact optimal solution, only approximate")
        print("• Hyperparameter tuning: Requires careful selection of learning rate")
        print("• Iterative: Takes multiple steps vs single matrix computation")
        print("• Local minima: Can get stuck in local minima (though not in convex OLS)")
        
        print("\n1.(b) Discretizing Output Space for Regression")
        print("-" * 60)
        print("NO, this approach would NOT be sufficient to solve the regression problem because:")
        print("• Information Loss: Discretization loses the continuous nature of regression")
        print("• Ordering Loss: Intervals don't preserve the ordering relationship between values")
        print("• Precision: Limited by interval size - can't predict exact values")
        print("• Interpolation: Can't predict values between interval boundaries")
        print("• Regression Goal: We want to predict continuous values, not categories")
        
        print("\n1.(c) One-vs-All vs Multi-class Classification")
        print("-" * 60)
        print("ADVANTAGE of One-vs-All:")
        print("• Simplicity: Can use any binary classifier (SVM, logistic regression)")
        print("• Interpretability: Each classifier is easier to understand")
        print("• Scalability: Can add new classes without retraining all classifiers")
        
        print("\nDISADVANTAGE of One-vs-All:")
        print("• Class Imbalance: Creates imbalanced datasets (one class vs all others)")
        print("• Inconsistent Predictions: May not sum to 1, requires normalization")
        print("• Training Time: Need to train k separate classifiers")
        print("• Decision Boundaries: May not be optimal for multi-class problems")
        
        print("\n1.(d) Polynomial Regression Complexity")
        print("-" * 60)
        print("i. Dimension of polynomial features:")
        print("   For d variables and degree p: O(d^p)")
        print("   This is the number of ways to choose p items from d variables with repetition")
        
        print("\nii. Computational complexity of polynomial least squares:")
        print("   O(d^p * n) where n is the number of data points")
        print("   This is because we need to compute X^T X which is (d^p) × (d^p)")
        
        print("\niii. Real-world implications:")
        print("   • Curse of Dimensionality: Exponential growth in features")
        print("   • Overfitting: High-dimensional feature space leads to overfitting")
        print("   • Computational Cost: Becomes prohibitively expensive for high p or d")
        print("   • Regularization: Essential to prevent overfitting")
        print("   • Feature Selection: Important to reduce dimensionality")
    
    def newton_raphson_logistic(self):
        """Solutions to Newton-Raphson for logistic regression."""
        print("\n" + "=" * 80)
        print("NEWTON-RAPHSON FOR LOGISTIC REGRESSION")
        print("=" * 80)
        
        print("\n2.(a) Log-likelihood derivation:")
        print("-" * 60)
        print("Given: P(Y=1|X) = σ(θX) where σ(x) = 1/(1 + exp(-x))")
        print("Likelihood: L(θ) = ∏[i=1 to n] p(x^(i))^(y^(i)) (1-p(x^(i)))^(1-y^(i))")
        print("where p(x^(i)) = σ(θx^(i))")
        print("\nTaking log:")
        print("ℓ(θ) = Σ[i=1 to n] [y^(i) log(p(x^(i))) + (1-y^(i)) log(1-p(x^(i)))]")
        print("\nUsing σ'(x) = σ(x)(1-σ(x)) and the hint:")
        print("ℓ(θ) = Σ[i=1 to n] [y^(i)θx^(i) - log(1 + exp(θx^(i)))]")
        
        print("\n2.(b) Newton-Raphson update:")
        print("-" * 60)
        print("f(θ) = ∂ℓ(θ)/∂θ = Σ[i=1 to n] x^(i)(y^(i) - p(x^(i))) = 0")
        print("f'(θ) = ∂f(θ)/∂θ = -Σ[i=1 to n] (x^(i))^2 p(x^(i))(1-p(x^(i)))")
        print("\nNewton-Raphson update:")
        print("θ^(t+1) = θ^t - f(θ^t)/f'(θ^t)")
        
        print("\n2.(c) Matrix form and weighted least squares:")
        print("-" * 60)
        print("In matrix form: θ^(t+1) = (X^T W X)^(-1) X^T W z")
        print("where:")
        print("• X = [x^(1), x^(2), ..., x^(n)]^T (feature matrix)")
        print("• W = diag(w^(1), w^(2), ..., w^(n)) (weight matrix)")
        print("• w^(i) = p(x^(i))(1-p(x^(i))) (weights)")
        print("• z = [z^(1), z^(2), ..., z^(n)]^T (working response)")
        print("• z^(i) = θ^t x^(i) + (y^(i) - p(x^(i)))/w^(i)")
        print("\nThis shows that logistic regression can be solved using")
        print("iteratively reweighted least squares (IRLS)!")
    
    def maximum_likelihood_estimation(self):
        """Solutions to Maximum Likelihood Estimation problem."""
        print("\n" + "=" * 80)
        print("MAXIMUM LIKELIHOOD ESTIMATION")
        print("=" * 80)
        
        # Given dataset
        data = [
            ('R', 'goal'), ('B', 'goal'), ('R', 'no_goal'), ('R', 'no_goal'),
            ('B', 'goal'), ('B', 'goal'), ('Y', 'no_goal'), ('G', 'goal')
        ]
        
        print("\nGiven dataset:")
        for i, (ball, outcome) in enumerate(data, 1):
            print(f"  {i}. Ball: {ball}, Outcome: {outcome}")
        
        print("\n3.(a) Probabilistic Model:")
        print("-" * 60)
        print("Model: P(Y=goal|X=ball_color) = θ_ball")
        print("Parameters: θ_R, θ_B, θ_G, θ_Y (probabilities for each ball color)")
        print("Constraint: 0 ≤ θ_ball ≤ 1 for all ball colors")
        print("This is a Bernoulli distribution for each ball color.")
        
        print("\n3.(b) Log-likelihood formula:")
        print("-" * 60)
        print("L(θ) = ∏[i=1 to n] P(y^(i)|x^(i))")
        print("     = θ_R^2 (1-θ_R)^1 × θ_B^3 (1-θ_B)^0 × θ_G^1 (1-θ_G)^0 × θ_Y^0 (1-θ_Y)^1")
        print("     = θ_R^2 (1-θ_R) × θ_B^3 × θ_G × (1-θ_Y)")
        print("\nℓ(θ) = 2log(θ_R) + log(1-θ_R) + 3log(θ_B) + log(θ_G) + log(1-θ_Y)")
        print("\nWe want to maximize this because:")
        print("• Higher likelihood means better fit to observed data")
        print("• MLE finds parameters that make observed data most probable")
        print("• It's a principled way to estimate parameters from data")
        
        print("\n3.(c) Maximum Likelihood Estimates:")
        print("-" * 60)
        
        # Calculate MLEs
        # For each ball color, MLE is (number of goals) / (total attempts)
        ball_counts = {}
        goal_counts = {}
        
        for ball, outcome in data:
            ball_counts[ball] = ball_counts.get(ball, 0) + 1
            if outcome == 'goal':
                goal_counts[ball] = goal_counts.get(ball, 0) + 1
        
        print("MLE calculations:")
        for ball in ['R', 'B', 'G', 'Y']:
            if ball in ball_counts:
                mle = goal_counts.get(ball, 0) / ball_counts[ball]
                print(f"  θ_{ball} = {goal_counts.get(ball, 0)}/{ball_counts[ball]} = {mle:.3f}")
            else:
                print(f"  θ_{ball} = 0/0 = undefined (no data)")
        
        print("\nInterpretation:")
        print("• Red balls: 2 goals out of 3 attempts = 66.7% success rate")
        print("• Blue balls: 3 goals out of 3 attempts = 100% success rate")
        print("• Green balls: 1 goal out of 1 attempt = 100% success rate")
        print("• Yellow balls: 0 goals out of 1 attempt = 0% success rate")
        
        print("\n3.(d) Potential inaccuracies:")
        print("-" * 60)
        print("This approach might yield inaccurate estimates when:")
        print("• Small sample size: Only 8 total observations")
        print("• Sparse data: Some ball colors have very few observations")
        print("• Overfitting: MLE can overfit to small samples")
        print("• No regularization: Doesn't account for prior knowledge")
        print("• Extreme estimates: 100% and 0% probabilities are unrealistic")
        print("\nExample: θ_B = 1.0 suggests blue balls ALWAYS score, which is")
        print("unlikely to be true in reality. We need more data or regularization.")
    
    def demonstrate_ols_derivation(self):
        """Demonstrate OLS derivation with code."""
        print("\n" + "=" * 80)
        print("OLS DERIVATION DEMONSTRATION")
        print("=" * 80)
        
        # Generate sample data
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 2)
        true_theta = np.array([2.5, -1.3, 0.8])  # [intercept, x1, x2]
        X_with_bias = np.column_stack([np.ones(n), X])
        y = X_with_bias @ true_theta + 0.1 * np.random.randn(n)
        
        print("Sample data generated:")
        print(f"  n = {n} samples")
        print(f"  p = 2 features")
        print(f"  True parameters: {true_theta}")
        
        # OLS solution
        XTX = X_with_bias.T @ X_with_bias
        XTX_inv = np.linalg.inv(XTX)
        theta_ols = XTX_inv @ X_with_bias.T @ y
        
        print(f"\nOLS solution: {theta_ols}")
        print(f"Error from true: {np.linalg.norm(theta_ols - true_theta):.6f}")
        
        # Demonstrate the normal equations
        print("\nNormal equations verification:")
        print("X^T X θ = X^T y")
        print(f"LHS: X^T X θ = {XTX @ theta_ols}")
        print(f"RHS: X^T y = {X_with_bias.T @ y}")
        print(f"Difference: {np.linalg.norm(XTX @ theta_ols - X_with_bias.T @ y):.10f}")
    
    def run_all_exercises(self):
        """Run all written exercises."""
        print("CS5785 Assignment 1 - Written Exercises Solutions")
        print("=" * 80)
        
        self.conceptual_questions()
        self.newton_raphson_logistic()
        self.maximum_likelihood_estimation()
        self.demonstrate_ols_derivation()
        
        print("\n" + "=" * 80)
        print("ALL WRITTEN EXERCISES COMPLETE!")
        print("=" * 80)

if __name__ == "__main__":
    exercises = WrittenExercises()
    exercises.run_all_exercises()
