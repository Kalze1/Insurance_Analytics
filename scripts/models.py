# Import necessary libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Linear Regression
def build_linear_regression(X_train, y_train):
    """
    Build and train a Linear Regression model.
    """
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    return lr_model

# Decision Tree Regressor
def build_decision_tree(X_train, y_train, max_depth=None):
    """
    Build and train a Decision Tree Regressor model.
    """
    dt_model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    dt_model.fit(X_train, y_train)
    return dt_model

# Random Forest Regressor
def build_random_forest(X_train, y_train, n_estimators=100, max_depth=None):
    """
    Build and train a Random Forest Regressor model.
    """
    rf_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model

# XGBoost Regressor
def build_xgboost(X_train, y_train, n_estimators=100, learning_rate=0.1, max_depth=3):
    """
    Build and train an XGBoost Regressor model.
    """
    xgb_model = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42)
    xgb_model.fit(X_train, y_train)
    return xgb_model

# Evaluation Function
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using Mean Squared Error and R-squared score.
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2

# Main function to train all models and evaluate
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Train and evaluate Linear Regression, Decision Trees, Random Forests, and XGBoost models.
    Returns a dictionary with model names and their performance.
    """
    results = {}

    # # Train and evaluate Linear Regression
    lr_model = build_linear_regression(X_train, y_train)
    lr_mse, lr_r2 = evaluate_model(lr_model, X_test, y_test)
    results['Linear Regression'] = {'MSE': lr_mse, 'R2': lr_r2}

    # Train and evaluate Decision Tree
    dt_model = build_decision_tree(X_train, y_train)
    dt_mse, dt_r2 = evaluate_model(dt_model, X_test, y_test)
    results['Decision Tree'] = {'MSE': dt_mse, 'R2': dt_r2}

    # Train and evaluate Random Forest
    rf_model = build_random_forest(X_train, y_train)
    rf_mse, rf_r2 = evaluate_model(rf_model, X_test, y_test)
    results['Random Forest'] = {'MSE': rf_mse, 'R2': rf_r2}

    # Train and evaluate XGBoost
    xgb_model = build_xgboost(X_train, y_train)
    xgb_mse, xgb_r2 = evaluate_model(xgb_model, X_test, y_test)
    results['XGBoost'] = {'MSE': xgb_mse, 'R2': xgb_r2}

    return results

    
