import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from joblib import dump

logging.basicConfig(level=logging.INFO)

def load_and_clean_data(path: str, target_column: str):
    data = pd.read_csv(path)
    data = handle_missing_data(data)
    target_column_actual = next((col for col in data.columns if col.lower() == target_column.lower()), None)

    if not target_column_actual:
        raise ValueError(f"Column '{target_column}' not found in the dataset.")

    X = data.drop(target_column_actual, axis=1)
    y = data[target_column_actual]
    logging.info(f"Data loaded with {len(X)} rows and {X.shape[1]} features.")
    return train_test_split(X, y, test_size=0.2, random_state=42)

def handle_missing_data(data):
    strategy_map = {'mean': 'mean', 'median': 'median', 'mode': 'most_frequent', 'drop': 'drop'}
    strategy = input("Choose imputation strategy (mean, median, mode, drop): ").lower()
    if strategy == "drop":
        return data.dropna()
    elif strategy in strategy_map:
        imputer = SimpleImputer(strategy=strategy_map[strategy])
        return pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    else:
        logging.warning("Invalid strategy. Using default 'drop'.")
        return data.dropna()

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test)

def choose_model(model_type):
    models = {
        'linear': LinearRegression(),
        'ridge': Ridge(),
        'lasso': Lasso(),
        'polynomial': make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
        'logistic': LogisticRegression(max_iter=10000)
    }
    return models.get(model_type)

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    if issubclass(type(model), (LinearRegression, Ridge, Lasso)):
        accuracy = r2_score(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        return accuracy, mse, 'regression'
    else:
        accuracy = accuracy_score(y_test, predictions.round())
        mse = 'Not applicable for classification'
        return accuracy, mse, 'classification'

def plot_predictions(y_true, y_pred, model_type, task_type):
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'True vs Predicted Values for {model_type.capitalize()} Model')
    plt.show(block=False)
    plt.pause(3)
    plt.close()

def prepare_for_logistic(y_train, y_test):
    bins = np.linspace(0, y_train.max(), 5)
    y_train_binned = np.digitize(y_train, bins) - 1
    y_test_binned = np.digitize(y_test, bins) - 1
    return y_train_binned, y_test_binned

def main():
    path = input("Enter the path to your CSV dataset: ")
    target_column = input("Please enter the target column name: ")
    X_train, X_test, y_train, y_test = load_and_clean_data(path, target_column)
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    continue_testing = 'yes'
    while continue_testing.lower() == 'yes':
        model_type = input("Choose a model type (linear, polynomial, ridge, lasso, logistic): ").lower()
        model = choose_model(model_type)
        
        if not model:
            logging.error("Unknown model type.")
            continue
        
        if model_type == "logistic":
            y_train, y_test = prepare_for_logistic(y_train, y_test)
            cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        else:
            cv_strategy = 3  # Use default K-Fold for regression tasks
        
        scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_strategy)
        logging.info(f"Cross-validation scores: {scores}")
        logging.info(f"Average score: {scores.mean():.2f}")
        model.fit(X_train_scaled, y_train)

        accuracy, mse, task_type = evaluate_model(model, X_test_scaled, y_test)

        if task_type == 'regression':
            rmse = np.sqrt(mse)
            logging.info(f"R2 Score for {model_type.capitalize()} Model: {accuracy:.2f}")
            logging.info(f"Root Mean Squared Error for {model_type.capitalize()} Model: {rmse}")
        else:
            logging.info(f"Accuracy for {model_type.capitalize()} Model: {accuracy*100:.2f}%")

        plot_predictions(y_test, model.predict(X_test_scaled), model_type, task_type)

        save_option = input("Would you like to save the trained model? (yes/no): ").lower()
        if save_option == "yes":
            dump(model, f"{model_type}_model.pkl")
            logging.info(f"Model saved as {model_type}_model.pkl")

        continue_testing = input("Would you like to test another model? (yes/no): ")

if __name__ == "__main__":
    main()



"""
# MLModelEvaluator

## Description
MLModelEvaluator is a Python-based tool designed to simplify the process of training and evaluating regression models. Users can easily load datasets, handle missing values, choose from multiple regression models, and get insights into model performance via cross-validation, R2 score, and mean squared error. Additionally, this tool provides a visualization of true vs predicted values, offering a clearer understanding of model behavior. Once satisfied, users can save their trained models for future predictions or deployment.

## Features
- **Data Loading & Preprocessing:** Streamlined data loading from CSV, with an interactive method to handle missing values.
- **Model Selection:** Users can choose from Linear Regression, Ridge, Lasso, Polynomial Regression, or Logistic Regression.
- **Evaluation Metrics:** Evaluate model performance using cross-validation, R2 score, and mean squared error.
- **Visualization:** Plot true vs. predicted values to visually assess model accuracy.
- **Model Saving:** Option to save the trained model in a .pkl format for future use.

## Usage
1. Clone this repository.
2. Run `MLModelEvaluator.py`.
3. Follow the interactive prompts to load data, preprocess, select a model, and evaluate.
4. Optionally, save the model for future use.

## Dependencies
- pandas
- numpy
- scikit-learn
- matplotlib

## Contribution
Feel free to fork this repository and enhance the tool! Contributions, feedback, and issues are always welcome.
"""
