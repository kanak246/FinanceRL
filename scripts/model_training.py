import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, RandomizedSearchCV 
from scripts.model_setup import create_model
from sklearn.metrics import r2_score

def cross_validate_model(X_train, y_train, model_type='sgd', cv_folds=5):
    """
    Perform cross-validation on the given model and return the mean MSE and R-squared across folds.
    """
    model_cv = create_model(model_type)

    # Calculate MSE and R-squared via cross-validation
    cv_mse_scores = cross_val_score(model_cv, X_train, y_train, cv=cv_folds, scoring='neg_mean_squared_error')
    cv_r2_scores = cross_val_score(model_cv, X_train, y_train, cv=cv_folds, scoring='r2')

    # Return positive mean MSE and restricted R-squared
    mean_mse = -cv_mse_scores.mean()
    mean_r2 = max(0, min(1, cv_r2_scores.mean()))
    print(f"Cross-Validation MSE ({model_type}): {mean_mse}")
    print(f"Cross-Validation R-squared ({model_type}): {mean_r2}")
    
    return mean_mse, mean_r2

def tune_hyperparameters(X_train, y_train, model_type):
    """
    Tune hyperparameters for the given model type using RandomizedSearchCV.
    """
    if model_type == 'random_forest':
        param_dist = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': [None, 'sqrt', 'log2']
        }
    elif model_type == 'sgd':
        param_dist = {
            'alpha': [1e-4, 1e-3, 1e-2, 1e-1],
            'max_iter': [1000, 3000, 5000],
            'tol': [1e-3, 1e-4, 1e-5],
            'learning_rate': ['constant', 'optimal', 'adaptive']
        }
    elif model_type == 'decision_tree':
        param_dist = {
            'max_depth': [5, 10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': [None, 'sqrt', 'log2']
        }
    elif model_type == 'xgboost':
        param_dist = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 6, 9],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        }
    elif model_type == 'lasso':
        param_dist = {
            'alpha': [0.01, 0.1, 1, 10, 100]
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Get the model using create_model
    model = create_model(model_type)

    # Perform Randomized Search with 10 random combinations and 3-fold cross-validation
    random_search = RandomizedSearchCV(
        estimator=model, 
        param_distributions=param_dist, 
        n_iter=10,  # Limits the number of random combinations
        cv=3,       # 3-fold cross-validation
        scoring='neg_mean_squared_error', 
        verbose=2, 
        n_jobs=-1,  # Use all available cores
        random_state=42  # Set for reproducibility
    )
    random_search.fit(X_train, y_train)

    print(f"Best parameters for {model_type}: {random_search.best_params_}")
    print(f"Best CV MSE for {model_type}: {-random_search.best_score_}")

    # Return the best model, best parameters, and best score
    return random_search.best_estimator_, random_search.best_params_, -random_search.best_score_


def train_model(training_df, target_date, model_type, is_Treasury=False, is_incremental=False, tune_params=False):
    """
    Train the model (SGD, RandomForest, or DecisionTree) either from scratch or incrementally.
    Optional hyperparameter tuning before training.
    """
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')

    if is_Treasury:
        model_file = os.path.join(model_dir, f'{model_type}_treasury_yield_model.pkl')
        X_scaler_file = os.path.join(model_dir, f'X_scaler_treasury_{model_type}.pkl')
        y_scaler_file = os.path.join(model_dir, f'y_scaler_treasury_{model_type}.pkl')
    else:
        model_file = os.path.join(model_dir, f'{model_type}_bond_yield_model.pkl')
        X_scaler_file = os.path.join(model_dir, f'X_scaler_{model_type}.pkl')
        y_scaler_file = os.path.join(model_dir, f'y_scaler_{model_type}.pkl')

    # Feature and target extraction
    if is_Treasury:
        X_train = training_df[['time_to_maturity']]
    else:
        X_train = training_df[['bond_sym_id_encoded', 'total_daily_volume', 'time_to_maturity']]
    
    y_train = training_df['volume_weighted_yield'].values.reshape(-1, 1)

    # Initialize scalers or load existing ones
    if is_incremental and os.path.exists(model_file) and os.path.exists(X_scaler_file) and os.path.exists(y_scaler_file):
        model = joblib.load(model_file)
        X_scaler = joblib.load(X_scaler_file)
        y_scaler = joblib.load(y_scaler_file)
        print(f"Loaded existing {model_type} model and scalers for incremental training...")
    else:
        X_scaler = StandardScaler()
        y_scaler = StandardScaler()

    # Scale the features and target
    X_train_scaled = X_scaler.fit_transform(X_train)
    y_train_scaled = y_scaler.fit_transform(y_train)

    # If hyperparameter tuning is enabled, tune the model
    if tune_params:
        print(f"Tuning hyperparameters for {model_type} model...")
        model, best_params, best_score = tune_hyperparameters(X_train_scaled, y_train_scaled.ravel(), model_type)
        print(f"Best parameters: {best_params}")
        print(f"Best score (MSE): {best_score}")
    else:
        # Create the model if it's not already created
        model = create_model(model_type)

    # Train the model
    if hasattr(model, 'partial_fit'):
        model.partial_fit(X_train_scaled, y_train_scaled.ravel())
    else:
        model.fit(X_train_scaled, y_train_scaled.ravel())  # Use fit for models that don't support partial_fit

    # Save the scalers and model
    joblib.dump(X_scaler, X_scaler_file)
    joblib.dump(y_scaler, y_scaler_file)
    joblib.dump(model, model_file)

    print(f"Model ({model_type}) trained and saved to {model_file}.")

    return model, X_scaler, y_scaler
