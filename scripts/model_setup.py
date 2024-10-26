from sklearn.linear_model import SGDRegressor, Lasso, Ridge, ElasticNet, BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor 
from lightgbm import LGBMRegressor

def create_model(model_type):
    """
    Create a model based on the given model type.
    """
    if model_type == 'sgd':
        return SGDRegressor(max_iter=5000, tol=1e-4, alpha=0.01, learning_rate='optimal')
    elif model_type == 'random_forest':
        return RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'decision_tree':
        return DecisionTreeRegressor(max_depth=10, min_samples_split=5, random_state=42)
    elif model_type == 'xgboost':
        return XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42)
    elif model_type == 'lasso':
        return Lasso(alpha=0.001)
    elif model_type == 'ridge':
        return Ridge(alpha=0.001)  # Default alpha for Ridge
    elif model_type == 'elasticnet':
        return ElasticNet(alpha=1.0, l1_ratio=0.5)  # Default values for ElasticNet
    elif model_type == 'svr':
        return SVR(kernel='rbf', C=1.0, gamma='scale')  # Default RBF kernel with SVR
    elif model_type == 'knn':
        return KNeighborsRegressor(n_neighbors=5)  # Default 5 nearest neighbors for KNN
    elif model_type == 'lightgbm':
        return LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42)
    elif model_type == 'bayesian_ridge':
        return BayesianRidge()  # Bayesian Ridge default model
    else:
        raise ValueError(f"Unknown model type: {model_type}")
