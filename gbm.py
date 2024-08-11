from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV
from model import *

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [-1, 10, 20, 30],
    'min_child_samples': [10, 20, 30],
    'feature_fraction': [0.7, 0.8, 0.9],
    'bagging_fraction': [0.7, 0.8, 0.9],
    'lambda_l1': [0, 0.1, 0.5],
    'lambda_l2': [0, 0.1, 0.5]
}

# Initialize model
model = LGBMRegressor()

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=50, cv=3, random_state=42, n_jobs=-1)

# Fit model
random_search.fit(X_train, y_train)

# Best parameters
print("Best Parameters:", random_search.best_params_)

# Best model
best_model = random_search.best_estimator_

# Make predictions and evaluate the model
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error with Optimized LightGBM: {mse}")
print(f"R^2 Score with Optimized LightGBM: {r2}")

# Export the trained model
model_filename = 'flight_delay_prediction_model_lightgbm_optimized.pkl'
joblib.dump(best_model, model_filename)
print(f"Model saved as {model_filename}")
