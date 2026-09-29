import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. LOAD THE DATASET
df = pd.read_csv("data/bodyfat.csv")

print("First 5 rows:")
print(df.head())
print("\nDataset info:")
print(df.info())
print("\nMissing values:")
print(df.isnull().sum())


# 2. CLEAN / PREPARE THE DATA
# Remove Density because it directly relates to BodyFat
if "Density" in df.columns:
    df = df.drop(columns=["Density"])

# Make sure the target column name matches your dataset exactly
target_column = "BodyFat"

# Separate features and target
X = df.drop(columns=[target_column])
y = df[target_column]

print("\nFeature columns:")
print(X.columns.tolist())


# 3. TRAIN / TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 4. CREATE MODELS
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}


# 5. TRAIN AND EVALUATE MODELS
results = {}
best_model = None
best_model_name = None
best_r2 = float("-inf")

for name, model in models.items():
    print(f"\n--- {name} ---")

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    preds = model.predict(X_test)

    # Evaluate
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, preds)

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    cv_mean = cv_scores.mean()

    results[name] = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "CV_R2_Mean": cv_mean
    }

    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²: {r2:.3f}")
    print(f"Mean CV R²: {cv_mean:.3f}")

    # Track best model
    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_model_name = name


# 6. SAVE THE BEST MODEL
joblib.dump(best_model, "models/bodyfat_model.pkl")

print(f"\nBest model: {best_model_name}")
print("Saved as models/bodyfat_model.pkl")


# 7. SHOW FINAL RESULTS
print("\nFinal results summary:")
for model_name, metrics in results.items():
    print(f"\n{model_name}")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.3f}")