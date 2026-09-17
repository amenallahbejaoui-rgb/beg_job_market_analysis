from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "ds_salaries_clean.csv"


def regression_metrics(actual, predicted):
    return {
        "MAE_USD": mean_absolute_error(actual, predicted),
        "RMSE_USD": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted),
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {DATA_PATH}\n"
            "Run notebooks/01_data_cleaning.ipynb first."
        )

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows and {df.shape[1]} columns")

    target = "salary_in_usd"
    feature_columns = [
        "work_year",
        "experience_level",
        "employment_type",
        "job_title",
        "remote_ratio",
        "company_size",
        "employee_continent",
        "company_continent",
    ]
    categorical_features = [
        "experience_level",
        "employment_type",
        "job_title",
        "company_size",
        "employee_continent",
        "company_continent",
    ]
    numeric_features = ["work_year", "remote_ratio"]

    required_columns = feature_columns + [target]
    missing_columns = sorted(set(required_columns) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    model_df = df[required_columns].dropna().copy()
    X = model_df[feature_columns]
    y = model_df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")

    baseline = DummyRegressor(strategy="median")
    baseline.fit(X_train, y_train)
    baseline_predictions = baseline.predict(X_test)
    baseline_metrics = regression_metrics(y_test, baseline_predictions)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
            ("numeric", "passthrough", numeric_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=300,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    cross_validator = KFold(n_splits=5, shuffle=True, random_state=42)
    cross_validation = cross_validate(
        model,
        X,
        y,
        cv=cross_validator,
        scoring={
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
        },
        n_jobs=-1,
    )
    cross_validation_summary = pd.Series({
        "MAE_USD": -cross_validation["test_mae"].mean(),
        "RMSE_USD": -cross_validation["test_rmse"].mean(),
        "R2": cross_validation["test_r2"].mean(),
    }, name="5-fold cross-validation")
    print("\n5-fold cross-validation:")
    print(cross_validation_summary.round(2).to_string())

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    model_metrics = regression_metrics(y_test, predictions)

    metrics = pd.DataFrame(
        [baseline_metrics, model_metrics],
        index=["Baseline", "Random forest"],
    )
    print("\nEvaluation metrics:")
    print(metrics.round(2).to_string())

    results = X_test.copy()
    results["actual_salary_usd"] = y_test
    results["predicted_salary_usd"] = predictions
    results["absolute_error_usd"] = (
        results["actual_salary_usd"] - results["predicted_salary_usd"]
    ).abs()
    print("\nLargest prediction errors:")
    print(
        results.sort_values("absolute_error_usd", ascending=False)
        [["actual_salary_usd", "predicted_salary_usd", "absolute_error_usd"]]
        .head(10)
        .round(2)
        .to_string()
    )

    fitted_preprocessor = model.named_steps["preprocessor"]
    fitted_regressor = model.named_steps["regressor"]
    feature_names = fitted_preprocessor.get_feature_names_out()
    importance = pd.Series(
        fitted_regressor.feature_importances_,
        index=feature_names,
        name="importance",
    ).sort_values(ascending=False)
    print("\nTop model features:")
    print(importance.head(15).round(4).to_string())

    example_job = pd.DataFrame(
        [{
            "work_year": 2022,
            "experience_level": "SE",
            "employment_type": "FT",
            "job_title": "Data Scientist",
            "remote_ratio": 100,
            "company_size": "M",
            "employee_continent": "North America",
            "company_continent": "North America",
        }]
    )
    example_prediction = model.predict(example_job)[0]
    print(f"\nExample estimated salary: ${example_prediction:,.0f} per year")


if __name__ == "__main__":
    main()
