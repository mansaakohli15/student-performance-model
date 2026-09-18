"""
Model training, cross-validation benchmarking, and hyperparameter tuning module.
"""

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_validate, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    ExtraTreesRegressor,
    VotingRegressor,
)
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error


class ModelTrainer:
    """Manages model evaluation, benchmarking, and hyperparameter optimization."""

    def __init__(self, preprocessor, cv_folds: int = 5, random_state: int = 42):
        self.preprocessor = preprocessor
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        self.candidate_models = self._initialize_model_zoo()

    def _initialize_model_zoo(self) -> Dict[str, Any]:
        """Defines candidate regression algorithms."""
        return {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=1.0, random_state=self.random_state),
            "Lasso Regression": Lasso(alpha=0.05, random_state=self.random_state),
            "ElasticNet": ElasticNet(alpha=0.05, l1_ratio=0.5, random_state=self.random_state),
            "Huber Regressor": HuberRegressor(max_iter=200),
            "Random Forest": RandomForestRegressor(
                n_estimators=100, max_depth=6, random_state=self.random_state
            ),
            "Gradient Boosting": GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.05, max_depth=3, random_state=self.random_state
            ),
            "HistGradientBoosting": HistGradientBoostingRegressor(
                max_iter=100, learning_rate=0.05, max_depth=3, random_state=self.random_state
            ),
            "Extra Trees": ExtraTreesRegressor(
                n_estimators=100, max_depth=6, random_state=self.random_state
            ),
        }

    def benchmark_models(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series
    ) -> pd.DataFrame:
        """
        Runs 5-fold cross-validation across all candidate models and evaluates on holdout test set.
        """
        results = []
        scoring = {
            "r2": "r2",
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
        }

        for name, regressor in self.candidate_models.items():
            print(f"  -> Cross-validating {name}...", flush=True)
            pipe = Pipeline([("prep", self.preprocessor), ("reg", regressor)])

            cv_res = cross_validate(
                pipe, X_train, y_train, cv=self.cv, scoring=scoring, n_jobs=None
            )

            # Fit on full training set and evaluate on holdout test set
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            test_r2 = r2_score(y_test, y_pred)
            test_rmse = root_mean_squared_error(y_test, y_pred)
            test_mae = mean_absolute_error(y_test, y_pred)

            results.append(
                {
                    "Model": name,
                    "CV_R2_Mean": cv_res["test_r2"].mean(),
                    "CV_R2_Std": cv_res["test_r2"].std(),
                    "CV_RMSE_Mean": -cv_res["test_rmse"].mean(),
                    "CV_MAE_Mean": -cv_res["test_mae"].mean(),
                    "Test_R2": test_r2,
                    "Test_RMSE": test_rmse,
                    "Test_MAE": test_mae,
                    "Fit_Time_Sec": cv_res["fit_time"].mean(),
                }
            )

        # Voting Ensemble
        print("  -> Cross-validating Voting Ensemble (Ridge + GBDT)...", flush=True)
        ensemble = VotingRegressor(
            estimators=[
                ("ridge", Ridge(alpha=1.0, random_state=self.random_state)),
                (
                    "gbr",
                    GradientBoostingRegressor(
                        n_estimators=100,
                        learning_rate=0.05,
                        max_depth=3,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )
        ens_pipe = Pipeline([("prep", self.preprocessor), ("reg", ensemble)])
        ens_cv = cross_validate(ens_pipe, X_train, y_train, cv=self.cv, scoring=scoring, n_jobs=None)
        ens_pipe.fit(X_train, y_train)
        y_ens_pred = ens_pipe.predict(X_test)

        results.append(
            {
                "Model": "Voting Ensemble (Ridge + GBDT)",
                "CV_R2_Mean": ens_cv["test_r2"].mean(),
                "CV_R2_Std": ens_cv["test_r2"].std(),
                "CV_RMSE_Mean": -ens_cv["test_rmse"].mean(),
                "CV_MAE_Mean": -ens_cv["test_mae"].mean(),
                "Test_R2": r2_score(y_test, y_ens_pred),
                "Test_RMSE": root_mean_squared_error(y_test, y_ens_pred),
                "Test_MAE": mean_absolute_error(y_test, y_ens_pred),
                "Fit_Time_Sec": ens_cv["fit_time"].mean(),
            }
        )

        benchmark_df = pd.DataFrame(results).sort_values(by="CV_R2_Mean", ascending=False)
        return benchmark_df

    def tune_best_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_type: str = "ridge",
    ) -> Tuple[Pipeline, Dict[str, Any]]:
        """
        Performs systematic grid search cross-validation for hyperparameter optimization.
        """
        if model_type == "ridge":
            pipeline = Pipeline(
                [("prep", self.preprocessor), ("reg", Ridge(random_state=self.random_state))]
            )
            param_grid = {
                "reg__alpha": [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
                "reg__solver": ["auto", "svd", "cholesky", "lsqr"],
            }
        elif model_type == "gradient_boosting":
            pipeline = Pipeline(
                [
                    ("prep", self.preprocessor),
                    ("reg", GradientBoostingRegressor(random_state=self.random_state)),
                ]
            )
            param_grid = {
                "reg__n_estimators": [50, 100, 150],
                "reg__learning_rate": [0.03, 0.05, 0.1],
                "reg__max_depth": [2, 3, 4],
            }
        elif model_type == "random_forest":
            pipeline = Pipeline(
                [
                    ("prep", self.preprocessor),
                    ("reg", RandomForestRegressor(random_state=self.random_state)),
                ]
            )
            param_grid = {
                "reg__n_estimators": [50, 100],
                "reg__max_depth": [4, 6, 8],
            }
        else:
            raise ValueError(f"Unsupported model_type for tuning: {model_type}")

        grid_search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=self.cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=None,
            return_train_score=True,
        )

        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_, grid_search.best_params_
