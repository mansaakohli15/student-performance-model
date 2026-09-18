"""
End-to-end ML training and evaluation orchestrator.
"""

from typing import Dict, Any, Tuple
import os
import joblib
import pandas as pd
from src.data_loader import DataLoader
from src.feature_engineering import build_preprocessor, get_feature_column_types
from src.model_trainer import ModelTrainer
from src.model_evaluator import ModelEvaluator


class StudentPerformancePipeline:
    """Orchestrates end-to-end machine learning pipeline."""

    def __init__(
        self,
        data_path: str = "student_data.csv",
        output_dir: str = "artifacts",
        target_name: str = "math_score",
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        self.data_path = data_path
        self.output_dir = output_dir
        self.target_name = target_name
        self.test_size = test_size
        self.random_state = random_state

        self.models_dir = os.path.join(output_dir, "models")
        self.reports_dir = os.path.join(output_dir, "reports")
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)

        self.data_loader = DataLoader(data_path=self.data_path)
        self.evaluator = ModelEvaluator(output_dir=self.output_dir)

    def run(self, tune_hyperparameters: bool = True, save_plots: bool = True) -> Dict[str, Any]:
        """
        Executes full workflow:
        1. Ingest & validate data
        2. Build column transformation pipeline
        3. Benchmark candidate models via 5-fold CV
        4. Tune best candidate model
        5. Evaluate diagnostics & save artifacts
        """
        print(f"[*] Ingesting data for target: '{self.target_name}'...")
        X_train, X_test, y_train, y_test = self.data_loader.prepare_data(
            target_name=self.target_name,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        print(f"[*] Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

        # Build preprocessor
        nominal_cols, ordinal_cols, numeric_cols = get_feature_column_types(X_train)
        preprocessor = build_preprocessor(
            nominal_cols=nominal_cols,
            ordinal_cols=ordinal_cols,
            numeric_cols=numeric_cols,
            scale_numeric=True,
        )

        # Benchmark candidate models
        print("\n[*] Running 5-Fold Cross-Validation across candidate models...")
        trainer = ModelTrainer(preprocessor=preprocessor, cv_folds=5, random_state=self.random_state)
        benchmark_df = trainer.benchmark_models(X_train, y_train, X_test, y_test)

        benchmark_csv_path = os.path.join(self.reports_dir, "model_benchmarks.csv")
        benchmark_df.to_csv(benchmark_csv_path, index=False)
        print(f"[+] Benchmark results saved to: {benchmark_csv_path}")

        # Choose best model (Ridge/Tuned Ridge or GBDT depending on target)
        if tune_hyperparameters:
            print("\n[*] Optimizing hyperparameters via GridSearchCV...")
            best_model, best_params = trainer.tune_best_model(X_train, y_train, model_type="ridge")
            print(f"[+] Best Hyperparameters: {best_params}")
        else:
            best_model_name = benchmark_df.iloc[0]["Model"]
            print(f"[*] Selecting top benchmark model: {best_model_name}")
            # Build and fit top model
            reg = trainer.candidate_models.get(best_model_name, trainer.candidate_models["Ridge Regression"])
            from sklearn.pipeline import Pipeline
            best_model = Pipeline([("prep", preprocessor), ("reg", reg)])
            best_model.fit(X_train, y_train)
            best_params = {}

        # Evaluate best model
        y_test_pred = best_model.predict(X_test)
        metrics = self.evaluator.evaluate_performance(
            y_true=y_test.values,
            y_pred=y_test_pred,
            num_features=X_train.shape[1],
        )

        # Extract feature names & importances
        prep_fitted = best_model.named_steps["prep"]
        feature_names = list(prep_fitted.get_feature_names_out())
        df_importance = self.evaluator.extract_feature_importance(best_model, feature_names)
        importance_csv_path = os.path.join(self.reports_dir, "feature_importance.csv")
        df_importance.to_csv(importance_csv_path, index=False)

        # Save model pipeline
        model_artifact_path = os.path.join(self.models_dir, f"best_model_{self.target_name}.joblib")
        metadata = {
            "target_name": self.target_name,
            "feature_columns": list(X_train.columns),
            "metrics": metrics,
            "best_params": best_params,
        }
        joblib.dump({"pipeline": best_model, "metadata": metadata}, model_artifact_path)
        print(f"[+] Serialized model artifact saved to: {model_artifact_path}")

        # Generate plots
        if save_plots:
            print("\n[*] Generating diagnostic figures...")
            self.evaluator.plot_actual_vs_predicted(
                y_true=y_test.values,
                y_pred=y_test_pred,
                r2=metrics["R2_Score"],
                rmse=metrics["RMSE"],
                filename="actual_vs_predicted.png",
            )
            self.evaluator.plot_residuals_diagnostics(
                y_true=y_test.values,
                y_pred=y_test_pred,
                filename="residuals_diagnostics.png",
            )
            self.evaluator.plot_feature_importance(
                df_importance=df_importance,
                top_n=10,
                filename="feature_importance.png",
            )
            self.evaluator.plot_benchmark_comparison(
                benchmark_df=benchmark_df,
                filename="model_benchmarks.png",
            )
            print(f"[+] Diagnostic figures saved in: {self.evaluator.figures_dir}")

        return {
            "benchmark_df": benchmark_df,
            "metrics": metrics,
            "best_params": best_params,
            "model_path": model_artifact_path,
            "feature_importance": df_importance,
        }
