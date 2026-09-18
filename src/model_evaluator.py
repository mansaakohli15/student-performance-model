"""
Model evaluation, residual diagnostics, feature importance extraction, and plotting module.
"""

from typing import Dict, Any, List, Optional
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    root_mean_squared_error,
    median_absolute_error,
    mean_absolute_percentage_error,
)
from sklearn.pipeline import Pipeline


# Configure clean plotting style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 10, "axes.labelsize": 11, "figure.titlesize": 13})


class ModelEvaluator:
    """Computes comprehensive regression diagnostics and generates evaluation figures."""

    def __init__(self, output_dir: str = "artifacts"):
        self.output_dir = output_dir
        self.figures_dir = os.path.join(output_dir, "figures")
        self.reports_dir = os.path.join(output_dir, "reports")
        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)

    def evaluate_performance(
        self, y_true: np.ndarray, y_pred: np.ndarray, num_features: int
    ) -> Dict[str, float]:
        """
        Computes standard and robust regression evaluation metrics.
        """
        n = len(y_true)
        p = num_features

        r2 = r2_score(y_true, y_pred)
        # Adjusted R²
        adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / max(1, (n - p - 1))
        rmse = root_mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        med_ae = median_absolute_error(y_true, y_pred)
        
        # Calculate MAPE excluding zero denominators
        non_zero_mask = y_true > 0
        if np.any(non_zero_mask):
            mape = float(np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])))
        else:
            mape = 0.0

        residuals = y_true - y_pred
        skewness = float(stats.skew(residuals))
        kurtosis = float(stats.kurtosis(residuals))

        return {
            "R2_Score": r2,
            "Adjusted_R2": adj_r2,
            "RMSE": rmse,
            "MAE": mae,
            "Median_AE": med_ae,
            "MAPE": mape,
            "Residual_Mean": float(np.mean(residuals)),
            "Residual_Std": float(np.std(residuals)),
            "Residual_Skewness": skewness,
            "Residual_Kurtosis": kurtosis,
        }

    def extract_feature_importance(
        self, pipeline: Pipeline, feature_names: List[str]
    ) -> pd.DataFrame:
        """
        Extracts feature importances (for tree models) or coefficients (for linear models).
        """
        model = pipeline.named_steps["reg"]

        # If voting regressor or custom ensemble, handle sub-estimators
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            metric_name = "Importance"
        elif hasattr(model, "coef_"):
            importances = model.coef_
            metric_name = "Coefficient"
        elif hasattr(model, "estimators_"):
            # If ensemble, average available importances/coefs
            first_est = model.estimators_[0]
            if hasattr(first_est, "coef_"):
                importances = first_est.coef_
                metric_name = "Coefficient"
            else:
                importances = np.ones(len(feature_names)) / len(feature_names)
                metric_name = "Importance"
        else:
            importances = np.zeros(len(feature_names))
            metric_name = "Importance"

        # Match length if feature names count differs
        if len(importances) != len(feature_names):
            feature_names = [f"Feature_{i}" for i in range(len(importances))]

        df_importance = pd.DataFrame(
            {"Feature": feature_names, metric_name: importances, "AbsValue": np.abs(importances)}
        ).sort_values(by="AbsValue", ascending=False)

        return df_importance

    def plot_actual_vs_predicted(
        self, y_true: np.ndarray, y_pred: np.ndarray, r2: float, rmse: float, filename: str = "actual_vs_predicted.png"
    ) -> str:
        """Plots scatter of actual vs predicted scores with identity reference line."""
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        ax.scatter(y_true, y_pred, alpha=0.6, edgecolors="k", linewidths=0.5, color="#1f77b4", label="Test Samples")
        
        min_val = min(min(y_true), min(y_pred)) - 2
        max_val = max(max(y_true), max(y_pred)) + 2
        ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Ideal (y = ŷ)")

        ax.set_title(f"Actual vs Predicted Student Performance\n$R^2 = {r2:.4f}$ | $\\mathrm{{RMSE}} = {rmse:.2f}$", pad=12)
        ax.set_xlabel("Actual Score")
        ax.set_ylabel("Predicted Score")
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.legend(loc="upper left")
        plt.tight_layout()

        filepath = os.path.join(self.figures_dir, filename)
        plt.savefig(filepath)
        plt.close(fig)
        return filepath

    def plot_residuals_diagnostics(
        self, y_true: np.ndarray, y_pred: np.ndarray, filename: str = "residuals_diagnostics.png"
    ) -> str:
        """Plots residual distribution and Q-Q normality plot."""
        residuals = y_true - y_pred
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

        # 1. Residuals vs Predicted
        axes[0].scatter(y_pred, residuals, alpha=0.6, edgecolors="k", linewidths=0.5, color="#2ca02c")
        axes[0].axhline(0, color="r", linestyle="--", lw=2)
        axes[0].set_title("Residuals vs Fitted Values (Homoscedasticity Check)")
        axes[0].set_xlabel("Fitted Values (ŷ)")
        axes[0].set_ylabel("Residuals (y - ŷ)")

        # 2. Q-Q Plot
        stats.probplot(residuals, dist="norm", plot=axes[1])
        axes[1].set_title("Normal Q-Q Plot of Residuals")

        plt.tight_layout()
        filepath = os.path.join(self.figures_dir, filename)
        plt.savefig(filepath)
        plt.close(fig)
        return filepath

    def plot_feature_importance(
        self, df_importance: pd.DataFrame, top_n: int = 10, filename: str = "feature_importance.png"
    ) -> str:
        """Plots horizontal bar chart of top predictive features."""
        df_top = df_importance.head(top_n).sort_values(by="AbsValue", ascending=True)
        metric_col = "Coefficient" if "Coefficient" in df_top.columns else "Importance"

        fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
        colors = ["#2b5c8f" if v >= 0 else "#c0392b" for v in df_top[metric_col]]
        bars = ax.barh(df_top["Feature"], df_top[metric_col], color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)

        ax.set_title(f"Top {top_n} Features by {metric_col} (Magnitude)")
        ax.set_xlabel(f"{metric_col} Value")
        ax.axvline(0, color="black", linestyle="-", lw=0.8)

        # Add text labels on bars
        for bar in bars:
            width = bar.get_width()
            offset = 0.05 if width >= 0 else -0.05
            ha = "left" if width >= 0 else "right"
            ax.annotate(
                f"{width:.2f}",
                xy=(width + offset, bar.get_y() + bar.get_height() / 2),
                va="center",
                ha=ha,
                fontsize=8,
                color="black",
            )

        plt.tight_layout()
        filepath = os.path.join(self.figures_dir, filename)
        plt.savefig(filepath)
        plt.close(fig)
        return filepath

    def plot_benchmark_comparison(
        self, benchmark_df: pd.DataFrame, filename: str = "model_benchmarks.png"
    ) -> str:
        """Plots bar chart comparing CV R² and Test R² across all evaluated models."""
        fig, ax = plt.subplots(figsize=(11, 6), dpi=300)

        df_plot = benchmark_df.sort_values(by="CV_R2_Mean", ascending=True)
        y_pos = np.arange(len(df_plot))
        height = 0.35

        ax.barh(
            y_pos - height / 2,
            df_plot["CV_R2_Mean"],
            height,
            xerr=df_plot["CV_R2_Std"],
            label="5-Fold CV $R^2$ (Mean ± Std)",
            color="#3498db",
            alpha=0.85,
            edgecolor="black",
            capsize=3,
        )
        ax.barh(
            y_pos + height / 2,
            df_plot["Test_R2"],
            height,
            label="Holdout Test $R^2$",
            color="#2ecc71",
            alpha=0.85,
            edgecolor="black",
        )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_plot["Model"])
        ax.set_xlabel("$R^2$ Score (Higher is better)")
        ax.set_title("Model Zoo Performance Comparison across Cross-Validation and Holdout Test")
        ax.legend(loc="lower right")
        plt.tight_layout()

        filepath = os.path.join(self.figures_dir, filename)
        plt.savefig(filepath)
        plt.close(fig)
        return filepath
