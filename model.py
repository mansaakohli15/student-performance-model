"""
Student Performance Prediction - Production Runner Script.
Maintains backward compatibility while executing the modular production pipeline.
"""

from src.pipeline import StudentPerformancePipeline


def main():
    print("=" * 60)
    print(" Running Student Performance Prediction Pipeline")
    print("=" * 60)

    # Execute pipeline predicting math score (and composite score)
    pipeline = StudentPerformancePipeline(
        data_path="student_data.csv",
        output_dir="artifacts",
        target_name="math_score",
    )
    results = pipeline.run(tune_hyperparameters=True, save_plots=True)

    print("\n" + "=" * 60)
    print(" MODEL PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Holdout Test R² Score        : {results['metrics']['R2_Score']:.4f}")
    print(f"Adjusted R² Score            : {results['metrics']['Adjusted_R2']:.4f}")
    print(f"Root Mean Squared Error (RMSE): {results['metrics']['RMSE']:.2f}")
    print(f"Mean Absolute Error (MAE)    : {results['metrics']['MAE']:.2f}")
    print(f"Mean Absolute % Error (MAPE) : {results['metrics']['MAPE'] * 100:.2f}%")
    print("=" * 60)
    print("\nTop Predictive Features:")
    print(results["feature_importance"].head(10).to_string(index=False))
    print("\nModel artifact & plots saved to 'artifacts/' folder.")


if __name__ == "__main__":
    main()