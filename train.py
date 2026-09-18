"""
CLI Training Entrypoint for Student Performance Prediction Pipeline.
"""

import argparse
import sys
from src.pipeline import StudentPerformancePipeline


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate student performance machine learning models."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="student_data.csv",
        help="Path to CSV dataset (default: student_data.csv)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="math_score",
        choices=["math_score", "composite_score", "reading_score", "writing_score"],
        help="Target score to predict (default: math_score)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts",
        help="Directory to save model artifacts and plots (default: artifacts)",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        default=True,
        help="Perform hyperparameter tuning via GridSearchCV (default: True)",
    )
    parser.add_argument(
        "--no-tune",
        dest="tune",
        action="store_false",
        help="Skip hyperparameter tuning and use top cross-validated baseline",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        default=True,
        help="Generate and save diagnostic evaluation plots (default: True)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("      STUDENT PERFORMANCE MACHINE LEARNING TRAINING PIPELINE")
    print("=" * 70)
    print(f"Dataset      : {args.data_path}")
    print(f"Target       : {args.target}")
    print(f"Output Dir   : {args.output_dir}")
    print(f"Tuning       : {args.tune}")
    print("=" * 70)

    try:
        pipeline = StudentPerformancePipeline(
            data_path=args.data_path,
            output_dir=args.output_dir,
            target_name=args.target,
        )
        results = pipeline.run(tune_hyperparameters=args.tune, save_plots=args.save_plots)

        print("\n" + "=" * 70)
        print("                   FINAL EVALUATION SUMMARY")
        print("=" * 70)
        print(f"Holdout Test R² Score        : {results['metrics']['R2_Score']:.4f}")
        print(f"Adjusted R² Score            : {results['metrics']['Adjusted_R2']:.4f}")
        print(f"Root Mean Squared Error (RMSE): {results['metrics']['RMSE']:.2f}")
        print(f"Mean Absolute Error (MAE)    : {results['metrics']['MAE']:.2f}")
        print(f"Mean Absolute % Error (MAPE) : {results['metrics']['MAPE'] * 100:.2f}%")
        print(f"Residual Skewness            : {results['metrics']['Residual_Skewness']:.3f}")
        print(f"Residual Kurtosis            : {results['metrics']['Residual_Kurtosis']:.3f}")
        print("=" * 70)
        print(f"[+] Pipeline completed successfully. Artifacts saved in '{args.output_dir}'.")

    except Exception as e:
        print(f"\n[!] Training failed with error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
