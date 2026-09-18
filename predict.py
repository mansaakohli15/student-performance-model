"""
Inference Module for Student Performance Prediction.
Loads serialized pipeline artifact and runs single or batch predictions.
"""

from typing import Union, Dict, Any, List
import os
import argparse
import joblib
import pandas as pd
import numpy as np


class StudentPerformancePredictor:
    """Predictor class that encapsulates model loading and inference."""

    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model artifact not found at: {model_path}")

        artifact = joblib.load(model_path)
        self.pipeline = artifact["pipeline"]
        self.metadata = artifact.get("metadata", {})
        self.expected_features = self.metadata.get("feature_columns", [])
        self.target_name = self.metadata.get("target_name", "score")

    def predict(self, input_data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]) -> np.ndarray:
        """
        Executes prediction on input dictionary, list of dicts, or DataFrame.
        """
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, list):
            df = pd.DataFrame(input_data)
        elif isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        else:
            raise TypeError("Input data must be a dict, list of dicts, or pandas DataFrame.")

        # Validate presence of expected features
        if self.expected_features:
            missing = set(self.expected_features) - set(df.columns)
            if missing:
                raise ValueError(f"Input data is missing expected features: {missing}")
            df = df[self.expected_features]

        predictions = self.pipeline.predict(df)
        # Clip scores to valid academic range [0, 100]
        return np.clip(predictions, 0.0, 100.0)


def main():
    parser = argparse.ArgumentParser(description="Student Performance Inference CLI")
    parser.add_argument(
        "--model-path",
        type=str,
        default="artifacts/models/best_model_math_score.joblib",
        help="Path to serialized model artifact (.joblib)",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Run inference on example test cases",
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default=None,
        help="Path to CSV file for batch predictions",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Path to save batch predictions CSV",
    )

    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"[!] Model not found at '{args.model_path}'. Please train the model first by running `python train.py`.")
        return

    predictor = StudentPerformancePredictor(model_path=args.model_path)
    print(f"[*] Loaded model for target: '{predictor.target_name}'")

    if args.sample or (not args.input_csv):
        print("\n" + "=" * 60)
        print("                  SAMPLE INFERENCE RUN")
        print("=" * 60)
        sample_cases = [
            {
                "gender": "female",
                "race/ethnicity": "group C",
                "parental level of education": "bachelor's degree",
                "lunch": "standard",
                "test preparation course": "completed",
                "reading score": 85,
                "writing score": 88,
            },
            {
                "gender": "male",
                "race/ethnicity": "group A",
                "parental level of education": "high school",
                "lunch": "free/reduced",
                "test preparation course": "none",
                "reading score": 52,
                "writing score": 49,
            },
            {
                "gender": "female",
                "race/ethnicity": "group E",
                "parental level of education": "master's degree",
                "lunch": "standard",
                "test preparation course": "completed",
                "reading score": 95,
                "writing score": 98,
            },
        ]

        df_sample = pd.DataFrame(sample_cases)
        preds = predictor.predict(df_sample)
        df_sample[f"predicted_{predictor.target_name}"] = np.round(preds, 2)
        print(df_sample.to_string(index=False))
        print("=" * 60)

    if args.input_csv:
        print(f"\n[*] Processing batch predictions from: {args.input_csv}")
        df_input = pd.read_csv(args.input_csv)
        preds = predictor.predict(df_input)
        df_input[f"predicted_{predictor.target_name}"] = np.round(preds, 2)
        
        out_path = args.output_csv or "batch_predictions.csv"
        df_input.to_csv(out_path, index=False)
        print(f"[+] Saved batch predictions to: {out_path}")


if __name__ == "__main__":
    main()
