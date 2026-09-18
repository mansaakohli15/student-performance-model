"""
Data loading, validation, and splitting module.
"""

from typing import Tuple, List, Optional
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class DataLoader:
    """Loads and validates the student performance dataset."""

    EXPECTED_COLUMNS = [
        "gender",
        "race/ethnicity",
        "parental level of education",
        "lunch",
        "test preparation course",
        "math score",
        "reading score",
        "writing score",
    ]

    def __init__(self, data_path: str = "student_data.csv"):
        self.data_path = data_path

    def load_data(self) -> pd.DataFrame:
        """Load CSV data and perform schema validation."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at path: {self.data_path}")

        df = pd.read_csv(self.data_path)
        self._validate_schema(df)
        return df

    def _validate_schema(self, df: pd.DataFrame) -> None:
        """Validates columns, data types, missing values, and value ranges."""
        missing_cols = set(self.EXPECTED_COLUMNS) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns in dataset: {missing_cols}")

        # Check for missing values
        null_counts = df[self.EXPECTED_COLUMNS].isnull().sum()
        if null_counts.any():
            raise ValueError(f"Dataset contains null values:\n{null_counts[null_counts > 0]}")

        # Validate score ranges [0, 100]
        score_cols = ["math score", "reading score", "writing score"]
        for col in score_cols:
            if (df[col] < 0).any() or (df[col] > 100).any():
                raise ValueError(f"Column '{col}' contains values outside valid range [0, 100].")

    def prepare_data(
        self,
        target_name: str = "math_score",
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare features (X) and target (y) and split into train and test sets.

        Supported target_name:
            - 'math_score': Predict math score from demographics + reading & writing scores
            - 'composite_score': Predict average score across all subjects from demographics only
            - 'reading_score': Predict reading score
            - 'writing_score': Predict writing score
        """
        df = self.load_data().copy()

        # Add composite score
        df["composite_score"] = (df["math score"] + df["reading score"] + df["writing score"]) / 3.0

        if target_name == "math_score":
            feature_cols = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course",
                "reading score",
                "writing score",
            ]
            target_col = "math score"
        elif target_name == "composite_score":
            feature_cols = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course",
            ]
            target_col = "composite_score"
        elif target_name == "reading_score":
            feature_cols = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course",
                "math score",
                "writing score",
            ]
            target_col = "reading score"
        elif target_name == "writing_score":
            feature_cols = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course",
                "math score",
                "reading score",
            ]
            target_col = "writing score"
        else:
            raise ValueError(f"Unknown target_name: {target_name}. Supported: 'math_score', 'composite_score', 'reading_score', 'writing_score'")

        X = df[feature_cols]
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        return X_train, X_test, y_train, y_test
