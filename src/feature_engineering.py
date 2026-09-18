"""
Feature engineering and preprocessing pipeline construction.
"""

from typing import List, Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline


EDUCATION_HIERARCHY = [
    [
        "some high school",
        "high school",
        "some college",
        "associate's degree",
        "bachelor's degree",
        "master's degree",
    ]
]


def build_preprocessor(
    nominal_cols: List[str],
    ordinal_cols: List[str],
    numeric_cols: List[str],
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """
    Constructs a scikit-learn ColumnTransformer for preprocessing heterogeneous data.

    Parameters:
    -----------
    nominal_cols : List[str]
        Categorical features with no intrinsic ordering (One-Hot Encoded).
    ordinal_cols : List[str]
        Categorical features with hierarchical ordering (Ordinal Encoded).
    numeric_cols : List[str]
        Continuous numerical features (Optionally Standard Scaled).
    scale_numeric : bool
        Whether to apply StandardScaler to numerical features.

    Returns:
    --------
    ColumnTransformer:
        Fitted or un-fitted transformer object.
    """
    transformers = []

    if nominal_cols:
        transformers.append(
            (
                "nominal",
                OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
                nominal_cols,
            )
        )

    if ordinal_cols:
        transformers.append(
            (
                "ordinal",
                OrdinalEncoder(categories=EDUCATION_HIERARCHY),
                ordinal_cols,
            )
        )

    if numeric_cols:
        num_transformer = StandardScaler() if scale_numeric else "passthrough"
        transformers.append(
            (
                "numeric",
                num_transformer,
                numeric_cols,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def get_feature_column_types(X: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """Automatically partition input columns into nominal, ordinal, and numeric."""
    nominal_candidates = ["gender", "race/ethnicity", "lunch", "test preparation course"]
    ordinal_candidates = ["parental level of education"]
    numeric_candidates = ["reading score", "writing score", "math score"]

    nominal_cols = [col for col in nominal_candidates if col in X.columns]
    ordinal_cols = [col for col in ordinal_candidates if col in X.columns]
    numeric_cols = [col for col in numeric_candidates if col in X.columns]

    return nominal_cols, ordinal_cols, numeric_cols
