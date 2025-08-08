import pandas as pd
from typing import Dict, Any, List, Tuple


def get_column_data_for_visualization(
    df: pd.DataFrame, col1: str, col2: str, max_points: int = 50
) -> Tuple[List, List]:
    """
    Extract data for two columns, sampling if necessary for visualization.

    Args:
        df: Input dataframe
        col1: First column name
        col2: Second column name
        max_points: Maximum number of points to return

    Returns:
        Tuple of (col1_data, col2_data) as lists
    """
    if col1 not in df.columns or col2 not in df.columns:
        return [], []

    clean_df = df[[col1, col2]].copy()
    clean_df[col1] = pd.to_numeric(clean_df[col1], errors="coerce")
    clean_df[col2] = pd.to_numeric(clean_df[col2], errors="coerce")
    clean_df = clean_df.dropna()

    if len(clean_df) == 0:
        return [], []

    # Sample data if too many points.
    if len(clean_df) > max_points:
        clean_df = clean_df.sample(n=max_points, random_state=42)

    return clean_df[col1].tolist(), clean_df[col2].tolist()


def validate_data_for_suggestions(
    df: pd.DataFrame, numeric_cols: List[str]
) -> Tuple[bool, str]:
    """
    Validate if the data is suitable for generating column suggestions.

    Args:
        df: Input dataframe
        numeric_cols: List of numeric column names

    Returns:
        Tuple of (is_valid, error_message)
    """
    if df.empty:
        return False, "Dataset is empty"

    if len(numeric_cols) < 2:
        return (
            False,
            "Dataset needs at least 2 numeric columns for comparison suggestions",
        )

    if len(df) < 5:
        return False, "Dataset needs at least 5 rows for meaningful analysis"

    return True, ""


def create_line_chart_data(data: List[float], max_points: int = 20) -> List[float]:
    """
    Prepare data for line chart visualization in streamlit dataframe.

    Args:
        data: List of numeric values
        max_points: Maximum number of points for the chart

    Returns:
        List of values suitable for line chart
    """
    if not data:
        return []

    if len(data) > max_points:
        # Sample evenly across the data for accurate representation.
        step = len(data) // max_points
        return [data[i] for i in range(0, len(data), step)][:max_points]

    return data
