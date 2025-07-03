from dataclasses import dataclass, field
from sklearn.decomposition import PCA
import pandas as pd


@dataclass
class PCAState:
    """Stores the state of a Principal Component Analysis (PCA) transformation.

    Attributes:
        pca_object: Fitted scikit-learn PCA model instance
        pca_cols: Column names used for PCA transformation
        explained_variance: Variance explained by each principal component
        cumulative_variance: Cumulative variance explained by components
        min_components_95_variance: Number of components needed to explain >=95% variance
    """

    pca_object: PCA | None = None
    pca_cols: list[str] = field(default_factory=list)
    explained_variance: list[float] = field(default_factory=list)
    cumulative_variance: list[float] = field(default_factory=list)
    min_components_95_variance: int = 0


@dataclass
class AppState:
    """Main application state container for data analysis.

    Attributes:
        cleaned_df: Preprocessed dataset
        numeric_cols: Numeric column names from cleaned_df
        x_axis: Currently selected feature for X-axis in visualizations
        y_axis: Currently selected feature for Y-axis in visualizations
        filtered_df: Subset of cleaned_df after applying filters
        pca_state: Current state of PCA dimensionality reduction
    """

    cleaned_df: pd.DataFrame
    numeric_cols: list[str]
    x_axis: str
    y_axis: str
    filtered_df: pd.DataFrame
    pca_state: PCAState
