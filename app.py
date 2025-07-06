# Imports, setup and functions
import streamlit as st
import pandas as pd
import plotly.io as pio
from datatypes import PCAState, AppState
from impl import (
    render_github_footer,
    render_cumulative_variance_tab,
    render_scree_tab,
    render_scatter_tab,
    render_viz_config_tab,
    render_pca_formulas_tab,
    render_pca_tab
)

# Set explicit headless parameters for Chromium for streamlit cloud
# Remove if only deploying locally
pio.kaleido.scope.chromium_args = (
  "--headless",
  "--no-sandbox",
  "--single-process",
  "--disable-gpu",
)


@st.cache_data
def clean_csv_file(csv_file):
    df = pd.read_csv(csv_file)
    cleaned_df = df.dropna()
    numeric_cols = cleaned_df.select_dtypes(
        include=["float64", "int64"]
    ).columns.tolist()
    return cleaned_df, numeric_cols


# WebApp Code
st.set_page_config(layout="wide")

st.title("Manufacturing Data Analysis Dashboard")

with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

uploaded_csv_file = st.file_uploader("Upload dataset", type="csv")

# Initialize session state
if "app_state" not in st.session_state:
    st.session_state.app_state = AppState(
        cleaned_df=pd.DataFrame(),
        numeric_cols=[],
        x_axis="",
        y_axis="",
        filtered_df=pd.DataFrame(),
        pca_state=PCAState(
            pca_object=None,
            pca_cols=[],
            explained_variance=[],
            cumulative_variance=[],
            min_components_95_variance=0,
        ),
    )

app_state = st.session_state.app_state
cleaned_df = app_state.cleaned_df
numeric_cols = app_state.numeric_cols
x_axis = app_state.x_axis
y_axis = app_state.y_axis
filtered_df = app_state.filtered_df
pca_results_dict = app_state.pca_state

if uploaded_csv_file is not None:
    cleaned_df, numeric_cols = clean_csv_file(uploaded_csv_file)

    if len(numeric_cols) >= 2:
        x_axis = numeric_cols[0]
        y_axis = numeric_cols[1]
        filtered_df = cleaned_df.copy()
    else:
        st.warning("No numeric columns found for visualization.")

    left_col, right_col = st.columns([4, 6])

    with left_col:
        tab_pca, tab_pca_formulas, tab_viz = st.tabs(
            ["PCA Config", "PCA Formulas", "Visualization Config"]
        )
        with tab_pca:
            cleaned_df, numeric_cols, app_state = render_pca_tab(
                cleaned_df, numeric_cols, app_state
            )

        with tab_pca_formulas:
            render_pca_formulas_tab(app_state)

        with tab_viz:
            filtered_df, x_axis, y_axis, app_state = render_viz_config_tab(
                cleaned_df, numeric_cols, x_axis, y_axis, app_state
            )

    with right_col:
        # Create tabs for visualizations
        tab_scatter, tab_scree, tab_cumulative_variance = st.tabs(
            ["Scatter Plot", "Scree Plot", "Cumulative Explained Variance"]
        )

        with tab_scatter:
            render_scatter_tab(filtered_df, x_axis, y_axis, uploaded_csv_file)

        with tab_scree:
            render_scree_tab(app_state)

        with tab_cumulative_variance:
            render_cumulative_variance_tab(app_state)

else:
    st.info("Please upload a CSV file to begin analysis")

render_github_footer()
