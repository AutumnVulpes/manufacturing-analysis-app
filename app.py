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
    render_pca_tab,
    clean_csv_file,
    load_css,
    render_ai_helper_tab,
    render_dashboard_title,
)


# Set explicit headless parameters for Chromium for streamlit cloud.
# Remove if only deploying locally.
pio.kaleido.scope.chromium_args = (
    "--headless",
    "--no-sandbox",
    "--single-process",
    "--disable-gpu",
)

# =================================================================================================
# WebApp Code
# =================================================================================================

# Setup -------------------------------------------------------------------------------------------

load_css()
st.set_page_config(layout="wide")

# Initialize session state.
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
        active_provider="",
        active_api_key="",
        generated_title="",
        last_processed_filename="",
        chat_history=[],
        title_needs_typewriter=False,
        previous_title="",
    )

app_state = st.session_state.app_state

app_state = render_dashboard_title(app_state)

uploaded_csv_file = st.file_uploader("Upload dataset", type="csv")

# App -------------------------------------------------------------------------------------------

if uploaded_csv_file is not None:
    app_state.cleaned_df, app_state.numeric_cols = clean_csv_file(uploaded_csv_file)

    if len(app_state.numeric_cols) >= 2:
        app_state.x_axis = app_state.numeric_cols[0]
        app_state.y_axis = app_state.numeric_cols[1]
        app_state.filtered_df = app_state.cleaned_df.copy()
    else:
        st.warning("No numeric columns found for visualization.")

    left_col, right_col = st.columns([4, 6])

    with left_col:
        tab_pca, tab_pca_formulas, tab_viz, tab_ai = st.tabs(
            ["PCA Config", "PCA Formulas", "Visualization Config", "AI Data Assistant"]
        )
        with tab_pca:
            app_state.cleaned_df, app_state.numeric_cols, app_state = render_pca_tab(
                app_state.cleaned_df, app_state.numeric_cols, app_state
            )

        with tab_pca_formulas:
            render_pca_formulas_tab(app_state)

        with tab_viz:
            app_state.filtered_df, app_state.x_axis, app_state.y_axis, _ = (
                render_viz_config_tab(
                    app_state.cleaned_df,
                    app_state.numeric_cols,
                    app_state.x_axis,
                    app_state.y_axis,
                    app_state,
                )
            )

        with tab_ai:
            app_state, rerun_needed = render_ai_helper_tab(app_state, uploaded_csv_file)
            if rerun_needed:
                st.rerun()

    with right_col:
        tab_scatter, tab_scree, tab_cumulative_variance = st.tabs(
            ["Scatter Plot", "Scree Plot", "Cumulative Explained Variance"]
        )

        with tab_scatter:
            render_scatter_tab(
                app_state.filtered_df,
                app_state.x_axis,
                app_state.y_axis,
                uploaded_csv_file,
            )

        with tab_scree:
            render_scree_tab(app_state)

        with tab_cumulative_variance:
            render_cumulative_variance_tab(app_state)

else:
    st.info("Please upload a CSV file to begin analysis")

render_github_footer()
