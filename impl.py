import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Widgets
def render_cumulative_variance_tab(app_state):
    """
    Renders cumulative explained variance visualization tab
    """
    st.subheader("Cumulative Explained Variance")
    if app_state.pca_state.cumulative_variance:
        cumulative_variance = app_state.pca_state.cumulative_variance
        min_components_95_variance = app_state.pca_state.min_components_95_variance

        # Create cumulative variance plot data
        component_indices = list(range(len(cumulative_variance)))
        component_labels = [f"PC{i + 1}" for i in component_indices]
        df_cumulative = pd.DataFrame(
            {
                "Component Index": component_indices,
                "Component": component_labels,
                "Cumulative Variance": cumulative_variance,
            }
        )
        fig = px.line(
            df_cumulative,
            x="Component Index",
            y="Cumulative Variance",
            title="Cumulative Explained Variance",
            markers=True,
        )

        # Set custom x-axis tick labels
        fig.update_xaxes(tickvals=component_indices, ticktext=component_labels)

        # Add 95% threshold line and annotation
        fig.add_hline(
            y=0.95,
            line_dash="dash",
            line_color="red",
            annotation_text="95% Variance",
            annotation_position="bottom right",
        )

        # Add marker for optimal component count
        if min_components_95_variance > 0:
            fig.add_vline(
                x=min_components_95_variance - 1,
                line_dash="dash",
                line_color="green",
                annotation_text=f"Optimal: {min_components_95_variance} components",
                annotation_position="top right",
            )

        create_download_plotly_figure_button(fig, "cumulative-variance.png")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run PCA in the 'PCA Config' tab to see the cumulative variance plot.")


def render_scree_tab(app_state):
    """
    Renders scree plot visualization tab
    """
    st.subheader("Scree Plot")
    if app_state.pca_state.explained_variance:
        # Create scree plot data
        explained_variance = app_state.pca_state.explained_variance
        df_scree = pd.DataFrame(
            {
                "Component": [f"PC{i + 1}" for i in range(len(explained_variance))],
                "Explained Variance": explained_variance,
            }
        )
        fig = px.line(
            df_scree,
            x="Component",
            y="Explained Variance",
            title="Variance Explained by Principal Components",
            markers=True,
        )
        # Add download button
        create_download_plotly_figure_button(fig, "scree-plot.png")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run PCA in the 'PCA Config' tab to see the scree plot.")


def render_scatter_tab(filtered_df, x_axis, y_axis, uploaded_csv_file):
    """
    Renders scatter plot visualization tab
    """
    st.subheader("Scatter Plot Visualization")
    if is_valid_plot_config(filtered_df, x_axis, y_axis):
        fig = px.scatter(filtered_df, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")
        create_download_plotly_figure_button(fig, "scatter-plot.png")
        st.plotly_chart(fig, use_container_width=True)
    elif uploaded_csv_file is None:
        st.info("Upload a CSV file to begin analysis and visualization.")
    else:
        st.info(
            "Select valid X and Y axes in the 'Visualization Config' tab on the left to see the plot."
        )


def render_viz_config_tab(cleaned_df, numeric_cols, x_axis, y_axis, app_state):
    """
    Renders visualization configuration UI
    Returns updated filtered_df, x_axis, y_axis, and app_state
    """
    st.subheader("Scatter Plot Configuration")
    filtered_df = pd.DataFrame()

    if len(numeric_cols) >= 2:
        col1_plot_config, col2_plot_config = st.columns(2)
        x_axis = col1_plot_config.selectbox(
            "X-axis",
            options=numeric_cols,
            index=numeric_cols.index(x_axis) if x_axis in numeric_cols else 0,
            key="x_axis_select",
        )
        y_axis = col2_plot_config.selectbox(
            "Y-axis",
            options=numeric_cols,
            index=numeric_cols.index(y_axis)
            if y_axis in numeric_cols
            else (1 if len(numeric_cols) > 1 else 0),
            key="y_axis_select",
        )

        st.subheader("Data Filtering")
        filters = {}
        cols_to_filter = []
        if x_axis:
            cols_to_filter.append(x_axis)
        if y_axis and y_axis != x_axis:
            cols_to_filter.append(y_axis)

        current_filtered_df = cleaned_df.copy()
        for col in cols_to_filter:
            if col in cleaned_df.columns:
                min_val_df = float(cleaned_df[col].min())
                max_val_df = float(cleaned_df[col].max())
                step = (
                    (max_val_df - min_val_df) / 100
                    if (max_val_df - min_val_df) > 0
                    else 1.0
                )

                filters[col] = st.slider(
                    f"Filter {col}",
                    min_value=min_val_df,
                    max_value=max_val_df,
                    value=(min_val_df, max_val_df),
                    step=step,
                    key=f"filter_{col}",
                )

        if not current_filtered_df.empty:
            for col, (min_val, max_val) in filters.items():
                current_filtered_df = current_filtered_df[
                    (current_filtered_df[col] >= min_val)
                    & (current_filtered_df[col] <= max_val)
                ]

        filtered_df = current_filtered_df
    else:
        st.warning("No numeric columns available to configure visualization.")

    return filtered_df, x_axis, y_axis, app_state


def render_pca_formulas_tab(app_state):
    """
    Renders the PCA formulas tab UI
    """
    st.subheader("Principal Component Formulas")
    if app_state.pca_state.pca_object is not None and app_state.pca_state.pca_cols:
        pca_obj = app_state.pca_state.pca_object
        pca_cols_for_formula = app_state.pca_state.pca_cols

        n_optimal = app_state.pca_state.min_components_95_variance

        # Render latex formulas for optimal principal components
        for comp_index in range(n_optimal):
            with st.expander(f"PC{comp_index + 1} Formula Details"):
                st.write(f"PC{comp_index + 1} = ")
                for i, col in enumerate(pca_cols_for_formula):
                    if pca_obj.components_.shape[1] > i:
                        formula = (
                            rf"{pca_obj.components_[comp_index][i]:.4f} \times "
                            rf"\frac{{{col} - \mu_{{{col}}}}}{{\sigma_{{{{col}}}}}}"
                        )
                        st.latex(formula)
    else:
        st.info("Upload data and run PCA in the 'PCA Config' tab to see formulas.")


def render_pca_tab(cleaned_df, numeric_cols, app_state):
    # Renders PCA configuration UI and handles PCA processing
    # Returns updated df, numeric_cols, and app_state
    st.subheader("Principal Component Analysis")
    if numeric_cols:
        exclude_cols = st.multiselect(
            "Exclude columns from PCA", numeric_cols, key="pca_exclude_cols"
        )
        pca_cols = [col for col in numeric_cols if col not in exclude_cols]
    else:
        exclude_cols = []
        pca_cols = []
        st.info("No numeric columns available for PCA.")

    if len(pca_cols) >= 2:
        st.info(f"PCA will be performed on: {', '.join(pca_cols)}")

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cleaned_df[pca_cols])

        pca = PCA()
        pca_components = pca.fit_transform(scaled_data)

        app_state.pca_state.pca_object = pca
        app_state.pca_state.pca_cols = pca_cols
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = explained_variance.cumsum()
        app_state.pca_state.explained_variance = explained_variance.tolist()
        app_state.pca_state.cumulative_variance = cumulative_variance.tolist()

        # Determine number of optimal principal components to reach 95% variance
        min_components_95_variance = next(
            (i + 1 for i, v in enumerate(cumulative_variance) if v >= 0.95),
            len(cumulative_variance),
        )
        app_state.pca_state.min_components_95_variance = min_components_95_variance

        # Append columns with optimal principal components to df
        for i in range(min_components_95_variance):
            col_name = f"PC{i + 1}"
            cleaned_df[col_name] = pca_components[:, i]
            if col_name not in numeric_cols:
                numeric_cols.append(col_name)

        st.subheader("Data with Principal Components")
        st.write(cleaned_df.head())

        csv_pca = cleaned_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Cleaned Data with PCA",
            csv_pca,
            "cleaned_data_with_pca.csv",
            "text/csv",
            key="download-csv-pca",
        )
    else:
        if exclude_cols:
            st.warning(
                f"Only {len(pca_cols)} column(s) available for PCA. Please exclude fewer columns to have at least two."
            )
        else:
            st.warning("The dataset needs at least 2 numeric columns for PCA.")

        st.subheader("Cleaned Data Preview")
        st.write(cleaned_df.head())

        csv_cleaned = cleaned_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Cleaned Data",
            csv_cleaned,
            "cleaned_data.csv",
            "text/csv",
            key="download-csv-cleaned",
        )

    return cleaned_df, numeric_cols, app_state


def create_download_plotly_figure_button(fig, filename="plot.png"):
    """Convert Plotly figure to PNG and create download button"""
    img_bytes = fig.to_image(format="png")
    st.download_button(
        "Download Graph as PNG",
        img_bytes,
        filename,
        "image/png",
        key=f"download-png-{filename.replace('.', '-')}",
    )


def render_github_footer():
    st.markdown("---")
    theme_base = st.get_option("theme.base")
    github_repo_url = "https://github.com/AutumnVulpes/manufacturing-analysis-app"

    if theme_base == "dark":
        fill_color = "rgb(145, 152, 161)"  # Dark theme color
    else:
        fill_color = "rgb(89, 99, 110)"  # Light theme color

    github_svg = f'''
    <svg xmlns="http://www.w3.org/2000/svg" height="20" viewBox="0 0 16 16" width="20" aria-hidden="true" class="d-block">
        <path fill="{fill_color}" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path>
    </svg>
    '''

    # Embedded github logo and name with hyperlinks to repo
    st.markdown(
        f'<div class="github-logo-container">'
        f'<a href="{github_repo_url}" target="_blank">'
        f"{github_svg}"
        f"</a>"
        f'<span style="margin-left: 10px;">Made by <a href="{github_repo_url}" target="_blank">@AutumnVulpes</a></span>'
        f"</div>",
        unsafe_allow_html=True,
    )


# Computations
def is_valid_plot_config(df, x, y):
    return not df.empty and x and y and x in df.columns and y in df.columns
