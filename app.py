import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import kaleido

st.set_page_config(layout="wide")

st.title("Manufacturing Data Analysis Dashboard")

# Custom CSS
st.markdown(
    """
    <style>
    /* Ensure the main app container itself takes full viewport height */
    .stApp {
        min-height: 100vh;
    }

    /* Target the specific stVerticalBlock within the *first* column for scrollability */
    [data-testid="stHorizontalBlock"] > div:first-child [data-testid="stVerticalBlock"] {
        height: calc(100vh - 180px) !important; /* Adjust 180px based on actual space taken by title/footer/header elements outside the columns */
        overflow-y: auto !important; /* Enable vertical scrolling */
        padding-right: 15px; /* Add space for the scrollbar */
        border: 1px solid #e0e0e0; /* Optional: for visual demarcation */
        border-radius: 5px; /* Optional: subtle rounding */
        padding-left: 15px; /* Add some internal padding */
    }

    /* Optional: Fine-tune margins for a cleaner look */
    h1 {
        margin-top: 0;
        padding-top: 20px;
        padding-bottom: 20px;
    }
    .stFileUploader {
        margin-bottom: 20px;
    }
    h2 {
        margin-top: 25px;
        margin-bottom: 15px;
    }

    /* GitHub Logo and text container alignment (Centralized) */
    .github-logo-container {
        display: flex;
        justify-content: center; /* Aligns items to the center */
        align-items: center; /* Vertically aligns items if they have different heights */
        padding: 10px 0;
        margin-top: 10px; /* Add some space from the horizontal rule */
    }
    .github-logo-container img {
        margin-right: 10px; /* Logo margin remains unchanged */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# File uploader label remains "Upload dataset"
uploaded_file = st.file_uploader("Upload dataset", type="csv")

# Initialize variables outside the if uploaded_file block to prevent errors on initial load
cleaned_df = pd.DataFrame()
numeric_cols = []
x_axis, y_axis = None, None
filtered_df = pd.DataFrame() # Initialize as empty DataFrame
pca_results = {"pca_object": None, "pca_cols": []} # To store PCA results for formula tab

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    cleaned_df = df.dropna()

    numeric_cols = cleaned_df.select_dtypes(
        include=['float64', 'int64']).columns.tolist()

    # Set initial values for x_axis and y_axis if numeric_cols exist
    if len(numeric_cols) >= 2:
        x_axis = numeric_cols[0]
        y_axis = numeric_cols[1]
        filtered_df = cleaned_df.copy() # Initially, filtered_df is the full cleaned_df
    elif len(numeric_cols) == 1:
        x_axis = numeric_cols[0]
        y_axis = numeric_cols[0]
        filtered_df = cleaned_df.copy()
    else:
        st.warning("No numeric columns found for visualization.")


    # Create main layout columns
    left_col, right_col = st.columns([4, 6])

    with left_col:
        # Create tabs inside the left column
        tab_pca, tab_pca_formulas, tab_viz = st.tabs(
            ["PCA Config", "PCA Formulas", "Visualization Config"]
        )

        with tab_pca:
            st.subheader("Principal Component Analysis")
            if numeric_cols:
                exclude_cols = st.multiselect("Exclude columns from PCA", numeric_cols, key="pca_exclude_cols")
                pca_cols = [col for col in numeric_cols if col not in exclude_cols]
            else:
                exclude_cols = []
                pca_cols = []
                st.info("No numeric columns available for PCA.")


            if len(pca_cols) >= 2:
                st.info(f"PCA will be performed on: {', '.join(pca_cols)}")

                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(cleaned_df[pca_cols])

                pca = PCA(n_components=2)
                pca_components = pca.fit_transform(scaled_data)

                # Store PCA results in a dictionary to pass to the formulas tab
                pca_results["pca_object"] = pca
                pca_results["pca_cols"] = pca_cols

                # Add PC columns to cleaned_df for display/download
                cleaned_df['PC1'] = pca_components[:, 0]
                cleaned_df['PC2'] = pca_components[:, 1]
                # Update numeric_cols to include PC1 and PC2 for subsequent use in visualization tab
                if 'PC1' not in numeric_cols: # Prevent adding duplicates on re-run
                    numeric_cols.extend(['PC1', 'PC2'])

                st.subheader("Data with Principal Components")
                st.write(cleaned_df.head())

                # Download button for PCA data
                csv_pca = cleaned_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Cleaned Data with PCA",
                    csv_pca,
                    "cleaned_data_with_pca.csv",
                    "text/csv",
                    key='download-csv-pca'
                )
            else:
                if exclude_cols:
                    st.warning(
                        f"Only {len(pca_cols)} column(s) available for PCA. Please exclude fewer columns to have at least two.")
                else:
                    st.warning("The dataset needs at least 2 numeric columns for PCA.")

                st.subheader("Cleaned Data Preview")
                st.write(cleaned_df.head())

                # Download button for cleaned data
                csv_cleaned = cleaned_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Cleaned Data",
                    csv_cleaned,
                    "cleaned_data.csv",
                    "text/csv",
                    key='download-csv-cleaned'
                )

        with tab_pca_formulas:
            st.subheader("Principal Component Formulas")
            if pca_results["pca_object"] is not None and pca_results["pca_cols"]:
                pca_obj = pca_results["pca_object"]
                pca_cols_for_formula = pca_results["pca_cols"]

                with st.expander("PC1 Formula Details"):
                    st.write("PC1 = ")
                    for i, col in enumerate(pca_cols_for_formula):
                        if pca_obj.components_.shape[1] > i:
                            st.latex(
                                fr"{pca_obj.components_[0][i]:.4f} \times \frac{{{col} - \mu_{{{col}}}}}{{\sigma_{{{{col}}}}}}")

                with st.expander("PC2 Formula Details"):
                    st.write("PC2 = ")
                    for i, col in enumerate(pca_cols_for_formula):
                        if pca_obj.components_.shape[1] > i:
                            st.latex(
                                fr"{pca_obj.components_[1][i]:.4f} \times \frac{{{col} - \mu_{{{col}}}}}{{\sigma_{{{{col}}}}}}")
            else:
                st.info("Upload data and run PCA in the 'PCA Config' tab to see formulas.")


        with tab_viz:
            st.subheader("Scatter Plot Configuration")
            if len(numeric_cols) >= 2:
                col1_plot_config, col2_plot_config = st.columns(2)
                # Use updated numeric_cols which might include PC1/PC2
                x_axis = col1_plot_config.selectbox("X-axis", options=numeric_cols, index=numeric_cols.index(x_axis) if x_axis in numeric_cols else 0, key="x_axis_select")
                y_axis = col2_plot_config.selectbox("Y-axis", options=numeric_cols, index=numeric_cols.index(y_axis) if y_axis in numeric_cols else (1 if len(numeric_cols) > 1 else 0), key="y_axis_select")

                st.subheader("Data Filtering")
                filters = {}
                cols_to_filter = []
                if x_axis: cols_to_filter.append(x_axis)
                if y_axis and y_axis != x_axis: cols_to_filter.append(y_axis)

                current_filtered_df = cleaned_df.copy()
                for col in cols_to_filter:
                    if col in cleaned_df.columns:
                        min_val_df = float(cleaned_df[col].min())
                        max_val_df = float(cleaned_df[col].max())
                        step = (max_val_df - min_val_df) / 100 if (max_val_df - min_val_df) > 0 else 1.0

                        filters[col] = st.slider(
                            f"Filter {col}",
                            min_value=min_val_df,
                            max_value=max_val_df,
                            value=(min_val_df, max_val_df), # Always initialize with full range for clarity
                            step=step,
                            key=f"filter_{col}"
                        )

                if not current_filtered_df.empty:
                    for col, (min_val, max_val) in filters.items():
                        current_filtered_df = current_filtered_df[(current_filtered_df[col] >= min_val) & (
                            current_filtered_df[col] <= max_val)]

                filtered_df = current_filtered_df

            else:
                st.warning("No numeric columns available to configure visualization.")
                filtered_df = pd.DataFrame()


    with right_col:
        st.subheader("Scatter Plot Visualization")

        if not filtered_df.empty and x_axis and y_axis and x_axis in filtered_df.columns and y_axis in filtered_df.columns:
            fig = px.scatter(
                filtered_df,
                x=x_axis,
                y=y_axis,
                title=f"{y_axis} vs {x_axis}"
            )   
            img_bytes = fig.to_image(format="png")

            st.download_button(
                "Download Graph as PNG",
                img_bytes,
                "manufacturing_plot.png",
                "image/png",
                key='download-png-plot'
            )
            st.plotly_chart(fig, use_container_width=True)

        elif uploaded_file is None:
            st.info("Upload a CSV file to begin analysis and visualization.")
        else:
            st.info("Select valid X and Y axes in the 'Visualization Config' tab on the left to see the plot.")

else:
    st.info("Please upload a CSV file to begin analysis")

# GitHub footer with theme-appropriate SVG icon
st.markdown("---")
theme_base = st.config.get_option("theme.base")
github_repo_url = "https://github.com/AutumnVulpes/manufacturing-analysis-app"

# Set color based on theme
if theme_base == "dark":
    fill_color = "rgb(145, 152, 161)"  # Dark theme color
else:
    fill_color = "rgb(89, 99, 110)"    # Light theme color

# SVG with dynamic fill color
github_svg = f'''
<svg xmlns="http://www.w3.org/2000/svg" height="20" viewBox="0 0 16 16" width="20" aria-hidden="true" class="d-block">
    <path fill="{fill_color}" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path>
</svg>
'''

# Embed the SVG and text
st.markdown(
    f'<div class="github-logo-container">'
    f'<a href="{github_repo_url}" target="_blank">'
    f'{github_svg}'
    f'</a>'
    f'<span style="margin-left: 10px;">Made by <a href="{github_repo_url}" target="_blank">@AutumnVulpes</a></span>'
    f'</div>',
    unsafe_allow_html=True
)
