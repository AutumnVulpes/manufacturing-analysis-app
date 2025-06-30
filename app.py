import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.title("Manufacturing Data Analysis Dashboard")

uploaded_file = st.file_uploader("Upload manufacturing data CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    cleaned_df = df.dropna()

    numeric_cols = cleaned_df.select_dtypes(
        include=['float64', 'int64']).columns.tolist()

    # Principal Component Analysis
    # (Automatically includes all numeric columns, allows user to exclude)
    st.subheader("Principal Component Analysis")
    exclude_cols = st.multiselect("Exclude columns from PCA", numeric_cols)
    pca_cols = [col for col in numeric_cols if col not in exclude_cols]

    if len(pca_cols) >= 2:
        st.info(f"PCA will be performed on: {', '.join(pca_cols)}")

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cleaned_df[pca_cols])

        pca = PCA(n_components=2)
        pca_components = pca.fit_transform(scaled_data)

        cleaned_df['PC1'] = pca_components[:, 0]
        cleaned_df['PC2'] = pca_components[:, 1]
        numeric_cols += ['PC1', 'PC2']

        st.subheader("Data with Principal Components")
        st.write(cleaned_df.head())

        # Add download button for cleaned data with PCA values
        csv = cleaned_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Cleaned Data with PCA",
            csv,
            "cleaned_data_with_pca.csv",
            "text/csv",
            key='download-csv'
        )

        st.subheader("Principal Component Formulas")
        with st.expander("PC1 Formula Details"):
            st.write("PC1 = ")
            for i, col in enumerate(pca_cols):
                st.latex(
                    fr"PC1 = PC1 + ({pca.components_[0][i]:.4f} \times \frac{{{col} - \mu_{{{col}}}}}{{\sigma_{{{{col}}}}}}")

        with st.expander("PC2 Formula Details"):
            st.write("PC2 = ")
            for i, col in enumerate(pca_cols):
                st.latex(
                    fr"PC2 = PC2 + ({pca.components_[1][i]:.4f} \times \frac{{{col} - \mu_{{{col}}}}}{{\sigma_{{{{col}}}}}}")
    else:
        if exclude_cols:
            st.warning(
                f"Only {len(pca_cols)} column(s) available for PCA. Please exclude fewer columns to have at least two.")
        else:
            st.warning("The dataset needs at least 2 numeric columns for PCA.")

        st.subheader("Cleaned Data Preview")
        st.write(cleaned_df.head())

        # Add download button for cleaned data
        csv = cleaned_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Cleaned Data",
            csv,
            "cleaned_data.csv",
            "text/csv",
            key='download-csv'
        )

    if len(numeric_cols) >= 2:

        st.subheader("Scatter Plot Configuration")
        col1, col2 = st.columns(2)
        x_axis = col1.selectbox("X-axis", options=numeric_cols, index=0)
        y_axis = col2.selectbox("Y-axis", options=numeric_cols, index=1)

        # Create two columns for sliders and plot
        left_col, right_col = st.columns(2)

        with left_col:
            st.subheader("Data Filtering")
            filters = {}
            for col in [x_axis, y_axis]:
                if col in numeric_cols:
                    min_val = float(cleaned_df[col].min())
                    max_val = float(cleaned_df[col].max())
                    step = (max_val - min_val) / 100
                    filters[col] = st.slider(
                        f"Filter {col}",
                        min_value=min_val,
                        max_value=max_val,
                        value=(min_val, max_val),
                        step=step
                    )

            filtered_df = cleaned_df.copy()
            for col, (min_val, max_val) in filters.items():
                filtered_df = filtered_df[(filtered_df[col] >= min_val) & (
                    filtered_df[col] <= max_val)]

        with right_col:
            fig = px.scatter(
                filtered_df,
                x=x_axis,
                y=y_axis,
                title=f"{y_axis} vs {x_axis}"
            )
            st.plotly_chart(fig)

            # Add download button for the graph
            if 'fig' in locals():
                img_bytes = fig.to_image(format="png")
                st.download_button(
                    "Download Graph as PNG",
                    img_bytes,
                    "manufacturing_plot.png",
                    "image/png",
                    key='download-png'
                )

    else:
        st.warning("Dataset needs at least 2 numeric columns for visualization")
else:
    st.info("Please upload a CSV file to begin analysis")

# Add GitHub repo link at the bottom
st.markdown("---")
st.markdown(
    "[GitHub Repository](https://github.com/AutumnVulpes/manufacturing-analysis-app)")
