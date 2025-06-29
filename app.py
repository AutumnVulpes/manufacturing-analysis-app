import streamlit as st
import pandas as pd
import plotly.express as px

# Set app title
st.title("Manufacturing Data Analysis Dashboard")

# File uploader
uploaded_file = st.file_uploader("Upload manufacturing data CSV", type="csv")

if uploaded_file is not None:
    # Read and clean data
    df = pd.read_csv(uploaded_file)
    cleaned_df = df.dropna()
    
    # Show cleaned data
    st.subheader("Cleaned Data Preview")
    st.write(cleaned_df.head())
    
    # Get numeric columns
    numeric_cols = cleaned_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    if len(numeric_cols) >= 2:
        # Column selection
        st.subheader("Scatter Plot Configuration")
        col1, col2 = st.columns(2)
        x_axis = col1.selectbox("X-axis", options=numeric_cols, index=0)
        y_axis = col2.selectbox("Y-axis", options=numeric_cols, index=1)
        
        # Create sliders for selected columns
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
        
        # Apply filters
        filtered_df = cleaned_df.copy()
        for col, (min_val, max_val) in filters.items():
            filtered_df = filtered_df[(filtered_df[col] >= min_val) & (filtered_df[col] <= max_val)]
        
        # Generate scatter plot
        fig = px.scatter(
            filtered_df,
            x=x_axis,
            y=y_axis,
            title=f"{y_axis} vs {x_axis}"
        )
        st.plotly_chart(fig)
        
    else:
        st.warning("Dataset needs at least 2 numeric columns for visualization")
else:
    st.info("Please upload a CSV file to begin analysis")
