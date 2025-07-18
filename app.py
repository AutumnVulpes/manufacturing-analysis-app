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
)
from pydantic import BaseModel
import instructor
import json
from openai import OpenAI
from typing import Optional

# Define Pydantic model for title suggestions
class TitleSuggestion(BaseModel):
    title: str

# Updated function for title generation with provider support
def generate_title_from_csv(provider: str, api_key: str, filename: str, columns: list[str]) -> str:
    try:
        # Configure client based on provider
        if provider == "OpenRouter":
            base_url = "https://openrouter.ai/api/v1"
            default_headers = {
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://manufacturing-analysis-app.streamlit.app",
                "X-Title": "Manufacturing Analysis Dashboard"
            }
        elif provider == "Gemini":
            base_url = "https://generativelanguage.googleapis.com/v1beta"
            default_headers = {"x-goog-api-key": api_key}
        else:  # OpenAI
            base_url = "https://api.openai.com/v1"
            default_headers = {"Authorization": f"Bearer {api_key}"}

        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            default_headers=default_headers,
        )
        
        # Updated prompt for 3-word titles without "Analysis Dashboard"
        prompt = f"Generate a concise, professional title (max 3 words) for a manufacturing data analysis dashboard based on CSV file '{filename}' with columns: {', '.join(columns)}. Do NOT include 'Analysis' or 'Dashboard' in the title."
        
        response = client.chat.completions.create(
            model="mistralai/mistral-7b-instruct:free",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.7
        )
        # Extract content and try to parse as JSON
        content = response.choices[0].message.content
        if content:
            try:
                # Try parsing as JSON
                data = json.loads(content)
                # Handle both object-formatted and string-formatted responses
                if isinstance(data, dict):
                    title_suggestion = TitleSuggestion(**data)
                    return title_suggestion.title
                elif isinstance(data, str):
                    return data
                else:
                    return str(data)
            except json.JSONDecodeError:
                # If not JSON, use content directly as title
                return content
        return ""
    except Exception as e:
        st.error(f"Title generation failed: {e}")
        return ""  # Return empty string instead of None

# Set explicit headless parameters for Chromium for streamlit cloud
# Remove if only deploying locally
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
        generated_title="",
        last_processed_filename="",
    )

app_state = st.session_state.app_state

# Set title conditionally
if app_state.generated_title:
    st.title(f"{app_state.generated_title} Analysis Dashboard")
else:
    st.title("Vulpes' Data Analysis Dashboard")

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
            ["PCA Config", "PCA Formulas", "Visualization Config", "AI Helper"]
        )
        with tab_pca:
            app_state.cleaned_df, app_state.numeric_cols, app_state = render_pca_tab(
                app_state.cleaned_df, app_state.numeric_cols, app_state
            )

        with tab_pca_formulas:
            render_pca_formulas_tab(app_state)

        with tab_viz:
            app_state.filtered_df, app_state.x_axis, app_state.y_axis, app_state = (
                render_viz_config_tab(
                    app_state.cleaned_df,
                    app_state.numeric_cols,
                    app_state.x_axis,
                    app_state.y_axis,
                    app_state,
                )
            )
            
        # AI Helper tab implementation
    with tab_ai:
        st.subheader("API Key Management")
        
        with st.expander("Add API Key"):
            key_name = st.text_input("Name", key="new_key_name")
            provider = st.selectbox(
                "Provider", 
                ["OpenRouter", "Gemini", "OpenAI"], 
                key="new_key_provider"
            )
            key_value = st.text_input("Key", type="password", key="new_key_value")
            if st.button("Add Key") and key_name and key_value:
                # Store key with provider info
                app_state.api_keys[key_name] = (provider, key_value)
                app_state.active_api_key_name = key_name
                app_state.active_provider = provider
                st.session_state.app_state = app_state
                st.success(f"Added key: {key_name} for {provider}")
        
        if app_state.api_keys:
            selected_key = st.selectbox(
                "Active API Key",
                options=list(app_state.api_keys.keys()),
                index=list(app_state.api_keys.keys()).index(app_state.active_api_key_name) 
                if app_state.active_api_key_name and app_state.active_api_key_name in app_state.api_keys 
                else 0,
                format_func=lambda key: f"{key} ({app_state.api_keys[key][0]})"
            )
            if selected_key != app_state.active_api_key_name:
                app_state.active_api_key_name = selected_key
                # Update active provider when key changes
                app_state.active_provider = app_state.api_keys[selected_key][0]
                st.session_state.app_state = app_state
        else:
            st.info("No API keys added yet")
        
        # Generate title when conditions are met
        if (app_state.active_api_key_name and 
            uploaded_csv_file and 
            app_state.last_processed_filename != uploaded_csv_file.name):
            with st.spinner("Generating title using AI..."):
                # Get the key value from the stored tuple
                _, key_value = app_state.api_keys[app_state.active_api_key_name]
                title = generate_title_from_csv(
                    app_state.active_provider,
                    key_value, 
                    uploaded_csv_file.name, 
                    app_state.cleaned_df.columns.tolist()
                )
                if title:
                    app_state.generated_title = title
                    app_state.last_processed_filename = uploaded_csv_file.name
                    st.session_state.app_state = app_state
                    st.success(f"Generated title: {title}")
                    # Update title immediately and rerun
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
