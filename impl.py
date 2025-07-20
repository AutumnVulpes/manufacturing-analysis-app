import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from datatypes import TitleSuggestion, ColumnPair, ColumnComparisonSuggestions, AppState
from openai import OpenAI
import instructor
import json

# =================================================================================================
# States and Computations
# =================================================================================================


@st.cache_data
def clean_csv_file(csv_file) -> tuple[pd.DataFrame, list[str]]:
    """
    Clean CSV file by removing rows with missing values and extract numeric columns.
    
    Args:
        csv_file: Uploaded CSV file object from Streamlit file uploader
        
    Returns:
        tuple: (cleaned_dataframe, list_of_numeric_column_names)
    """
    df = pd.read_csv(csv_file)
    cleaned_df = df.dropna()
    numeric_cols = cleaned_df.select_dtypes(
        include=["float64", "int64"]
    ).columns.tolist()
    return cleaned_df, numeric_cols


@st.cache_data
def load_css() -> None:
    """
    Load and apply CSS styles from styles.css file to the Streamlit app.
    
    This function is cached to avoid reloading CSS on every app rerun.
    """
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def is_valid_plot_config(df: pd.DataFrame, x: str, y: str) -> bool:
    """
    Validate if the plot configuration is valid for creating visualizations.
    
    Args:
        df: DataFrame to check
        x: X-axis column name
        y: Y-axis column name
        
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    return not df.empty and x and y and x in df.columns and y in df.columns


def get_api_client_config(provider: str, api_key: str) -> tuple[str, dict, str]:
    """
    Get API client configuration based on provider.
    
    Args:
        provider: API provider name ("OpenRouter", "Gemini", or "OpenAI")
        api_key: API key for the selected provider
        
    Returns:
        tuple: (base_url, default_headers, model) configuration for the provider
    """
    if provider == "OpenRouter":
        base_url = "https://openrouter.ai/api/v1"
        default_headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referrer": "https://manufacturing-analysis-app.streamlit.app",
            "X-Title": "Manufacturing Analysis Dashboard",
        }
        model = "mistralai/mistral-7b-instruct:free"
    elif provider == "Gemini":
        base_url = "https://generativelanguage.googleapis.com/v1beta"
        default_headers = {"x-goog-api-key": api_key}
        model = "gemini-pro"
    else:  # OpenAI
        base_url = "https://api.openai.com/v1"
        default_headers = {"Authorization": f"Bearer {api_key}"}
        model = "gpt-3.5-turbo"
    
    return base_url, default_headers, model


def generate_title_from_csv(
    provider: str, api_key: str, filename: str, columns: list[str]
) -> str:
    """
    Generate a concise title for the CSV file using AI.
    
    Args:
        provider: API provider name ("OpenRouter", "Gemini", or "OpenAI")
        api_key: API key for the selected provider
        filename: Name of the uploaded CSV file
        columns: List of column names from the CSV file
        
    Returns:
        str: Generated title (max 3 words) or empty string if generation fails
    """
    try:
        base_url, default_headers, model = get_api_client_config(provider, api_key)

        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            default_headers=default_headers,
        )

        prompt = (
            f"Generate a concise, professional title (max 3 words) for a manufacturing data analysis "
            f"dashboard based on CSV file '{filename}' with columns: {', '.join(columns)}. Do NOT include 'Analysis' or 'Dashboard' in the title."
        )

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.7,
        )
        title = response.choices[0].message.content
        if title:
            try:
                data = json.loads(title)
                if isinstance(data, dict):
                    title_suggestion = TitleSuggestion(**data)
                    return title_suggestion.title
                elif isinstance(data, str):
                    return data
                else:
                    return str(data)
            except json.JSONDecodeError:
                # Use title directly because response is a useful string regardless.
                return title
        return ""
    except Exception as e:
        st.error(f"Title generation failed: {e}")
        return ""


def generate_column_comparison_suggestions_stream(
    provider: str, api_key: str, numeric_cols: list[str], df_sample: pd.DataFrame
) -> object | None:
    """
    Generate streaming multiple column comparison suggestions using Instructor with partial tokens.
    
    Args:
        provider: API provider name ("OpenRouter", "Gemini", or "OpenAI")
        api_key: API key for the selected provider
        numeric_cols: List of numeric column names from the dataset
        df_sample: Sample DataFrame for statistical analysis
        
    Returns:
        object | None: Streaming iterator of ColumnComparisonSuggestions or None if failed
    """
    try:
        base_url, default_headers, model = get_api_client_config(provider, api_key)

        # Create sample statistics for context, ensuring we handle numeric data properly
        stats_info = []
        for col in numeric_cols[:10]:
            if col in df_sample.columns:
                # Ensure we're working with numeric data only
                numeric_data = pd.to_numeric(df_sample[col], errors='coerce').dropna()
                if len(numeric_data) > 0:
                    stats = {
                        "column": col,
                        "mean": float(numeric_data.mean()),
                        "std": float(numeric_data.std()),
                        "min": float(numeric_data.min()),
                        "max": float(numeric_data.max()),
                        "count": int(len(numeric_data))
                    }
                    stats_info.append(stats)

        # OpenRouter doesn't support function calling, use a simpler approach
        if provider == "OpenRouter":
            return _generate_suggestions_openrouter(base_url, default_headers, model, api_key, numeric_cols, stats_info)
        else:
            # Use Instructor for Gemini and OpenAI
            openai_client = OpenAI(
                base_url=base_url,
                api_key=api_key,
                default_headers=default_headers,
            )
            
            client = instructor.from_openai(openai_client, mode=instructor.Mode.JSON)

            prompt = f"""Analyze the following numeric columns from a manufacturing dataset and suggest multiple pairs of columns to compare in visualizations.

            Available columns: {", ".join(numeric_cols)}
            
            Sample statistics:
            {json.dumps(stats_info, indent=2)}
            
            Provide 3-5 different column pair recommendations, ranked by priority (1=highest, 2=medium, 3=lowest).
            
            Consider factors like:
            - Potential correlations between variables
            - Manufacturing process relationships
            - Data variance and distribution
            - Business insights that could be gained
            - Different types of relationships (linear, non-linear, categorical vs continuous)
            
            For each recommendation, provide detailed reasoning explaining why this pair would be valuable to analyze.
            Also provide an overall analysis summary explaining the dataset characteristics and analysis strategy."""

            # Streaming to indicate that the suggestions are AI generated.
            suggestions_stream = client.chat.completions.create_partial(
                model=model,
                response_model=ColumnComparisonSuggestions,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                stream=True,
            )

            return suggestions_stream

    except Exception as e:
        st.error(f"Column comparison suggestions failed: {e}")
        return None


def _generate_suggestions_openrouter(base_url: str, default_headers: dict, model: str, api_key: str, numeric_cols: list[str], stats_info: list) -> object:
    """
    Generate column suggestions for OpenRouter using regular chat completions (no function calling).
    
    Args:
        base_url: API base URL
        default_headers: Request headers
        model: Model name
        api_key: API key
        numeric_cols: List of numeric column names
        stats_info: Statistical information about columns
        
    Returns:
        Generator yielding partial ColumnComparisonSuggestions objects
    """
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        default_headers=default_headers,
    )

    # Simplified prompt for better streaming experience
    prompt = f"""Analyze these manufacturing dataset columns and suggest column pairs for visualization:

Available columns: {", ".join(numeric_cols)}

First, provide an overall analysis of the dataset characteristics and strategy.
Then suggest 3-4 column pairs ranked by priority (1=highest priority).

For each pair, explain why it would be valuable to analyze.

Format your response as:

OVERALL ANALYSIS:
[Your analysis here]

RECOMMENDATIONS:
1. [Column1] vs [Column2] - Priority 1
   Reasoning: [explanation]

2. [Column1] vs [Column2] - Priority 2  
   Reasoning: [explanation]

[Continue for 3-4 pairs total]"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            stream=True,
        )

        # Stream the response with text-based parsing
        full_response = ""
        import re
        import time
        
        # Track streaming state
        last_yielded_length = 0
        analysis_complete = False
        
        for chunk in response:
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                
                # Add delay for visible streaming
                time.sleep(0.03)
                
                # Stream partial analysis
                if "OVERALL ANALYSIS:" in full_response and not analysis_complete:
                    # Extract analysis section
                    analysis_start = full_response.find("OVERALL ANALYSIS:") + len("OVERALL ANALYSIS:")
                    analysis_end = full_response.find("RECOMMENDATIONS:")
                    
                    if analysis_end == -1:
                        # Still building analysis
                        current_analysis = full_response[analysis_start:].strip()
                    else:
                        # Analysis complete
                        current_analysis = full_response[analysis_start:analysis_end].strip()
                        analysis_complete = True
                    
                    # Only yield if we have significant new content
                    if len(current_analysis) > last_yielded_length + 20:
                        last_yielded_length = len(current_analysis)
                        
                        # Create partial suggestion object
                        partial_suggestions = type('PartialSuggestions', (), {
                            'overall_analysis': current_analysis,
                            'recommendations': []
                        })()
                        yield partial_suggestions
        
        # Parse final response into structured format
        try:
            # Extract overall analysis
            analysis_match = re.search(r'OVERALL ANALYSIS:\s*(.*?)\s*RECOMMENDATIONS:', full_response, re.DOTALL)
            overall_analysis = analysis_match.group(1).strip() if analysis_match else "Analysis not found"
            
            # Extract recommendations
            recommendations = []
            rec_pattern = r'(\d+)\.\s*([^-\s]+)\s*vs\s*([^-\s]+)\s*-\s*Priority\s*(\d+)\s*Reasoning:\s*([^\n]+)'
            matches = re.findall(rec_pattern, full_response, re.IGNORECASE)
            
            for match in matches[:4]:  # Limit to 4 recommendations
                rec_num, col1, col2, priority, reasoning = match
                
                # Clean column names
                col1 = col1.strip().strip('[]()').strip()
                col2 = col2.strip().strip('[]()').strip()
                
                # Find matching columns from available list
                col1_match = next((col for col in numeric_cols if col.lower() in col1.lower() or col1.lower() in col.lower()), col1)
                col2_match = next((col for col in numeric_cols if col.lower() in col2.lower() or col2.lower() in col.lower()), col2)
                
                recommendations.append(ColumnPair(
                    column1=col1_match,
                    column2=col2_match,
                    reasoning=reasoning.strip(),
                    visualization_type="scatter",
                    priority=int(priority) if priority.isdigit() else 1
                ))
            
            # Create final suggestions object
            final_suggestions = ColumnComparisonSuggestions(
                overall_analysis=overall_analysis,
                recommendations=recommendations
            )
            
            yield final_suggestions
            
        except Exception as e:
            st.error(f"Failed to parse OpenRouter response: {e}")
            # Show raw response for debugging
            st.text_area("Raw response for debugging:", full_response[:1000], height=200)
            
    except Exception as e:
        st.error(f"OpenRouter API call failed: {e}")
        return None


# =================================================================================================
# Widgets
# =================================================================================================

# Components --------------------------------------------------------------------------------------


def create_download_plotly_figure_button(fig, filename: str = "plot.png") -> None:
    """
    Convert Plotly figure to PNG and create a Streamlit download button.
    
    Args:
        fig: Plotly figure object to convert and download
        filename: Name for the downloaded file (default: "plot.png")
    """
    img_bytes = fig.to_image(format="png")
    st.download_button(
        "Download Graph as PNG",
        img_bytes,
        filename,
        "image/png",
        key=f"download-png-{filename.replace('.', '-')}",
    )


def render_github_footer() -> None:
    """
    Render a GitHub footer with repository link and author information.
    
    Displays a GitHub logo and link that adapts to the current Streamlit theme
    (dark or light mode) with appropriate colors.
    """
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


# Tabs --------------------------------------------------------------------------------------------


def render_cumulative_variance_tab(app_state):
    """Renders cumulative explained variance visualization tab."""
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


def render_ai_helper_tab(app_state: AppState, uploaded_csv_file):
    """
    Renders the AI Helper tab UI.

    Returns updated app_state and a rerun flag.
    """
    st.subheader("API Key Management")
    rerun_needed = False

    with st.expander("Add API Key"):
        key_name = st.text_input("Name", key="new_key_name")
        provider = st.selectbox(
            "Provider", ["OpenRouter", "Gemini", "OpenAI"], key="new_key_provider"
        )
        key_value = st.text_input("Key", type="password", key="new_key_value")
        if st.button("Add Key") and key_name and key_value:
            app_state.api_keys[key_name] = (provider, key_value)
            app_state.active_api_key_name = key_name
            app_state.active_provider = provider
            st.success(f"Added key: {key_name} for {provider}")

    # TODO(Fox) - Store single API key in session state only.
    if app_state.api_keys:
        selected_key = st.selectbox(
            "Active API Key",
            options=list(app_state.api_keys.keys()),
            index=list(app_state.api_keys.keys()).index(app_state.active_api_key_name)
            if app_state.active_api_key_name
            and app_state.active_api_key_name in app_state.api_keys
            else 0,
            format_func=lambda key: f"{key} ({app_state.api_keys[key][0]})",
        )
        if selected_key != app_state.active_api_key_name:
            app_state.active_api_key_name = selected_key
            app_state.active_provider = app_state.api_keys[selected_key][0]

    # Generate title when conditions are met.
    if (
        app_state.active_api_key_name
        and uploaded_csv_file
        and app_state.last_processed_filename != uploaded_csv_file.name
    ):
        with st.spinner("Generating title using AI..."):
            _, key_value = app_state.api_keys[app_state.active_api_key_name]
            title = generate_title_from_csv(
                app_state.active_provider,
                key_value,
                uploaded_csv_file.name,
                app_state.cleaned_df.columns.tolist(),
            )
            if title:
                app_state.generated_title = title
                app_state.last_processed_filename = uploaded_csv_file.name
                st.success(f"Generated title: {title}")
                rerun_needed = True

    # Column Comparison Suggestion Feature.
    # TODO(Fox) - Recommendations should be MULTIPLE pairs.
    st.subheader("Column Comparison Suggestions")

    if (
        app_state.active_api_key_name
        and not app_state.cleaned_df.empty
        and len(app_state.numeric_cols) >= 2
    ):
        if st.button(
            "🤖 Get AI Column Comparison Suggestions", key="column_suggestion_btn"
        ):
            _, key_value = app_state.api_keys[app_state.active_api_key_name]

            # Create placeholders for streaming content.
            suggestion_container = st.container()

            with suggestion_container:
                st.write("**Generating column recommendations...**")

                try:
                    suggestions_stream = generate_column_comparison_suggestions_stream(
                        app_state.active_provider,
                        key_value,
                        app_state.numeric_cols,
                        app_state.cleaned_df.head(100),  # Use sample for analysis.
                    )

                    if suggestions_stream:
                        # Initialize display variables.
                        current_suggestions = None
                        
                        # Create placeholders for streaming content
                        overall_analysis_placeholder = st.empty()
                        recommendations_placeholder = st.empty()

                        # Stream the partial responses.
                        for partial_suggestions in suggestions_stream:
                            current_suggestions = partial_suggestions

                            # Update overall analysis
                            if (
                                hasattr(partial_suggestions, "overall_analysis")
                                and partial_suggestions.overall_analysis
                            ):
                                overall_analysis_placeholder.write(
                                    f"**Overall Analysis:** {partial_suggestions.overall_analysis}"
                                )

                            # Update recommendations
                            if (
                                hasattr(partial_suggestions, "recommendations")
                                and partial_suggestions.recommendations
                            ):
                                with recommendations_placeholder.container():
                                    st.write("**Column Pair Recommendations:**")
                                    for i, rec in enumerate(partial_suggestions.recommendations):
                                        if hasattr(rec, 'column1') and hasattr(rec, 'column2'):
                                            priority_emoji = "🥇" if rec.priority == 1 else "🥈" if rec.priority == 2 else "🥉"
                                            st.write(f"{priority_emoji} **Pair {i+1}:** {rec.column1} vs {rec.column2}")
                                            if hasattr(rec, 'reasoning') and rec.reasoning:
                                                st.write(f"   *Reasoning:* {rec.reasoning}")
                                            if hasattr(rec, 'visualization_type') and rec.visualization_type:
                                                st.write(f"   *Visualization:* {rec.visualization_type}")
                                            st.write("")

                        # Final suggestions received.
                        if current_suggestions and hasattr(current_suggestions, 'recommendations') and current_suggestions.recommendations:
                            st.success("✅ Analysis Complete!")

                            # Add buttons to apply suggestions
                            st.write("**Apply a recommendation:**")
                            cols = st.columns(min(len(current_suggestions.recommendations), 3))
                            
                            for i, rec in enumerate(current_suggestions.recommendations[:3]):  # Show max 3 buttons
                                with cols[i]:
                                    priority_emoji = "🥇" if rec.priority == 1 else "🥈" if rec.priority == 2 else "🥉"
                                    if st.button(
                                        f"{priority_emoji} Apply Pair {i+1}", 
                                        key=f"apply_suggestion_{i}_btn"
                                    ):
                                        if (
                                            rec.column1 in app_state.numeric_cols
                                            and rec.column2 in app_state.numeric_cols
                                        ):
                                            app_state.x_axis = rec.column1
                                            app_state.y_axis = rec.column2
                                            st.success(f"Applied: {rec.column1} vs {rec.column2}")
                                            rerun_needed = True
                                        else:
                                            st.error("Suggested columns not found in dataset")
                            
                            # Clear button
                            if st.button("Clear Suggestions", key="clear_suggestions_btn"):
                                st.rerun()

                except Exception as e:
                    st.error(f"Failed to generate column suggestions: {e}")

    elif not app_state.active_api_key_name:
        st.info("Add an API key to use AI column comparison suggestions")
    elif app_state.cleaned_df.empty:
        st.info("Upload a CSV file to get column comparison suggestions")
    elif len(app_state.numeric_cols) < 2:
        st.info("Dataset needs at least 2 numeric columns for comparison suggestions")

    return app_state, rerun_needed


def render_pca_formulas_tab(app_state):
    """Renders the PCA formulas tab UI."""
    st.subheader("Principal Component Formulas")
    if app_state.pca_state.pca_object is not None and app_state.pca_state.pca_cols:
        pca_obj = app_state.pca_state.pca_object
        pca_cols_for_formula = app_state.pca_state.pca_cols

        n_optimal = app_state.pca_state.min_components_95_variance

        # Render latex formulas for optimal principal components.
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
    # Renders PCA configuration UI and handles PCA processing.
    # Returns updated df, numeric_cols, and app_state.
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

        # Determine number of optimal principal components to reach 95% variance.
        min_components_95_variance = next(
            (i + 1 for i, v in enumerate(cumulative_variance) if v >= 0.95),
            len(cumulative_variance),
        )
        app_state.pca_state.min_components_95_variance = min_components_95_variance

        # Append columns with optimal principal components to df.
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
