"""
LLM Prompts for Manufacturing Analysis App

This module contains all the prompt strings used by the LLM client for various tasks
including column suggestions, title generation, chat responses, and validation.
"""

# =================================================================================================
# Column Suggestion Prompts
# =================================================================================================

COLUMN_SUGGESTION_BASE_PROMPT = """You are a data analysis expert specializing in manufacturing data. Analyze the following dataset and suggest meaningful column pairings for scatter plot visualization that could reveal insights, trends, correlations, or patterns.

Dataset Information:
- Total rows: {total_rows}
- Total columns: {total_columns}
- All columns are numerical data for scatter plot visualization

Available Columns: {column_names}

Column Statistics:
{column_statistics}

Sample Data:
{sample_data}

Please suggest 3-5 pairs of columns that would be most insightful to visualize together as scatter plots. For each pair, provide:
1. The exact column names (must match the column names provided above exactly)
2. A brief reasoning explaining what insights could be gained from the scatter plot

Focus on combinations that could reveal:
- Correlations between manufacturing variables
- Process relationships and dependencies
- Quality control patterns
- Efficiency metrics relationships
- Linear or non-linear relationships
- Clustering patterns or outliers

Return your response in the following JSON format:

{{
  "overall_analysis": "Brief analysis of the dataset characteristics and strategy",
  "suggestions": [
    {{
      "column1_name": "exact column name from the list above",
      "column2_name": "exact column name from the list above", 
      "reasoning": "explanation of insights this pairing could reveal"
    }}
  ]
}}

Ensure all column names match exactly with the provided column list."""

COLUMN_SUGGESTION_SYSTEM_PROMPT = (
    "You are a data analysis expert who suggests meaningful column pairings for "
    "manufacturing data visualization. Always respond with valid JSON in the specified format."
)

OPENROUTER_COLUMN_SUGGESTION_SYSTEM_PROMPT = (
    "You are a data analysis expert who suggests meaningful column pairings for "
    "manufacturing data visualization. Respond with valid JSON containing 'suggestions' array "
    "and 'overall_analysis' string."
)


# =================================================================================================
# Title Generation Prompts
# =================================================================================================

TITLE_GENERATION_PROMPT = (
    "Generate a concise, professional title (3-4 words) that implies the tool's purpose for "
    "analyzing manufacturing data from CSV file '{filename}' with columns: {columns}. Examples: "
    "'Manufacturing Data Analysis', 'Production Dashboard', 'Quality Control Hub', "
    "'Process Analytics Tool'."
)

OPENROUTER_TITLE_GENERATION_PROMPT = (
    "Generate a concise, professional title (3-4 words) for analyzing manufacturing data from "
    "'{filename}' with columns: {columns}. Return only the title, nothing else. Examples: "
    "Manufacturing Data Analysis, Production Dashboard, Quality Control Hub"
)

TITLE_GENERATION_SYSTEM_PROMPT = (
    "You are an expert at creating concise, professional titles for data analysis tools. "
    "Generate exactly 3-4 words that clearly indicate the purpose of the tool."
)

OPENROUTER_TITLE_GENERATION_SYSTEM_PROMPT = (
    "Generate exactly 3-4 words for a data analysis tool title. Return only the title."
)


# =================================================================================================
# Data Context Templates and Functions
# =================================================================================================

import pandas as pd
from typing import List, Optional, Any

DATA_CONTEXT_TEMPLATE = """
CURRENT DATASET CONTEXT:
- Total rows: {total_rows}
- Total columns: {total_columns}
- Numeric columns: {numeric_columns_count}
- Column names: {column_names}

BASIC STATISTICS:
{basic_statistics}
{pca_results}{column_suggestions_info}"""

PCA_RESULTS_TEMPLATE = """
PCA ANALYSIS RESULTS:
- Principal components generated: {min_components_95_variance}
- Columns used for PCA: {pca_cols}
- Explained variance by components: {explained_variance}
- Cumulative variance (first 3 PCs): {cumulative_variance}
"""

COLUMN_SUGGESTIONS_INFO_TEMPLATE = """
GENERATED COLUMN SUGGESTIONS:
- Number of suggestions: {suggestions_count}
- Suggested pairs: {suggested_pairs}
"""


def create_column_suggestion_prompt(data_summary: dict) -> str:
    """
    Create a prompt for the LLM to suggest column pairings.

    Parameters
    ----------
    data_summary : dict
        Dictionary containing data information including statistics,
        column names, and sample data

    Returns
    -------
    str
        Formatted prompt string ready for LLM consumption
    """
    import json

    # Build column statistics string
    column_statistics = ""
    for stat in data_summary["statistics"]:
        column_statistics += f"- {stat['column']}: mean={stat['mean']:.2f}, std={stat['std']:.2f}, range=[{stat['min']:.2f}, {stat['max']:.2f}]\n"

    # Format the prompt using the template
    prompt = COLUMN_SUGGESTION_BASE_PROMPT.format(
        total_rows=data_summary["total_rows"],
        total_columns=data_summary["total_columns"],
        column_names=", ".join(data_summary["column_names"]),
        column_statistics=column_statistics,
        sample_data=json.dumps(data_summary["sample_data"], indent=2),
    )

    return prompt


def create_data_context(
    df: pd.DataFrame,
    numeric_cols: List[str],
    pca_state: Optional[Any] = None,
    column_suggestions: Optional[Any] = None,
) -> str:
    """
    Create data context for chat interactions.

    Parameters
    ----------
    df : pd.DataFrame
        Current dataframe containing the dataset
    numeric_cols : List[str]
        List of numeric column names
    pca_state : Optional[Any], default=None
        PCA state object if PCA analysis has been performed
    column_suggestions : Optional[Any], default=None
        Column suggestions object if suggestions have been generated

    Returns
    -------
    str
        Formatted data context string for LLM consumption
    """
    # Build basic statistics
    basic_statistics = ""
    for col in numeric_cols[:5]:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(series) > 0:
                basic_statistics += f"- {col}: mean={series.mean():.2f}, std={series.std():.2f}, range=[{series.min():.2f}, {series.max():.2f}]\n"

    # Build PCA results section
    pca_results = ""
    if pca_state and pca_state.pca_object is not None:
        pca_results = PCA_RESULTS_TEMPLATE.format(
            min_components_95_variance=pca_state.min_components_95_variance,
            pca_cols=", ".join(pca_state.pca_cols),
            explained_variance=[f"{v:.3f}" for v in pca_state.explained_variance[:3]],
            cumulative_variance=[f"{v:.3f}" for v in pca_state.cumulative_variance[:3]],
        )

    # Build column suggestions section
    column_suggestions_info = ""
    if column_suggestions and hasattr(column_suggestions, "suggestions"):
        suggested_pairs = ", ".join(
            [
                f"({s.column1_name}, {s.column2_name})"
                for s in column_suggestions.suggestions[:3]
            ]
        )
        column_suggestions_info = COLUMN_SUGGESTIONS_INFO_TEMPLATE.format(
            suggestions_count=len(column_suggestions.suggestions),
            suggested_pairs=suggested_pairs,
        )

    # Build column names display
    column_names_display = ", ".join(numeric_cols[:10])
    if len(numeric_cols) > 10:
        column_names_display += "..."

    # Use the template to build the complete context
    return DATA_CONTEXT_TEMPLATE.format(
        total_rows=len(df),
        total_columns=len(df.columns),
        numeric_columns_count=len(numeric_cols),
        column_names=column_names_display,
        basic_statistics=basic_statistics,
        pca_results=pca_results,
        column_suggestions_info=column_suggestions_info,
    )


# =================================================================================================
# Chat System Prompts
# =================================================================================================

DATA_ENGINEER_SYSTEM_PROMPT = """You are a data analysis assistant. Your primary goal is to answer the user's question directly and concisely.

CRITICAL RULES:
1. Answer ONLY what the user specifically asked
2. Do NOT provide unsolicited data analysis or dataset overviews
3. Be concise and focused on the exact question
4. Only mention dataset details if the user specifically asks about the data

Guidelines:
- If asked about yourself: Give a brief, direct answer about being a data analysis assistant
- If asked about specific data analysis: Provide targeted insights for that specific question
- If asked about the dataset: Then and only then provide relevant dataset information
- If asked about correlations/patterns: Focus only on what was asked
- Always be concise and avoid information dumping

Remember: The user has data context available, but only provide it when specifically relevant to their exact question."""


# =================================================================================================
# Question Relevance Check Prompts
# =================================================================================================

QUESTION_RELEVANCE_SYSTEM_PROMPT = """
You are a highly discerning assistant for data visualization and analysis.
Your task is to classify user questions as either requiring specific data context (true) or not requiring data context (false).
Be strict in filtering out truly irrelevant topics, but allow for foundational data science and analysis concepts, and bias towards 'true' if truly unsure.

Return 'true' if the question clearly and directly asks about:
- The content, structure, or properties of the current dataset (e.g., columns, rows, data types, unique values, specific values, important features).
- Analysis, insights, or interpretations derived directly from the current dataset (e.g., trends, patterns, correlations, distributions, statistical summaries, data visualization, machine learning results applied to this data).
- Actions related to processing or exploring this specific dataset (e.g., cleaning, preprocessing, transformation suggestions for this data).
- General data science concepts, methods, or visualizations that are foundational to data analysis (e.g., explaining what a scatter plot is, defining correlation, describing PCA).

Return 'false' if the question asks about:
- Your identity, functionality, or general capabilities as an AI.
- Any topic completely unrelated to data analysis or the concepts within it (e.g., general knowledge, personal advice, current events, entertainment, cooking, programming languages not related to data processing).

Output a single boolean value: true or false.

Examples:
Dataset columns: "cycle_time", "temperature"

Q: Which column is the most important to explore deeper?
A: true

Q: Can you offer some insights?
A: true

Q: What are the columns in this dataset?
A: true

Q: Explain what a scatter plot is.
A: true

Q: What is the capital of France?
A: false

Q: How do I bake a cake?
A: false
"""

QUESTION_RELEVANCE_PROMPT = (
    "Does this question need information about the current dataset: '{user_message}'"
)


# =================================================================================================
# Response Validation Prompts
# =================================================================================================

RESPONSE_VALIDATION_SYSTEM_PROMPT = """You are an expert at validating AI responses for quality and relevance.

Evaluate if the response:
1. Is concise and to the point (not overly verbose)
2. Directly addresses the user's question
3. Doesn't force unrelated topics into the answer

A good response should be focused, relevant, and appropriately sized for the question asked."""

RESPONSE_VALIDATION_PROMPT = (
    "User asked: '{user_message}'\n\nAI responded: '{response}'\n\nIs this response concise "
    "and does it address the question?"
)
