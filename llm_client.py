"""
LLM Client using Strategy Pattern.

This module provides a unified interface for interacting with different LLM providers
using the strategy pattern to handle provider-specific implementations.
"""

import time
import json
import pandas as pd
from typing import Dict, Any, List

from open_router_llm_strategy import OpenRouterLLMStrategy
from instructor_llm_strategy import InstructorLLMStrategy

import models
import prompts


class LLMClient:
    """Client for interacting with LLM APIs using strategy pattern."""

    def __init__(self, provider: str, api_key: str):
        """
        Initialize the LLM client with the appropriate strategy.

        Args:
            provider: API provider name ("OpenRouter", "Gemini", or "OpenAI")
            api_key: API key for the selected provider
        """
        if provider == "OpenRouter":
            self._strategy = OpenRouterLLMStrategy(provider, api_key)
        else:
            self._strategy = InstructorLLMStrategy(provider, api_key)

    def _create_prompt(self, data_summary: Dict[str, Any]) -> str:
        """
        Create a prompt for the LLM to suggest column pairings.

        Args:
            data_summary: Dictionary containing data information

        Returns:
            Formatted prompt string
        """
        # Build column statistics string
        column_statistics = ""
        for stat in data_summary["statistics"]:
            column_statistics += f"- {stat['column']}: mean={stat['mean']:.2f}, std={stat['std']:.2f}, range=[{stat['min']:.2f}, {stat['max']:.2f}]\n"

        # Format the prompt using the template from prompts module
        prompt = prompts.COLUMN_SUGGESTION_BASE_PROMPT.format(
            total_rows=data_summary["total_rows"],
            total_columns=data_summary["total_columns"],
            column_names=", ".join(data_summary["column_names"]),
            column_statistics=column_statistics,
            sample_data=json.dumps(data_summary["sample_data"], indent=2),
        )

        return prompt

    def get_column_suggestions(
        self, data_summary: Dict[str, Any], max_retries: int = 3
    ) -> models.ColumnSuggestions:
        """
        Get column pairing suggestions from the LLM with retry logic.

        Args:
            data_summary: Dictionary containing data information
            max_retries: Maximum number of retry attempts

        Returns:
            ColumnSuggestions object

        Raises:
            Exception: If all retry attempts fail
        """
        prompt = self._create_prompt(data_summary)

        for attempt in range(max_retries):
            try:
                response = self._strategy.get_column_suggestions(prompt, data_summary)
                return response

            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(
                        f"Failed to get suggestions after {max_retries} attempts. Last error: {str(e)}"
                    )

                wait_time = 2**attempt
                time.sleep(wait_time)
                continue

        raise Exception("Unexpected error in retry loop")

    def generate_title(self, filename: str, columns: list[str]) -> str:
        """"""
        return self._strategy.generate_title(filename, columns)

    def _create_data_context(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str],
        pca_state=None,
        column_suggestions=None,
    ) -> str:
        """
        Create data context for chat interactions.

        Args:
            df: Current dataframe
            numeric_cols: List of numeric columns
            pca_state: PCA state if available
            column_suggestions: Column suggestions if available

        Returns:
            Formatted data context string
        """
        context = f"""
CURRENT DATASET CONTEXT:
- Total rows: {len(df)}
- Total columns: {len(df.columns)}
- Numeric columns: {len(numeric_cols)}
- Column names: {", ".join(numeric_cols[:10])}{"..." if len(numeric_cols) > 10 else ""}

BASIC STATISTICS:
"""
        # Add basic statistics for first few numeric columns
        for col in numeric_cols[:5]:
            if col in df.columns:
                series = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(series) > 0:
                    context += f"- {col}: mean={series.mean():.2f}, std={series.std():.2f}, range=[{series.min():.2f}, {series.max():.2f}]\n"

        if pca_state and pca_state.pca_object is not None:
            context += f"""
PCA ANALYSIS RESULTS:
- Principal components generated: {pca_state.min_components_95_variance}
- Columns used for PCA: {", ".join(pca_state.pca_cols)}
- Explained variance by components: {[f"{v:.3f}" for v in pca_state.explained_variance[:3]]}
- Cumulative variance (first 3 PCs): {[f"{v:.3f}" for v in pca_state.cumulative_variance[:3]]}
"""

        if column_suggestions and hasattr(column_suggestions, "suggestions"):
            context += f"""
GENERATED COLUMN SUGGESTIONS:
- Number of suggestions: {len(column_suggestions.suggestions)}
- Suggested pairs: {", ".join([f"({s.column1_name}, {s.column2_name})" for s in column_suggestions.suggestions[:3]])}
"""

        return context

    def stream_data_insights(
        self,
        user_message: str,
        chat_history: List[models.ChatMessage],
        df: pd.DataFrame,
        numeric_cols: List[str],
        pca_state=None,
        column_suggestions=None,
    ):
        """
        Stream data insights response for real-time chat experience with relevance checking.

        Args:
            user_message: User's question/message
            chat_history: Previous chat messages
            df: Current dataframe
            numeric_cols: List of numeric columns
            pca_state: PCA state if available
            column_suggestions: Column suggestions if available

        Yields:
            Response chunks for streaming
        """
        print(f"[DEBUG] LLMClient.stream_data_insights called with user_message: {user_message}")
        print(f"[DEBUG] Chat history length: {len(chat_history)}")
        print(f"[DEBUG] DataFrame shape: {df.shape}")
        print(f"[DEBUG] Numeric columns: {numeric_cols}")
        
        try:
            print(f"[DEBUG] Checking question relevance...")
            relevance_check = self._strategy.check_question_relevance(user_message)
            print(f"[DEBUG] Relevance check result: {relevance_check.is_data_related}, reasoning: {relevance_check.reasoning}")

            if not relevance_check.is_data_related:
                rejection_msg = (
                    "Sorry, I don't think that's relevant to analyzing this data!"
                )
                print(f"[DEBUG] Question not relevant, yielding rejection: {rejection_msg}")
                yield rejection_msg
                return rejection_msg

            print(f"[DEBUG] Question is relevant, proceeding with response generation...")
            system_prompt = prompts.DATA_ENGINEER_SYSTEM_PROMPT
            messages = [{"role": "system", "content": system_prompt}]

            # Only add data context if the question specifically needs dataset information
            # Use a simple heuristic to determine if data context is needed
            message_lower = user_message.lower()
            data_related_terms = [
                "dataset", "data", "columns", "statistics", "pca", "correlation", 
                "pattern", "trend", "analysis", "insight", "visualization", "rows",
                "values", "distribution", "mean", "std", "variance", "component"
            ]
            needs_data_context = any(term in message_lower for term in data_related_terms)
            print(f"[DEBUG] Needs data context: {needs_data_context}")

            if needs_data_context:
                data_context = self._create_data_context(
                    df, numeric_cols, pca_state, column_suggestions
                )
                print(f"[DEBUG] Created data context: {data_context[:200]}...")
                messages.append(
                    {
                        "role": "system",
                        "content": f"CURRENT DATA CONTEXT:\n{data_context}",
                    }
                )

            for msg in chat_history[-5:]:
                messages.append({"role": msg.role, "content": msg.content})

            messages.append({"role": "user", "content": user_message})
            
            print(f"[DEBUG] Final messages array has {len(messages)} messages")
            print(f"[DEBUG] Calling strategy.stream_data_insights...")

            # Delegate to strategy
            full_response = ""
            chunk_count = 0
            for content in self._strategy.stream_data_insights(messages):
                chunk_count += 1
                print(f"[DEBUG] LLMClient received chunk {chunk_count}: {content}")
                full_response += content
                yield content

            print(f"[DEBUG] LLMClient streaming completed. Total chunks: {chunk_count}")
            print(f"[DEBUG] LLMClient full response: {full_response}")
            return full_response

        except Exception as e:
            error_msg = f"Error getting insights: {str(e)}"
            print(f"[ERROR] LLMClient exception: {error_msg}")
            print(f"[ERROR] Exception type: {type(e)}")
            yield error_msg
            return error_msg
