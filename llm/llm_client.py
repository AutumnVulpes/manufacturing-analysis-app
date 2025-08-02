"""
LLM Client using Strategy Pattern.

This module provides a unified interface for interacting with different LLM providers
using the strategy pattern to handle provider-specific implementations.
"""

import pandas as pd
from typing import Dict, Any, List, Optional, Generator

from .open_router_llm_strategy import OpenRouterLLMStrategy
from .instructor_llm_strategy import InstructorLLMStrategy

import models
from . import prompts
from .prompts import create_data_context, create_column_suggestion_prompt


class LLMClient:
    """
    Client for interacting with LLM APIs using strategy pattern.

    This class provides a unified interface for different LLM providers,
    automatically selecting the appropriate strategy based on the provider.
    """

    def __init__(self, provider: str, api_key: str):
        """
        Initialize the LLM client with the appropriate strategy.

        Parameters
        ----------
        provider : str
            API provider name ("OpenRouter", "Gemini", or "OpenAI")
        api_key : str
            API key for the selected provider
        """
        if provider == "OpenRouter":
            self._strategy = OpenRouterLLMStrategy(provider, api_key)
        else:
            self._strategy = InstructorLLMStrategy(provider, api_key)

    def get_column_suggestions(
        self, data_summary: Dict[str, Any]
    ) -> models.ColumnSuggestions:
        """
        Get column pairing suggestions from the LLM.

        Parameters
        ----------
        data_summary : Dict[str, Any]
            Dictionary containing data information including statistics,
            column names, and sample data

        Returns
        -------
        models.ColumnSuggestions
            Object containing suggested column pairings and analysis

        Raises
        ------
        Exception
            If all retry attempts fail
        """
        prompt = create_column_suggestion_prompt(data_summary)
        return self._strategy.get_column_suggestions(prompt, data_summary)

    def generate_title(self, filename: str, columns: list[str]) -> str:
        """Generate a title for the data analysis dashboard."""
        return self._strategy.generate_title(filename, columns)

    def stream_data_insights(
        self,
        user_message: str,
        chat_history: List[models.ChatMessage],
        df: pd.DataFrame,
        numeric_cols: List[str],
        pca_state: Optional[Any] = None,
        column_suggestions: Optional[Any] = None,
    ) -> Generator[str, None, str]:
        """
        Stream data insights response for real-time chat experience with relevance checking.

        Parameters
        ----------
        user_message : str
            User's question/message
        chat_history : List[models.ChatMessage]
            Previous chat messages for context
        df : pd.DataFrame
            Current dataframe containing the dataset
        numeric_cols : List[str]
            List of numeric column names
        pca_state : optional
            PCA state object if PCA analysis has been performed
        column_suggestions : optional
            Column suggestions object if suggestions have been generated

        Yields
        ------
        str
            Response chunks for streaming display
        """
        print(
            f"[DEBUG] LLMClient.stream_data_insights called with user_message: {user_message}"
        )
        print(f"[DEBUG] Chat history length: {len(chat_history)}")
        print(f"[DEBUG] DataFrame shape: {df.shape}")
        print(f"[DEBUG] Numeric columns: {numeric_cols}")

        try:
            print(f"[DEBUG] Checking question relevance...")
            relevance_check = self._strategy.check_question_relevance(user_message)
            print(
                f"[DEBUG] Relevance check result: {relevance_check.is_data_related}, reasoning: {relevance_check.reasoning}"
            )

            if not relevance_check.is_data_related:
                rejection_msg = (
                    "Sorry, I don't think that's relevant to analyzing this data!"
                )
                print(
                    f"[DEBUG] Question not relevant, yielding rejection: {rejection_msg}"
                )
                yield rejection_msg
                return rejection_msg

            print(
                f"[DEBUG] Question is relevant, proceeding with response generation..."
            )
            system_prompt = prompts.DATA_ENGINEER_SYSTEM_PROMPT
            messages = [{"role": "system", "content": system_prompt}]

            # Only add data context if the question specifically needs dataset information
            # Use a simple heuristic to determine if data context is needed
            message_lower = user_message.lower()
            data_related_terms = [
                "dataset",
                "data",
                "columns",
                "statistics",
                "pca",
                "correlation",
                "pattern",
                "trend",
                "analysis",
                "insight",
                "visualization",
                "rows",
                "values",
                "distribution",
                "mean",
                "std",
                "variance",
                "component",
            ]
            needs_data_context = any(
                term in message_lower for term in data_related_terms
            )
            print(f"[DEBUG] Needs data context: {needs_data_context}")

            if needs_data_context:
                data_context = create_data_context(
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
