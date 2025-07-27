"""
Abstract base class for LLM strategies.

This module defines the strategy pattern interface for different LLM providers,
allowing for clean separation of provider-specific logic while maintaining
a consistent interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
import models

class LLMStrategyInterface(ABC):
    """Abstract base class for LLM provider strategies."""

    def __init__(self, provider: str, api_key: str):
        """
        Initialize the strategy with provider and API key.

        Args:
            provider: API provider name
            api_key: API key for the provider
        """
        self.provider = provider
        self.api_key = api_key
        self.base_url, self.default_headers, self.model = self._get_api_config()

    @abstractmethod
    def _get_api_config(self) -> tuple[str, dict, str]:
        """
        Get API client configuration based on provider.

        Returns:
            tuple: (base_url, default_headers, model) configuration for the provider
        """
        pass

    @abstractmethod
    def get_column_suggestions(
        self, prompt: str, data_summary: Dict[str, Any]
    ) -> models.ColumnSuggestions:
        """
        Get column pairing suggestions from the LLM.

        Args:
            prompt: The formatted prompt for column suggestions
            data_summary: Dictionary containing data information

        Returns:
            ColumnSuggestions object
        """
        pass

    @abstractmethod
    def generate_title(self, filename: str, columns: List[str]) -> str:
        """
        Generate a concise title for the CSV file.

        Args:
            filename: Name of the uploaded CSV file
            columns: List of column names from the CSV file

        Returns:
            Generated title string
        """
        pass

    @abstractmethod
    def check_question_relevance(self, user_message: str) -> models.IsChatQueryRelevant:
        """
        Check if the user's question is relevant to data engineering/analysis.

        Args:
            user_message: User's question/message

        Returns:
            IsChatQueryRelevant object indicating if question is data-related
        """
        pass

    @abstractmethod
    def validate_response(
        self, response: str, user_message: str
    ) -> models.IsChatResponseValid:
        """
        Validate that the response is concise and addresses the user's question.

        Args:
            response: The AI response to validate
            user_message: The original user question

        Returns:
            IsChatResponseValid object with validation results
        """
        pass

    @abstractmethod
    def stream_data_insights(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ):
        """
        Stream data insights response for real-time chat experience.

        Args:
            messages: List of message dictionaries for the conversation
            **kwargs: Additional parameters for streaming

        Yields:
            Response chunks for streaming
        """
        pass
