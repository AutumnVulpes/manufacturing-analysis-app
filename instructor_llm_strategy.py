"""
Instructor LLM Strategy implementation.

This module implements the LLM strategy for providers that support function calling
using instructor for structured responses with Pydantic models.
"""

import instructor
from openai import OpenAI
import streamlit as st
from typing import Dict, Any, List

from llm_strategy_interface import LLMStrategyInterface
import models
import prompts


class InstructorLLMStrategy(LLMStrategyInterface):
    """Strategy implementation for providers supporting instructor."""

    def __init__(self, provider: str, api_key: str):
        """Initialize Instructor strategy."""
        super().__init__(provider, api_key)
        openai_client = OpenAI(
            base_url=self.base_url,
            api_key=api_key,
            default_headers=self.default_headers,
        )
        self.client = instructor.from_openai(
            openai_client, mode=instructor.Mode.JSON
        )

    def _get_api_config(self) -> tuple[str, dict, str]:
        """Get API configuration based on provider."""
        if self.provider == "Gemini":
            base_url = "https://generativelanguage.googleapis.com/v1beta"
            default_headers = {"x-goog-api-key": self.api_key}
            model = "gemini-pro"
        else:  # OpenAI
            base_url = "https://api.openai.com/v1"
            default_headers = {"Authorization": f"Bearer {self.api_key}"}
            model = "gpt-3.5-turbo"

        return base_url, default_headers, model

    def get_column_suggestions(
        self, prompt: str, data_summary: Dict[str, Any]
    ) -> models.ColumnSuggestions:
        """Get column suggestions using instructor."""
        response = self.client.chat.completions.create(
            model=self.model,
            response_model=models.ColumnSuggestions,
            messages=[
                {
                    "role": "system",
                    "content": prompts.COLUMN_SUGGESTION_SYSTEM_PROMPT,
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=1500,
        )

        # Validate suggestions against available columns
        valid_suggestions = []
        available_columns = set(data_summary["column_names"])

        for suggestion in response.suggestions:
            if (
                suggestion.column1_name in available_columns
                and suggestion.column2_name in available_columns
                and suggestion.column1_name != suggestion.column2_name
            ):
                valid_suggestions.append(suggestion)

        return models.ColumnSuggestions(
            suggestions=valid_suggestions,
            overall_analysis=response.overall_analysis,
        )

    def generate_title(self, filename: str, columns: List[str]) -> str:
        """Generate title using instructor."""
        try:
            prompt = prompts.TITLE_GENERATION_PROMPT.format(
                filename=filename,
                columns=', '.join(columns)
            )

            response = self.client.chat.completions.create(
                model=self.model,
                response_model=models.TitleSuggestion,
                messages=[
                    {
                        "role": "system",
                        "content": prompts.TITLE_GENERATION_SYSTEM_PROMPT,
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=50,
            )

            return response.title.strip()

        except Exception as e:
            st.error(f"Title generation failed: {e}")
            return ""

    def check_question_relevance(self, user_message: str) -> models.IsChatQueryRelevant:
        """Check question relevance using instructor."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                response_model=models.IsChatQueryRelevant,
                messages=[
                    {
                        "role": "system",
                        "content": prompts.QUESTION_RELEVANCE_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": prompts.QUESTION_RELEVANCE_PROMPT.format(user_message=user_message),
                    },
                ],
                temperature=0.1,
                max_tokens=200,
            )
            print(response.reasoning)
            print("USING INSTRUCTOR")
            return response
        except Exception:
            # Default to treating as data-related if check fails
            return models.IsChatQueryRelevant(
                is_data_related=True,
                reasoning="Relevance check failed, defaulting to data-related",
            )

    def validate_response(
        self, response: str, user_message: str
    ) -> models.IsChatResponseValid:
        """Validate response using instructor."""
        try:
            validation = self.client.chat.completions.create(
                model=self.model,
                response_model=models.IsChatResponseValid,
                messages=[
                    {
                        "role": "system",
                        "content": prompts.RESPONSE_VALIDATION_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": prompts.RESPONSE_VALIDATION_PROMPT.format(
                            user_message=user_message,
                            response=response
                        ),
                    },
                ],
                temperature=0.1,
                max_tokens=300,
            )
            return validation
        except Exception:
            return models.IsChatResponseValid(
                response=response, is_concise=True, addresses_question=True
            )

    def stream_data_insights(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ):
        """Stream data insights using instructor."""
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,  # Lower temperature for more focused responses
                max_tokens=400,  # Reduced for more concise responses
                stream=True,
            )

            full_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield content

            return full_response

        except Exception as e:
            error_msg = f"Error getting insights: {str(e)}"
            yield error_msg
            return error_msg
