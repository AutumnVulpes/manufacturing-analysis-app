"""
Instructor LLM Strategy implementation.

This module implements the LLM strategy for providers that support function calling
using instructor for structured responses with Pydantic models.
"""

import instructor
from openai import OpenAI
import streamlit as st
from typing import Dict, Any, List, Generator

from .llm_strategy_interface import LLMStrategyInterface
import models
from . import prompts
from .retry_config import (
    retry_column_suggestions,
    retry_title_generation,
    retry_chat_operations,
    retry_validation_operations,
)


class InstructorLLMStrategy(LLMStrategyInterface):
    """
    Strategy implementation for providers supporting instructor.

    This strategy uses the instructor library for structured responses
    with Pydantic models, providing type-safe LLM interactions.
    """

    def __init__(self, provider: str, api_key: str):
        """
        Initialize Instructor strategy.

        Parameters
        ----------
        provider : str
            API provider name ("Gemini" or "OpenAI")
        api_key : str
            API key for the selected provider
        """
        super().__init__(provider, api_key)
        # Raw client for streaming
        self.raw_client = OpenAI(
            base_url=self.base_url,
            api_key=api_key,
            default_headers=self.default_headers,
        )
        # Instructor client for structured operations
        self.client = instructor.from_openai(self.raw_client, mode=instructor.Mode.JSON)

    def _get_api_config(self) -> tuple[str, dict, str]:
        """Get API configuration based on provider."""
        if self.provider == "Gemini":
            base_url = "https://generativelanguage.googleapis.com/v1beta"
            default_headers = {"x-goog-api-key": self.api_key}
            model = "gemini-1.5-pro"
        else:  # OpenAI
            base_url = "https://api.openai.com/v1"
            default_headers = {"Authorization": f"Bearer {self.api_key}"}
            model = "gpt-3.5-turbo"

        return base_url, default_headers, model

    @retry_column_suggestions
    def get_column_suggestions(
        self, prompt: str, data_summary: Dict[str, Any]
    ) -> models.ColumnSuggestions:
        """
        Get column suggestions using instructor with retry logic.

        Parameters
        ----------
        prompt : str
            Formatted prompt for the LLM
        data_summary : Dict[str, Any]
            Dictionary containing data information for validation

        Returns
        -------
        models.ColumnSuggestions
            Validated column suggestions object
        """
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

    @retry_title_generation
    def generate_title(self, filename: str, columns: List[str]) -> str:
        """Generate title using instructor with retry logic."""
        try:
            prompt = prompts.TITLE_GENERATION_PROMPT.format(
                filename=filename, columns=", ".join(columns)
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

    @retry_validation_operations
    def check_question_relevance(self, user_message: str) -> models.IsChatQueryRelevant:
        """Check question relevance using instructor with retry logic."""
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
                        "content": prompts.QUESTION_RELEVANCE_PROMPT.format(
                            user_message=user_message
                        ),
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

    @retry_validation_operations
    def validate_response(
        self, response: str, user_message: str
    ) -> models.IsChatResponseValid:
        """Validate response using instructor with retry logic."""
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
                            user_message=user_message, response=response
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

    @retry_chat_operations
    def _get_streaming_response(self, messages: List[Dict[str, str]]):
        """Get streaming response with retry logic."""
        return self.raw_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            max_tokens=400,
            stream=True,
        )

    def stream_data_insights(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> Generator[str, None, str]:
        """Stream data insights using hybrid approach: raw streaming + instructor validation."""
        print(
            f"[DEBUG] Starting hybrid stream_data_insights with {len(messages)} messages"
        )
        print(f"[DEBUG] Messages: {messages}")

        try:
            print(f"[DEBUG] Getting streaming response with retry logic")
            stream = self._get_streaming_response(messages)

            print(f"[DEBUG] Got raw stream object: {type(stream)}")

            # Step 1: Collect streaming chunks and yield them in real-time
            full_content = ""
            chunk_count = 0

            for chunk in stream:
                chunk_count += 1
                print(f"[DEBUG] Processing raw chunk {chunk_count}: {type(chunk)}")

                # Extract content from OpenAI streaming format
                if hasattr(chunk, "choices") and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        content = delta.content
                        print(f"[DEBUG] Got streaming content: {content}")
                        full_content += content
                        yield content
                else:
                    print(f"[DEBUG] No content in chunk or no choices")

            print(f"[DEBUG] Raw streaming completed. Total chunks: {chunk_count}")
            print(f"[DEBUG] Full content length: {len(full_content)}")
            print(f"[DEBUG] Full content: {full_content}")

            # Step 2: Validate complete response with instructor (post-processing)
            try:
                print(f"[DEBUG] Validating complete response with instructor...")
                validated_response = models.ChatStreamResponse(content=full_content)
                print(f"[DEBUG] Response validation successful")
                return validated_response.content
            except Exception as validation_error:
                # If validation fails, return raw content with warning
                print(f"[WARNING] Response validation failed: {validation_error}")
                print(f"[WARNING] Returning raw content without validation")
                return full_content

        except Exception as e:
            error_str = str(e)
            print(f"[ERROR] Hybrid streaming failed: {error_str}")
            print(f"[ERROR] Exception type: {type(e)}")
            print(f"[ERROR] Exception details: {e}")
            error_msg = f"Error getting insights: {error_str}"
            yield error_msg
            return error_msg
