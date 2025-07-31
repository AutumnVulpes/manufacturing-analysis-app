"""
OpenRouter LLM Strategy implementation.

This module implements the LLM strategy for OpenRouter providers that don't support
function calling, using regular OpenAI client with manual JSON parsing.
"""

from openai import OpenAI
import json
import re
import streamlit as st
from typing import Dict, Any, List

from llm_strategy_interface import LLMStrategyInterface
import models
import prompts


class OpenRouterLLMStrategy(LLMStrategyInterface):
    """Strategy implementation for OpenRouter providers."""

    def __init__(self, provider: str, api_key: str):
        """Initialize OpenRouter strategy."""
        super().__init__(provider, api_key)
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=api_key,
            default_headers=self.default_headers,
        )

    def _get_api_config(self) -> tuple[str, dict, str]:
        """Get API configuration for OpenRouter."""
        base_url = "https://openrouter.ai/api/v1"
        default_headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referrer": "https://manufacturing-analysis-app.streamlit.app",
            "X-Title": "Manufacturing Analysis Dashboard",
        }
        model = "mistralai/mistral-7b-instruct:free"
        return base_url, default_headers, model

    def get_column_suggestions(
        self, prompt: str, data_summary: Dict[str, Any]
    ) -> models.ColumnSuggestions:
        """Get column suggestions using OpenRouter."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": prompts.OPENROUTER_COLUMN_SUGGESTION_SYSTEM_PROMPT,
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=1500,
        )

        content = response.choices[0].message.content

        try:
            data = json.loads(content)

            suggestions = []
            for item in data.get("suggestions", []):
                suggestions.append(
                    models.ColumnSuggestion(
                        column1_name=item.get("column1_name", ""),
                        column2_name=item.get("column2_name", ""),
                        reasoning=item.get("reasoning", ""),
                    )
                )

            return models.ColumnSuggestions(
                suggestions=suggestions,
                overall_analysis=data.get("overall_analysis", ""),
            )

        except json.JSONDecodeError:
            # Fallback: try to extract information from text
            return self._parse_text_response(content)

    def _parse_text_response(self, content: str) -> models.ColumnSuggestions:
        """Parse text response when JSON parsing fails."""
        suggestions = []

        # Try to extract column pairs and reasoning
        pattern = r"(\w+)\s*(?:vs|and|\&)\s*(\w+).*?[Rr]easoning?:?\s*([^\n]+)"
        matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)

        for i, (col1, col2, reasoning) in enumerate(matches[:5]):
            suggestions.append(
                models.ColumnSuggestion(
                    column1_name=col1.strip(),
                    column2_name=col2.strip(),
                    reasoning=reasoning.strip(),
                )
            )

        # Extract overall analysis (first paragraph usually)
        lines = content.split("\n")
        overall_analysis = ""
        for line in lines:
            if len(line.strip()) > 50:  # Assume first substantial line is analysis
                overall_analysis = line.strip()
                break

        return models.ColumnSuggestions(
            suggestions=suggestions, overall_analysis=overall_analysis
        )

    def generate_title(self, filename: str, columns: List[str]) -> str:
        """Generate title using OpenRouter."""
        try:
            prompt = prompts.OPENROUTER_TITLE_GENERATION_PROMPT.format(
                filename=filename,
                columns=', '.join(columns)
            )

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": prompts.OPENROUTER_TITLE_GENERATION_SYSTEM_PROMPT,
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=20,
            )

            title = response.choices[0].message.content.strip()
            # Clean up the response to take only the first line and limit to reasonable length
            title = title.split("\n")[0].strip()
            words = title.split()
            if len(words) > 6:
                title = " ".join(words[:4])

            return title

        except Exception as e:
            st.error(f"OpenRouter title generation failed: {e}")
            return ""

    def check_question_relevance(self, user_message: str) -> models.IsChatQueryRelevant:
        """Check question relevance using LLM-based reasoning for OpenRouter."""
        try:
            # Simple, direct prompt that forces the model to end with true/false
            openrouter_relevance_prompt = f"""Analyze this question: "{user_message}"

Is this question about data analysis, datasets, or data science concepts?

Provide brief reasoning, then end your response with exactly one word: true or false

Your response must end with: true or false"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": openrouter_relevance_prompt}
                ],
                temperature=0.0,  # Zero temperature for consistency
                max_tokens=100,
            )

            content = response.choices[0].message.content.strip()
            
            # Debug: Print the actual response
            print(f"DEBUG - OpenRouter response: '{content}'")
            
            # Extract the final word and check if it's true/false
            words = content.split()
            if words:
                final_word = words[-1].lower().rstrip('.,!?')
                is_data_related = final_word == "true"
                
                # Extract reasoning (everything except the final word)
                reasoning = " ".join(words[:-1]) if len(words) > 1 else "LLM analysis"
            else:
                # Fallback if no words found
                is_data_related = True
                reasoning = "Empty response, defaulting to data-related"
            
            print(f"DEBUG - Final word: '{final_word}', Decision: {is_data_related}")
            
            return models.IsChatQueryRelevant(
                is_data_related=is_data_related,
                reasoning=reasoning[:200],
            )
            
        except Exception as e:
            print(f"DEBUG - Exception: {e}")
            return models.IsChatQueryRelevant(
                is_data_related=True,
                reasoning=f"Error occurred, defaulting to data-related: {str(e)}",
            )

    def validate_response(
        self, response: str, user_message: str
    ) -> models.IsChatResponseValid:
        """Validate response using simple heuristics for OpenRouter."""
        try:
            is_concise = len(response.split()) < 200  # Less than 200 words
            addresses_question = len(response) > 10  # At least some content

            return models.IsChatResponseValid(
                response=response,
                is_concise=is_concise,
                addresses_question=addresses_question,
            )
        except Exception:
            return models.IsChatResponseValid(
                response=response, is_concise=True, addresses_question=True
            )

    def stream_data_insights(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ):
        """Stream data insights using OpenRouter."""
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
