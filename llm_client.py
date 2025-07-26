import instructor
from openai import OpenAI
from models import SuggestionsResponse, ColumnSuggestion, TitleResponse
from typing import Dict, Any, Tuple
import time
import json
import streamlit as st


class LLMClient:
    """Client for interacting with LLM APIs using instructor for structured responses."""

    def __init__(self, provider: str, api_key: str):
        """
        Initialize the LLM client.

        Args:
            provider: API provider name ("OpenRouter", "Gemini", or "OpenAI")
            api_key: API key for the selected provider
        """
        self.provider = provider
        self.api_key = api_key
        self.base_url, self.default_headers, self.model = self._get_api_config()

        if provider == "OpenRouter":
            # OpenRouter doesn't support function calling, use regular OpenAI client.
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=api_key,
                default_headers=self.default_headers,
            )
            self.use_instructor = False
        else:
            openai_client = OpenAI(
                base_url=self.base_url,
                api_key=api_key,
                default_headers=self.default_headers,
            )
            self.client = instructor.from_openai(
                openai_client, mode=instructor.Mode.JSON
            )
            self.use_instructor = True

    def _get_api_config(self) -> tuple[str, dict, str]:
        """
        Get API client configuration based on provider.

        Returns:
            tuple: (base_url, default_headers, model) configuration for the provider
        """
        if self.provider == "OpenRouter":
            base_url = "https://openrouter.ai/api/v1"
            default_headers = {
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referrer": "https://manufacturing-analysis-app.streamlit.app",
                "X-Title": "Manufacturing Analysis Dashboard",
            }
            model = "mistralai/mistral-7b-instruct:free"
        elif self.provider == "Gemini":
            base_url = "https://generativelanguage.googleapis.com/v1beta"
            default_headers = {"x-goog-api-key": self.api_key}
            model = "gemini-pro"
        else:  # OpenAI
            base_url = "https://api.openai.com/v1"
            default_headers = {"Authorization": f"Bearer {self.api_key}"}
            model = "gpt-3.5-turbo"

        return base_url, default_headers, model

    def _create_prompt(self, data_summary: Dict[str, Any]) -> str:
        """
        Create a prompt for the LLM to suggest column pairings.

        Args:
            data_summary: Dictionary containing data information

        Returns:
            Formatted prompt string
        """
        prompt = f"""You are a data analysis expert specializing in manufacturing data. Analyze the following dataset and suggest meaningful column pairings for scatter plot visualization that could reveal insights, trends, correlations, or patterns.

Dataset Information:
- Total rows: {data_summary["total_rows"]}
- Total columns: {data_summary["total_columns"]}
- All columns are numerical data for scatter plot visualization

Available Columns: {", ".join(data_summary["column_names"])}

Column Statistics:
"""
        for stat in data_summary["statistics"]:
            prompt += f"- {stat['column']}: mean={stat['mean']:.2f}, std={stat['std']:.2f}, range=[{stat['min']:.2f}, {stat['max']:.2f}]\n"

        prompt += f"""
Sample Data:
{json.dumps(data_summary["sample_data"], indent=2)}

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

        return prompt

    def get_column_suggestions(
        self, data_summary: Dict[str, Any], max_retries: int = 3
    ) -> SuggestionsResponse:
        """
        Get column pairing suggestions from the LLM with retry logic.

        Args:
            data_summary: Dictionary containing data information
            max_retries: Maximum number of retry attempts

        Returns:
            SuggestionsResponse object

        Raises:
            Exception: If all retry attempts fail
        """
        prompt = self._create_prompt(data_summary)

        for attempt in range(max_retries):
            try:
                if self.use_instructor:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        response_model=SuggestionsResponse,
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a data analysis expert who suggests meaningful column pairings for manufacturing data visualization. Always respond with valid JSON in the specified format.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.7,
                        max_tokens=1500,
                    )
                else:
                    response = self._get_openrouter_suggestions(prompt)

                valid_suggestions = []
                available_columns = set(data_summary["column_names"])

                for suggestion in response.suggestions:
                    if (
                        suggestion.column1_name in available_columns
                        and suggestion.column2_name in available_columns
                        and suggestion.column1_name != suggestion.column2_name
                    ):
                        valid_suggestions.append(suggestion)

                return SuggestionsResponse(
                    suggestions=valid_suggestions,
                    overall_analysis=response.overall_analysis,
                )

            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(
                        f"Failed to get suggestions after {max_retries} attempts. Last error: {str(e)}"
                    )

                wait_time = 2**attempt
                time.sleep(wait_time)
                continue

        raise Exception("Unexpected error in retry loop")

    def _get_openrouter_suggestions(self, prompt: str) -> SuggestionsResponse:
        """
        Get suggestions from OpenRouter using regular chat completions.

        Args:
            prompt: The formatted prompt

        Returns:
            SuggestionsResponse object
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a data analysis expert who suggests meaningful column pairings for manufacturing data visualization. Respond with valid JSON containing 'suggestions' array and 'overall_analysis' string.",
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
                    ColumnSuggestion(
                        column1_name=item.get("column1_name", ""),
                        column2_name=item.get("column2_name", ""),
                        reasoning=item.get("reasoning", ""),
                    )
                )

            return SuggestionsResponse(
                suggestions=suggestions,
                overall_analysis=data.get("overall_analysis", ""),
            )

        except json.JSONDecodeError:
            # Fallback: try to extract information from text.
            return self._parse_text_response(content)

    def _parse_text_response(self, content: str) -> SuggestionsResponse:
        """
        Parse text response when JSON parsing fails.

        Args:
            content: Raw text response

        Returns:
            SuggestionsResponse object
        """
        import re

        suggestions = []

        # Try to extract column pairs and reasoning
        pattern = r"(\w+)\s*(?:vs|and|\&)\s*(\w+).*?[Rr]easoning?:?\s*([^\n]+)"
        matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)

        for i, (col1, col2, reasoning) in enumerate(matches[:5]):
            suggestions.append(
                ColumnSuggestion(
                    column1_name=col1.strip(),
                    column2_name=col2.strip(),
                    reasoning=reasoning.strip(),
                )
            )

        # Extract overall analysis (first paragraph usually).
        lines = content.split("\n")
        overall_analysis = ""
        for line in lines:
            if len(line.strip()) > 50:  # Assume first substantial line is analysis.
                overall_analysis = line.strip()
                break

        return SuggestionsResponse(
            suggestions=suggestions, overall_analysis=overall_analysis
        )

    def generate_title(self, filename: str, columns: list[str]) -> str:
        """
        Generate a concise title for the CSV file using AI with instructor for structured responses.

        Args:
            filename: Name of the uploaded CSV file
            columns: List of column names from the CSV file

        Returns:
            str: Generated title (3-4 words) or empty string if generation fails
        """
        try:
            if self.provider == "OpenRouter":
                return self._generate_title_openrouter(filename, columns)

            prompt = (
                f"Generate a concise, professional title (3-4 words) that implies the tool's purpose for analyzing "
                f"manufacturing data from CSV file '{filename}' with columns: {', '.join(columns)}. "
                f"Examples: 'Manufacturing Data Analysis', 'Production Dashboard', 'Quality Control Hub', 'Process Analytics Tool'."
            )

            response = self.client.chat.completions.create(
                model=self.model,
                response_model=TitleResponse,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at creating concise, professional titles for data analysis tools. Generate exactly 3-4 words that clearly indicate the purpose of the tool.",
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

    def _generate_title_openrouter(self, filename: str, columns: list[str]) -> str:
        """
        Generate title for OpenRouter using regular chat completions (no function calling).

        Args:
            filename: CSV filename
            columns: List of column names

        Returns:
            str: Generated title or empty string if failed
        """
        try:
            prompt = (
                f"Generate a concise, professional title (3-4 words) for analyzing manufacturing data from '{filename}' "
                f"with columns: {', '.join(columns)}. Return only the title, nothing else. "
                f"Examples: Manufacturing Data Analysis, Production Dashboard, Quality Control Hub"
            )

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Generate exactly 3-4 words for a data analysis tool title. Return only the title.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=20,
            )

            title = response.choices[0].message.content.strip()
            # Clean up the response to take only the first line and limit to reasonable length.
            title = title.split("\n")[0].strip()
            words = title.split()
            if len(words) > 6:
                title = " ".join(words[:4])

            return title

        except Exception as e:
            st.error(f"OpenRouter title generation failed: {e}")
            return ""

    def validate_api_key(self) -> bool:
        """
        Validate the API key by making a simple test request.

        Returns:
            True if API key is valid, False otherwise
        """
        try:
            if self.use_instructor:
                # Simple test request for instructor-based clients.
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=10,
                )
            else:
                # Simple test request for regular OpenAI client.
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=10,
                )
            return True
        except Exception:
            return False


def get_api_client_config(provider: str, api_key: str) -> Tuple[str, dict, str]:
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
    else:  # OpenAI case.
        base_url = "https://api.openai.com/v1"
        default_headers = {"Authorization": f"Bearer {api_key}"}
        model = "gpt-3.5-turbo"

    return base_url, default_headers, model
