import instructor
from openai import OpenAI
from models import SuggestionsResponse, ColumnSuggestion, TitleResponse, ChatMessage, RelevanceCheck, ValidatedResponse
from typing import Dict, Any, Tuple, List
import time
import json
import streamlit as st
import pandas as pd


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
        else:
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

    def _create_data_context(self, df: pd.DataFrame, numeric_cols: List[str], 
                           pca_state=None, column_suggestions=None) -> str:
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
- Column names: {', '.join(numeric_cols[:10])}{'...' if len(numeric_cols) > 10 else ''}

BASIC STATISTICS:
"""
        # Add basic statistics for first few numeric columns
        for col in numeric_cols[:5]:
            if col in df.columns:
                series = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(series) > 0:
                    context += f"- {col}: mean={series.mean():.2f}, std={series.std():.2f}, range=[{series.min():.2f}, {series.max():.2f}]\n"

        if pca_state and pca_state.pca_object is not None:
            context += f"""
PCA ANALYSIS RESULTS:
- Principal components generated: {pca_state.min_components_95_variance}
- Columns used for PCA: {', '.join(pca_state.pca_cols)}
- Explained variance by components: {[f'{v:.3f}' for v in pca_state.explained_variance[:3]]}
- Cumulative variance (first 3 PCs): {[f'{v:.3f}' for v in pca_state.cumulative_variance[:3]]}
"""

        if column_suggestions and hasattr(column_suggestions, 'suggestions'):
            context += f"""
GENERATED COLUMN SUGGESTIONS:
- Number of suggestions: {len(column_suggestions.suggestions)}
- Suggested pairs: {', '.join([f'({s.column1_name}, {s.column2_name})' for s in column_suggestions.suggestions[:3]])}
"""

        return context

    def _check_question_relevance(self, user_message: str) -> RelevanceCheck:
        """
        Check if the user's question is relevant to data engineering/analysis.

        Args:
            user_message: User's question/message

        Returns:
            RelevanceCheck object indicating if question is data-related
        """
        try:
            if self.use_instructor:
                response = self.client.chat.completions.create(
                    model=self.model,
                    response_model=RelevanceCheck,
                    messages=[
                        {
                            "role": "system",
                            "content": """You are an expert at determining if questions require data analysis or are about the current dataset.

A question IS data-related and needs data context if it asks about:
- The current dataset, its columns, patterns, or insights
- Analysis of the uploaded data
- Correlations, trends, or patterns in the data
- Specific data visualization recommendations
- Statistical analysis of the current data
- PCA results or principal components
- Data preprocessing suggestions for this dataset

A question is NOT data-related (or doesn't need data context) if it asks about:
- What the chatbot is or how it works
- General data science concepts without reference to current data
- Cooking recipes, food, or restaurants
- General knowledge unrelated to data
- Personal advice or opinions
- Entertainment, sports, or hobbies
- Travel, weather, or current events
- Programming languages unrelated to data analysis
- Any topic completely unrelated to data or analytics

Be strict: only return true if the question specifically needs information about the current dataset."""
                        },
                        {"role": "user", "content": f"Does this question need information about the current dataset: '{user_message}'"}
                    ],
                    temperature=0.1,
                    max_tokens=200,
                )
                return response
            else:
                # For OpenRouter, use a more specific heuristic
                dataset_keywords = [
                    'this data', 'dataset', 'columns', 'my data', 'current data',
                    'correlation', 'pattern', 'trend', 'outlier', 'distribution',
                    'pca', 'component', 'insight', 'analysis', 'visualization'
                ]
                
                # Questions about the chatbot itself
                chatbot_keywords = [
                    'are you', 'what are you', 'who are you', 'how do you work',
                    'what can you do', 'your capabilities', 'human or robot'
                ]
                
                message_lower = user_message.lower()
                
                # If it's asking about the chatbot, don't treat as needing data context.
                if any(keyword in message_lower for keyword in chatbot_keywords):
                    is_data_related = False
                else:
                    is_data_related = any(keyword in message_lower for keyword in dataset_keywords)
                
                return RelevanceCheck(
                    is_data_related=is_data_related,
                    reasoning="Enhanced keyword-based check for OpenRouter"
                )
        except Exception:
            # Default to treating as data-related if check fails.
            return RelevanceCheck(
                is_data_related=True,
                reasoning="Relevance check failed, defaulting to data-related"
            )

    def _create_data_engineer_system_prompt(self) -> str:
        """
        Create system prompt for data engineering focused responses.

        Returns:
            System prompt string for data engineering context
        """
        return """You are a data analysis assistant. Your primary goal is to answer the user's question directly and concisely.

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


    def _validate_response(self, response: str, user_message: str) -> ValidatedResponse:
        """
        Validate that the response is concise and addresses the user's question.

        Args:
            response: The AI response to validate
            user_message: The original user question

        Returns:
            ValidatedResponse object with validation results
        """
        try:
            if self.use_instructor:
                validation = self.client.chat.completions.create(
                    model=self.model,
                    response_model=ValidatedResponse,
                    messages=[
                        {
                            "role": "system",
                            "content": """You are an expert at validating AI responses for quality and relevance.

Evaluate if the response:
1. Is concise and to the point (not overly verbose)
2. Directly addresses the user's question
3. Doesn't force unrelated topics into the answer

A good response should be focused, relevant, and appropriately sized for the question asked."""
                        },
                        {
                            "role": "user", 
                            "content": f"User asked: '{user_message}'\n\nAI responded: '{response}'\n\nIs this response concise and does it address the question?"
                        }
                    ],
                    temperature=0.1,
                    max_tokens=300,
                )
                return validation
            else:
                is_concise = len(response.split()) < 200  # Less than 200 words.
                addresses_question = len(response) > 10  # At least some content.
                
                return ValidatedResponse(
                    response=response,
                    is_concise=is_concise,
                    addresses_question=addresses_question
                )
        except Exception:
            return ValidatedResponse(
                response=response,
                is_concise=True,
                addresses_question=True
            )


    def stream_data_insights(self, user_message: str, chat_history: List[ChatMessage],
                           df: pd.DataFrame, numeric_cols: List[str], 
                           pca_state=None, column_suggestions=None):
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
        try:
            relevance_check = self._check_question_relevance(user_message)
            
            if not relevance_check.is_data_related:
                rejection_msg = "Sorry, I don't think that's relevant to analyzing this data!"
                yield rejection_msg
                return rejection_msg
            
            system_prompt = self._create_data_engineer_system_prompt()
            messages = [{"role": "system", "content": system_prompt}]
            
            # Only add data context if the question specifically needs dataset information.
            message_lower = user_message.lower()
            needs_data_context = any(keyword in message_lower for keyword in [
                'dataset', 'data', 'columns', 'statistics', 'pca', 'correlation', 
                'pattern', 'trend', 'analysis', 'insight', 'visualization'
            ])
            
            if needs_data_context:
                data_context = self._create_data_context(df, numeric_cols, pca_state, column_suggestions)
                messages.append({"role": "system", "content": f"CURRENT DATA CONTEXT:\n{data_context}"})
            
            for msg in chat_history[-5:]:
                messages.append({"role": msg.role, "content": msg.content})
            
            messages.append({"role": "user", "content": user_message})

            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,  # Lower temperature for more focused responses
                max_tokens=400,   # Reduced for more concise responses
                stream=True
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
