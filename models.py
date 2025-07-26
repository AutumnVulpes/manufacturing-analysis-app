from pydantic import BaseModel, Field
from typing import List


class ColumnSuggestion(BaseModel):
    """
    Model for a single column pairing suggestion from the LLM.

    This model represents one suggested pair of columns that could be meaningfully
    compared in a scatter plot visualization, along with the reasoning for why
    this pairing would be valuable for analysis.

    Attributes:
        column1_name: The exact name of the first suggested column from the CSV
        column2_name: The exact name of the second suggested column from the CSV
        reasoning: A concise explanation of what insight could be gained by comparing these two columns
    """

    column1_name: str = Field(
        description="The exact name of the first suggested column from the CSV"
    )
    column2_name: str = Field(
        description="The exact name of the second suggested column from the CSV"
    )
    reasoning: str = Field(
        description="A concise explanation of what insight could be gained by comparing these two columns"
    )


class SuggestionsResponse(BaseModel):
    """
    Model for the complete response containing multiple column suggestions.

    This model represents the full response from the LLM when requesting column
    pairing suggestions, including both the individual suggestions and an overall
    analysis of the dataset characteristics.

    Attributes:
        suggestions: List of ColumnSuggestion objects representing suggested column pairings
        overall_analysis: Overall analysis of the dataset characteristics and strategy
    """

    suggestions: List[ColumnSuggestion] = Field(
        description="List of column pairing suggestions"
    )
    overall_analysis: str = Field(
        default="", description="Overall analysis of the dataset characteristics"
    )


class TitleResponse(BaseModel):
    """
    Model for AI-generated dashboard title.

    This model represents the structured response when requesting an AI-generated
    title for the data analysis dashboard based on the uploaded CSV file and its columns.

    Attributes:
        title: A concise, professional 3-4 word title that implies the tool's purpose for analyzing the data
    """

    title: str = Field(
        description="A concise, professional 3-4 word title that implies the tool's purpose for analyzing the data"
    )
