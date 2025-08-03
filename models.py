from pydantic import BaseModel, Field, field_validator
from typing import List
from datetime import datetime


class ColumnSuggestion(BaseModel):
    """Model for a single column pairing suggestion from the LLM."""

    column1_name: str = Field(
        description="The exact name of the first suggested column from the CSV"
    )
    column2_name: str = Field(
        description="The exact name of the second suggested column from the CSV"
    )
    reasoning: str = Field(
        description="A concise explanation of what insight could be gained by comparing these two columns"
    )


class ColumnSuggestions(BaseModel):
    """Model for the complete response containing multiple column suggestions."""

    suggestions: List[ColumnSuggestion] = Field(
        description="List of column pairing suggestions"
    )
    overall_analysis: str = Field(
        default="", description="Overall analysis of the dataset characteristics"
    )


class TitleSuggestion(BaseModel):
    """Model for AI-generated dashboard title."""

    title: str = Field(
        description="A concise, professional 3-4 word title that implies the tool's purpose for analyzing the data"
    )


class ChatMessage(BaseModel):
    """Model for a single chat message in the data insights chatbox."""

    role: str = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Content of the message")
    timestamp: datetime = Field(default_factory=datetime.now)

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v not in ["user", "assistant", "system"]:
            raise ValueError("Role must be user, assistant, or system")
        return v


class IsChatQueryRelevant(BaseModel):
    """Model for checking if a user question is relevant to data engineering/analysis."""

    is_data_related: bool = Field(
        ...,
        description="True if question is related to data analysis, engineering, or the current dataset",
    )
    reasoning: str = Field(
        ..., description="Brief explanation of why it is or isn't data-related"
    )


class IsChatResponseValid(BaseModel):
    """Model for validating that responses are concise and appropriate."""

    response: str = Field(..., description="The validated response text")
    is_concise: bool = Field(
        ..., description="Whether the response is concise and to the point"
    )
    addresses_question: bool = Field(
        ..., description="Whether the response directly addresses the user's question"
    )


class ChatStreamResponse(BaseModel):
    """Model for streaming chat responses."""

    content: str = Field(..., description="The chat response content")
    is_complete: bool = Field(
        default=False, description="Whether the response is complete"
    )
