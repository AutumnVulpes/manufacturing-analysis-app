"""
Centralized retry configuration using tenacity.

This module provides standardized retry policies for different LLM operations,
with proper error classification and configurable backoff strategies.
"""

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
    before_sleep_log,
    after_log,
)
import logging
from typing import Callable
import openai

logger = logging.getLogger(__name__)


def is_rate_limit_error(exception: Exception) -> bool:
    """Check if exception is a rate limit error."""
    error_str = str(exception).lower()
    return (
        "429" in error_str
        or "rate limit" in error_str
        or "too many requests" in error_str
        or (hasattr(exception, 'status_code') and exception.status_code == 429)
    )


def is_server_error(exception: Exception) -> bool:
    """Check if exception is a server error (5xx)."""
    error_str = str(exception).lower()
    return (
        "500" in error_str
        or "502" in error_str
        or "503" in error_str
        or "504" in error_str
        or "internal server error" in error_str
        or "bad gateway" in error_str
        or "service unavailable" in error_str
        or "gateway timeout" in error_str
        or (hasattr(exception, 'status_code') and 500 <= exception.status_code < 600)
    )


def is_network_error(exception: Exception) -> bool:
    """Check if exception is a network-related error."""
    error_str = str(exception).lower()
    return (
        "connection" in error_str
        or "timeout" in error_str
        or "network" in error_str
        or isinstance(exception, (ConnectionError, TimeoutError))
    )


def should_retry_error(exception: Exception) -> bool:
    """Determine if an error should be retried."""
    # Don't retry client errors (4xx except 429).
    error_str = str(exception).lower()
    client_errors = ["400", "401", "403", "404", "422"]
    if any(code in error_str for code in client_errors):
        return False
    
    return (
        is_rate_limit_error(exception)
        or is_server_error(exception)
        or is_network_error(exception)
        or isinstance(exception, openai.APIError)
    )


# Retry decorators for different operation types --------------------------------------------------------------------------------------------------------------

def retry_column_suggestions(func: Callable) -> Callable:
    """Retry decorator for column suggestion operations (aggressive retry)."""
    return retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=retry_if_exception(should_retry_error),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        after=after_log(logger, logging.INFO),
    )(func)


def retry_title_generation(func: Callable) -> Callable:
    """Retry decorator for title generation (moderate retry)."""
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception(should_retry_error),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        after=after_log(logger, logging.INFO),
    )(func)


def retry_chat_operations(func: Callable) -> Callable:
    """Retry decorator for chat operations (conservative retry)."""
    return retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception(should_retry_error),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        after=after_log(logger, logging.INFO),
    )(func)


def retry_validation_operations(func: Callable) -> Callable:
    """Retry decorator for validation operations (minimal retry)."""
    return retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception(should_retry_error),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        after=after_log(logger, logging.INFO),
    )(func)


def retry_rate_limit_heavy(func: Callable) -> Callable:
    """Retry decorator for operations prone to rate limiting."""
    return retry(
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=2, min=2, max=120),
        retry=retry_if_exception(lambda e: is_rate_limit_error(e) or should_retry_error(e)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        after=after_log(logger, logging.INFO),
    )(func)
