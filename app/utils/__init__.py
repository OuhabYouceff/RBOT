"""
Utils package for helper functions and utilities.
"""

from .prompt_templates import (
    SYSTEM_PROMPT_FR,
    SYSTEM_PROMPT_AR,
    QUESTION_SEGMENTATION_PROMPT,
    format_context,
    get_no_results_response,
    format_final_response
)
from .language_detector import LanguageDetector
from .response_formatter import ResponseFormatter

__all__ = [
    'SYSTEM_PROMPT_FR',
    'SYSTEM_PROMPT_AR',
    'QUESTION_SEGMENTATION_PROMPT',
    'format_context',
    'get_no_results_response',
    'format_final_response',
    'LanguageDetector',
    'ResponseFormatter'
]