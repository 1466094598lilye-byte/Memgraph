#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility Module Package
Contains core functions including LLM client, error handling, data validation, etc.
"""

from .llm_client import LLMClient, create_client
from .error_handler import (
    ErrorType,
    ProcessingResult,
    LLMErrorHandler,
    CheckpointManager,
    create_error_handler,
    create_checkpoint_manager
)
from .data_validator import (
    MemoryPoint,
    SessionSummary,
    DialogueTurn,
    DialogueData,
    ValidationErrorReport,
    validate_memory_point,
    validate_session_summary,
    validate_dialogue_data,
    batch_validate_and_report
)

__all__ = [
    # LLM client
    'LLMClient',
    'create_client',

    # Error handling
    'ErrorType',
    'ProcessingResult',
    'LLMErrorHandler',
    'CheckpointManager',
    'create_error_handler',
    'create_checkpoint_manager',

    # Data validation
    'MemoryPoint',
    'SessionSummary',
    'DialogueTurn',
    'DialogueData',
    'ValidationErrorReport',
    'validate_memory_point',
    'validate_session_summary',
    'validate_dialogue_data',
    'batch_validate_and_report',
]