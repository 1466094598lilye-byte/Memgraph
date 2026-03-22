#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline Module
Contains complete dialogue generation pipeline processors
"""

from .base_processor import BaseProcessor, ProcessorResult
from .project_outline_processor import ProjectOutlineProcessor
from .event_processor import EventProcessor
from .summary_processor import SummaryProcessor
from .multi_agent_dialogue_processor import ConversationController

__all__ = [
    # Base classes
    'BaseProcessor',
    'ProcessorResult',

    # Core processors
    'ProjectOutlineProcessor',
    'EventProcessor',
    'SummaryProcessor',

    # Multi-Agent system
    'ConversationController',
]