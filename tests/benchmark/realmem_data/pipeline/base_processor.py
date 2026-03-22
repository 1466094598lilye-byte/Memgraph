#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base Processor Class
Defines base interfaces and common functionality for all processors
"""

import json
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path

from utils import (
    LLMErrorHandler,
    CheckpointManager,
    ProcessingResult as UtilsProcessingResult,
    validate_session_summary,
    validate_dialogue_data,
    validate_memory_point
)


@dataclass
class ProcessorResult:
    """Processor result"""
    success: bool
    data: Any = None
    error_message: str = ""
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    validation_errors: List[str] = field(default_factory=list)

    def add_error(self, error: str):
        """Add error message"""
        self.error_message = error
        self.success = False

    def add_validation_error(self, error: str):
        """Add validation error"""
        self.validation_errors.append(error)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'success': self.success,
            'data': self.data,
            'error_message': self.error_message,
            'processing_time': self.processing_time,
            'metadata': self.metadata,
            'validation_errors': self.validation_errors
        }


class BaseProcessor(ABC):
    """Base processor abstract class"""

    def __init__(self, llm_client, checkpoint_dir: str = "output/checkpoints"):
        """
        Initialize processor

        Args:
            llm_client: LLM client
            checkpoint_dir: Checkpoint directory
        """
        self.llm_client = llm_client
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize error handling and checkpoint management
        self.error_handler = LLMErrorHandler(llm_client)
        self.checkpoint_manager = CheckpointManager(
            str(self.checkpoint_dir / f"{self.__class__.__name__}_checkpoint.json")
        )

        self.processor_name = self.__class__.__name__

    def load_prompt(self, prompt_file: str) -> str:
        """
        Load prompt file

        Args:
            prompt_file: Prompt file path

        Returns:
            Prompt content
        """
        try:
            prompt_path = Path("prompts") / prompt_file
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise FileNotFoundError(f"Unable to load prompt file {prompt_file}: {str(e)}")

    def save_checkpoint(self, data: Dict[str, Any]) -> bool:
        """Save checkpoint"""
        checkpoint_data = {
            "processor": self.processor_name,
            "timestamp": time.time(),
            "data": data
        }
        return self.checkpoint_manager.save_checkpoint(checkpoint_data)

    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load checkpoint"""
        checkpoint = self.checkpoint_manager.load_checkpoint()
        if checkpoint and checkpoint.get("processor") == self.processor_name:
            return checkpoint.get("data")
        return None

    def clear_checkpoint(self) -> bool:
        """Clear checkpoint"""
        return self.checkpoint_manager.clear_checkpoint()

    def safe_llm_generate(self, prompt: str, system_prompt: Optional[str] = None,
                         temperature: float = 0.7, max_tokens: int = 20000) -> ProcessorResult:
        """
        Safe LLM generation call

        Args:
            prompt: Prompt text
            system_prompt: System prompt text
            temperature: Temperature parameter
            max_tokens: Maximum number of tokens

        Returns:
            Processing result
        """
        llm_result = self.error_handler.safe_generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )

        if llm_result.success:
            return ProcessorResult(
                success=True,
                data=llm_result.result,
                processing_time=llm_result.processing_time
            )
        else:
            return ProcessorResult(
                success=False,
                error_message=f"LLM call failed: {llm_result.error_message}",
                processing_time=llm_result.processing_time,
                metadata={"error_type": llm_result.error_type.value if llm_result.error_type else None}
            )

    def extract_json_from_response(self, response: str) -> ProcessorResult:
        """
        Extract JSON from LLM response

        Args:
            response: LLM response

        Returns:
            Processing result containing JSON data
        """
        json_result = self.error_handler.extract_json_from_response(response)

        if json_result.success:
            return ProcessorResult(
                success=True,
                data=json_result.result
            )
        else:
            return ProcessorResult(
                success=False,
                error_message=f"JSON extraction failed: {json_result.error_message}",
                metadata={"error_type": json_result.error_type.value if json_result.error_type else None}
            )

    def process(self, data: Any, use_checkpoint: bool = True) -> ProcessorResult:
        """
        Complete processing workflow

        Args:
            data: Input data
            use_checkpoint: Whether to use checkpoint

        Returns:
            Processing result
        """
        start_time = time.time()

        # Try loading checkpoint
        if use_checkpoint:
            checkpoint_data = self.load_checkpoint()
            if checkpoint_data:
                print(f"{self.processor_name}: Restored from checkpoint")
                return ProcessorResult(
                    success=True,
                    data=checkpoint_data,
                    metadata={"restored_from_checkpoint": True}
                )

        try:
            # Core processing
            process_result = self.process_core(data)
            if not process_result.success:
                return process_result

            # Save checkpoint
            if use_checkpoint:
                self.save_checkpoint({
                    "input_data": data,
                    "output_data": process_result.data,
                    "processing_time": process_result.processing_time
                })

            process_result.processing_time = time.time() - start_time
            print(f"{self.processor_name}: Processing successful ({process_result.processing_time:.2f}s)")

            return process_result

        except Exception as e:
            error_msg = f"{self.processor_name} processing exception: {str(e)}"
            print(f"Error: {error_msg}")
            return ProcessorResult(
                success=False,
                error_message=error_msg,
                processing_time=time.time() - start_time
            )

    def get_status(self) -> Dict[str, Any]:
        """Get processor status"""
        checkpoint = self.load_checkpoint()
        return {
            "processor": self.processor_name,
            "has_checkpoint": checkpoint is not None,
            "checkpoint_timestamp": checkpoint.get("timestamp") if checkpoint else None,
            "input_schema": self.get_input_schema(),
            "output_schema": self.get_output_schema()
        }