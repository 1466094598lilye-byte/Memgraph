#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Error Handling Module
Includes LLM call error handling, checkpoint mechanism, and JSON parsing
"""

import json
import os
import re
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from .llm_client import LLMClient


class ErrorType(Enum):
    """Error type enumeration"""
    TIMEOUT = "timeout"
    API_ERROR = "api_error"
    JSON_EXTRACT = "json_extract"
    JSON_PARSE = "json_parse"
    VALIDATION = "validation"
    UNKNOWN = "unknown"


@dataclass
class ProcessingResult:
    """Processing result"""
    success: bool
    result: Any = None
    error_type: Optional[ErrorType] = None
    error_message: str = ""
    retry_count: int = 0
    processing_time: float = 0.0


class LLMErrorHandler:
    """LLM call error handler"""

    def __init__(self, llm_client: LLMClient, max_retries: int = 3):
        """
        Initialize error handler

        Args:
            llm_client: LLM client instance
            max_retries: Maximum number of retries
        """
        self.llm_client = llm_client
        self.max_retries = max_retries
        self.setup_logging()

    def setup_logging(self):
        """Setup logging configuration"""
        os.makedirs('output', exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('output/llm_error_handler.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )

        self.logger = logging.getLogger(__name__)

    def categorize_error(self, exception: Exception) -> ErrorType:
        """
        Categorize error type

        Args:
            exception: Exception object

        Returns:
            Error type
        """
        if "timeout" in str(exception).lower():
            return ErrorType.TIMEOUT
        elif "api" in str(exception).lower() or "connection" in str(exception).lower():
            return ErrorType.API_ERROR
        elif "json" in str(exception).lower():
            return ErrorType.JSON_PARSE
        else:
            return ErrorType.UNKNOWN

    def retry_llm_call(self, func, *args, **kwargs) -> Tuple[bool, Any]:
        """
        LLM call with retry mechanism

        Args:
            func: Function to call
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            (success_flag, result)
        """
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    # Exponential backoff
                    wait_time = 2 ** attempt
                    self.logger.info(f"Retry attempt {attempt}, waiting {wait_time} seconds...")
                    time.sleep(wait_time)

                    # Adjust parameters for degradation
                    if 'temperature' in kwargs:
                        kwargs['temperature'] = max(0.1, kwargs['temperature'] * 0.8)
                    if 'max_tokens' in kwargs:
                        kwargs['max_tokens'] = max(100, kwargs['max_tokens'] // 2)

                result = func(*args, **kwargs)
                self.logger.info(f"LLM call successful, retry count: {attempt}")
                return True, result

            except Exception as e:
                last_error = e
                error_type = self.categorize_error(e)
                self.logger.warning(f"LLM call failed (attempt {attempt + 1}/{self.max_retries + 1}): {error_type.value} - {str(e)}")

                # Some error types are not suitable for retry
                if error_type in [ErrorType.VALIDATION, ErrorType.JSON_PARSE]:
                    break

        error_type = self.categorize_error(last_error)
        self.logger.error(f"LLM call ultimately failed: {error_type.value} - {str(last_error)}")
        return False, ProcessingResult(
            success=False,
            error_type=error_type,
            error_message=str(last_error),
            retry_count=self.max_retries
        )

    def safe_generate(self, prompt: str, system_prompt: Optional[str] = None,
                     temperature: float = 0.7, max_tokens: int = 2000) -> ProcessingResult:
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
        start_time = time.time()

        def generate_call():
            return self.llm_client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )

        success, result = self.retry_llm_call(generate_call)

        processing_time = time.time() - start_time

        if success:
            return ProcessingResult(
                success=True,
                result=result,
                processing_time=processing_time
            )
        else:
            return result if isinstance(result, ProcessingResult) else ProcessingResult(
                success=False,
                error_type=ErrorType.UNKNOWN,
                error_message="Unknown error occurred",
                processing_time=processing_time
            )

    def extract_json_from_response(self, response: str) -> ProcessingResult:
        """
        Extract JSON from LLM response

        Args:
            response: LLM response text

        Returns:
            Processing result containing JSON data
        """
        try:
            # Try extracting from code block
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_str = response[start:end].strip()
                return ProcessingResult(
                    success=True,
                    result=json.loads(json_str)
                )

            # Try extracting from code block (no language identifier)
            if "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                json_str = response[start:end].strip()
                try:
                    return ProcessingResult(
                        success=True,
                        result=json.loads(json_str)
                    )
                except:
                    pass

            # Try finding standalone JSON (including objects and arrays)
            # Use smarter method to find complete JSON structure
            for start_char, end_char in [('{', '}'), ('[', ']')]:
                start_idx = response.find(start_char)
                if start_idx != -1:
                    # Use bracket matching to find complete JSON structure
                    brace_count = 0
                    in_string = False
                    escape_next = False

                    for i, char in enumerate(response[start_idx:], start_idx):
                        if escape_next:
                            escape_next = False
                            continue

                        if char == '\\' and in_string:
                            escape_next = True
                            continue

                        if char == '"' and not escape_next:
                            in_string = not in_string
                            continue

                        if not in_string:
                            if char == start_char:
                                brace_count += 1
                            elif char == end_char:
                                brace_count -= 1
                                if brace_count == 0:
                                    # Found complete JSON structure
                                    json_str = response[start_idx:i+1]
                                    try:
                                        parsed_json = json.loads(json_str)
                                        return ProcessingResult(
                                            success=True,
                                            result=parsed_json
                                        )
                                    except json.JSONDecodeError:
                                        # If parsing fails, continue searching for next possible JSON
                                        break
  
            return ProcessingResult(
                success=False,
                error_type=ErrorType.JSON_EXTRACT,
                error_message="Unable to extract valid JSON from response"
            )

        except json.JSONDecodeError as e:
            return ProcessingResult(
                success=False,
                error_type=ErrorType.JSON_PARSE,
                error_message=f"JSON parsing error: {str(e)}"
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                error_type=ErrorType.UNKNOWN,
                error_message=f"Unknown error: {str(e)}"
            )


class CheckpointManager:
    """Checkpoint manager - supports resume from interruption"""

    def __init__(self, checkpoint_file: str = "output/checkpoint.json"):
        """
        Initialize checkpoint manager

        Args:
            checkpoint_file: Checkpoint file path
        """
        self.checkpoint_file = checkpoint_file
        os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)

    def save_checkpoint(self, data: Dict[str, Any]) -> bool:
        """
        Save checkpoint

        Args:
            data: Data to save

        Returns:
            Whether save was successful
        """
        try:
            checkpoint_data = {
                "timestamp": time.time(),
                "data": data
            }

            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)

            print(f"Checkpoint saved to {self.checkpoint_file}")
            return True

        except Exception as e:
            print(f"Failed to save checkpoint: {str(e)}")
            return False

    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint

        Returns:
            Checkpoint data, or None if doesn't exist
        """
        if not os.path.exists(self.checkpoint_file):
            return None

        try:
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)

            print(f"Checkpoint loaded from {self.checkpoint_file}")
            return checkpoint_data.get("data")

        except Exception as e:
            print(f"Failed to load checkpoint: {str(e)}")
            return None

    def clear_checkpoint(self) -> bool:
        """
        Clear checkpoint

        Returns:
            Whether clear was successful
        """
        try:
            if os.path.exists(self.checkpoint_file):
                os.remove(self.checkpoint_file)
                print(f"Checkpoint cleared: {self.checkpoint_file}")
            return True
        except Exception as e:
            print(f"Failed to clear checkpoint: {str(e)}")
            return False


# Convenience functions
def create_error_handler(llm_client: LLMClient, max_retries: int = 3) -> LLMErrorHandler:
    """
    Create error handler instance

    Args:
        llm_client: LLM client
        max_retries: Maximum number of retries

    Returns:
        Error handler instance
    """
    return LLMErrorHandler(llm_client, max_retries)


def create_checkpoint_manager(checkpoint_file: str = "output/checkpoint.json") -> CheckpointManager:
    """
    Create checkpoint manager instance

    Args:
        checkpoint_file: Checkpoint file path

    Returns:
        Checkpoint manager instance
    """
    return CheckpointManager(checkpoint_file)