#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Validation Module
Using Pydantic for data validation and error reporting
"""

import json
import os
import re
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, validator, ValidationError
from pydantic.json import pydantic_encoder


class MemoryPoint(BaseModel):
    """Memory point model"""
    label: str = Field(..., description="Memory point label")
    content: str = Field(..., description="Detailed description of memory point")

    @validator('label')
    def validate_label(cls, v):
        """Validate label field"""
        if not v or not isinstance(v, str):
            raise ValueError("label field must be a non-empty string")

        # Standardize label names
        valid_labels = {
            'BodyMetrics', 'TrainingPlan', 'DietPlan', 'StoryOutline',
            'CharacterProfiles', 'ProjectState', 'Other'
        }

        # If not in predefined list, classify as Other but keep original value
        if v not in valid_labels:
            v = 'Other'

        return v

    @validator('content')
    def validate_content(cls, v):
        """Validate content field"""
        if not v or not isinstance(v, str):
            raise ValueError("content field must be a non-empty string")

        if len(v.strip()) < 5:
            raise ValueError("content field is too short, requires at least 5 characters")

        return v.strip()


class SessionSummary(BaseModel):
    """Session summary model"""
    SessionID: str = Field(..., description="Session ID, format like 'S1_01'")
    Summary: str = Field(..., description="Session summary content")
    MemoryPoints: List[MemoryPoint] = Field(default_factory=list, description="List of memory points")
    EventID: Optional[int] = Field(None, description="Event ID")

    @validator('SessionID')
    def validate_session_id(cls, v):
        """Validate SessionID format"""
        if not isinstance(v, str):
            raise ValueError("SessionID must be a string")

        if not re.match(r'^S\d+_\d+$', v):
            raise ValueError(f"Invalid SessionID format, should be 'S<event_id>_<session_num>', got: {v}")

        return v

    @validator('Summary')
    def validate_summary(cls, v):
        """Validate summary content"""
        if not isinstance(v, str):
            raise ValueError("Summary must be a string")

        if len(v.strip()) < 10:
            raise ValueError("Summary content is too short, requires at least 10 characters")

        return v.strip()

    @validator('EventID')
    def validate_event_id(cls, v):
        """Validate EventID"""
        if v is not None and (not isinstance(v, int) or v < 1):
            raise ValueError("EventID must be a positive integer")

        return v


class DialogueTurn(BaseModel):
    """Dialogue turn model"""
    speaker: str = Field(..., description="Speaker, 'User' or 'Assistant'")
    content: str = Field(..., description="Dialogue content")

    @validator('speaker')
    def validate_speaker(cls, v):
        """Validate speaker"""
        if v not in ['User', 'Assistant']:
            raise ValueError("speaker must be 'User' or 'Assistant'")
        return v

    @validator('content')
    def validate_content(cls, v):
        """Validate dialogue content"""
        if not isinstance(v, str):
            raise ValueError("content must be a string")

        if len(v.strip()) < 1:
            raise ValueError("Dialogue content cannot be empty")

        return v


class DialogueData(BaseModel):
    """Dialogue data model"""
    session_id: str = Field(..., description="Session ID")
    dialogue_turns: List[DialogueTurn] = Field(..., description="List of dialogue turns")

    @validator('session_id')
    def validate_session_id(cls, v):
        """Validate session ID"""
        if not isinstance(v, str):
            raise ValueError("session_id must be a string")

        if not v.strip():
            raise ValueError("session_id cannot be empty")

        return v

    @validator('dialogue_turns')
    def validate_dialogue_turns(cls, v):
        """Validate dialogue turns"""
        if not isinstance(v, list):
            raise ValueError("dialogue_turns must be a list")

        if len(v) < 2:
            raise ValueError("Dialogue must contain at least 2 turns")

        return v


class ValidationErrorReport:
    """Validation error reporter"""

    @staticmethod
    def validate_memory_point(data: Dict[str, Any]) -> tuple[bool, Optional[MemoryPoint], str]:
        """
        Validate memory point data

        Args:
            data: Data to validate

        Returns:
            (success, validated_object, error_message)
        """
        try:
            memory_point = MemoryPoint(**data)
            return True, memory_point, ""
        except ValidationError as e:
            error_msg = f"Memory point validation failed: {str(e)}"
            return False, None, error_msg
        except Exception as e:
            error_msg = f"Memory point validation exception: {str(e)}"
            return False, None, error_msg

    @staticmethod
    def validate_session_summary(data: Dict[str, Any]) -> tuple[bool, Optional[SessionSummary], str]:
        """
        Validate session summary data

        Args:
            data: Data to validate

        Returns:
            (success, validated_object, error_message)
        """
        try:
            session_summary = SessionSummary(**data)
            return True, session_summary, ""
        except ValidationError as e:
            error_msg = f"Session summary validation failed: {str(e)}"
            return False, None, error_msg
        except Exception as e:
            error_msg = f"Session summary validation exception: {str(e)}"
            return False, None, error_msg

    @staticmethod
    def validate_dialogue_data(data: Dict[str, Any]) -> tuple[bool, Optional[DialogueData], str]:
        """
        Validate dialogue data

        Args:
            data: Data to validate

        Returns:
            (success, validated_object, error_message)
        """
        try:
            dialogue_data = DialogueData(**data)
            return True, dialogue_data, ""
        except ValidationError as e:
            error_msg = f"Dialogue data validation failed: {str(e)}"
            return False, None, error_msg
        except Exception as e:
            error_msg = f"Dialogue data validation exception: {str(e)}"
            return False, None, error_msg

    @staticmethod
    def batch_validate_session_summaries(data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Batch validate session summaries

        Args:
            data_list: List of session summary data

        Returns:
            Validation result dictionary
        """
        results = {
            'valid': [],
            'invalid': [],
            'total_count': len(data_list),
            'success_count': 0,
            'error_count': 0
        }

        for i, data in enumerate(data_list):
            success, validated_obj, error_msg = ValidationErrorReport.validate_session_summary(data)

            if success:
                results['valid'].append({
                    'index': i,
                    'data': validated_obj.dict(),
                    'session_id': validated_obj.SessionID
                })
                results['success_count'] += 1
            else:
                results['invalid'].append({
                    'index': i,
                    'data': data,
                    'error': error_msg
                })
                results['error_count'] += 1

        return results

    @staticmethod
    def generate_validation_report(results: Dict[str, Any]) -> str:
        """
        Generate validation report

        Args:
            results: Validation result dictionary

        Returns:
            Formatted validation report string
        """
        report_lines = [
            "=== Data Validation Report ===",
            f"Total: {results['total_count']} items",
            f"Success: {results['success_count']} items",
            f"Failed: {results['error_count']} items",
            f"Success rate: {results['success_count']/results['total_count']*100:.1f}%",
            ""
        ]

        if results['invalid']:
            report_lines.append("=== Error Details ===")
            for item in results['invalid']:
                report_lines.append(f"[{item['index']}] {item['error']}")
                if 'SessionID' in item['data']:
                    report_lines.append(f"  SessionID: {item['data']['SessionID']}")
                report_lines.append("")

        return "\n".join(report_lines)

    @staticmethod
    def save_validation_report(results: Dict[str, Any], output_file: str = "output/validation_report.json"):
        """
        Save validation report to file

        Args:
            results: Validation result dictionary
            output_file: Output file path
        """
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=pydantic_encoder)

            print(f"Validation report saved to: {output_file}")

        except Exception as e:
            print(f"Failed to save validation report: {str(e)}")


# Convenience functions
def validate_memory_point(data: Dict[str, Any]) -> tuple[bool, Optional[MemoryPoint], str]:
    """Convenience function to validate memory point data"""
    return ValidationErrorReport.validate_memory_point(data)


def validate_session_summary(data: Dict[str, Any]) -> tuple[bool, Optional[SessionSummary], str]:
    """Convenience function to validate session summary data"""
    return ValidationErrorReport.validate_session_summary(data)


def validate_dialogue_data(data: Dict[str, Any]) -> tuple[bool, Optional[DialogueData], str]:
    """Convenience function to validate dialogue data"""
    return ValidationErrorReport.validate_dialogue_data(data)


def batch_validate_and_report(data_list: List[Dict[str, Any]],
                            output_file: str = "output/validation_report.json") -> Dict[str, Any]:
    """Convenience function for batch validation and report generation"""
    results = ValidationErrorReport.batch_validate_session_summaries(data_list)

    # Generate report
    report = ValidationErrorReport.generate_validation_report(results)
    print(report)

    # Save report
    ValidationErrorReport.save_validation_report(results, output_file)

    return results