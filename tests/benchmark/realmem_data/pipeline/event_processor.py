#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project Event Generation Processor
Generates detailed project event sequences based on project blueprints
"""

import json
from typing import Any, Dict, List
from pathlib import Path
from pydantic import BaseModel, Field

from .base_processor import BaseProcessor, ProcessorResult


class EventInput(BaseModel):
    """Event input model"""
    user_profile: Dict[str, Any] = Field(..., description="User profile")
    project_state: Dict[str, Any] = Field(..., description="Current project state")
    project_blueprint: Dict[str, Any] = Field(..., description="Project evolution blueprint")

    

class DynamicUpdate(BaseModel):
    """Dynamic update model"""
    attribute: str = Field(..., description="Attribute being updated")
    key_changed: str = Field(..., description="Key name that was changed")
    content: str = Field(..., description="Update content")


class ProjectEvent(BaseModel):
    """Project event model"""
    event_index: int = Field(..., description="Event index")
    anchor_node_id: str = Field(..., description="Anchor node ID")
    event_name: str = Field(..., description="Event name")
    event_time: str = Field(..., description="Event time")
    stage_description: str = Field(..., description="Stage description")
    event_description: str = Field(..., description="Event description")
    dynamic_updates: List[DynamicUpdate] = Field(default_factory=list, description="Dynamic updates")
    event_result: str = Field(..., description="Event result")
    reasoning: str = Field(..., description="Reasoning explanation")

    

class EventOutput(BaseModel):
    """Event output model"""
    events: List[ProjectEvent] = Field(..., description="List of events")
    total_events: int = Field(..., description="Total number of events")
    project_timeline: str = Field(..., description="Project timeline")


class EventProcessor(BaseProcessor):
    """Event processor"""

    def get_input_schema(self) -> Dict[str, Any]:
        """Get input data schema"""
        return {
            "type": "object",
            "properties": {
                "user_profile": {
                    "type": "object",
                    "description": "User profile"
                },
                "project_state": {
                    "type": "object",
                    "description": "Current project state"
                },
                "project_blueprint": {
                    "type": "object",
                    "description": "Project evolution blueprint",
                    "properties": {
                        "project_goal": {"type": "string"},
                        "anchor_nodes": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "core_task": {"type": "string"},
                                    "asset_focused_tasks": {"type": "object"}
                                }
                            }
                        }
                    }
                }
            },
            "required": ["user_profile", "project_state", "project_blueprint"]
        }

    def get_output_schema(self) -> Dict[str, Any]:
        """Get output data schema"""
        return {
            "type": "object",
            "properties": {
                "events": {
                    "type": "array",
                    "description": "List of events",
                    "items": {
                        "type": "object",
                        "properties": {
                            "event_index": {"type": "integer"},
                            "anchor_node_id": {"type": "string"},
                            "event_name": {"type": "string"},
                            "event_time": {"type": "string"},
                            "stage_description": {"type": "string"},
                            "event_description": {"type": "string"},
                            "dynamic_updates": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "attribute": {"type": "string"},
                                        "key_changed": {"type": "string"},
                                        "content": {"type": "string"}
                                    }
                                }
                            },
                            "event_result": {"type": "string"},
                            "reasoning": {"type": "string"}
                        }
                    }
                },
                "total_events": {"type": "integer", "description": "Total number of events"},
                "project_timeline": {"type": "string", "description": "Project timeline"}
            },
            "required": ["events", "total_events", "project_timeline"]
        }

    def validate_input(self, data: Any) -> ProcessorResult:
        """Validate input data - Core logic: input not validated, used directly"""
        return ProcessorResult(
            success=True,
            data=data,
            metadata={"validated": False}
        )

    def validate_output(self, data: Any) -> ProcessorResult:
        """Validate output data - Core logic: only validate if JSON can be extracted"""
        # Check if basic JSON structure can be extracted
        if isinstance(data, list) or (isinstance(data, dict) and ("events" in data or isinstance(data, list))):
            return ProcessorResult(
                success=True,
                data=data,
                metadata={"validated": False}
            )
        else:
            return ProcessorResult(
                success=False,
                error_message="Unable to extract valid JSON event data structure"
            )

    def build_prompt(self, input_data: Dict[str, Any]) -> str:
        """Build prompt"""
        # Load base prompt template
        prompt_template = self.load_prompt("event.txt")

        # Build context variables
        user_profile_json = json.dumps(input_data["user_profile"], ensure_ascii=False, indent=2)
        project_state_json = json.dumps(input_data["project_state"], ensure_ascii=False, indent=2)
        blueprint_json = json.dumps(input_data["project_blueprint"], ensure_ascii=False, indent=2)

        # Replace placeholders in template
        prompt = prompt_template.replace("{UserInputProfile_JSON}", user_profile_json)
        prompt = prompt.replace("{ProjectState_JSON}", project_state_json)
        prompt = prompt.replace("{ProjectEvolutionBlueprint_JSON}", blueprint_json)

        return prompt

    def process_core(self, data: Any) -> ProcessorResult:
        """Core processing logic - Core logic: directly replace prompt, extract JSON"""
        # Use input data directly, no validation
        input_data = data

        # Build prompt directly
        prompt = self.build_prompt(input_data)

        # Call LLM
        llm_result = self.safe_llm_generate(
            prompt=prompt,
            temperature=0.5,
            max_tokens=128000  # Increase token limit to ensure complete JSON output
        )

        if not llm_result.success:
            return llm_result

        # Extract JSON directly, no output structure validation
        json_result = self.extract_json_from_response(llm_result.data)

        if not json_result.success:
            # Save raw response for debugging
            debug_file = Path("output/debug/event_processor_raw_response.txt")
            debug_file.parent.mkdir(parents=True, exist_ok=True)
            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write(f"LLM raw response:\n{llm_result.data}\n\n")
                f.write(f"Extraction error: {json_result.error_message}\n")
            return json_result

        output_data = json_result.data

        # Core logic 2: Output only validates JSON extraction, does not modify LLM raw output
        # Return raw data extracted by LLM directly, no processing
        return ProcessorResult(
            success=True,
            data=output_data,
            metadata={
                "llm_tokens_used": len(str(llm_result.data)),
                "processor_type": "event_processor"
            }
        )

    def _generate_timeline(self, events: List[Dict[str, Any]]) -> str:
        """Generate project timeline"""
        if not events:
            return "No timeline information"

        start_time = events[0].get("event_time", "Unknown time")
        end_time = events[-1].get("event_time", "Unknown time")

        return f"From {start_time} to {end_time}, total {len(events)} events"