#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Session Summary Generation Processor
Reverse project events to generate session summary sequence
"""

import json
from typing import Any, Dict, List
from pathlib import Path
from pydantic import BaseModel, Field

from .base_processor import BaseProcessor, ProcessorResult


class SummaryInput(BaseModel):
    """Summary input model"""
    user_profile: Dict[str, Any] = Field(..., description="User profile")
    project_blueprint: Dict[str, Any] = Field(..., description="Project blueprint")
    full_event_log: List[Dict[str, Any]] = Field(..., description="Complete event log")
    target_event: Dict[str, Any] = Field(..., description="Target event")

    # Remove all validators - Core logic 1: input not validated


class SessionSummary(BaseModel):
    """Session summary model"""
    session_id: str = Field(..., description="Session ID")
    event_id: int = Field(..., description="Event ID")
    timestamp: str = Field(..., description="Timestamp")
    session_summary: str = Field(..., description="Session summary")

    # Remove all validators


class SummaryOutput(BaseModel):
    """Summary output model"""
    sessions: List[SessionSummary] = Field(..., description="Session summary list")
    total_sessions: int = Field(..., description="Total sessions")
    target_event_id: int = Field(..., description="Target event ID")

    # Remove all validators


class SummaryProcessor(BaseProcessor):
    """Summary processor"""

    def get_input_schema(self) -> Dict[str, Any]:
        """Get input data schema"""
        return {
            "type": "object",
            "properties": {
                "user_profile": {"type": "object"},
                "project_blueprint": {"type": "object"},
                "full_event_log": {"type": "array"},
                "target_event": {"type": "object"}
            },
            "required": ["user_profile", "project_blueprint", "full_event_log", "target_event"]
        }

    def get_output_schema(self) -> Dict[str, Any]:
        """Get output data schema"""
        return {
            "type": "object",
            "properties": {
                "sessions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "session_id": {"type": "string"},
                            "event_id": {"type": "integer"},
                            "timestamp": {"type": "string"},
                            "session_summary": {"type": "string"}
                        }
                    }
                },
                "total_sessions": {"type": "integer"},
                "target_event_id": {"type": "integer"}
            }
        }

    def validate_input(self, data: Any) -> ProcessorResult:
        """Validate input data - Core logic 1: input not validated, used directly"""
        return ProcessorResult(
            success=True,
            data=data,
            metadata={"validated": False}
        )

    def validate_output(self, data: Any) -> ProcessorResult:
        """Validate output data - Core logic 2: only validate if can extract JSON"""
        # any validJSONstructure is accepted（array or object）
        if isinstance(data, (dict, list)):
            return ProcessorResult(
                success=True,
                data=data,
                metadata={"validated": False}
            )
        else:
            return ProcessorResult(
                success=False,
                error_message="Unable to extract valid JSON data structure"
            )

    def build_prompt(self, input_data: Dict[str, Any]) -> str:
        """Build prompt"""
        prompt_template = self.load_prompt("summary.txt")

        user_profile_json = json.dumps(input_data["user_profile"], ensure_ascii=False)
        blueprint_json = json.dumps(input_data["project_blueprint"], ensure_ascii=False)
        event_log_json = json.dumps(input_data["full_event_log"], ensure_ascii=False)
        target_event_json = json.dumps(input_data["target_event"], ensure_ascii=False)

        prompt = prompt_template.replace("{UserInputProfile_JSON}", user_profile_json)
        prompt = prompt.replace("{ProjectEvolutionBlueprint_JSON}", blueprint_json)
        prompt = prompt.replace("{FullEventLog_JSON}", event_log_json)
        prompt = prompt.replace("{TargetEvent_JSON}", target_event_json)

        return prompt

    def process_core(self, data: Any) -> ProcessorResult:
        """Core processing logic - Core logic: directly replace prompt, extract JSON"""
        # Use input data directly, no validation - Core logic 1
        input_data = data

        # Build prompt directly
        prompt = self.build_prompt(input_data)

        # Call LLM, no token limit - Core logic 2
        llm_result = self.safe_llm_generate(
            prompt=prompt,
            temperature=0.6
            # Remove max_tokens limit
        )

        if not llm_result.success:
            return llm_result


        #print(llm_result.data)
        # Extract JSON directly, no output structure validation - Core logic 2
        json_result = self.extract_json_from_response(llm_result.data)

        if not json_result.success:
            # Save raw response for debugging
            debug_file = Path("output/debug/summary_processor_raw_response.txt")
            debug_file.parent.mkdir(parents=True, exist_ok=True)
            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write(f"LLMraw response:\n{llm_result.data}\n\n")
                f.write(f"Extraction error: {json_result.error_message}\n")
            
            # Try using LLM to repair JSON
            print(f"⚠️ Summary Processor JSON Error: {json_result.error_message}")
            print(f"⚠️ Attempting LLM-based repair for invalid JSON...")
            
            # Build repair prompt
            repair_prompt = f"""
The following text contains a JSON object but it has syntax errors (e.g., missing commas, quotes, brackets, trailing commas).
Please repair it and output ONLY the valid JSON string. Do not add any markdown formatting or explanations.

<invalid_text>
{llm_result.data}
</invalid_text>
"""
            try:
                # Call LLM to repair JSON
                repair_result = self.safe_llm_generate(
                    prompt=repair_prompt,
                    temperature=0.1
                )
                

                #print(f"repair_result.data: {repair_result.data}")
                if repair_result.success:
                    # Try extracting JSON from repaired response
                    repaired_json = self.extract_json_from_response(repair_result.data)
                    
                    if repaired_json.success:
                        print("✅ JSON successfully repaired by LLM")
                        output_data = repaired_json.data
                        
                        # Save repaired JSON
                        repaired_debug_file = Path("output/debug/summary_processor_repaired.json")
                        with open(repaired_debug_file, 'w', encoding='utf-8') as f:
                            json.dump(output_data, f, ensure_ascii=False, indent=2)
                        
                        # Continue processing repaired data
                        return ProcessorResult(
                            success=True,
                            data=output_data,
                            metadata={
                                "llm_tokens_used": len(str(repair_result.data)),
                                "processor_type": "summary_processor",
                                "repaired": True
                            }
                        )
                    else:
                        print(f"❌ LLM repair failed: {repaired_json.error_message}")
                        print(f"Raw Repair Response: {repair_result.data}")
                else:
                    print(f"❌ LLM repair call failed")
            except Exception as repair_error:
                print(f"❌ Error during JSON repair: {str(repair_error)}")
            
            # If repair fails, return original error
            return json_result

        output_data = json_result.data

        # Save successfully extracted JSON for debugging
        debug_file = Path("output/debug/summary_processor_extracted.json")
        debug_file.parent.mkdir(parents=True, exist_ok=True)
        with open(debug_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        # Core logic 2: output only validates JSON extraction, does not modify LLM raw output
        # Return raw data extracted from LLM directly, no processing
        return ProcessorResult(
            success=True,
            data=output_data,
            metadata={
                "llm_tokens_used": len(str(llm_result.data)),
                "processor_type": "summary_processor"
            }
        )