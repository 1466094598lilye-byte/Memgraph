#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project Outline Generation Processor
Converts user input into structured project blueprint
"""

import json
from typing import Any, Dict, List
from pydantic import BaseModel, Field

from .base_processor import BaseProcessor, ProcessorResult


class ProjectOutlineProcessor(BaseProcessor):
    """Project outline processor"""

    def build_prompt(self, input_data: Dict[str, Any]) -> str:
        """Build prompt"""
        # Load base prompt template
        prompt_template = self.load_prompt("project_outline.txt")

        # Extract project_goal from persona
        project_goal = input_data.get("primary_goal", "")

        # Build context variables
        persona_json = json.dumps(input_data["persona"], ensure_ascii=False, indent=2)
        attributes_json = json.dumps(input_data["project_attributes"], ensure_ascii=False)

        #print(f"project_goal: {project_goal}")
        #print(f"persona_json: {persona_json}")
        #print(f"attributes_json: {attributes_json}")

        # Replace placeholders in template
        prompt = prompt_template.replace("{persona}", persona_json)
        prompt = prompt.replace("{project_goal}", project_goal)
        prompt = prompt.replace("{project_attributes}", attributes_json)

        return prompt

    def process_core(self, data: Any) -> ProcessorResult:
        """Core processing logic"""

        input_data = data

        # Build prompt
        prompt = self.build_prompt(input_data)

        # Call LLM
        llm_result = self.safe_llm_generate(
            prompt=prompt,
            temperature=0.3,  # Lower temperature to ensure structured output
            max_tokens=128000
        )

        if not llm_result.success:
            return llm_result

        # ExtractJSONresponse
        json_result = self.extract_json_from_response(llm_result.data)

        if not json_result.success:
            return json_result

        output_data = json_result.data

        # Core logic 2: output only validates JSON extraction, does not modify LLM raw output
        # Return LLM raw data extracted, no processing
        return ProcessorResult(
            success=True,
            data=output_data,
            metadata={
                "llm_tokens_used": len(str(llm_result.data)),
                "input_attributes_count": len(input_data["project_attributes"])
            }
        )