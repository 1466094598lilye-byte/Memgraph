#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Agent Dialogue Generation Processor
Uses multiple agents to collaboratively generate goal-oriented dialogues
"""

import json
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from pydantic import BaseModel, Field

from .base_processor import BaseProcessor, ProcessorResult
from utils.llm_client import LLMClient
from utils.error_handler import LLMErrorHandler
from utils.dialogue_postprocessor import DialoguePostprocessor


class ConversationPhase(Enum):
    """Conversation phase"""
    OPENING = "opening"          # Opening
    EXPLORATION = "exploration" # Exploring needs
    SOLUTION = "solution"        # Providing solution
    CONFIRMATION = "confirmation" # Confirming understanding
    CLOSURE = "closure"         # Closing


@dataclass
class DialogueTurn:
    """Dialogue turn"""
    speaker: str
    content: str
    phase: ConversationPhase
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GoalEvaluation:
    """Goal evaluation result"""
    overall_score: float  # Total score 0-100
    problem_understanding: float  # Problem understanding 0-20
    solution_provided: float  # Solution provided 0-30
    user_confirmed: float  # User confirmation 0-30
    task_closed: float  # Task closure 0-20
    is_complete: bool  # Is complete
    reasoning: str  # Evaluation reasoning


class UserAgent:
    """User Agent - Simulating User Behavior (Optimized for Result-Orientation)"""

    def __init__(self, llm_client, user_profile: Dict[str, Any], event_log: List[Dict[str, Any]], current_event_session_summary_list: List[Dict[str, Any]] = None, history_dialogue: List[Dict[str, Any]] = None):
        self.llm_client = llm_client
        self.user_profile = user_profile
        self.event_log = event_log
        self.current_event_session_summary_list = current_event_session_summary_list or []
        # Pre-process dialogue history
        self.history_dialogue_text = ""
        if history_dialogue:
            self.history_dialogue_text = "\n".join([
                f"[{turn.get('speaker', 'Unknown')}: {turn.get('content', '')}]"
                for turn in history_dialogue
            ])

    def generate_opening(self, task_summary: str, current_time: str = "") -> str:
        """Generate Opening Message (Optimized for Fuzzy Continuity & Conditional Data)"""
        
        # Extract critical information from the last session
        last_turn_content = "None"
        if hasattr(self, 'history_dialogue_text') and self.history_dialogue_text:
            history_lines = self.history_dialogue_text.strip().split('\n')
            if history_lines:
                last_turn_content = history_lines[-1]

        prompt = f"""
# Role
You are {self.user_profile.get('name', 'User')}.
Please immerse yourself fully in this role. Initiate this conversation based on your background settings and the events you have recently experienced.

# Context Data
<current_time>
{current_time}
</current_time>

<user_profile>
{json.dumps(self.user_profile, ensure_ascii=False, indent=2)}
</user_profile>

<event_history_summary>
{json.dumps(self.event_log[:2], ensure_ascii=False, indent=2)}
</event_history_summary>

<last_session_conclusion>
State at the end of the previous session: {last_turn_content}
</last_session_conclusion>

<current_task>
{task_summary}
</current_task>

# Instructions
You need to send the first message to the AI assistant. Please follow these steps:

1. **Status Retrospective**: Review <event_history_summary>.
   - If this is a new event/session, focus on initiating contact.
   - Tone: Natural and conversational.

2. **Dialogue Strategy**:
   - **Start with Intent**: State clearly what you want to do (e.g., "I want to start a fitness plan").
   - **Simulate Human Pacing**: Do not list numbers/data immediately unless answering a pending question.

3. **Dialogue History Review & Continuity Strategy (CRITICAL)**: 
   - Check <last_session_conclusion>. analyze the context of the very last sentence.
   
   - **SCENARIO A: Cold Start / New Topic** (History is Empty OR Task has changed)
     - **Action**: Treat this as walking into a room for the first time or sending the very first email.
     - **Constraint**: **ABSOLUTELY FORBIDDEN** to use connecting words like "Okay", "So", "Well", "Phew", "Finally", "Great".
     - **Phrasing**: Start directly with a greeting ("Hi", "Hello") or a direct statement of intent ("I decided to...", "I want to...").
     - **Context**: If <event_history_summary> shows a recent related event, you can reference the *event itself* (e.g., "I just finished brainstorming"), but NOT a previous *conversation*.
   
   - **SCENARIO B: Explicit Pending Question** (The AI ended by asking for specific data, e.g., "What is your weight?")
     - **Action**: You are allowed to provide *that specific data* immediately to answer the pending question.
     - **Constraint**: Only answer what was asked. Do not dump extra unrequested data.
   
   - **SCENARIO C: General Continuation** (History exists, but no specific pending question)
     - **Action**: Use **"Fuzzy Continuity"**. Resume the flow naturally.
     - **Phrasing**: Use phrases like "I'm back," "Ready to move on to the next step," or "Let's continue with the plan."
     - **Constraint**: **Do NOT quote the previous message.** Do not say "Since you said X...". Do not be mechanical. Just pick up the vibe.

4. **Initiate Dialogue**: 
   - Connect with <current_task>.
   - If Scenario A or C applies: Ask for guidance ("What do you need?").
   - If Scenario B applies: Give the answer.

# Constraints
- **NO DATA DUMPING**: Unless falling under SCENARIO B, do NOT list specific metrics in this opening message.
- **Forbidden**: Do not reference system info.
- Keep length under 60 words.

Please generate the opening message in English ONLY:
"""
        return self.llm_client.generate(prompt, temperature=0.7)

    def generate_response(self, ai_message: str, conversation_history: List,
                         current_goal: str, phase, current_time: str = "") -> str:
        """Generate User Response (Updated with Anti-Loop Logic)"""
        # Extract all dialogue history
        current_history_text = "\n".join([f"{turn.speaker}: {turn.content}" for turn in conversation_history])

        if self.history_dialogue_text:
            history_text = f"{self.history_dialogue_text}\n\n{current_history_text}"
        else:
            history_text = current_history_text

        prompt = f"""
# Role
You are {self.user_profile.get('name', 'User')}. You are conversing with an AI assistant.

# Context Data
<current_time>
{current_time}
</current_time>

<user_profile>
{json.dumps(self.user_profile, ensure_ascii=False, indent=2)}
</user_profile>

<event_history_summary>
{json.dumps(self.event_log[:2], ensure_ascii=False, indent=2)}
</event_history_summary>

<history_dialogue>
{self.history_dialogue_text}
</history_dialogue>

<conversation_state>
Current Phase: {phase.value}
Current Goal: {current_goal}
</conversation_state>

<recent_history>
{current_history_text}
</recent_history>

<latest_ai_message>
{ai_message}
</latest_ai_message>

# Response Strategy (CRITICAL)
Currently in the **{phase.value}** phase. React based on the AI's latest message:

* **OPENING**:
    - Supplement specific background information.

* **EXPLORATION**:
    - Answer questions. If technical details are confusing, say so.

* **SOLUTION & CONFIRMATION (Anti-Loop Logic)**:
    - **CHECK**: Did the AI *actually* provide the full content?
    - **IF AI ASKS "Ready to see it?" OR "Shall I proceed?"**:
        - **ACTION**: Do NOT just say "Yes". Say "Yes, show it to me immediately." or "Stop asking and list the plan now."
    - **IF NO (AI says "I am working on it" or "I will send it")**:
        - **DO NOT** be polite. **DO NOT** say "take your time" or "no rush".
        - **ACTION**: Urge the AI to provide the result immediately. Say: "Please show me the draft now."

* **CLOSURE**:
    - **Pre-condition**: Only end the conversation if you have received the actual solution.
    - If you haven't seen the result yet, **refuse to close** and ask for the result again.

# Task Focus (Important!)
- **Strict Limit**: Focus only on discussing content related to <current_task>.
- **Realignment**: If the AI deviates from the current task, you must remind them.
- **Refuse Jumping Ahead**: Explicitly refuse to discuss topics beyond the scope.

# Constraints
1. **Tone Consistency**: Always maintain the speaking style of {self.user_profile.get('role', 'User')}.
2. **Behavior**: **Do not accept empty promises.** You want results, not status updates.
3. **Length Limit**: Keep under 150 words.

Please generate the response in English ONLY:
"""
        return self.llm_client.generate(prompt, temperature=0.8)


class AssistantAgent:
    """AI Assistant Agent - Features memory citation tracking, smart deduplication AND Temporal Awareness"""

    def __init__(self, llm_client, history_dialogue: List[Dict[str, Any]] = None):
        self.llm_client = llm_client
        # Pre-process dialogue history
        self.history_dialogue_text = ""
        if history_dialogue:
            self.history_dialogue_text = "\n".join([
                f"[Dialogue History] {turn.get('speaker', 'Unknown')}: {turn.get('content', '')}"
                for turn in history_dialogue
            ])

    def generate_response(self, user_message: str, conversation_history: List,
                         memory_context: str, phase,
                         current_time: str = "",            # [NEW] Current simulation time
                         current_plan_items: List[Dict[str, Any]] = None,    # [NEW] Semantic plan item list
                         current_goal: str = "") -> str:
        """
        Generate AI response with strict Information Firewall, Passive Task Activation, and Temporal Logic.
        """

        # 1. Prepare History Context
        current_history_text = "\n".join([f"{turn.speaker}: {turn.content}" for turn in conversation_history])

        if self.history_dialogue_text:
            history_text = f"{self.history_dialogue_text}\n\n{current_history_text}"
        else:
            history_text = current_history_text

        has_memory = bool(memory_context and memory_context.strip())

        # Format Plan Items for Prompt
        if current_plan_items is None:
            current_plan_items = []
        # Extract id+content fields for display, while maintaining complete structure
        if current_plan_items and isinstance(current_plan_items[0], dict):
            # If object list, extract id+content for display
            plan_contents = [f"[{item.get('id', '?')}] {item.get('content', '')}" for item in current_plan_items if item.get("content")]
            schedule_text = json.dumps(plan_contents, indent=2, ensure_ascii=False)
        else:
            # If string list, use directly
            schedule_text = json.dumps(current_plan_items, indent=2, ensure_ascii=False)

        # 2. Define Critical Protocols (Firewall + Activation + Execution + Time)
        protocol_instruction = """
# CRITICAL PROTOCOLS (MUST FOLLOW)

## 1. INFORMATION FIREWALL (Do Not Leak Info)
The <current_task> contains the **Target Goal/Hidden State**.
- **CONSTRAINT**: You **CANNOT** use specific details from <current_task> unless the user has explicitly mentioned them.
- **BEHAVIOR**: Feign ignorance until the user reveals the info.

## 2. TEMPORAL & SCHEDULE LOGIC (Time Awareness) [NEW]
You have access to <current_time> and <current_schedule>.
- **Reference Frame**: Interpret relative time (e.g., "tomorrow", "this afternoon") based on <current_time>.
- **Conflict Check**: Before agreeing to a new time proposed by the user, CHECK <current_schedule>.
  - **IF CONFLICT**: Gently reject and explain (e.g., "You actually have the Python class at that time.").
  - **IF FREE**: You can proceed to agree.
- **Status Awareness**:
  - If <current_schedule> contains items, assume they are **REMAINING** tasks to be done.
  - If <current_schedule> is empty, congratulate the user or ask for new plans.
  - Do NOT ask about tasks that are NOT in the list (assume they are deleted/completed).

## 3. TASK ACTIVATION (Passive Context)
- **CASE A (Exploration/Guide)**: Guide user to provide missing info.
- **CASE B (Explicit OR Implicit Execution Request)**:
  - **Triggers**: User says "Give me the plan", "I'm ready", or confirms details.
  - **ACTION**: STOP ASKING QUESTIONS. JUST GENERATE IT.

## 4. IMMEDIATE EXECUTION (Anti-Stalling)
- **Applies ONLY to CASE B**.
- **NO SIMULATED DELAYS**: Generate the FULL content immediately.
"""

        # 3. Construct Prompt
        base_context = f"""
# Role
You are a professional AI Assistant managing the user's schedule and tasks.

# Context Data
<current_time>
{current_time}
</current_time>

<current_schedule>
{schedule_text}
</current_schedule>

<history_dialogue>
{history_text}
</history_dialogue>

<user_latest_message>
{user_message}
</user_latest_message>

<current_phase>
{phase.value}
</current_phase>

<current_task>
{current_goal}
(WARNING: HIDDEN STATE. Do NOT use these details unless user mentioned them in history. See Protocol 1.)
</current_task>

{protocol_instruction}
"""

        if has_memory:
            prompt = f"""
{base_context}

<long_term_memory>
{memory_context}
</long_term_memory>

# Instructions (Memory & Logic)
1. **De-duplication**: Check if memory content has already been discussed in <history_dialogue>.
2. **Firewall Check**: Even if memory/task has the answer, ask the user to confirm if it's not in the active dialogue.
3. **Phase Strategy**:
   - **OPENING/EXPLORATION**: Use memory to answer questions but maintain the Firewall.
   - **SOLUTION**: Wait for the user's trigger (CASE B). Once triggered, STOP TALKING, START SOLVING.

4. **Generate Output**:
   - Include <memory_analysis> and <response>.
   - If memory is not needed based on the "De-duplication Check", <memory_analysis> must be filled with "None".

# Few-Shot Examples

### Example 1: EXPLORATION Phase (Information Firewall Demo)
**Context**: Task="Plan surprise party for Mom at 6PM". History=User says "I need to plan an event."
**Bad Response (Leakage)**: "Sure, for your Mom's surprise party at 6PM?" (FAIL: User didn't say Mom or 6PM yet).
**Correct Output**:
<memory_analysis>
None
</memory_analysis>
<response>
I can help with that. What kind of event are you planning, and who is it for?
</response>

### Example 2: SCHEDULE CONFLICT (Temporal Logic)
**Context**:
- Time: Friday 14:00.
- Schedule: ["Dentist appointment at 3 PM today"]
**User Message**: "Let's schedule a call for 3 PM today."
**Correct Output**:
<memory_analysis>
- [Citation]: Schedule item (Dentist appointment) -> [Application]: Detect Conflict
</memory_analysis>
<response>
I'm afraid 3 PM today won't work. You have a dentist appointment scheduled then. Would 4:30 PM work instead?
</response>

### Example 3: EMPTY SCHEDULE (Status Awareness)
**Context**: Schedule=[]. User Message: "What's next?"
**Correct Output**:
<memory_analysis>
None
</memory_analysis>
<response>
Good news! Your schedule is completely clear for now. We've finished everything. Do you want to take a break or start a new plan?
</response>

### Example 4: EXPLORATION Phase (Memory Deduplication)
**Context**: Memory="User is a Python Expert". History includes: "AI: Since you are an expert in Python..."
**User Message**: "How do I reverse a list?"
**Goal**: Answer directly. Do not repeat "As a Python expert" again.
**Correct Output**:
<memory_analysis>
None
</memory_analysis>
<response>
You can simply use the slicing method `list[::-1]` or the `list.reverse()` method in place.
</response>

### Example 5: OPENING Phase (Active Memory Usage)
**Context**: Memory="User budget is $500". Phase=OPENING. History=Empty.
**User Message**: "I want to buy a new phone."
**Correct Output**:
<memory_analysis>
- [Citation]: User budget is $500 -> [Application]: Filter phone recommendations
</memory_analysis>
<response>
Welcome back! With your budget of $500, we should look at mid-range options like the Pixel A-series or the iPhone SE. Shall we start there?
</response>

### Example 6: SOLUTION Phase (CRITICAL - Immediate Execution)
**Context**: Phase=SOLUTION. User says: "Okay, give me the finalized itinerary."
**Bad Response**: "I am working on the itinerary now. Please wait a moment."
**Correct Output**:
<memory_analysis>
None
</memory_analysis>
<response>
Here is the complete itinerary designed for you:

**Day 1: Arrival & City Tour**
- 10:00 AM: Check-in at Hotel...
- 02:00 PM: Visit the Museum...

**Day 2: Mountain Hike**
... (Full content provided immediately)
</response>

# Output Format
<memory_analysis>
List memories that *actually need* to be cited. If not needed, enter "None".
</memory_analysis>

<response>
The formal response content.
</response>

Please generate the response in English ONLY:
"""
        else:
            # No Memory Mode (Still needs Schedule Logic!)
            prompt = f"""
{base_context}

# Instructions
1. **Firewall Adherence**: Strictly distinguish between what you know (Hidden State) and what the user has told you.
2. **Temporal Logic**: USE <current_time> and <current_schedule> to validate all time-related requests.
3. **Execution**: If CASE B, provide the solution IMMEDIATELY.

Please directly generate the response content in English ONLY:
"""

        #print("------------------------------------")
        #print("Assistantprompt：" + prompt)
        #print("------------------------------------")

        return self.llm_client.generate(prompt, temperature=0.5)


class SemanticScheduleAgent:
    """Semantic Schedule Agent - Updates user's Future Plan List based on time passage and conversation"""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def _extract_content_from_items(self, plan_items: list) -> list:
        """
        Extract content from plan items
        Args:
            plan_items: Can be a list of strings or structured objects
        Returns:
            List of content strings
        """
        if not plan_items:
            return []

        # If string list, return directly
        if isinstance(plan_items[0], str):
            return plan_items

        # If object list, extract content
        if isinstance(plan_items[0], dict):
            return [item.get("content", "") for item in plan_items if item.get("content")]

        return []

    def _merge_structured_items(self, original_items: list, updated_contents: list) -> list:
        """
        Intelligently merge original structured items with updated content list
        Args:
            original_items: Original structured item list
            updated_contents: Updated content list
        Returns:
            Merged structured item list
        """
        if not original_items:
            # If original is empty, return simple string list (maintain backward compatibility)
            return updated_contents

        # If original is string list, return updated content directly
        if isinstance(original_items[0], str):
            return updated_contents

        # If original is structured object, perform intelligent merge
        if isinstance(original_items[0], dict):
            # Create content to original item mapping
            content_to_item = {item.get("content", ""): item for item in original_items}
            merged_items = []

            for content in updated_contents:
                if content in content_to_item:
                    # Keep original item
                    merged_items.append(content_to_item[content])
                else:
                    # New item as string (will get ID during save)
                    merged_items.append(content)

            return merged_items

        return updated_contents

    def process(self, dialogue_history: str, current_time_str: str, plan_items: list) -> list:
        """
        Process and update the user's future plan list based on time and conversation

        Args:
            dialogue_history: Conversation history text
            current_time_str: Current time string (e.g., "Friday 14:00")
            plan_items: Current list of plan items (can be string list or structured objects)

        Returns:
            Updated list of plan items (maintains input format)
        """

        # --- Extract content forLLMprocessing ---
        content_items = self._extract_content_from_items(plan_items)

        # --- Construct Prompt ---
        prompt = f"""
# Role
You are a **Strict Schedule Manager**. Your ONLY goal is to manage the user's "Concrete Schedule List".
You are NOT a habit coach or a note-taker for general advice.

# Inputs
<current_time>
{current_time_str}
</current_time>

<dialogue_history>
{dialogue_history}
</dialogue_history>

<current_memory_list>
{json.dumps(content_items, indent=2, ensure_ascii=False)}
</current_memory_list>

# DEFINITION OF A VALID PLAN
A valid plan item MUST have two components:
1. **Specific Action**: A concrete, do-able task.
2. **Specific Timing**: A specific date, time, frequency, OR a status like "(Postponed)".

**❌ INVALID (REMOVE):**
- Vague: "Aim for movement", "Try to eat healthy".
- General Advice: "Drink water", "Read books".

**✅ VALID (KEEP):**
- "Flight to Tokyo on Dec 5th" (One-off)
- "Gym every Mon/Wed at 8 PM" (Recurring)
- "Submit report (Postponed)" (Status based)

# Instructions (Follow Strictly in Order)

1. **CLEANUP (Sanity Check)**: 
   - Remove items that are VAGUE goals or GENERAL advice immediately.

2. **RECURRING PLAN PROTECTION**: 
   - **Rule**: Recurring items (e.g., "Every Monday") are STRUCTURAL.
   - **Action**: DO NOT modify/delete them based on temporary slips (e.g., "I missed the gym today").
   - **Exception**: ONLY modify if user EXPLICITLY says "Change my schedule", "I quit", or "Move all future classes to...".

3. **ONE-OFF TASK UPDATES**:
   - **Completed**: If user says "Done"/"Finished", REMOVE.
   - **Postponed/Rescheduled**: If time has passed OR user says "do it later":
     - **DO NOT DELETE**. 
     - **UPDATE** to the new specific time if provided.
     - **MARK** as "(Postponed)" if no specific time is given but task is not done.

4. **NEW ITEMS**: 
   - Add only if it has a SPECIFIC action and SPECIFIC time/trigger.

5. **NO CHANGE DEFAULT (CRITICAL)**:
   - If the dialogue contains NO schedule-related changes (no completions, no new plans, no rescheduling requests) AND the current list requires no cleanup:
   - **OUTPUT THE `<current_memory_list>` EXACTLY AS IS.** - DO NOT rephrase, DO NOT reorder, DO NOT summarize.

## Few-Shot Examples

**Example 1: The "No Change" Case (CRITICAL)**
Input List: ["Gym every Monday at 7 PM", "Buy Milk"]
Current Time: Monday 10:00 AM
Dialogue: "The weather is really nice today."
Output:
{{
  "reasoning": "User is just chatting. No schedule updates, completions, or new plans detected. List is already valid. NO CHANGES.",
  "updated_list": ["Gym every Monday at 7 PM", "Buy Milk"]
}}

**Example 2: Protecting Recurring Plans (No Change)**
Input List: ["Gym every Monday at 7 PM"]
Current Time: Monday 8:00 PM
Dialogue: "I was too tired to go to the gym today."
Output:
{{
  "reasoning": "User missed ONE session, but did NOT ask to change the weekly schedule. Adhering to Recurring Protection rule. KEEP AS IS.",
  "updated_list": ["Gym every Monday at 7 PM"]
}}

**Example 3: Postponing One-off Tasks**
Input List: ["Submit Report by 2 PM"]
Current Time: Tuesday 3:00 PM (Time Passed)
Dialogue: "I'm running late, I'll finish the report by 5 PM."
Output:
{{
  "reasoning": "Task time passed, but user rescheduled it to '5 PM'. I will UPDATE the time.",
  "updated_list": ["Submit Report by 5 PM"]
}}

**Example 4: Explicitly Modifying Recurring Plans**
Input List: ["Team Meeting every Friday 9 AM"]
Current Time: Thursday
Dialogue: "We are changing the weekly team meeting to Mondays at 10 AM from now on."
Output:
{{
  "reasoning": "User explicitly requested a structural change ('from now on'). Updating the recurring item.",
  "updated_list": ["Team Meeting every Monday 10 AM"]
}}

# Output Format
Return a JSON object with `reasoning` and `updated_list`.
"""

        #print("------------------------------------")
        #print("SemanticSchedule prompt (Strict Mode):" + prompt)
        #print("------------------------------------")
        
        # --- Call LLM ---
        try:
            raw_response = self.llm_client.generate(prompt, temperature=0.0) # Lower temperature to increase strictness

            from utils.error_handler import LLMErrorHandler
            error_handler = LLMErrorHandler(self.llm_client)
            json_result = error_handler.extract_json_from_response(raw_response)

            if json_result.success:
                reasoning = json_result.result.get('reasoning', '')
                print(f"--- [SemanticScheduleAgent Reasoning] ---\n{reasoning}\n-------------------")

                # Get LLM-updated content list
                updated_contents = json_result.result.get("updated_list", content_items)

                # Intelligently merge back to original format
                merged_result = self._merge_structured_items(plan_items, updated_contents)

                return merged_result
            else:
                print(f"⚠️ SemanticScheduleAgent JSON Error: {json_result.error_message}")
                return plan_items

        except Exception as e:
            print(f"⚠️ SemanticScheduleAgent Error: {e}")
            return plan_items


class GoalEvaluatorAgent:
    """Goal Evaluator Agent - Evaluates the completion degree of dialogue goals"""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def evaluate_goal_completion(self, conversation_history: List,
                               original_goal: str):
        """Evaluate goal completion status"""
        conversation_text = "\n".join([f"{turn.speaker}: {turn.content}" for turn in conversation_history])

        prompt = f"""
Please evaluate the goal completion status of the following dialogue:

Original Goal: {original_goal}

Dialogue Content:
{conversation_text}

Please score based on the following criteria:
1. Problem Understanding (0-20 points): Did the AI accurately understand the user's needs?
2. Solution Provision (0-30 points): Did the AI provide specific and actionable solutions?
3. User Confirmation (0-30 points): Did the user understand and accept the solution?
4. Task Closure (0-20 points): Did the conversation end naturally with the task clearly completed?

Please return the evaluation results strictly in the following JSON format:
{{
    "problem_understanding": score,
    "solution_provided": score,
    "user_confirmed": score,
    "task_closed": score,
    "reasoning": "Detailed reasoning process"
}}
"""

        try:
            response = self.llm_client.generate(prompt, temperature=0.3)

            # Extract JSON
            import re

            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                # Clean control characters (Fix method 1)
                def clean_control_characters(text):
                    import re
                    return re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)

                cleaned_json = clean_control_characters(json_match.group())
                evaluation_data = json.loads(cleaned_json)
            else:
                # Default scores
                evaluation_data = {
                    "problem_understanding": 15,
                    "solution_provided": 20,
                    "user_confirmed": 15,
                    "task_closed": 10,
                    "reasoning": "Unable to parse evaluation response, using default scores"
                }

            # Calculate total score
            total_score = (
                evaluation_data["problem_understanding"] +
                evaluation_data["solution_provided"] +
                evaluation_data["user_confirmed"] +
                evaluation_data["task_closed"]
            )

            return GoalEvaluation(
                overall_score=total_score,
                problem_understanding=evaluation_data["problem_understanding"],
                solution_provided=evaluation_data["solution_provided"],
                user_confirmed=evaluation_data["user_confirmed"],
                task_closed=evaluation_data["task_closed"],
                # Completion Criteria: Total score >= 80 AND Dialogue turns >= 6 (prevent premature ending)
                is_complete=total_score >= 80 and len(conversation_history) >= 6,
                reasoning=evaluation_data["reasoning"]
            )

        except Exception as e:
            # Return default values on evaluation failure
            return GoalEvaluation(
                overall_score=50.0,
                problem_understanding=15.0,
                solution_provided=15.0,
                user_confirmed=10.0,
                task_closed=10.0,
                is_complete=False,
                reasoning=f"Evaluation failed: {str(e)}"
            )


class MemoryManagerAgent:
    """Memory Extraction Agent - Extracts long-term memory points after a session ends (Optimized Version)"""

    def __init__(self, llm_client, project_attributes_schema: str = ""):
        self.llm_client = llm_client
        self.project_attributes_schema = project_attributes_schema

    def extract_memory_points(self, dialogue_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract structured memory points and enrich them with raw text.
        """
        session_id = dialogue_data.get('session_id', 'unknown')
        dialogue_turns = dialogue_data.get('dialogue_turns', [])

        # 1. Preprocessing: Convert dialogue turns to indexed plain text for LLM citation
        dialogue_text = ""
        for i, turn in enumerate(dialogue_turns):
            role = turn.get('speaker', 'unknown')
            content = turn.get('content', '')
            dialogue_text += f"[Turn {i+1}] {role}: {content}\n"
        
        #print(f"dialogue_text: {dialogue_text}")

        prompt = f"""
You are an expert-level **Dialogue Memory Analyst**.
Your goal is to extract structured data points with long-term retention value from a dialogue session.

# Input Data
## 1. Project Schema (Dynamic Memory Definition)
The following concepts are key project variables. Any plans, numbers, or status changes regarding these variables belong to "Dynamic Memory":
<schema_definition>
{self.project_attributes_schema}
</schema_definition>

## 2. Dialogue Transcript
Session ID: {session_id}
<transcript>
{dialogue_text}
</transcript>

# Analysis Logic & Rules

## Phase 1: Identification
Iterate through the dialogue and identify candidates for two memory types:

### Type A: Dynamic Memory (Schema-bound)
* **Confirmed (High Confidence):** The user explicitly states a fact, or agrees to a specific proposal by the AI.
    * *Key Rule (Backtracking):* If the user says "I agree", "Yes", or "Do it", you **must** backtrack to the immediately preceding AI turn to extract specific details. In this case, the **Source Turn** is that AI turn (because the information is there).
* **Draft (Pending):** Specific proposals made by the AI currently on the table awaiting user action. **Ignore** loose suggestions directly ignored by the user.

### Type B: Static Memory (User Profile)
* **Extract:** Permanent user attributes (dietary habits, health status, profession, family, hard constraints).
* **Ignore:** Temporary states (hungry now), emotions, pleasantries, feedback on AI performance.

## Phase 2: Conflict Resolution
If the same attribute is discussed multiple times, determine the **final state** at the end of the conversation. Overwrite previous values with the latest confirmed state.

# Output Format
You must output a JSON object.

## JSON Structure
{{
  "thought_process": "Brief analysis of the dialogue flow, noting any corrections or backtracking logic used...",
  "memory_points": [
    {{
      "index": "DM-{{session_id}}-XX",
      "type": "Dynamic",
      "tag": "Confirmed" | "Draft",
      "content": "【Attribute Name】 Complete detailed statement...",
      "source_turn": integer
    }},
    {{
      "index": "SM-{{session_id}}-XX",
      "type": "Static",
      "tag": null,
      "content": "Content stated by user...",
      "source_turn": integer
    }}
  ]
}}

# Few-Shot Examples

## Example 1: Basic Static Memory & Backtracking Logic
<transcript>
[Turn 1] AI: I can book a flight for you. Do you have any dietary restrictions?
[Turn 2] User: I am a vegan.
[Turn 3] AI: Understood. I found a flight to NY on Friday for $300. Should I book it?
[Turn 4] User: Yes, please book it.
</transcript>
<output_example>
{{
  "thought_process": "User defined static attribute (vegan) in Turn 2. User confirmed AI proposal in Turn 4, needed to backtrack to Turn 3 for details.",
  "memory_points": [
    {{
      "index": "SM-E01-01",
      "type": "Static",
      "tag": null,
      "content": "User is vegan.",
      "source_turn": 2
    }},
    {{
      "index": "DM-E01-01",
      "type": "Dynamic",
      "tag": "Confirmed",
      "content": "【Flight Booking】 Destination: NY, Price: $300, Day: Friday.",
      "source_turn": 3
    }}
  ]
}}
</output_example>

## Example 2: Correction & Overwrite
<transcript>
[Turn 1] User: Schedule a meeting for 2 PM.
[Turn 2] AI: Okay, 2 PM is set.
[Turn 3] User: Actually, push it to 4 PM.
[Turn 4] AI: No problem, changed to 4 PM.
</transcript>
<output_example>
{{
  "thought_process": "User initially set 2 PM, but corrected to 4 PM in Turn 3. Extract only final state (4 PM).",
  "memory_points": [
    {{
      "index": "DM-E02-01",
      "type": "Dynamic",
      "tag": "Confirmed",
      "content": "【Meeting Time】 4:00 PM.",
      "source_turn": 3
    }}
  ]
}}
</output_example>

## Example 3: Rejection (No Memory)
<transcript>
[Turn 1] AI: Should we add a reminder for tomorrow morning?
[Turn 2] User: No, I don't need that.
</transcript>
<output_example>
{{
  "thought_process": "AI proposed a reminder, but user explicitly rejected it. No memory created.",
  "memory_points": []
}}
</output_example>

# Start Analysis
Analyze the dialogue transcript above and output JSON.
"""

        try:
            # 2. LLM Generation
            response = self.llm_client.generate(prompt, temperature=0.1)

            error_handler = LLMErrorHandler(self.llm_client)
            json_result = error_handler.extract_json_from_response(response)

            final_data = {"memory_points": []}

            if json_result.success:
                final_data = json_result.result
            else:
                print(f"⚠️ Memory Extraction JSON Error: {json_result.error_message}")
                print(f"⚠️ Attempting LLM-based repair for invalid JSON...")

                # Build Repair Prompt
                repair_prompt = f"""
The following text contains a JSON object but has syntax errors (e.g., missing commas, quotes, brackets, trailing commas).
Please fix it and **output only the valid JSON string**. Do not add any markdown formatting or explanations.

<invalid_text>
{response}
</invalid_text>
"""
                try:
                    # Call LLM to repair JSON
                    repair_response = self.llm_client.generate(repair_prompt, temperature=0.1)

                    # Attempt to extract JSON from repair response
                    repaired_result = error_handler.extract_json_from_response(repair_response)

                    if repaired_result.success:
                        print("✅ JSON successfully repaired by LLM")
                        final_data = repaired_result.result
                    else:
                        print(f"❌ LLM repair failed: {repaired_result.error_message}")
                        print(f"Raw Repair Response: {repair_response}")
                except Exception as repair_error:
                    print(f"❌ Error during JSON repair: {str(repair_error)}")

                print(f"Raw Original Response: {response}") # Debug helper

            # =========================================================
            # ✨ New: Post-processing step
            # =========================================================
            if final_data and "memory_points" in final_data:
                self._enrich_memory_with_source_content(final_data["memory_points"], dialogue_turns)

            return final_data

        except Exception as e:
            print(f"⚠️ Memory Extraction Critical Error: {str(e)}")
            return {"memory_points": []}

    def _enrich_memory_with_source_content(self, memory_points: List[Dict], dialogue_turns: List[Dict]):
        """
        Look up original dialogue content by source_turn index and inject it into memory_points.
        """
        for point in memory_points:
            source_turn_index = point.get("source_turn")

            # Robustness check: Ensure index exists and is an integer
            if isinstance(source_turn_index, int):
                # Conversion logic: LLM output is 1-based (Turn 1), list is 0-based
                list_index = source_turn_index - 1

                # Check index bounds (Prevent LLM hallucinating non-existent line numbers)
                if 0 <= list_index < len(dialogue_turns):
                    original_turn = dialogue_turns[list_index]

                    # Inject original content, clean XML tags
                    # from utils.dialogue_postprocessor import DialoguePostprocessor
                    dialogue_postprocessor = DialoguePostprocessor()
                    cleaned_content = dialogue_postprocessor.clean_all_xml_tags(original_turn.get("content", ""))

                    point["source_content_snapshot"] = cleaned_content
                    point["source_role_snapshot"] = original_turn.get("speaker", "")
                else:
                    point["source_content_snapshot"] = "<Error: Source Turn Out of Bounds>"
            else:
                point["source_content_snapshot"] = "<Error: Invalid Source Turn Index>"


class MemoryDeduplicationAgent:
    """Memory Deduplication Agent - Consolidates redundant memory points"""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def deduplicate_memory_points(self, memory_data: Dict[str, Any], batch_size: int = 20) -> Dict[str, Any]:
        """
        Deduplicates and optimizes memory points in batches.

        Args:
            memory_data: Dictionary containing the list of memory points {"memory_points": [...]}
            batch_size: Number of items to process in a single LLM call (Default 50)

        Returns:
            Dictionary containing valid (non-discarded) memory points.
        """
        all_points = memory_data.get('memory_points', [])

        if not all_points:
            return {"memory_points": []}

        final_valid_points = []
        total_count = len(all_points)
        
        # Calculate total batches required
        total_batches = (total_count + batch_size - 1) // batch_size
        
        print(f"🔄 Starting Batch Deduplication: {total_count} points total, {total_batches} batches (Batch Size: {batch_size})")

        # Process in batches
        for i in range(0, total_count, batch_size):
            batch_index = (i // batch_size) + 1
            current_batch = all_points[i:i + batch_size]
            
            print(f"  - Processing Batch {batch_index}/{total_batches} ({len(current_batch)} points)...")
            
            # Process current batch
            batch_valid_points = self._process_single_batch(current_batch)
            final_valid_points.extend(batch_valid_points)

        # Calculate count of active points (filter out those where discard=true)
        active_points_count = len([m for m in final_valid_points if not m.get('discard', False)])
        print(f"✅ Deduplication Complete: {total_count} -> {active_points_count} active points.")
        return {"memory_points": final_valid_points}

    def _process_single_batch(self, memory_points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a single batch of memory points.
        """
        if not memory_points:
            return []

        # Preprocessing: Build plain text representation & Index mapping for recovery
        id_to_original = {}
        memory_text = ""

        for i, point in enumerate(memory_points):
            index = point.get('index', f'Point-{i+1}')
            content = point.get('content', '')

            # Build index map for subsequent data recovery
            id_to_original[index] = point

            # Format line: [Index] Content
            memory_text += f"[{index}] {content}\n"

        prompt = f"""
# Role
You are an expert **"Memory Deduplication & Status Tagger"**.
Your task is to analyze the provided list of memory points to identify duplicate, redundant, or conflicting information.
**Key Requirement: You are strictly forbidden from directly deleting any memory points.** You must mark the `discard` status for each point and optimize the content of valid points.

# Context Data
<current_memory_points>
{memory_text}
</current_memory_points>

# Processing Rules

## 1. Status Tagging Rules (`discard` field)
Analyze each point to determine if it should be kept or discarded:
- **discard: true (Discard)**:
    - The point is a duplicate of another point in the list.
    - The point's information is less detailed than another point covering the same topic.
    - The point is logically obsolete based on newer information.
- **discard: false (Valid)**:
    - The point contains unique, valid, and the most current information.
    - The point is a "Master Version" that effectively consolidates information from other discarded points.

## 2. Content Optimization Rules (For `discard: false` points)
- **Merge Complementary Info**: If a kept point can absorb information from a discarded point (e.g., merging "Budget" + "Location"), please modify its `content` to include the complete information.
- **Keep As Is**: If no merging is required, keep the original content unchanged.

## 3. Priority Judgment
- **Detail Oriented**: "Sushi at 7 PM" > "Eat Sushi".
- **Completeness**: Merged/Consolidated info > Partial info.

# Output Format
Output a single JSON object containing the processed list of memory points. **You must include every input memory point** (keeping its original index).

## JSON Structure
{{
  "memory_points": [
    {{
      "index": "Original Index",
      "content": "Optimized content (keep original if discard:true)",
      "discard": trueOrFalse  // true = Redundant/Obsolete, false = Valid/Active
    }}
  ]
}}

# Few-Shot Examples

## Example 1: Consolidation
**Input:**
[DM-01] User wants Italian food.
[DM-02] User confirmed Italian food at 8 PM.

**Output:**
{{
  "memory_points": [
    {{
      "index": "DM-01",
      "content": "User wants Italian food.",
      "discard": true
    }},
    {{
      "index": "DM-02",
      "content": "User confirmed Italian food at 8 PM.",
      "discard": false
    }}
  ]
}}

## Example 2: Merging Details
**Input:**
[SM-01] User is vegan.
[SM-02] User is vegan and allergic to nuts.

**Output:**
{{
  "memory_points": [
    {{
      "index": "SM-01",
      "content": "User is vegan.",
      "discard": true
    }},
    {{
      "index": "SM-02",
      "content": "User is vegan and allergic to nuts.",
      "discard": false
    }}
  ]
}}

# Start Processing
Analyze <current_memory_points> and output JSON.
"""

        try:
            # Low temperature to ensure logical consistency
            response = self.llm_client.generate(prompt, temperature=0.1)

            error_handler = LLMErrorHandler(self.llm_client)
            json_result = error_handler.extract_json_from_response(response)

            valid_points = []

            if json_result.success:
                processed_points = json_result.result.get('memory_points', [])

                for p in processed_points:
                    # Critical Fix: Use the original object as the Base
                    idx = p.get('index')
                    # Clean up brackets that LLM might return in index
                    if idx:
                        idx = idx.strip('[]')
                    original = id_to_original.get(idx)

                    if original:
                        # [Correction Point]: Create a shallow copy of the original data
                        # This preserves all original fields: type, tag, session_id, source_turn
                        # As well as the newly added source_content_snapshot, source_role_snapshot
                        final_point = original.copy()

                        # Only update fields allowed to be modified by LLM: content
                        # LLM might have merged or optimized content, so we must overwrite original content with LLM output
                        if 'content' in p:
                            final_point['content'] = p['content']

                        # Keep discard status info, but do not use for filtering here (filtering happens in main loop)
                        if 'discard' in p:
                            final_point['discard'] = p['discard']

                        valid_points.append(final_point)
                    else:
                        # Defensive programming: If original index not found (theoretically shouldn't happen), temporarily trust LLM output
                        valid_points.append(p)

                return valid_points
            else:
                print(f"⚠️ Batch Deduplication JSON Error: {json_result.error_message}")
                # If current batch fails, safely return the original data of this batch (skip deduplication)
                return memory_points

        except Exception as e:
            print(f"⚠️ Batch Deduplication Critical Error: {str(e)}")
            return memory_points

class MemoryRetrieveAgent:
    """
    Dual-Track Memory Retrieval Agent
    Track 1: Dynamic Memory (Project Execution Flow) - Strict matching, relies on Tags.
    Track 2: Static Memory (User Profile Flow) - Associative matching, relies on Content Semantics.
    """

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def _format_memories_for_prompt(self, memories: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        """
        Helper: Converts memory objects into an LLM-readable text format.
        Also establishes an index map for retrieving original objects later.
        """
        text_lines = []
        id_map = {}

        for m in memories:
            # Use 'index' from the data structure as the unique identifier
            m_id = m.get('index', 'UNKNOWN_ID')
            content = m.get('content', '')
            tag = m.get('tag', '') # e.g., Confirmed, Draft, etc.
            # memory_type = m.get('type', '') # Dynamic or Static
            source_turn = m.get('source_turn', '')

            id_map[m_id] = m

            # Formatting Strategy:
            # If a tag exists (Dynamic Memory), show [ID] [Tag] [Source] to emphasize status, type, and source.
            # If no tag (Static Memory), show [ID] [Source] content directly.
            if tag:
                line = f"[{m_id}] [{tag}] [Turn {source_turn}] {content}"
            else:
                line = f"[{m_id}] [Turn {source_turn}] {content}"

            text_lines.append(line)

        return "\n".join(text_lines), id_map

    def retrieve_project_memories(self, dynamic_memories: List[Dict[str, Any]], task_summary: str) -> List[Dict[str, Any]]:
        """
        [Track 1] Retrieve Dynamic Memories - Batch Version
        Core Logic: Find entries that constitute "Constraints" or "Foundation" for the current task.

        Processing Strategy:
        - Filter out memory points marked as 'discard'
        - Split memories into batches of 100, retrieve top 10 per batch
        - Consolidate and deduplicate results across batches
        """
        if not dynamic_memories:
            return []

        # Filter out discarded memories
        active_memories = [m for m in dynamic_memories if not m.get('discard', False)]
        if not active_memories:
            return []

        all_selected_memories = []
        batch_size = 100
        target_per_batch = 10

        # Process memories in batches
        for i in range(0, len(active_memories), batch_size):
            batch_memories = active_memories[i:i + batch_size]
            print(f"      🔄 Processing Batch {i//batch_size + 1}/{(len(active_memories) + batch_size - 1)//batch_size} ({len(batch_memories)} memories)")

            mem_text, id_map = self._format_memories_for_prompt(batch_memories)

            prompt = f"""
# Role
You are a "Zero Omission" Project Memory Retriever.
The system is about to execute task <Current_Task>. To prevent execution errors, you must retrieve **all potentially relevant** information from the <Project_History>.

# Input
<Current_Task>
{task_summary}
</Current_Task>

<Project_History>
{mem_text}
</Project_History>

# Retrieval Criteria (Principle of Over-Inclusion)
Please scan every memory. If it meets ANY of the following conditions, you **MUST** add its index to the list:

1.  **Hard Constraints**: Involves time, location, budget, personnel, technical specs, etc. (Even if it is a 'Draft').
2.  **Context Relevance**: Explains the origin, purpose, or prerequisites of the task.
3.  **Keyword Hits**: The memory contains entity words found in the task description (e.g., specific client names, place names, item names).
4.  **Uncertainty Principle**: If you are hesitant about whether a memory is useful, **SELECT IT**. Do not try to "save" Tokens for me; I would rather read too much than miss something.

**Negative Filter (Exclude Only)**: Exclude only completely irrelevant chit-chat or obsolete information clearly belonging to other closed projects.

# Output Control
* **Strictly Forbid** outputting any reasoning process, explanatory text, or Markdown tags (like ```json).
* **Only** return a standard JSON object.
* The format must be strictly as follows:
{{
    "selected_indices": ["INDEX_1", "INDEX_2"]
}}
"""
            # Dynamic retrieval requires high precision, use low Temperature
            batch_selected = self._execute_llm_call(prompt, id_map, temperature=0.1)
            all_selected_memories.extend(batch_selected)

            if batch_selected:
                print(f"      ✅ Selected {len(batch_selected)} memories from Batch {i//batch_size + 1}")
            else:
                print(f"      ⚠️ No relevant memories found in Batch {i//batch_size + 1}")

        # Cross-batch deduplication (using index as unique identifier)
        seen_indices = set()
        deduplicated_memories = []

        for memory in all_selected_memories:
            memory_index = memory.get('index', '')
            if memory_index and memory_index not in seen_indices:
                seen_indices.add(memory_index)
                deduplicated_memories.append(memory)

        print(f"      📊 Batch Retrieval Summary: Selected {len(all_selected_memories)} total → {len(deduplicated_memories)} unique memories after deduplication")

        return deduplicated_memories

    def retrieve_static_memories(self, static_memories: List[Dict[str, Any]], task_summary: str) -> List[Dict[str, Any]]:
        """
        [Track 2] Retrieve Static Memories
        Core Logic: Associative Reasoning (Contextual Association)

        Processing Strategy:
        - Filter out memory points marked as 'discard'
        """
        if not static_memories:
            return []

        # Filter out discarded memories
        active_memories = [m for m in static_memories if not m.get('discard', False)]
        if not active_memories:
            return []

        mem_text, id_map = self._format_memories_for_prompt(active_memories)

        prompt = f"""
# Role
You are a Personal Assistant with "Mind-Reading Capabilities."
When assisting the user in completing <Current_Task>, your core task is to: Retrieve all <User_Preferences> that might improve the experience, avoid pitfalls, or provide convenience.

# Input
<Current_Task>
{task_summary}
</Current_Task>

<User_Preferences>
{mem_text}
</User_Preferences>

# Retrieval Logic (Broad Association & Pitfall Avoidance)
Please scan every memory. If there is **any dimension of association**, you must select it. Do not worry about selecting too many; I need the richest possible context.

**Please scan based on these three dimensions:**

1.  **Context & Resources**:
    * **Environment**: Does the task involve outdoors/weather/noise? Retrieve relevant preferences (e.g., "likes window seats," "hates rain").
    * **Benefits**: Does the task involve spending/booking? Retrieve membership cards, coupons, or preferred brands.
    * **Tools**: Does the task involve specific software or devices? Retrieve usage habits.

2.  **Style & Taste**:
    * Retrieve user aesthetic tendencies in relevant fields (e.g., color, tone, design style, flavor).
    * *Example*: If the task is "Write an email," you must retrieve memories like "User prefers concise style."

3.  **Pitfall Guide (Constraints & Taboos) [Highest Priority]**:
    * **Must Select** all records regarding "Dislikes," "Allergies," "Hates," or "Blacklists" that are potentially related to the task.
    * *Principle*: Memories that stop me from making mistakes are more important than memories that tell me how to do better.

# Decision Threshold
* **Over-Inclusion**: If you are unsure if a preference applies, **keep it**.
* **Fuzzy Matching**: Literal matching is not required. If the task is "Buy coffee" and the memory is "Recently quitting sugar," this is a strong correlation and must be selected.

# Output Control
* **Strictly Forbid** outputting any reasoning process, explanatory text, or Markdown tags.
* **Only** return a standard JSON object.
* The format must be strictly as follows:
{{
    "selected_indices": ["SM-S12_01-01", "SM-S12_04-02"]
}}
"""
        # Static retrieval requires associative capability, use slightly higher Temperature
        return self._execute_llm_call(prompt, id_map, temperature=0.25)

    def _execute_llm_call(self, prompt: str, id_map: Dict, temperature: float) -> List[Dict]:
        """Unified execution of LLM calls and parsing."""
        try:
            response = self.llm_client.generate(prompt, temperature=temperature)

            # Assuming you have an error handler tool
            error_handler = LLMErrorHandler(self.llm_client)
            json_result = error_handler.extract_json_from_response(response)

            selected_items = []
            if json_result.success:
                indices = json_result.result.get("selected_indices", [])
                for idx in indices:
                    if idx in id_map:
                        selected_items.append(id_map[idx])

            return selected_items
        except Exception as e:
            print(f"Retrieval Error: {e}")
            return []


class ConversationController:
    """Conversation Controller - Coordinates various agents"""

    def __init__(self, dialogue_client: LLMClient, evaluation_client: LLMClient, memory_client: LLMClient, memory_retrieve_client: LLMClient = None, dedup_client: LLMClient = None, semantic_schedule_client: LLMClient = None, project_attributes_schema: str = ""):
        self.dialogue_client = dialogue_client
        self.evaluation_client = evaluation_client
        self.memory_client = memory_client
        # If memory_retrieve_client not provided, default to memory_client
        self.memory_retrieve_client = memory_retrieve_client or memory_client
        # If dedup_client not provided, default to memory_client
        self.dedup_client = dedup_client or memory_client
        # If semantic_schedule_client not provided, default to memory_client
        self.semantic_schedule_client = semantic_schedule_client or memory_client
        self.user_agent = None
        self.assistant_agent = None
        self.goal_evaluator = GoalEvaluatorAgent(evaluation_client)  # Use dedicated evaluation client
        self.memory_manager = None
        self.memory_deduplicator = MemoryDeduplicationAgent(self.dedup_client)  # Use dedicated dedup client
        self.memory_retriever = MemoryRetrieveAgent(self.memory_retrieve_client)  # Use dedicated memory retrieval client
        self.semantic_schedule_agent = SemanticScheduleAgent(self.semantic_schedule_client)  # [NEW] Semantic schedule agent
        self.project_attributes_schema = project_attributes_schema
        # self.dialogue_postprocessor = DialoguePostprocessor()  # Postprocessor for cleaning XML tags

    def initialize_agents(self, user_profile: Dict[str, Any], full_event_log: List[Dict[str, Any]], current_event_session_summary_list: List[Dict[str, Any]] = None, history_dialogue: List[Dict[str, Any]] = None):
        """Initialize all agents"""
        self.user_agent = UserAgent(self.dialogue_client, user_profile, full_event_log, current_event_session_summary_list, history_dialogue)  # Pass history dialogue
        self.assistant_agent = AssistantAgent(self.dialogue_client, history_dialogue)  # Pass history dialogue
        # MemoryManagerAgent is now used for memory extraction after dialogue ends, no need to initialize during conversation

    def create_memory_manager(self) -> MemoryManagerAgent:
        """Create memory manager agent for memory extraction after session ends"""
        return MemoryManagerAgent(self.memory_client, self.project_attributes_schema)  # Use dedicated memory client

    def update_schedule_plan(self, dialogue_history: List[DialogueTurn], current_time: str, current_plan_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Update schedule plan using SemanticScheduleAgent

        Args:
            dialogue_history: Dialogue history
            current_time: Current time
            current_plan_items: Current plan item list

        Returns:
            Updated plan item list
        """
        # Convert dialogue history to text format
        dialogue_text = "\n".join([f"{turn.speaker}: {turn.content}" for turn in dialogue_history])

        # Use SemanticScheduleAgent to process
        return self.semantic_schedule_agent.process(dialogue_text, current_time, current_plan_items)

    def determine_conversation_phase(self, conversation_history: List[DialogueTurn]) -> ConversationPhase:
        """Determine current conversation phase"""
        turn_count = len(conversation_history)

        if turn_count == 0:
            return ConversationPhase.OPENING
        elif turn_count <= 2:
            return ConversationPhase.EXPLORATION
        elif turn_count <= 4:
            return ConversationPhase.SOLUTION
        elif turn_count <= 6:
            return ConversationPhase.CONFIRMATION
        else:
            return ConversationPhase.CLOSURE

    def conduct_conversation(self, session_summary: str, session_id: str,
                           max_turns: int = 18, session_context: Dict[str, Any] = None, history_dialogue: List[Dict[str, Any]] = None,
                          current_time: str = "", current_plan_items: List[Dict[str, Any]] = None) -> Tuple[List[DialogueTurn], GoalEvaluation, Dict[str, Any], List[Dict[str, Any]]]:
        """Conduct conversation"""
        conversation_history = []
        current_goal = session_summary

        # Process plan item data
        if current_plan_items is None:
            current_plan_items = []


        #print("------------------------------------------")
        #print("current_plan_items: " + str(current_plan_items))
        #print("-------------------------------------")

        # Use memory retrieval agent to extract relevant memories
        memory_context = ""
        if session_context and "memory_points" in session_context:
            memory_points = session_context["memory_points"]

            if isinstance(memory_points, list) and memory_points:
                # Separate dynamic and static memories
                dynamic_memories = [m for m in memory_points if m.get('type') == 'Dynamic']
                static_memories = [m for m in memory_points if m.get('type') == 'Static']

                # Use memory retrieval agent to get relevant memories (this logic remains unchanged)
                retrieved_dynamic = self.memory_retriever.retrieve_project_memories(dynamic_memories, session_summary)
                retrieved_static = self.memory_retriever.retrieve_static_memories(static_memories, session_summary)

                # Save retrieved memory data for return
                retrieved_memory_data = {
                    "dynamic": retrieved_dynamic,
                    "static": retrieved_static,
                    "all": retrieved_dynamic + retrieved_static
                }

                # Format retrieved memories as strings
                formatted_memories = []

                # --- Helper function: Format single memory point ---
                def format_memory_string(point, category):
                    index = point.get('index', 'Unknown')
                    content = point.get('content', '')

                    # Get newly added fields (use .get to prevent errors with old data)
                    src_content = point.get('source_content_snapshot')
                    src_role = point.get('source_role_snapshot')

                    # Build base string
                    if category == 'Dynamic':
                        tag = point.get('tag', 'Draft')
                        base_str = f"【{index}】[Dynamic][{tag}] {content}"
                    else:
                        base_str = f"【{index}】[Static] {content}"

                    # If reference content exists, add to end as evidence (Grounding)
                    if src_content:
                        # Format example: (Ref AI: "Shall we book...")
                        role_str = f" {src_role}" if src_role else ""
                        base_str += f' (Ref{role_str}: "{src_content}")'

                    return base_str

                # 1. Format dynamic memories
                for point in retrieved_dynamic:
                    formatted_memories.append(format_memory_string(point, 'Dynamic'))

                # 2. Format static memories
                for point in retrieved_static:
                    formatted_memories.append(format_memory_string(point, 'Static'))

                memory_context = "\n".join(formatted_memories)

                # Print retrieval results
                print(f"      🔍 Memory Retrieval: Dynamic {len(retrieved_dynamic)}/{len(dynamic_memories)}, Static {len(retrieved_static)}/{len(static_memories)}")


        #print(memory_context)
        #print("Build conversation phase--------------------------------")

        # User opening
        phase = ConversationPhase.OPENING
        opening_message = self.user_agent.generate_opening(current_goal, current_time)
        conversation_history.append(DialogueTurn("User", opening_message, phase))

        # Conversation loop
        for turn_num in range(max_turns):
            # Determine current phase
            phase = self.determine_conversation_phase(conversation_history)

            # Check if it's the last round and user just spoke, ensure AI can respond
            if turn_num == max_turns - 1 and len(conversation_history) % 2 == 1:
                # Last round, force AI response to end conversation
                last_user_message = conversation_history[-1].content
                ai_response = self.assistant_agent.generate_response(
                    last_user_message, conversation_history, memory_context,
                    ConversationPhase.CLOSURE, current_time, current_plan_items, current_goal
                )
                conversation_history.append(DialogueTurn("Assistant", ai_response, ConversationPhase.CLOSURE))
                break

            if len(conversation_history) % 2 == 1:  # AI response
                last_user_message = conversation_history[-1].content
                # Pass actual memory context, including time and schedule
                ai_response = self.assistant_agent.generate_response(
                    last_user_message, conversation_history, memory_context, phase, current_time, current_plan_items, current_goal
                )
                conversation_history.append(DialogueTurn("Assistant", ai_response, phase))

            else:  # User response
                last_ai_message = conversation_history[-1].content
                user_response = self.user_agent.generate_response(
                    last_ai_message, conversation_history, current_goal, phase, current_time
                )
                conversation_history.append(DialogueTurn("User", user_response, phase))

            # Evaluate goal completion every 2 rounds (after AI speaks, ensuring complete User-AI dialogue turn)
            if len(conversation_history) >= 6 and len(conversation_history) % 2 == 0:  # Even length (AI just finished speaking)
                evaluation = self.goal_evaluator.evaluate_goal_completion(conversation_history, current_goal)

                if evaluation.is_complete:
                    # Dialogue complete, current AI response is the final summary, no additional response needed
                    break

        # Final evaluation
        final_evaluation = self.goal_evaluator.evaluate_goal_completion(conversation_history, current_goal)

        # Use SemanticScheduleAgent to update plan items (can update as long as there's current time)
        updated_plan_items = current_plan_items if current_plan_items else []
        if current_time:
            print(f"📅 Updating plan items...")
            updated_plan_items = self.update_schedule_plan(conversation_history, current_time, updated_plan_items)
            print(f"📅 Plan items update complete: {len(current_plan_items) if current_plan_items else 0} -> {len(updated_plan_items)} items")

        # If no retrieved memory, create empty structure
        if 'retrieved_memory_data' not in locals():
            retrieved_memory_data = {
                "dynamic": [],
                "static": [],
                "all": []
            }

        return conversation_history, final_evaluation, retrieved_memory_data, updated_plan_items

    def extract_session_memory(self, dialogue_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract memory points after dialogue ends

        Args:
            dialogue_data: Dictionary containing dialogue data

        Returns:
            Memory point data, format consistent with memory.json
        """
        memory_manager = self.create_memory_manager()
        return memory_manager.extract_memory_points(dialogue_data)

    def deduplicate_memory_points(self, memory_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deduplicate memory points

        Args:
            memory_data: Dictionary containing all memory points, format is {"memory_points": [...]}

        Returns:
            Deduplicated memory point data, format same as input
        """
        return self.memory_deduplicator.deduplicate_memory_points(memory_data)


class MultiAgentDialogueProcessor(BaseProcessor):
    """Multi-Agent Dialogue Generation Processor"""

    def get_input_schema(self) -> Dict[str, Any]:
        """Retrieve input data schema"""
        return {
            "type": "object",
            "properties": {
                "user_input_profile": {"type": "object", "description": "User Profile"},
                "full_event_log": {"type": "array", "description": "Full Event Log"},
                "current_event_session_summary_list": {"type": "array", "description": "Session Summary List"},
                "target_session": {"type": "object", "description": "Target Session"},
                "current_time": {"type": "string", "description": "Current simulation time (e.g., 'Friday 14:00')"},
                "current_plan_items": {"type": "array", "description": "Current plan items for semantic schedule processing", "items": {"type": "string"}},
                "max_turns": {"type": "integer", "description": "Maximum conversation turns"},
                "memory_context": {"type": "object", "description": "Memory context with memory_points"}
            },
            "required": ["user_input_profile", "full_event_log", "target_session"]
        }

    def get_output_schema(self) -> Dict[str, Any]:
        """Retrieve output data schema"""
        return {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "dialogue_turns": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "speaker": {"type": "string"},
                            "content": {"type": "string"},
                            "phase": {"type": "string"},
                            "timestamp": {"type": "number"}
                        }
                    }
                },
                "goal_evaluation": {
                    "type": "object",
                    "properties": {
                        "overall_score": {"type": "number"},
                        "problem_understanding": {"type": "number"},
                        "solution_provided": {"type": "number"},
                        "user_confirmed": {"type": "number"},
                        "task_closed": {"type": "number"},
                        "is_complete": {"type": "boolean"},
                        "reasoning": {"type": "string"}
                    }
                },
                "retrieved_memory": {"type": "object", "description": "Retrieved memory data with dynamic and static memories"},
                "updated_plan_items": {
                    "type": "array",
                    "description": "Semantically updated plan items after conversation",
                    "items": {"type": "string"}
                },
                "metadata": {"type": "object"}
            }
        }

    def validate_input(self, data: Any) -> ProcessorResult:
        """Validate input data"""
        return ProcessorResult(success=True, data=data)

    def validate_output(self, data: Any) -> ProcessorResult:
        """Validate output data"""
        if isinstance(data, (dict, list)):
            return ProcessorResult(success=True, data=data)
        else:
            return ProcessorResult(
                success=False,
                error_message="Output must be a valid JSON data structure"
            )

    def process_core(self, data: Any) -> ProcessorResult:
        """Core processing logic"""
        start_time = time.time()

        try:
            # Extract input data
            user_profile = data.get("user_input_profile", {})
            full_event_log = data.get("full_event_log", [])
            target_session = data.get("target_session", {})
            session_summary = target_session.get("session_summary", "")
            session_id = target_session.get("session_id", "unknown")
            current_event_session_summary_list = data.get("current_event_session_summary_list", [])
            max_turns = data.get("max_turns", 12)  # Default to 12 if not provided
            memory_context = data.get("memory_context", {"memory_points": []})  # Extract memory context
            # [NEW] Extract time and plan context
            current_time = data.get("current_time", "")  # Current simulation time
            current_plan_items = data.get("current_plan_items", [])  # Current plan items for semantic schedule processing

            # Initialize conversation controller with semantic schedule support
            controller = ConversationController(
                dialogue_client=self.llm_client,
                evaluation_client=self.llm_client,  # Defaulting to the same client
                memory_client=self.llm_client,       # Defaulting to the same client
                dedup_client=self.llm_client,        # Defaulting to the same client
                semantic_schedule_client=self.llm_client  # [NEW] Defaulting to the same client
            )
            controller.initialize_agents(user_profile, full_event_log, current_event_session_summary_list)

            # Conduct conversation with memory context, time, schedule, and plan items
            conversation_history, goal_evaluation, retrieved_memory, updated_plan_items = controller.conduct_conversation(
                session_summary, session_id,
                max_turns=max_turns,
                session_context=memory_context,
                current_time=current_time,
                current_plan_items=current_plan_items
            )

            # Construct output data
            output_data = {
                "session_id": session_id,
                "dialogue_turns": [
                    {
                        "speaker": turn.speaker,
                        "content": turn.content,
                        "phase": turn.phase.value,
                        "timestamp": turn.timestamp
                    }
                    for turn in conversation_history
                ],
                "goal_evaluation": {
                    "overall_score": goal_evaluation.overall_score,
                    "problem_understanding": goal_evaluation.problem_understanding,
                    "solution_provided": goal_evaluation.solution_provided,
                    "user_confirmed": goal_evaluation.user_confirmed,
                    "task_closed": goal_evaluation.task_closed,
                    "is_complete": goal_evaluation.is_complete,
                    "reasoning": goal_evaluation.reasoning
                },
                "retrieved_memory": retrieved_memory,  # Retrieved memory data
                "updated_plan_items": updated_plan_items,  # [NEW] Semantically updated plan items
                "metadata": {
                    "total_turns": len(conversation_history),
                    "processing_time": time.time() - start_time,
                    "processor_type": "multi_agent_dialogue_with_schedule"
                }
            }

            print(f"✅ Multi-Agent Dialogue Generation Complete: {len(conversation_history)} turns, Goal Completion: {goal_evaluation.overall_score:.1f}")

            return ProcessorResult(
                success=True,
                data=output_data,
                processing_time=time.time() - start_time
            )

        except Exception as e:
            return ProcessorResult(
                success=False,
                error_message=f"Multi-Agent Dialogue Generation Failed: {str(e)}",
                processing_time=time.time() - start_time
            )