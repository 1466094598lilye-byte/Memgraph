#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dialogue Postprocessor
Processes memory_analysis tags in Assistant responses, extracts memory points, and cleans response text
"""

import json
import re
import uuid
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timedelta


@dataclass
class MemoryPoint:
    """Memory point data structure"""
    index: str
    session_id: str
    content: str
    memory_type: str  # "Dynamic" or "Static"
    tag: Optional[str] = None  # "Confirmed" or "Draft"


class DialoguePostprocessor:
    """Dialogue postprocessor"""

    def __init__(self):
        self.memory_counter = 1

    def parse_memory_analysis(self, memory_text: str) -> List[str]:
        """
        Parse memory_analysis text and extract memory points

        Args:
            memory_text: Text within memory_analysis tags

        Returns:
            List of extracted memory points (in string format)
        """
        memory_points = []

        if not memory_text or memory_text.strip() == "None":
            return memory_points

        # Split by lines and process each memory reference
        lines = memory_text.strip().split('\n')

        for line in lines:
            line = line.strip()
            if not line or not line.startswith('-'):
                continue

            # Parse format: - [Citation]: [DM-S1_02-01][Dynamic][Confirmed] [Fitness Preparation Task] Prepare necessary exercise equipment -> [Application]: Confirm user has purchased exercise equipment
            try:
                # Extract complete citation and application parts
                citation_match = re.search(r'- \[Citation\]:\s*(.*?)\s*->\s*\[Application\]:\s*(.*)$', line)
                if not citation_match:
                    continue

                citation = citation_match.group(1).strip()
                application = citation_match.group(2).strip()

                # Generate memory point content following the example format
                content = f"{citation} -> [Application]: {application}"

                memory_points.append(content)

            except Exception as e:
                print(f"Warning: Failed to parse memory point: {line}, Error: {e}")
                continue

        return memory_points

    def clean_all_xml_tags(self, text: str) -> str:
        """
        Clean all XML tags and keep only text content

        Args:
            text: Original text (may contain various XML tags)

        Returns:
            Cleaned plain text
        """
        if not text:
            return ""

        # Remove memory_analysis tags and their content (already extracted as memory points)
        cleaned = re.sub(r'<memory_analysis>.*?</memory_analysis>', '', text, flags=re.DOTALL)

        # Process response tags: if entire text is wrapped in response tags, extract content
        # Match: <response>content</response>
        if re.match(r'^\s*<response>.*</response>\s*$', cleaned, re.DOTALL):
            # Extract content within response tags
            cleaned = re.sub(r'<response>(.*?)</response>', r'\1', cleaned, flags=re.DOTALL)
        else:
            # If response tags only wrap part of content, remove tags and keep content
            cleaned = re.sub(r'</?response>', '', cleaned)

        # Clean up excessive blank lines
        cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)

        return cleaned.strip()

    def clean_response(self, response_text: str) -> str:
        """
        Clean response text by removing XML tags (maintains backward compatibility)

        Args:
            response_text: Original response text (may contain memory_analysis and response tags)

        Returns:
            Cleaned plain text response
        """
        return self.clean_all_xml_tags(response_text)

    def process_dialogue_turn(self, turn: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """
        Process a single dialogue turn

        Args:
            turn: Dialogue turn data
            session_id: Session ID

        Returns:
            Processed dialogue turn data
        """
        if turn.get("speaker") != "Assistant":
            return turn

        content = turn.get("content", "")

        # Extract memory_analysis content
        memory_match = re.search(r'<memory_analysis>(.*?)</memory_analysis>', content, re.DOTALL)
        memory_text = memory_match.group(1).strip() if memory_match else ""

        # Parse memory points
        memory_used = []
        if memory_text and memory_text != "None":
            memory_used = self.parse_memory_analysis(memory_text)

        # Clean response content (remove all XML tags)
        cleaned_content = self.clean_all_xml_tags(content)

        # Update turn content
        processed_turn = turn.copy()
        processed_turn["content"] = cleaned_content

        # Add is_query field
        processed_turn["is_query"] = False  # Assistant response is not a query

        # Add memory_used field if there are memory points
        if memory_used:
            processed_turn["memory_used"] = memory_used

        return processed_turn

    def process_dialogue_file(self, input_file: str, output_file: str = None) -> Dict[str, Any]:
        """
        Process a complete dialogue file

        Args:
            input_file: Input file path
            output_file: Output file path, if None overwrites original file

        Returns:
            Processed dialogue data
        """
        # Read input file
        with open(input_file, 'r', encoding='utf-8') as f:
            dialogue_data = json.load(f)

        # Get session_id
        session_id = dialogue_data.get("session_id", "unknown")

        # Reset memory point counter
        self.memory_counter = 1

        # Count total memory points
        total_memory_points = 0

        # Process each dialogue turn
        processed_turns = []
        for turn in dialogue_data.get("dialogue_turns", []):
            processed_turn = self.process_dialogue_turn(turn, session_id)
            processed_turns.append(processed_turn)

            # Count memory points
            if "memory_used" in processed_turn:
                total_memory_points += len(processed_turn["memory_used"])

        # Update dialogue data
        processed_data = dialogue_data.copy()
        processed_data["dialogue_turns"] = processed_turns

        # Remove original extracted_memory section (memory points now stored as used_memory in turns)
        if "extracted_memory" in processed_data:
            del processed_data["extracted_memory"]

        # Update metadata
        metadata = processed_data.get("metadata", {})
        metadata["memory_points_extracted"] = total_memory_points
        metadata["postprocessed"] = True
        processed_data["metadata"] = metadata

        # Write to file
        output_path = output_file or input_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)

        print(f"Post-processing completed: {total_memory_points} memory points extracted")
        return processed_data

    def process_directory(self, input_dir: str, output_dir: str = None, pattern: str = "*_dialogue.json") -> List[Dict[str, Any]]:
        """
        Batch process dialogue files in a directory

        Args:
            input_dir: Input directory
            output_dir: Output directory, if None overwrites original files
            pattern: File matching pattern

        Returns:
            List of processed dialogue data
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

        output_path = Path(output_dir) if output_dir else input_path
        output_path.mkdir(parents=True, exist_ok=True)

        # Find matching files
        files = list(input_path.glob(pattern))
        if not files:
            print(f"Warning: No files matching {pattern} found in {input_dir}")
            return []

        processed_files = []
        for file_path in files:
            try:
                # Determine output file path
                if output_dir:
                    output_file = output_path / file_path.name
                else:
                    output_file = str(file_path)  # Overwrite original file

                # Process file
                processed_data = self.process_dialogue_file(str(file_path), str(output_file))
                processed_files.append(processed_data)

            except Exception as e:
                print(f"Error: Failed to process file {file_path}: {e}")
                continue

        return processed_files

    def process_combined_dialogues(self, input_file: str, output_file: str = None) -> Dict[str, Any]:
        """
        Process combined dialogue file (containing multiple projects)

        Args:
            input_file: Input file path
            output_file: Output file path, if None overwrites original file

        Returns:
            Processed dialogue data
        """
        # Read input file
        with open(input_file, 'r', encoding='utf-8') as f:
            dialogue_data = json.load(f)

        total_memory_points = 0

        # Process each project
        processed_data = dialogue_data.copy()
        for project_name, sessions in dialogue_data.items():
            processed_sessions = []

            for session in sessions:
                session_id = session.get("session_id", "unknown")

                # Process all dialogue turns in this session
                processed_turns = []
                for turn in session.get("dialogue_turns", []):
                    processed_turn = self.process_dialogue_turn(turn, session_id)
                    processed_turns.append(processed_turn)

                    # Count memory points
                    if "memory_used" in processed_turn:
                        total_memory_points += len(processed_turn["memory_used"])

                # Update session data
                processed_session = session.copy()
                processed_session["dialogue_turns"] = processed_turns
                processed_sessions.append(processed_session)

            processed_data[project_name] = processed_sessions

        # Add processing statistics
        processed_data["postprocessing_stats"] = {
            "total_memory_points_extracted": total_memory_points,
            "processed_projects": len(dialogue_data),
            "postprocessed": True
        }

        # Write to file
        output_path = output_file or input_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)

        print(f"Combined dialogue post-processing completed: {total_memory_points} memory points extracted")
        return processed_data

    # ========== Person dialogue merging functionality ==========

    @staticmethod
    def convert_to_safe_filename(name: str) -> str:
        """Convert name to safe filename"""
        safe_name = re.sub(r'[<>:"/\\|?*]', '_', name)
        safe_name = re.sub(r'\s+', '_', safe_name.strip(' .'))
        return safe_name or 'unknown'

    def load_memory_points(self, person_name: str, output_dir: str = "output") -> Dict[str, Any]:
        """
        Load memory file for specified person

        Args:
            person_name: Person name
            output_dir: Output directory

        Returns:
            Memory point dictionary {memory_id: memory_point_data}
        """
        safe_person_name = self.convert_to_safe_filename(person_name)
        memory_file = Path(output_dir) / safe_person_name / f"{safe_person_name}_memory.json"

        if not memory_file.exists():
            raise FileNotFoundError(f"Memory file does not exist: {memory_file}")

        try:
            with open(memory_file, 'r', encoding='utf-8') as f:
                memory_data = json.load(f)

            memory_points = {}
            for memory_point in memory_data.get("memory_points", []):
                memory_id = memory_point.get("index")
                if memory_id:
                    memory_points[memory_id] = memory_point

            return memory_points

        except Exception as e:
            raise RuntimeError(f"Failed to read memory file {memory_file}: {e}")

    @staticmethod
    def extract_memory_id_from_usage(memory_usage: str) -> Optional[str]:
        """Extract memory point ID from memory_usage string"""
        # Pattern 1: content (memory_id) -> [Application]: ...
        match = re.search(r'\(([A-Za-z0-9_]+-(?:DM|SM)-[A-Za-z0-9_]+-[0-9]+)\)', memory_usage)
        if match:
            return match.group(1)

        # Pattern 2: memory_id (content) -> [Application]: ...
        match = re.search(r'^([A-Za-z0-9_]+-(?:DM|SM)-[A-Za-z0-9_]+-[0-9]+)\s*\(', memory_usage)
        if match:
            return match.group(1)

        # Pattern 3: 【memory_id】content -> [Application]: ...
        match = re.search(r'【([A-Za-z0-9_]+-(?:DM|SM)-[A-Za-z0-9_]+-[0-9]+)】', memory_usage)
        if match:
            return match.group(1)

        # Pattern 4: Plan item format 【{session_identifier}-schedule-{id}】
        match = re.search(r'【([^【】]*-schedule-[0-9]+)】', memory_usage)
        if match:
            return match.group(1)

        # Pattern 4.5: Plan item format missing left bracket {session_identifier}-schedule-{id}】
        match = re.search(r'([A-Za-z0-9_]+:[A-Za-z0-9_]+-schedule-[0-9]+)】', memory_usage)
        if match:
            return match.group(1)

        # Pattern 5: Format starting directly with memory_id
        match = re.search(r'^([A-Za-z0-9_]+-(?:DM|SM)-[A-Za-z0-9_]+-[0-9]+)', memory_usage)
        if match:
            return match.group(1)

        return None

    @staticmethod
    def is_schedule_memory_id(memory_id: str) -> bool:
        """Check if this is a plan item format memory ID"""
        return bool(re.search(r'-schedule-[0-9]+', memory_id))

    @staticmethod
    def format_memory_usage_citations(memory_usage: str) -> str:
        """Format memory citation string with 【memory_id】 at the beginning"""
        if not memory_usage:
            return memory_usage

        # If already in 【】 format at beginning, return directly
        if re.search(r'^\s*【[^【】]+】', memory_usage.strip()):
            return memory_usage

        # Simple replacement: find all strings matching memory ID format and unify format
        memory_id_pattern = r'([A-Za-z0-9_]+-(?:DM|SM)-[A-Za-z0-9_]+-[0-9]+)'

        def replace_memory_id(match):
            memory_id = match.group(1)
            return f'【{memory_id}】'

        # Replace all memory IDs with 【memory_id】 format
        result = re.sub(memory_id_pattern, replace_memory_id, memory_usage)

        return result

    def process_current_plan_items_in_memory_analysis(self, session_data: dict, session_identifier: str, memory_points: dict = None) -> dict:
        """
        Process current_plan_items references in <memory_analysis>, add special markers for each reference

        Args:
            session_data: Session data
            session_identifier: Session identifier (format: "project_identifier:session_id")
            memory_points: Memory point dictionary {memory_id: memory_point_data}

        Returns:
            Updated session_data
        """
        # Extract current_plan_items
        current_plan_items = session_data.get("input_data", {}).get("current_plan_items", [])

        if not current_plan_items:
            return session_data

        # Create plan item mapping {id: content}
        plan_map = {}
        for item in current_plan_items:
            plan_map[item['id']] = item['content']

        # Process dialogue turns
        dialogue_turns = session_data.get("dialogue_turns", [])
        if not dialogue_turns:
            dialogue_turns = session_data.get("dialogue_output", {}).get("dialogue_turns", [])

        for turn in dialogue_turns:
            content = turn.get("content", "")

            # Check if contains <memory_analysis>
            if "<memory_analysis>" in content:
                # Create replacement mapping for each plan item
                for plan_id, plan_content in plan_map.items():
                    # Create plan item memory point ID
                    schedule_memory_id = f"{session_identifier}-schedule-{plan_id}"

                    # Extract actual content from memory_points
                    actual_content = plan_content  # Default to original content
                    if memory_points and schedule_memory_id in memory_points:
                        actual_content = memory_points[schedule_memory_id].get("content", plan_content)

                    # Create marker and actual content
                    schedule_mark = f"【{schedule_memory_id}】 {actual_content}"

                    # Find various reference forms of plan item content
                    patterns_to_replace = [
                        # Complete content match
                        (re.escape(plan_content), schedule_mark),
                        # ID reference format [id] content
                        (rf'\[{plan_id}\]\s*{re.escape(plan_content)}', schedule_mark),
                        # Schedule item format
                        (rf'Schedule item.*?\[{plan_id}\][^)]*{re.escape(plan_content[:30])}', schedule_mark),
                        (rf'Schedule item.*?{re.escape(plan_content[:30])}', schedule_mark),
                        # Simple ID reference
                        (rf'\[{plan_id}\]', schedule_mark),
                    ]

                    # Apply replacements
                    for pattern, replacement in patterns_to_replace:
                        content = re.sub(pattern, replacement, content, flags=re.IGNORECASE | re.DOTALL)

                # Update dialogue content
                turn["content"] = content

        # Clean Current prefix in 【Current ...】 format
        for turn in dialogue_turns:
            turn_content = turn.get("content", "")
            turn["content"] = re.sub(r'【Current\s+', '【', turn_content)

        return session_data

    def merge_person_dialogues(self, person_name: str,
                                output_dir: str = "output",
                                output_filename: str = None) -> str:
        """
        Merge all generated dialogues of a person in processed_session_ids order

        Args:
            person_name: Person name
            output_dir: Output directory
            output_filename: Output filename

        Returns:
            Merged file path
        """
        safe_person_name = self.convert_to_safe_filename(person_name)
        person_dir = Path(output_dir) / safe_person_name

        if not person_dir.exists():
            raise FileNotFoundError(f"Person directory not found: {person_dir}")

        # Find interleaved dialogue queue state file
        queue_state_file = person_dir / "interleaved_dialogue_queue_state.json"

        if not queue_state_file.exists():
            raise FileNotFoundError(f"Queue state file does not exist: {queue_state_file}")

        # Read queue state file
        try:
            with open(queue_state_file, 'r', encoding='utf-8') as f:
                queue_state = json.load(f)
        except Exception as e:
            raise FileNotFoundError(f"Cannot read queue state file: {queue_state_file}, Error: {e}")

        # Check status
        if queue_state.get("status") != "completed":
            raise RuntimeError(f"Dialogue generation not completed (status: {queue_state.get('status')})")

        # Get processing order
        processed_session_ids = queue_state.get("processed_session_ids", [])
        if not processed_session_ids:
            raise RuntimeError("No processed_session_ids found")

        # Load memory points
        memory_points = self.load_memory_points(person_name, output_dir)

        merged_dialogues = []
        session_used_memories = {}
        session_uuid_map = {}

        for session_identifier in processed_session_ids:
            # Format: "project_identifier:session_id"
            if ":" not in session_identifier:
                continue

            project_identifier, session_id = session_identifier.split(":", 1)
            session_file = person_dir / project_identifier / "dialogues" / f"{session_id}.json"

            if not session_file.exists():
                print(f"Warning: Dialogue file does not exist: {session_file}")
                continue

            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)

                # Process current_plan_items references
                session_data = self.process_current_plan_items_in_memory_analysis(session_data, session_identifier, memory_points)

                # Extract and process dialogue content
                dialogue_turns = session_data.get("dialogue_turns", [])
                processed_turns = []

                if session_id not in session_used_memories:
                    session_used_memories[session_id] = set()

                for i, turn in enumerate(dialogue_turns):
                    processed_turn = self.process_dialogue_turn(turn, session_id)

                    if "is_query" not in processed_turn:
                        processed_turn["is_query"] = False

                    if "memory_used" in processed_turn and processed_turn["memory_used"]:
                        validated_memory_used = []
                        memory_session_uuids = set()

                        for memory_usage in processed_turn["memory_used"]:
                            formatted_usage = self.format_memory_usage_citations(memory_usage)
                            memory_id = self.extract_memory_id_from_usage(formatted_usage)

                            if memory_id:
                                if self.is_schedule_memory_id(memory_id):
                                    # Handle schedule memory
                                    schedule_match = re.match(r'([^:]+:[^-]+)-schedule-(\d+)', memory_id)
                                    schedule_content = None

                                    if schedule_match:
                                        schedule_session_identifier = schedule_match.group(1)
                                        schedule_id = int(schedule_match.group(2))

                                        # Find session file
                                        schedule_session_file = self.find_session_file(person_dir, schedule_session_identifier)
                                        if schedule_session_file:
                                            plan_item = self.get_schedule_item_from_plan(schedule_session_file, schedule_id)
                                            if plan_item:
                                                schedule_content = plan_item.get("content", "")
                                                correct_session_identifier = plan_item.get("session")
                                                if correct_session_identifier and correct_session_identifier != schedule_session_identifier:
                                                    memory_id = memory_id.replace(schedule_session_identifier, correct_session_identifier, 1)
                                                    schedule_session_identifier = correct_session_identifier

                                        schedule_session_uuid = session_uuid_map.get(schedule_session_identifier, session_uuid)
                                    else:
                                        schedule_session_uuid = session_uuid

                                    # Try to extract content from formatted_usage
                                    if not schedule_content:
                                        content_match = re.search(r'【' + re.escape(memory_id) + r'】([^【【]*?)(?=【|$)', formatted_usage)
                                        if content_match:
                                            schedule_content = content_match.group(1).strip()
                                            schedule_content = re.sub(r'^\[?\d+\]?\s*', '', schedule_content).strip()
                                            schedule_content = re.sub(r'\s*->\s*\[Application\]:.*', '', schedule_content).strip()

                                    if schedule_content:
                                        memory_obj = {
                                            "memory_id": memory_id,
                                            "session_uuid": schedule_session_uuid,
                                            "content": schedule_content
                                        }
                                        memory_session_uuids.add(schedule_session_uuid)
                                        validated_memory_used.append(memory_obj)
                                    continue

                                if memory_id not in memory_points:
                                    print(f"Warning: Memory point does not exist: {memory_id} (session: {session_id})")
                                    continue

                                if memory_id in session_used_memories[session_id]:
                                    continue

                                session_used_memories[session_id].add(memory_id)

                                memory_content = memory_points[memory_id].get("content", "")
                                memory_session_uuid = memory_points[memory_id].get("session_uuid", session_id)

                                memory_session_uuids.add(memory_session_uuid)

                                memory_obj = {
                                    "memory_id": memory_id,
                                    "session_uuid": memory_session_uuid,
                                    "content": memory_content
                                }
                                validated_memory_used.append(memory_obj)
                            else:
                                continue

                        if validated_memory_used:
                            processed_turn["memory_used"] = validated_memory_used
                            processed_turn["memory_session_uuids"] = list(memory_session_uuids)
                        else:
                            processed_turn.pop("memory_used", None)

                    processed_turns.append(processed_turn)

                # Post-processing: set is_query to true for previous turn that has memory_used
                for i, turn in enumerate(processed_turns):
                    if "memory_used" in turn and turn["memory_used"] and i > 0:
                        processed_turns[i-1]["is_query"] = True

                # ========== Apply refinement logic ==========
                # 1. Clean "Current" prefix
                for turn in processed_turns:
                    turn_content = turn.get("content", "")
                    turn_content = re.sub(r'【Current\s+', '【', turn_content)
                    turn["content"] = turn_content

                # 2. Generate query_id
                query_id_counter = 1
                for turn in processed_turns:
                    if turn.get("is_query", False):
                        turn["query_id"] = f"Q-{query_id_counter:04d}"
                        query_id_counter += 1
                    else:
                        turn["query_id"] = None

                extracted_memory = session_data.get("new_memory_points", [])
                session_uuid = uuid.uuid4().hex[:8]

                for memory_point in extracted_memory:
                    memory_point["session_uuid"] = session_uuid

                # ========== Convert current_plan_items to schedule memory points and add to extracted_memory ==========
                # Only add schedules created in this session to extracted_memory
                current_plan_items = session_data.get("input_data", {}).get("current_plan_items", [])
                if current_plan_items:
                    for plan_item in current_plan_items:
                        plan_id = plan_item.get("id")
                        plan_content = plan_item.get("content", "")
                        plan_session = plan_item.get("session", "")

                        # Only add to extracted_memory if plan_item's session field points to current session
                        if plan_session == session_identifier:
                            schedule_memory_id = f"{plan_session}-schedule-{plan_id}"
                            schedule_memory = {
                                "index": schedule_memory_id,
                                "type": "Schedule",
                                "tag": "Current",
                                "content": plan_content,
                                "source_turn": -1,
                                "source_content_snapshot": plan_content,
                                "source_role_snapshot": "System",
                                "session_uuid": session_uuid,
                                "discard": False
                            }
                            extracted_memory.append(schedule_memory)

                # 3. Generate incremental timestamps
                current_time = ""
                try:
                    current_time = session_data.get("current_time", "")
                    if not current_time and dialogue_turns:
                        first_turn = dialogue_turns[0]
                        timestamp = first_turn.get("timestamp", 0)
                        if timestamp:
                            current_time = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d (%A)")
                except Exception as e:
                    print(f"Warning: Failed to read time {session_identifier}: {e}")
                    current_time = ""

                # If no time, generate based on position
                if not current_time:
                    position = len(merged_dialogues)
                    target_date = datetime(2025, 12, 15) + timedelta(days=position)
                    current_time = target_date.strftime("%Y-%m-%d (%A)")

                # 4. Check if Assistant response needs to be added
                if processed_turns and processed_turns[-1]["speaker"] == "User":
                    import random

                    random_responses = [
                        "Noted. Your plan is very well-organized.",
                        "I completely agree with your idea.",
                        "That's awesome! I think your idea is brilliant.",
                        "I totally understand! That makes perfect sense.",
                        "The logic is very clear; I will proceed exactly as you described.",
                        "No problem, we'll follow your pace.",
                        "Okay, fully understood.",
                        "OK, your requirements are very clear.",
                        "It's decided then. You've been very thorough."
                    ]

                    additional_turn = {
                        "speaker": "Assistant",
                        "content": random.choice(random_responses),
                        "phase": "confirmation",
                        "timestamp": datetime.now().timestamp(),
                        "is_query": False,
                        "query_id": None
                    }

                    processed_turns.append(additional_turn)

                session_uuid_map[session_identifier] = session_uuid

                session_info = {
                    "session_identifier": session_identifier,
                    "session_uuid": session_uuid,
                    "current_time": current_time,
                    "dialogue_turns": processed_turns,
                    "extracted_memory": extracted_memory
                }

                merged_dialogues.append(session_info)

            except Exception as e:
                print(f"Error: Failed to read dialogue file {session_file}: {e}")
                continue

        # ========== Process Schedule memories: add schedules in memory_used to corresponding session's extracted_memory ==========

        # Step 1: Collect all referenced schedule memories
        from collections import defaultdict
        schedule_map = defaultdict(list)  # {session_uuid: [schedule_memories]}

        for dialogue in merged_dialogues:
            session_uuid = dialogue.get('session_uuid', '')
            session_identifier = dialogue.get('session_identifier', '')

            for turn in dialogue.get('dialogue_turns', []):
                memory_used_list = turn.get('memory_used', [])

                for memory in memory_used_list:
                    memory_id = memory.get('memory_id', '')

                    if self.is_schedule_memory_id(memory_id):
                        # Use session_uuid from memory as source
                        memory_session_uuid = memory.get('session_uuid', '')

                        schedule_map[memory_session_uuid].append({
                            'memory_id': memory_id,
                            'content': memory.get('content', ''),
                            'session_identifier': session_identifier
                        })

        if schedule_map:
            # Step 2: Add schedule memories to corresponding session's extracted_memory
            added_count = 0
            skipped_count = 0

            for session_uuid, schedules in schedule_map.items():
                # Find corresponding dialogue
                dialogue = None
                for dlg in merged_dialogues:
                    if dlg.get('session_uuid') == session_uuid:
                        dialogue = dlg
                        break

                if not dialogue:
                    continue

                extracted = dialogue.get('extracted_memory', [])

                for schedule in schedules:
                    # Check if already exists (via original memory_id or content)
                    original_memory_id = schedule['memory_id']
                    schedule_content = schedule['content']

                    exists = any(
                        mem.get('original_index') == original_memory_id or
                        mem.get('content', '') == schedule_content
                        for mem in extracted
                    )

                    if exists:
                        skipped_count += 1
                        continue

                    # Use original memory_id as index
                    new_memory = {
                        "index": original_memory_id,
                        "type": "Schedule",
                        "tag": None,
                        "content": schedule_content,
                        "source_turn": 0,
                        "source_content_snapshot": "",
                        "source_role_snapshot": "System",
                        "session_uuid": session_uuid,
                        "discard": False
                    }

                    extracted.append(new_memory)
                    added_count += 1

                dialogue['extracted_memory'] = extracted

        # ========== Generate global memory ID and update extracted_memory and memory_used ==========

        # Create mapping: {original_index: global_memory_id}
        memory_id_map = {}
        global_memory_counter = 1

        # Step 1: Traverse all extracted_memory and generate global IDs (including schedules)
        for dialogue in merged_dialogues:
            extracted = dialogue.get('extracted_memory', [])

            for mem in extracted:
                original_index = mem.get('index')
                if original_index and original_index not in memory_id_map:
                    # Generate global ID
                    global_memory_id = f"mem-{global_memory_counter:04d}"
                    memory_id_map[original_index] = global_memory_id
                    global_memory_counter += 1

        # Step 2: Update index in extracted_memory
        for dialogue in merged_dialogues:
            extracted = dialogue.get('extracted_memory', [])

            for mem in extracted:
                original_index = mem.get('index')
                if original_index in memory_id_map:
                    mem['original_index'] = original_index
                    mem['index'] = memory_id_map[original_index]

        # Step 3: Update memory_id in memory_used (handle two cases)
        updated_count = 0
        schedule_updated_count = 0

        for dialogue in merged_dialogues:
            for turn in dialogue.get('dialogue_turns', []):
                memory_used_list = turn.get('memory_used', [])

                for memory in memory_used_list:
                    if isinstance(memory, dict):
                        original_memory_id = memory.get('memory_id', '')

                        # Check if in memory_id_map
                        if original_memory_id in memory_id_map:
                            memory['original_memory_id'] = original_memory_id
                            memory['memory_id'] = memory_id_map[original_memory_id]
                            updated_count += 1

                            if 'schedule' in original_memory_id.lower():
                                schedule_updated_count += 1

        # Clean irrelevant fields
        cleaned_dialogues = []
        removed_fields = {
            'dialogue_turns': ['phase', 'timestamp'],
            'extracted_memory': ['type', 'tag', 'original_index'],
            'session': ['session_identifier']
        }

        for dialogue in merged_dialogues:
            cleaned_dialogue = dialogue.copy()

            # Clean session level fields
            for field in removed_fields['session']:
                cleaned_dialogue.pop(field, None)

            # Clean dialogue_turns
            cleaned_turns = []
            for turn in dialogue.get('dialogue_turns', []):
                cleaned_turn = turn.copy()
                for field in removed_fields['dialogue_turns']:
                    cleaned_turn.pop(field, None)
                cleaned_turns.append(cleaned_turn)
            cleaned_dialogue['dialogue_turns'] = cleaned_turns

            # Clean extracted_memory
            cleaned_memories = []
            for mem in dialogue.get('extracted_memory', []):
                cleaned_mem = mem.copy()
                for field in removed_fields['extracted_memory']:
                    cleaned_mem.pop(field, None)
                cleaned_memories.append(cleaned_mem)
            cleaned_dialogue['extracted_memory'] = cleaned_memories

            cleaned_dialogues.append(cleaned_dialogue)

        # Save merged dialogues
        if not output_filename:
            output_filename = f"{safe_person_name}_dialogues.json"

        output_file = person_dir / output_filename
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Do not include _metadata, directly save dialogues
        final_output = {
            "dialogues": cleaned_dialogues
        }

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(final_output, f, ensure_ascii=False, indent=2)

            print(f"Merge completed: {output_file}")
            print(f"Total sessions: {len(cleaned_dialogues)}")

            return str(output_file)

        except Exception as e:
            print(f"Error: Failed to save merged file: {e}")
            raise

    # ========== Memory point refinement functionality ==========

    def get_schedule_item_from_plan(self, session_file: Path, schedule_id: int) -> Optional[dict]:
        """
        Find plan item corresponding to schedule_id from session file

        Returns:
            Dictionary containing id, content, session fields, returns None if not found
        """
        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
        except Exception as e:
            return None

        current_plan_items = session_data.get("input_data", {}).get("current_plan_items", [])
        if not current_plan_items:
            return None

        for plan_item in current_plan_items:
            item_id = plan_item.get("id")
            if str(item_id) == str(schedule_id):
                return plan_item

        return None

    def find_session_file(self, person_dir: Path, session_identifier: str) -> Optional[Path]:
        """
        Find corresponding session file based on session_identifier
        """
        if ":" not in session_identifier:
            return None

        project_identifier, session_id = session_identifier.split(":", 1)
        session_file = person_dir / project_identifier / "dialogues" / f"{session_id}.json"

        if session_file.exists():
            return session_file
        return None

    def refine_merged_dialogues(self, dialogue_file: str, person_name: str = None,
                                output_file: str = None) -> str:
        """
        Refine memory point references in merged dialogue file

        Args:
            dialogue_file: Dialogue file path
            person_name: Person name (if None, inferred from file path)
            output_file: Output file path (optional, defaults to overwriting original file)

        Returns:
            Output file path
        """
        # Infer person_name
        if person_name is None:
            dialogue_path = Path(dialogue_file)
            person_name = dialogue_path.parent.name

        # Load memory points
        try:
            memory_points = self.load_memory_points(person_name, str(dialogue_path.parent.parent))
        except Exception as e:
            print(f"Error: Failed to load memory points: {e}")
            raise

        # Load dialogue file
        dialogue_path = Path(dialogue_file)
        if not dialogue_path.exists():
            raise FileNotFoundError(f"Dialogue file does not exist: {dialogue_file}")

        with open(dialogue_path, 'r', encoding='utf-8') as f:
            dialogue_data = json.load(f)

        dialogues = dialogue_data.get("dialogues", [])

        session_used_memories = {}
        session_uuid_map = {}
        memory_id_to_session_uuid = {}

        # First collect all session uuids and memory point mappings
        for session in dialogues:
            session_identifier = session.get("session_identifier", "")
            session_uuid = session.get("session_uuid", "")

            if not session_uuid:
                session_uuid = str(uuid.uuid4())[:8]

            session_uuid_map[session_identifier] = session_uuid

            extracted_memory = session.get("extracted_memory", [])
            for memory_point in extracted_memory:
                memory_index = memory_point.get("index", "")
                if memory_index:
                    memory_point["session_uuid"] = session_uuid
                    memory_id_to_session_uuid[memory_index] = session_uuid

        refined_dialogues = []

        global_query_id_counter = 1
        used_times = set()
        base_date = datetime(2025, 12, 15)
        last_used_date = base_date - timedelta(days=1)

        for position, session in enumerate(dialogues):
            session_identifier = session.get("session_identifier", "")
            session_uuid = session_uuid_map.get(session_identifier, "")

            # Process timestamp
            current_time = session.get("current_time", "")
            if not current_time:
                target_date = last_used_date + timedelta(days=1)
                time_str = target_date.strftime("%Y-%m-%d (%A)")

                attempts = 0
                while time_str in used_times and attempts < 365:
                    target_date += timedelta(days=1)
                    time_str = target_date.strftime("%Y-%m-%d (%A)")
                    attempts += 1

                current_time = time_str

            used_times.add(current_time)
            last_used_date = datetime.strptime(current_time, "%Y-%m-%d (%A)")

            dialogue_turns = session.get("dialogue_turns", [])

            if ":" in session_identifier:
                _, session_id = session_identifier.split(":", 1)
            else:
                session_id = session_identifier

            if session_id not in session_used_memories:
                session_used_memories[session_id] = set()

            refined_turns = []
            for turn in dialogue_turns:
                refined_turn = turn.copy()

                # Clean "Current" prefix
                turn_content = refined_turn.get("content", "")
                turn_content = re.sub(r'【Current\s+', '【', turn_content)
                refined_turn["content"] = turn_content

                # Process memory points
                if "memory_used" in refined_turn and refined_turn["memory_used"]:
                    validated_memory_used = []
                    memory_session_uuids = set()

                    for memory_usage in refined_turn["memory_used"]:
                        if isinstance(memory_usage, dict):
                            memory_id = memory_usage.get("memory_id")
                            schedule_content = memory_usage.get("content", "")

                            if self.is_schedule_memory_id(memory_id):
                                # Handle plan items
                                schedule_match = re.match(r'([^:]+:[^-]+)-schedule-(\d+)', memory_id)
                                if schedule_match:
                                    schedule_session_identifier = schedule_match.group(1)
                                    schedule_id = int(schedule_match.group(2))

                                    # Find session file
                                    session_file = self.find_session_file(dialogue_path.parent, schedule_session_identifier)
                                    if session_file:
                                        plan_item = self.get_schedule_item_from_plan(session_file, schedule_id)
                                        if plan_item:
                                            schedule_content = plan_item.get("content", "")
                                            correct_session_identifier = plan_item.get("session")
                                            if correct_session_identifier and correct_session_identifier != schedule_session_identifier:
                                                memory_id = memory_id.replace(schedule_session_identifier, correct_session_identifier, 1)
                                                schedule_session_identifier = correct_session_identifier

                                    schedule_session_uuid = session_uuid_map.get(schedule_session_identifier, session_uuid)
                                else:
                                    schedule_session_uuid = session_uuid

                                if schedule_content:
                                    memory_obj = {
                                        "memory_id": memory_id,
                                        "session_uuid": schedule_session_uuid,
                                        "content": schedule_content
                                    }
                                    memory_session_uuids.add(schedule_session_uuid)
                                    validated_memory_used.append(memory_obj)
                                continue

                            # Handle regular memory points
                            if memory_id:
                                if memory_id not in memory_points:
                                    print(f"Warning: Memory point does not exist: {memory_id}")
                                    continue

                                if memory_id in session_used_memories[session_id]:
                                    continue

                                session_used_memories[session_id].add(memory_id)

                                memory_content = memory_points[memory_id].get("content", "")
                                memory_session_uuid = memory_id_to_session_uuid.get(memory_id, session_uuid)

                                memory_session_uuids.add(memory_session_uuid)

                                memory_obj = {
                                    "memory_id": memory_id,
                                    "session_uuid": memory_session_uuid,
                                    "content": memory_content
                                }
                                validated_memory_used.append(memory_obj)
                            else:
                                continue

                        else:
                            # String format memory_usage
                            formatted_usage = self.format_memory_usage_citations(memory_usage)
                            memory_id = self.extract_memory_id_from_usage(formatted_usage)

                            # Handle plan items
                            if memory_id and self.is_schedule_memory_id(memory_id):
                                schedule_match = re.match(r'([^:]+:[^-]+)-schedule-(\d+)', memory_id)
                                schedule_content = None

                                if schedule_match:
                                    schedule_session_identifier = schedule_match.group(1)
                                    schedule_id = int(schedule_match.group(2))

                                    session_file = self.find_session_file(dialogue_path.parent, schedule_session_identifier)
                                    if session_file:
                                        plan_item = self.get_schedule_item_from_plan(session_file, schedule_id)
                                        if plan_item:
                                            schedule_content = plan_item.get("content", "")
                                            correct_session_identifier = plan_item.get("session")
                                            if correct_session_identifier and correct_session_identifier != schedule_session_identifier:
                                                memory_id = memory_id.replace(schedule_session_identifier, correct_session_identifier, 1)
                                                schedule_session_identifier = correct_session_identifier

                                    schedule_session_uuid = session_uuid_map.get(schedule_session_identifier, session_uuid)
                                else:
                                    schedule_session_uuid = session_uuid

                                if not schedule_content:
                                    # Try to extract from formatted_usage
                                    content_match = re.search(r'【' + re.escape(memory_id) + r'】([^【【]*?)(?=【|$)', formatted_usage)
                                    if content_match:
                                        schedule_content = content_match.group(1).strip()
                                        schedule_content = re.sub(r'^\[?\d+\]?\s*', '', schedule_content).strip()
                                        schedule_content = re.sub(r'\s*->\s*\[Application\]:.*', '', schedule_content).strip()

                                if schedule_content:
                                    memory_obj = {
                                        "memory_id": memory_id,
                                        "session_uuid": schedule_session_uuid,
                                        "content": schedule_content
                                    }
                                    memory_session_uuids.add(schedule_session_uuid)
                                    validated_memory_used.append(memory_obj)
                                continue

                            # Handle regular memory points
                            if memory_id:
                                if memory_id not in memory_points:
                                    print(f"Warning: Memory point does not exist: {memory_id}")
                                    continue

                                if memory_id in session_used_memories[session_id]:
                                    continue

                                session_used_memories[session_id].add(memory_id)

                                memory_content = memory_points[memory_id].get("content", "")
                                memory_session_uuid = memory_id_to_session_uuid.get(memory_id, session_uuid)

                                memory_session_uuids.add(memory_session_uuid)

                                memory_obj = {
                                    "memory_id": memory_id,
                                    "session_uuid": memory_session_uuid,
                                    "content": memory_content
                                }
                                validated_memory_used.append(memory_obj)
                            else:
                                print(f"Warning: Cannot extract ID: {memory_usage[:50]}...")
                                continue

                    if validated_memory_used:
                        refined_turn["memory_used"] = validated_memory_used
                        refined_turn["memory_session_uuids"] = list(memory_session_uuids)
                    else:
                        refined_turn.pop("memory_used", None)

                refined_turns.append(refined_turn)

            # Add query_id
            for i, turn in enumerate(refined_turns):
                if turn.get("is_query", False):
                    turn["query_id"] = f"Q-{global_query_id_counter:04d}"
                    global_query_id_counter += 1
                else:
                    if "query_id" not in turn:
                        turn["query_id"] = None

            refined_session = {
                "session_identifier": session.get("session_identifier", ""),
                "session_uuid": session_uuid,
                "current_time": current_time if current_time else session.get("current_time", ""),
                "dialogue_turns": refined_turns,
                "extracted_memory": session.get("extracted_memory", [])
            }

            refined_dialogues.append(refined_session)

        # Build output data
        output_data = {
            "_metadata": dialogue_data.get("_metadata", {}),
            "dialogues": refined_dialogues
        }

        # Update metadata
        output_data["_metadata"]["memory_refined"] = True
        output_data["_metadata"]["refine_timestamp"] = datetime.now().isoformat()

        # Determine output file
        if output_file is None:
            output_file = dialogue_file

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"Refinement completed: {output_path}")
        print(f"Processed {len(refined_dialogues)} sessions")

        return str(output_path)


def main():
    """Command line entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Dialogue Postprocessor - Full version (supports single file, batch processing, merging Person dialogues, refining memory points)")

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--merge", action="store_true",
                           help="Merge mode: Merge all dialogues of a person (requires person_name parameter)")
    mode_group.add_argument("--directory", "-d", action="store_true",
                           help="Directory mode: Batch process dialogue files in directory")
    mode_group.add_argument("--combined", "-c", action="store_true",
                           help="Combined file mode: Process already merged dialogue file")
    mode_group.add_argument("--refine", "-r", action="store_true",
                           help="Refine mode: Refine memory point references in merged dialogue file")

    # Common parameters
    parser.add_argument("input", nargs="?", help="Input file/directory path (person_name in merge mode, dialogue file path in refine mode)")
    parser.add_argument("-o", "--output", help="Output file/directory path")
    parser.add_argument("-p", "--pattern", default="*_dialogue.json",
                       help="File matching pattern (used in directory mode, default: *_dialogue.json)")
    parser.add_argument("--output-dir", default="output",
                       help="Output directory root path (used in merge/refine mode, default: output)")
    parser.add_argument("--output-filename",
                       help="Output filename (used in merge mode)")
    parser.add_argument("--person-name",
                       help="Person name (optional in refine mode, defaults to inferring from file path)")

    args = parser.parse_args()

    processor = DialoguePostprocessor()

    try:
        if args.merge:
            # Merge mode: Merge all dialogues of a person
            if not args.input:
                parser.error("merge mode requires person_name parameter")

            output_path = processor.merge_person_dialogues(
                person_name=args.input,
                output_dir=args.output_dir,
                output_filename=args.output_filename
            )
            validate_and_fix_file(Path(output_path), backup=False, dry_run=True)

        elif args.refine:
            # Refine mode: Refine memory point references in merged dialogue file
            if not args.input:
                parser.error("refine mode requires dialogue file path")

            output_path = processor.refine_merged_dialogues(
                dialogue_file=args.input,
                person_name=args.person_name,
                output_file=args.output
            )
            validate_and_fix_file(Path(output_path), backup=False, dry_run=True)

        elif args.combined:
            # Combined file mode: Process already merged dialogue file
            if not args.input:
                parser.error("combined mode requires input file path")

            result = processor.process_combined_dialogues(args.input, args.output)
            memory_count = result.get("postprocessing_stats", {}).get("total_memory_points_extracted", 0)
            print(f"Memory points extracted: {memory_count}")

            # Automatically validate output file
            output_file = args.output or args.input.replace('.json', '_processed.json')
            if Path(output_file).exists():
                validate_and_fix_file(Path(output_file), backup=False, dry_run=True)

        elif args.directory:
            # Directory mode: Batch process dialogue files in directory
            if not args.input:
                parser.error("directory mode requires input directory path")

            results = processor.process_directory(args.input, args.output, args.pattern)

            # Automatically validate all files in output directory
            output_dir = Path(args.output) if args.output else Path(args.input)
            if output_dir.exists():
                for result in results:
                    if result.get('output_file') and Path(result['output_file']).exists():
                        validate_and_fix_file(Path(result['output_file']), backup=False, dry_run=True)

        else:
            # Default mode: Process single file
            if not args.input:
                parser.error("Input file path required or use --merge/--directory/--combined/--refine mode")

            result = processor.process_dialogue_file(args.input, args.output)
            memory_count = result.get("metadata", {}).get("memory_points_extracted", 0)
            print(f"Memory points extracted: {memory_count}")

            # Automatically validate output file
            output_file = args.output or args.input
            if Path(output_file).exists():
                validate_and_fix_file(Path(output_file), backup=False, dry_run=True)

    except Exception as e:
        print(f"Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


def validate_and_fix_dialogue(dialogue: Dict[str, Any]) -> tuple[int, int]:
    """
    Validate and fix is_query flags in a single dialogue

    If is_query=true query is followed by Assistant response without memory_used, change is_query to false

    Args:
        dialogue: Dialogue data dictionary

    Returns:
        (fixed_count, total_queries): Number of fixes and total queries
    """
    fixed_count = 0
    total_queries = 0
    dialogue_turns = dialogue.get('dialogue_turns', [])

    for i, turn in enumerate(dialogue_turns):
        if turn.get('is_query', False) and turn.get('speaker') == 'User':
            total_queries += 1

            has_memory_used = False
            if i + 1 < len(dialogue_turns):
                next_turn = dialogue_turns[i + 1]
                if next_turn.get('speaker') == 'Assistant':
                    memory_used = next_turn.get('memory_used', [])
                    if memory_used and isinstance(memory_used, list) and len(memory_used) > 0:
                        has_memory_used = True

            if not has_memory_used:
                turn['is_query'] = False
                fixed_count += 1

    return fixed_count, total_queries


def validate_and_fix_file(file_path: Path, backup: bool = True, dry_run: bool = False) -> Dict[str, Any]:
    """
    Validate and fix is_query flags in a single data file

    Args:
        file_path: File path
        backup: Whether to backup original file
        dry_run: Whether to only check without modifying

    Returns:
        Processing result statistics
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"  Error: Failed to read file: {e}")
        return {
            'file': str(file_path),
            'status': 'error',
            'error': str(e),
            'fixed_count': 0,
            'total_queries': 0,
            'total_dialogues': 0
        }

    if not isinstance(data, dict) or 'dialogues' not in data:
        print(f"  Error: Unsupported data format")
        return {
            'file': str(file_path),
            'status': 'error',
            'error': 'Unsupported data format',
            'fixed_count': 0,
            'total_queries': 0,
            'total_dialogues': 0
        }

    dialogues = data.get('dialogues', [])
    total_dialogues = len(dialogues)

    total_fixed = 0
    total_queries = 0
    fixed_dialogues = []

    for idx, dialogue in enumerate(dialogues):
        fixed_count, query_count = validate_and_fix_dialogue(dialogue)
        total_fixed += fixed_count
        total_queries += query_count

        if fixed_count > 0:
            fixed_dialogues.append({
                'index': idx,
                'session_identifier': dialogue.get('session_identifier', f'dialogue_{idx}'),
                'fixed_count': fixed_count,
                'query_count': query_count
            })

    if total_fixed > 0:
        print(f"Validation found {total_fixed} issues in {file_path.name}")
        if fixed_dialogues:
            print(f"  Affected dialogues: {len(fixed_dialogues)}")
            for fd in fixed_dialogues[:2]:
                print(f"    - {fd['session_identifier']}: Fixed {fd['fixed_count']}/{fd['query_count']} queries")
            if len(fixed_dialogues) > 2:
                print(f"    ... and {len(fixed_dialogues) - 2} more")

    if not dry_run and total_fixed > 0:
        if backup:
            from datetime import datetime
            import shutil
            backup_path = file_path.with_suffix(f'.json.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
            try:
                shutil.copy2(file_path, backup_path)
            except Exception as e:
                print(f"Warning: Failed to backup file: {e}")

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error: Failed to save file: {e}")
            return {
                'file': str(file_path),
                'status': 'error',
                'error': f'Save failed: {e}',
                'fixed_count': total_fixed,
                'total_queries': total_queries,
                'total_dialogues': total_dialogues
            }

    return {
        'file': str(file_path),
        'status': 'success' if total_fixed == 0 or not dry_run else 'dry_run',
        'fixed_count': total_fixed,
        'total_queries': total_queries,
        'total_dialogues': total_dialogues,
        'fixed_dialogues': fixed_dialogues
    }


def run_validation_mode(args):
    """
    Run validation and fix mode
    """
    data_dir = Path(args.input)

    if not data_dir.exists():
        print(f"Error: Directory does not exist: {args.input}")
        return 1

    json_files = list(data_dir.glob('*.json'))
    json_files.sort()

    if not json_files:
        print(f"Error: No JSON files found: {args.input}")
        return 1

    print(f"Found {len(json_files)} files")
    print(f"Mode: {'Preview (will not modify)' if args.dry_run else 'Modify files'}")

    results = []
    for file_path in json_files:
        result = validate_and_fix_file(file_path, backup=not args.no_backup, dry_run=args.dry_run)
        results.append(result)

    print(f"\nProcessing completed - Summary:")
    total_files = len(results)
    total_fixed = sum(r.get('fixed_count', 0) for r in results)
    total_queries = sum(r.get('total_queries', 0) for r in results)

    print(f"Files processed: {total_files}")
    if total_fixed > 0:
        print(f"Total fixes: {total_fixed} out of {total_queries} queries")

        files_with_fixes = [r for r in results if r.get('fixed_count', 0) > 0]
        print(f"Files with fixes: {len(files_with_fixes)}")
        for r in files_with_fixes[:5]:
            print(f"  - {Path(r['file']).name}: Fixed {r['fixed_count']} queries")
        if len(files_with_fixes) > 5:
            print(f"  ... and {len(files_with_fixes) - 5} more")
    else:
        print("All files are normal, no fixes needed")

    return 0


if __name__ == "__main__":
    exit(main())
