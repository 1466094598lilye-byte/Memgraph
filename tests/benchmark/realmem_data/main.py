#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from utils.llm_client import create_client
from pipeline.project_outline_processor import ProjectOutlineProcessor
from pipeline.event_processor import EventProcessor
from pipeline.summary_processor import SummaryProcessor
from pipeline.multi_agent_dialogue_processor import MultiAgentDialogueProcessor
from dotenv import load_dotenv
import time
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from tqdm import tqdm
from pypinyin import lazy_pinyin, Style
from datetime import datetime, timedelta
import random

import os

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
base_url = os.getenv('BASE_URL')


# Global variables to store persona and topic data
PERSONA_DATA = None
TOPIC_ATTR_DATA = None
PERSON_GOAL_DATA = None

# Global person-based memory management
# Each person gets their own independent memory space
PERSON_MEMORY_DATA = {}  # Format: {person_name: {"memory_points": [], "total_sessions": 0, "last_updated": timestamp}}

# Global person-based ConversationController management
# Each person gets their own ConversationController instance
PERSON_CONVERSATION_CONTROLLERS = {}  # Format: {person_name: ConversationController}

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Project blueprint generator")

    parser.add_argument("--names", type=str, default=None,
                       help="Specify person name to generate dialogue for, e.g.: --names 'Zhang San'")
    parser.add_argument("--projects", "-n", type=int, default=3,
                       help="Number of project blueprints to generate for each person (default: 1)")

    # Model parameters
    parser.add_argument("--blueprint-model", type=str, default="gemini-2.5-pro",
                       help="LLM model for project blueprint generation (default: gemini-2.5-pro)")
    parser.add_argument("--event-model", type=str, default="gemini-2.5-pro",
                       help="LLM model for event generation (default: gemini-2.5-pro)")
    parser.add_argument("--summary-model", type=str, default="gemini-2.5-pro",
                       help="LLM model for session summary generation (default: gemini-2.5-pro)")
    parser.add_argument("--dialogue-model", type=str, default="gemini-2.5-flash",
                       help="LLM model for dialogue generation (default: gemini-2.5-flash)")
    parser.add_argument("--evaluation-model", type=str, default="gemini-2.5-flash-lite",
                       help="LLM model for evaluation (default: gemini-2.5-flash-lite)")
    parser.add_argument("--memory-model", type=str, default="gemini-2.5-flash",
                       help="LLM model for memory management (default: gemini-2.5-flash)")
    parser.add_argument("--memory-retrieve-model", type=str, default="gemini-2.5-flash",
                       help="LLM model for memory retrieval (default: gemini-2.5-flash)")
    parser.add_argument("--dedup-model", type=str, default="gemini-2.5-flash",
                       help="LLM model for deduplication (default: gemini-2.5-flash)")
    parser.add_argument("--semantic-schedule-model", type=str, default="gemini-2.5-pro",
                       help="LLM model for semantic schedule processing (default: gpt-4o-mini)")

    # Processing parameters
    parser.add_argument("--max-turns", type=int, default=24,
                       help="Maximum dialogue turns (default: 10)")
    parser.add_argument("--max-retries", type=int, default=2,
                       help="Maximum LLM call retry count (default: 2)")

    # Output parameters
    parser.add_argument("--output", "-o", type=str, default="output",
                       help="Output directory path (default: output)")
    parser.add_argument("--log", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--smart-recovery", action="store_true",
                       help="Enable smart interrupt recovery")

    return parser.parse_args()

# Progress check and recovery functions
def validate_blueprint(blueprint_data: Dict[str, Any]) -> bool:
    """Validate blueprint data integrity"""
    required_fields = ["_metadata", "project_goal", "project_attributes_schema"]

    for field in required_fields:
        if field not in blueprint_data:
            return False

    # Validate metadata integrity
    metadata = blueprint_data.get("_metadata", {})
    if not metadata.get("person_name") or not metadata.get("total_attributes_used"):
        return False

    return True


def validate_events(events_data: Dict[str, Any]) -> bool:
    """Validate events data integrity"""
    if "events" not in events_data:
        return False

    events = events_data["events"]
    if not isinstance(events, list) or len(events) == 0:
        return False

    # Validate required fields for each event
    for event in events:
        if not isinstance(event, dict):
            return False
        if not event.get("event_name"):
            return False

    return True


def validate_summaries(summaries_data: Dict[str, Any], events_data: Dict[str, Any] = None) -> bool:
    """Validate session summaries data integrity"""
    if "sessions" not in summaries_data:
        return False

    sessions = summaries_data["sessions"]
    if not isinstance(sessions, list) or len(sessions) == 0:
        return False

    # Validate required fields for each session
    for session in sessions:
        if not isinstance(session, dict):
            return False
        if not session.get("session_id") or not session.get("session_summary"):
            return False

    # If events data provided, check if all events have corresponding session summaries
    if events_data:
        try:
            # fromeventsdataingetalleventID（Translated commentevent_indexandevent_idTranslated commentfieldTranslated comment）
            event_ids = set()
            if "events" in events_data:
                events_list = events_data["events"]
                for event in events_list:
                    if isinstance(event, dict):
                        # Translated commentuseevent_id，Translated commenthaveTranslated commentuseevent_index
                        event_id = event.get("event_id") or event.get("event_index")
                        if event_id:
                            event_ids.add(event_id)

            # fromsummariesdataingetTranslated commenteventID
            covered_event_ids = set()
            for session in sessions:
                if isinstance(session, dict) and session.get("event_id"):
                    covered_event_ids.add(session["event_id"])

            # checkisTranslated commentalleventTranslated commenthaveTranslated commentsession summary
            if event_ids != covered_event_ids:
                print(f"    ⚠️ event: eventID {event_ids} vs  {covered_event_ids}")
                return False

        except Exception as e:
            print(f"    ⚠️ validationevent: {str(e)}")
            return False

    return True


def validate_dialogue(dialogue_file: Path) -> bool:
    """Translated docstring"""
    try:
        with open(dialogue_file, 'r', encoding='utf-8') as f:
            dialogue_data = json.load(f)

        # checkTranslated commentneedfield
        if not dialogue_data.get("session_id"):
            return False

        # Translated commentdatastructure：Translated commentlayerdialogue_turnsorTranslated commentindialogue_outputin
        dialogue_turns = dialogue_data.get("dialogue_turns", [])

        # Translated commentlayerTranslated commenthavedialogue_turns，attemptindialogue_outputinTranslated comment
        if not dialogue_turns and "dialogue_output" in dialogue_data:
            dialogue_output = dialogue_data["dialogue_output"]
            dialogue_turns = dialogue_output.get("dialogue_turns", [])

        if not isinstance(dialogue_turns, list):
            return False

        # checkTranslated commenthaveTranslated commentdialogue
        if len(dialogue_turns) == 0:
            return False

        # checkeachTranslated commentdialogueTranslated commentneedfield
        for turn in dialogue_turns:
            if not isinstance(turn, dict):
                return False
            if not turn.get("speaker") or not turn.get("content"):
                return False

        return True

    except Exception:
        return False


def check_dialogue_sequence_integrity(dialogues_dir: Path, expected_sessions: List[str]) -> Dict[str, Any]:
    """
    checkdialoguecolumn，check

    Args:
        dialogues_dir: dialoguefiledirectory
        expected_sessions: session IDlist（）

    Returns:
        {
            "is_sequence_complete": bool,
            "last_complete_session": str or None,
            "sessions_to_regenerate": List[str],
            "intact_sessions": List[str]
        }
    """
    intact_sessions = []
    first_problem_index = None

    # Translated commentcheckeachitemssession
    for i, session_id in enumerate(expected_sessions):
        session_file = dialogues_dir / f"{session_id}.json"

        if session_file.exists() and validate_dialogue(session_file):
            intact_sessions.append(session_id)
        else:
            if first_problem_index is None:
                first_problem_index = i
            break  # Translated commentitemsTranslated comment，stopcheck

    # Translated commentneedwantTranslated commentnewgeneratesessions（fromTranslated commentitemsTranslated commentbeginallafterTranslated commentsessions）
    if first_problem_index is not None:
        sessions_to_regenerate = expected_sessions[first_problem_index:]
        last_complete_session = intact_sessions[-1] if intact_sessions else None
    else:
        sessions_to_regenerate = []
        last_complete_session = intact_sessions[-1] if intact_sessions else None

    return {
        "is_sequence_complete": len(sessions_to_regenerate) == 0,
        "last_complete_session": last_complete_session,
        "sessions_to_regenerate": sessions_to_regenerate,
        "intact_sessions": intact_sessions,
        "first_problem_index": first_problem_index
    }


def validate_dialogues_completeness(dialogues_dir: Path, summaries_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    validationallsessiondialogueisgenerate，columncheck

    Args:
        dialogues_dir: dialoguefiledirectory
        summaries_data: session summariesdata

    Returns:
        {
            "is_complete": bool,
            "expected_sessions": int,
            "completed_sessions": int,
            "missing_sessions": List[str],
            "existing_session_ids": List[str],
            "sequence_integrity": Dict  # newTranslated commentcolumnTranslated commentinformation
        }
    """
    if not summaries_data or "sessions" not in summaries_data:
        return {
            "is_complete": False,
            "expected_sessions": 0,
            "completed_sessions": 0,
            "missing_sessions": [],
            "existing_session_ids": [],
            "sequence_integrity": {
                "is_sequence_complete": False,
                "sessions_to_regenerate": [],
                "last_complete_session": None,
                "intact_sessions": [],
                "first_problem_index": None
            }
        }

    sessions = summaries_data["sessions"]
    # Translated commentgetsession IDs
    expected_sessions = [session.get("session_id") for session in sessions if isinstance(session, dict) and session.get("session_id")]
    expected_sessions_count = len(expected_sessions)

    # checkdialoguesdirectoryisTranslated commentin
    if not dialogues_dir.exists():
        return {
            "is_complete": False,
            "expected_sessions": expected_sessions_count,
            "completed_sessions": 0,
            "missing_sessions": expected_sessions,
            "existing_session_ids": [],
            "sequence_integrity": {
                "is_sequence_complete": False,
                "sessions_to_regenerate": expected_sessions,
                "last_complete_session": None,
                "intact_sessions": [],
                "first_problem_index": 0
            }
        }

    # usenewTranslated commentcolumnTranslated commentcheck
    sequence_integrity = check_dialogue_sequence_integrity(dialogues_dir, expected_sessions)

    # getallTranslated commentindialoguefile
    dialogue_files = list(dialogues_dir.glob("*.json"))
    existing_session_ids = set()

    # validationeachitemsdialoguefileTranslated comment
    valid_dialogue_files = []
    for dialogue_file in dialogue_files:
        if validate_dialogue(dialogue_file):
            try:
                with open(dialogue_file, 'r', encoding='utf-8') as f:
                    dialogue_data = json.load(f)
                    session_id = dialogue_data.get("session_id")
                    if session_id:
                        existing_session_ids.add(session_id)
                        valid_dialogue_files.append(dialogue_file)
            except Exception:
                # fileTranslated comment，Translated comment
                continue

    completed_sessions = len(valid_dialogue_files)
    missing_sessions = set(expected_sessions) - existing_session_ids

    return {
        "is_complete": sequence_integrity["is_sequence_complete"] and completed_sessions == expected_sessions_count,
        "expected_sessions": expected_sessions_count,
        "completed_sessions": completed_sessions,
        "missing_sessions": sorted(list(missing_sessions)),
        "existing_session_ids": sorted(list(existing_session_ids)),
        "valid_dialogue_files": [f.name for f in sorted(valid_dialogue_files)],
        "sequence_integrity": sequence_integrity
    }


def get_topic_name_by_id(topic_id: str) -> str:
    """Translated docstring"""
    topics = TOPIC_ATTR_DATA.get('topics', [])
    for topic in topics:
        if topic.get('topic_id') == topic_id:
            return topic.get('topic_name', f'topic_{topic_id}')
    return f'topic_{topic_id}'


def check_project_progress(safe_person_name: str, project_identifier: str) -> Dict[str, Any]:
    """Translated docstring"""
    base_dir = Path(f"output/{safe_person_name}/{project_identifier}")

    # 1. checkblueprint
    blueprint_file = base_dir / "project_blueprints" / f"{project_identifier}_blueprint.json"
    if not blueprint_file.exists():
        return {"status": "need_blueprint", "stage": "blueprint", "progress": 0}

    try:
        with open(blueprint_file, 'r', encoding='utf-8') as f:
            blueprint = json.load(f)
        # validationblueprintTranslated comment
        if not validate_blueprint(blueprint):
            return {"status": "blueprint_incomplete", "stage": "blueprint", "progress": 0}
    except Exception as e:
        return {"status": "blueprint_corrupted", "stage": "blueprint", "progress": 0, "error": str(e)}

    # 2. checkevents
    events_file = base_dir / "project_events" / f"{project_identifier}_events.json"
    if not events_file.exists():
        return {"status": "need_events", "stage": "events", "progress": 25}

    try:
        with open(events_file, 'r', encoding='utf-8') as f:
            events_data = json.load(f)
        if not validate_events(events_data):
            return {"status": "events_incomplete", "stage": "events", "progress": 25}
    except Exception as e:
        return {"status": "events_corrupted", "stage": "events", "progress": 25, "error": str(e)}

    # 3. checksession summaries
    summaries_file = base_dir / "session_summaries" / f"{project_identifier}_summary.json"
    if not summaries_file.exists():
        return {"status": "need_summaries", "stage": "summaries", "progress": 50}

    try:
        with open(summaries_file, 'r', encoding='utf-8') as f:
            summaries_data = json.load(f)

        if not validate_summaries(summaries_data, events_data):
            return {"status": "summaries_incomplete", "stage": "summaries", "progress": 50}
    except Exception as e:
        return {"status": "summaries_corrupted", "stage": "summaries", "progress": 50, "error": str(e)}

    # 4. checkdialogues（Translated commentcolumnTranslated commentcheck）
    dialogues_dir = base_dir / "dialogues"

    # usenewdialogueTranslated commentvalidationfunction
    dialogue_completeness = validate_dialogues_completeness(dialogues_dir, summaries_data)

    if dialogue_completeness["expected_sessions"] == 0:
        return {
            "status": "need_summaries",
            "stage": "summaries",
            "progress": 50,
            "error": "No sessions found in summaries data"
        }

    # useTranslated commentcolumnTranslated commentcheckresult
    sequence_integrity = dialogue_completeness["sequence_integrity"]

    if not sequence_integrity["is_sequence_complete"]:
        sessions_to_regenerate = sequence_integrity["sessions_to_regenerate"]
        last_complete_session = sequence_integrity["last_complete_session"]
        first_session_to_regenerate = sessions_to_regenerate[0] if sessions_to_regenerate else None

        return {
            "status": "dialogues_incomplete",
            "stage": "dialogues",
            "progress": 75,
            "start_from_session": first_session_to_regenerate,
            "sessions_to_regenerate": sessions_to_regenerate,
            "last_complete_session": last_complete_session,
            "total_sessions": dialogue_completeness["expected_sessions"],
            "intact_sessions": sequence_integrity["intact_sessions"],
            "existing_session_ids": dialogue_completeness["existing_session_ids"],
            "regeneration_reason": f"Sequence broken at {first_session_to_regenerate}"
        }

    # Translated commentcolumnTranslated comment（Translated commentonTranslated commentwillTranslated comment，Translated commentforTranslated commentcolumnTranslated comment）
    if not dialogue_completeness["is_complete"]:
        return {
            "status": "dialogue_corrupted",
            "stage": "dialogues",
            "progress": 75,
            "valid_files": dialogue_completeness["valid_dialogue_files"],
            "expected_files_count": dialogue_completeness["expected_sessions"]
        }

    # 5. checkproject memory
    project_memory_file = base_dir / "project_memories" / f"{project_identifier}_memory.json"
    if not project_memory_file.exists():
        return {"status": "need_project_memory", "stage": "memory", "progress": 95}

    # projectcomplete
    return {
        "status": "completed",
        "stage": "completed",
        "progress": 100,
        "total_sessions": dialogue_completeness["expected_sessions"],
        "sequence_integrity": sequence_integrity
    }


def convert_to_safe_filename(text: str) -> str:
    """
    willtextforfileformat
    infor，forunder

    Args:
        text: originaltext

    Returns:
        filestring
    """
    if not text:
        return "unknown"

    # willinTranslated commentforTranslated comment，Translated commenthaveTranslated comment
    pinyin_list = lazy_pinyin(text, style=Style.NORMAL, neutral_tone_with_five=True)
    pinyin_text = "".join(pinyin_list)

    # Translated commentnon-Translated comment，Translated commentforunderTranslated comment
    safe_name = "".join([c if c.isalnum() else "_" for c in pinyin_text])

    # removeTranslated commentunderTranslated comment
    while "__" in safe_name:
        safe_name = safe_name.replace("__", "_")

    # removeTranslated commentunderTranslated comment
    safe_name = safe_name.strip("_")

    return safe_name or "unknown"

def read_persona_topic_files(persona_file="dataset/all_persona_topic/persona_all.json",
                             topic_attr_file="dataset/all_persona_topic/topic&goal&attr.json",
                             person_goal_file="dataset/all_persona_topic/person&goal.json"):
    """Read and properly parse the persona and topic/goal/attribute JSON files."""
    global PERSONA_DATA, TOPIC_ATTR_DATA, PERSON_GOAL_DATA

    # If data already loaded, return cached values
    if PERSONA_DATA is not None and TOPIC_ATTR_DATA is not None and PERSON_GOAL_DATA is not None:
        return

    try:
        # Read and parse persona data (array of persona objects)
        with open(persona_file, 'r', encoding='utf-8') as f:
            PERSONA_DATA = json.load(f)

        # Read and parse topic/goal/attribute data (contains topics array)
        with open(topic_attr_file, 'r', encoding='utf-8') as f:
            TOPIC_ATTR_DATA = json.load(f)
        
        # Read and parse person/goal data (contains person array)
        with open(person_goal_file, 'r', encoding='utf-8') as f:
            PERSON_GOAL_DATA = json.load(f)

        # Validate structure
        if not isinstance(PERSONA_DATA, list):
            raise ValueError("persona file should contain an array of persona objects")

        if not isinstance(TOPIC_ATTR_DATA, dict) or "topics" not in TOPIC_ATTR_DATA:
            raise ValueError("topic&attr file should contain a 'topics' key")

        if not isinstance(PERSON_GOAL_DATA, list):
            raise ValueError("person&goal file should contain an array of person/goal pairs")

        print(f"✅ Loaded {len(PERSONA_DATA)} personas")
        print(f"✅ Loaded {len(TOPIC_ATTR_DATA['topics'])} topics")
        print(f"✅ Loaded {len(PERSON_GOAL_DATA)} person/goal pairs")

    except FileNotFoundError as e:
        print(f"❌ Error: File not found - {e}")
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON format - {e}")
    except ValueError as e:
        print(f"❌ Error: Invalid data structure - {e}")

    return


def get_person_memory(person_name: str) -> Dict[str, Any]:
    """Translated docstring"""
    global PERSON_MEMORY_DATA

    if person_name not in PERSON_MEMORY_DATA:
        # attemptfromnewpathloadTranslated commenthavememoryfile
        safe_person_name = convert_to_safe_filename(person_name)
        memory_file_path = Path(f"output/{safe_person_name}/{safe_person_name}_memory.json")

        if memory_file_path.exists():
            try:
                with open(memory_file_path, 'r', encoding='utf-8') as f:
                    loaded_memory = json.load(f)
                PERSON_MEMORY_DATA[person_name] = loaded_memory
                print(f"✅ Loaded existing memory for {person_name}: {memory_file_path}")
            except Exception as e:
                print(f"⚠️ Failed to load memory file {memory_file_path}: {str(e)}")
                # initializenewpersonmemoryTranslated commentbetween
                PERSON_MEMORY_DATA[person_name] = {
                    "memory_points": [],
                    "total_sessions": 0,
                    "last_updated": time.time(),
                    "metadata": {
                        "person_name": person_name,
                        "created_at": time.time(),
                        "last_session_id": None
                    }
                }
        else:
            # initializenewpersonmemoryTranslated commentbetween
            PERSON_MEMORY_DATA[person_name] = {
                "memory_points": [],
                "total_sessions": 0,
                "last_updated": time.time(),
                "metadata": {
                    "person_name": person_name,
                    "created_at": time.time(),
                    "last_session_id": None
                }
            }

    return PERSON_MEMORY_DATA[person_name]


def save_project_memory(person_name: str, project_identifier: str, memory_data: Dict[str, Any]) -> str:
    """
    saveprojectmemorydatatofile

    Args:
        person_name: personname
        project_identifier: projectID
        memory_data: memorydata

    Returns:
        savefilepath
    """
    safe_person_name = convert_to_safe_filename(person_name)

    # createprojectmemorydirectory
    project_memory_dir = Path(f"output/{safe_person_name}/{project_identifier}/project_memories")
    project_memory_dir.mkdir(parents=True, exist_ok=True)

    # generateprojectmemoryfile
    filename = f"{project_identifier}_memory.json"
    filepath = project_memory_dir / filename

    # saveprojectmemorydata
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(memory_data, f, ensure_ascii=False, indent=2)

    print(f"💾 Saved project memory for {project_identifier}: {filepath}")
    return str(filepath)


def save_person_memory(person_name: str, output_dir: str = None) -> str:
    """
    savepersonmemorydatatofile

    Args:
        person_name: personname
        output_dir: outputdirectory

    Returns:
        savefilepath
    """
    global PERSON_MEMORY_DATA

    if person_name not in PERSON_MEMORY_DATA:
        return None

    # Translated commenthaveTranslated commentoutputdirectory，usenewTranslated commentstructure
    if output_dir is None:
        safe_person_name = convert_to_safe_filename(person_name)
        output_dir = f"output/{safe_person_name}"

    # createoutputdirectory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # generatefileTranslated comment
    safe_person_name = convert_to_safe_filename(person_name)
    filename = f"{safe_person_name}_memory.json"
    filepath = output_path / filename

    # savememorydata
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(PERSON_MEMORY_DATA[person_name], f, ensure_ascii=False, indent=2)

    print(f"💾 Saved memory for {person_name}: {filepath}")
    return str(filepath)


def get_person_schedule(person_name: str, project_identifier: str = None) -> Dict[str, Any]:
    """Translated docstring"""
    safe_person_name = convert_to_safe_filename(person_name)
    schedule_file_path = Path(f"output/{safe_person_name}/schedule.json")

    # Translated commentdirectoryTranslated commentin
    schedule_file_path.parent.mkdir(parents=True, exist_ok=True)

    if schedule_file_path.exists():
        try:
            with open(schedule_file_path, 'r', encoding='utf-8') as f:
                schedule_data = json.load(f)
            print(f"✅ Loaded existing schedule for {person_name}/{project_identifier}: {schedule_file_path}")
            return schedule_data
        except Exception as e:
            print(f"⚠️ Failed to load schedule file {schedule_file_path}: {str(e)}")

    # returnTranslated commentstructure
    return {
        "plan_items": [],  # Translated commentprojectlist
        "current_date": "",  # currentdate
        "metadata": {
            "created_time": time.time(),
            "person_name": person_name,
            "max_plan_id": 0  # Translated commentIDfor0
        }
    }


def extract_plan_items_content(schedule_data: Dict[str, Any]) -> List[str]:
    """
    fromscheduledatainprojectcontentlist
    Args:
        schedule_data: includesprojectdata
    Returns:
        projectcontentstringlist
    """
    plan_items = schedule_data.get("plan_items", [])
    if not plan_items:
        return []

    # Translated commentisstringlist，Translated commentreturn
    if isinstance(plan_items[0], str):
        return plan_items

    # Translated commentisobjectlist，Translated commentcontentfield
    if isinstance(plan_items[0], dict):
        return [item.get("content", "") for item in plan_items if item.get("content")]

    return []

def get_plan_items_full(schedule_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    fromscheduledataingetprojectobjectlist
    Args:
        schedule_data: includesprojectdata
    Returns:
        projectobjectlist
    """
    plan_items = schedule_data.get("plan_items", [])
    if not plan_items:
        return []

    # Translated commentisstringlist，needwantTranslated commentforTranslated commentstructure（Translated commentafterTranslated comment）
    if isinstance(plan_items[0], str):
        return [{"content": content} for content in plan_items]

    # Translated commentisobjectlist，Translated commentreturn
    if isinstance(plan_items[0], dict):
        return plan_items

    return []

def process_plan_items_with_metadata(plan_items: List[Dict[str, Any]], project_identifier: str, session_id: str, existing_schedule_data: Dict[str, Any] = None) -> Tuple[List[Dict[str, Any]], int]:
    """
    forprojectaddsessionandidfield
    Args:
        plan_items: projectlist
        project_identifier: projectID
        session_id: sessionID
        existing_schedule_data: havedata，getID
    Returns:
        tuple: (includesmetadataprojectlist, newID)
    """
    if not plan_items:
        return [], 0

    # getTranslated commenthaveTranslated commentID：Translated commentfrommetadatainget，itstimesfromTranslated commenthaveprojectincalculate
    max_id = 0
    if existing_schedule_data:
        # Translated commentattemptfrommetadataingetrecordTranslated commentID
        metadata = existing_schedule_data.get("metadata", {})
        if "max_plan_id" in metadata:
            try:
                max_id = int(metadata["max_plan_id"])
            except (ValueError, TypeError):
                pass
        else:
            # Translated commentmetadatainTranslated commenthave，Translated commentfromTranslated commenthaveprojectincalculate
            if "plan_items" in existing_schedule_data:
                existing_items = existing_schedule_data["plan_items"]
                if existing_items and isinstance(existing_items[0], dict):
                    for item in existing_items:
                        if isinstance(item, dict) and "id" in item:
                            try:
                                max_id = max(max_id, int(item["id"]))
                            except (ValueError, TypeError):
                                pass

    processed_items = []
    for item in plan_items:
        if isinstance(item, dict):
            # Translated commentisTranslated commentstructure，checkisTranslated commentneedwantaddfield
            if "id" not in item:
                max_id += 1
                item["id"] = str(max_id)
                item["session"] = f"{project_identifier}:{session_id}"
                item["created_time"] = time.time()
            else:
                # Translated commenthaveid，updatemax_id
                try:
                    max_id = max(max_id, int(item["id"]))
                except (ValueError, TypeError):
                    pass
            processed_items.append(item)
        else:
            # Translated commentisstring，createTranslated commentstructure
            max_id += 1
            processed_items.append({
                "id": str(max_id),
                "content": item,
                "session": f"{project_identifier}:{session_id}",
                "created_time": time.time()
            })

    return processed_items, max_id

def save_person_schedule(person_name: str, schedule_data: Dict[str, Any], project_identifier: str = None, session_id: str = None) -> str:
    """Translated docstring"""
    safe_person_name = convert_to_safe_filename(person_name)
    schedule_file_path = Path(f"output/{safe_person_name}/schedule.json")

    # Translated commentdirectoryTranslated commentin
    schedule_file_path.parent.mkdir(parents=True, exist_ok=True)

    # initializevariable
    new_max_id = None

    # processTranslated commentproject，addmetadata
    if "plan_items" in schedule_data and project_identifier and session_id:
        # readTranslated commenthavedataTranslated commentgetTranslated commentID
        existing_schedule_data = {}
        if schedule_file_path.exists():
            try:
                with open(schedule_file_path, 'r', encoding='utf-8') as f:
                    existing_schedule_data = json.load(f)
            except Exception:
                pass

        # Translated commentprojectforstructureTranslated commentformat，getnewTranslated commentID
        plan_items = schedule_data["plan_items"]
        processed_items, new_max_id = process_plan_items_with_metadata(
            plan_items, project_identifier, session_id, existing_schedule_data
        )
        schedule_data["plan_items"] = processed_items

    # updateTranslated commentdata，includingTranslated commentID
    if "metadata" not in schedule_data:
        schedule_data["metadata"] = {}

    metadata_updates = {
        "updated_time": time.time(),
        "person_name": person_name
    }

    # savenewTranslated commentIDtometadatain
    if new_max_id is not None:
        metadata_updates["max_plan_id"] = new_max_id

    schedule_data["metadata"].update(metadata_updates)

    try:
        with open(schedule_file_path, 'w', encoding='utf-8') as f:
            json.dump(schedule_data, f, ensure_ascii=False, indent=2)
        print(f"💾 Schedule saved: {schedule_file_path}")
        return str(schedule_file_path)
    except Exception as e:
        print(f"❌ Failed to save schedule file {schedule_file_path}: {str(e)}")
        return ""


def generate_project_blueprint(person_data: Dict[str, Any], project_attributes: List[str], project_goal: str = None, model: str = None, topic_name: str = None, selected_task_id: str = None) -> Optional[Dict[str, Any]]:
    """
    Generate project blueprint for a specific person using ProjectOutlineProcessor

    Args:
        person_data: Persona data object
        project_attributes: List of project attributes from the selected topic
        project_goal: Specific goal/task description (optional)
        model: LLM model name
        topic_name: Selected topic name (optional)
        selected_task_id: Selected task ID (optional)

    Returns:
        Project blueprint data or None if failed
    """
    person_name = person_data.get('name', 'unknown_person')

    try:
        # Use provided model or default
        if model is None:
            model = "gemini-2.5-pro-thinking-*"

        # Initialize LLM client
        llm_client = create_client(api_key=api_key, base_url=base_url, model=model)
        print("✅ LLM client initialized")

        # Initialize ProjectOutlineProcessor
        processor = ProjectOutlineProcessor(
            llm_client=llm_client,
            checkpoint_dir="output/checkpoints/project_outline"
        )
        print("✅ ProjectOutlineProcessor initialized")

        # Prepare input data for processor
        processor_input = {
            "persona": person_data,
            "project_attributes": project_attributes
        }
        
        # Add project_goal if provided
        if project_goal:
            processor_input["primary_goal"] = project_goal

        # Process the person data
        result = processor.process(
            data=processor_input,
            use_checkpoint=False
        )

        if result.success:
            print("✅ Project blueprint generated successfully!")
            # Add metadata to the result
            blueprint = result.data
            blueprint["_metadata"] = {
                "person_name": person_name,
                "persona_role": person_data.get('role', 'Unknown'),
                "total_attributes_used": len(project_attributes)
            }

            # Add topic and task metadata if provided
            if topic_name:
                blueprint["selected_topic"] = topic_name
            if project_goal:
                blueprint["selected_goal"] = project_goal
            if selected_task_id:
                blueprint["selected_task_id"] = selected_task_id

            # addproject_attributes_schemafield - Translated commentvalidationfailedTranslated comment
            if project_attributes:
                blueprint["project_attributes_schema"] = project_attributes

            # Generate project identifier for directory structure
            safe_topic_name = convert_to_safe_filename(topic_name) if topic_name else "unknown_topic"
            safe_person_name = convert_to_safe_filename(person_name)

            # Debug: checkparametervalue
            print(f"    Debug generate_project_blueprint:")
            print(f"      topic_name: '{topic_name}'")
            print(f"      selected_task_id: '{selected_task_id}' (type: {type(selected_task_id)})")
            print(f"      safe_topic_name: '{safe_topic_name}'")

            project_identifier = f"{safe_topic_name}_{selected_task_id or 'unknown'}"
            print(f"      project_identifier: '{project_identifier}'")

            # Save blueprint to new directory structure
            output_dir = Path(f"output/{safe_person_name}/{project_identifier}/project_blueprints")
            output_dir.mkdir(parents=True, exist_ok=True)

            filename = f"{project_identifier}_blueprint.json"
            filepath = output_dir / filename

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(blueprint, f, ensure_ascii=False, indent=2)

            print(f"✅ Blueprint saved to: {filepath}")
            return blueprint
        else:
            print(f"❌ Failed to generate project blueprint: {result.error_message}")
            return None

    except Exception as e:
        print(f"❌ Error generating project blueprint: {str(e)}")
        return None


def generate_session_summaries(person_data: Dict[str, Any], blueprint: Dict[str, Any], events_data: Dict[str, Any], model: str = None) -> Optional[Dict[str, Any]]:
    """
    forprojecteventGenerate session summaries

    Args:
        person_data: roledata
        blueprint: projectblueprint
        events_data: eventdata
        model: LLMname

    Returns:
        summarydataorNone
    """
    person_name = person_data.get('name', 'unknown_person')

    # eventdatacancaninTranslated commentfieldin
    if 'events' in events_data:
        events_list = events_data['events']
    elif 'output_data' in events_data:
        if isinstance(events_data['output_data'], list):
            events_list = events_data['output_data']
        elif isinstance(events_data['output_data'], dict):
            events_list = events_data['output_data'].get('events', [])
        else:
            events_list = []
    else:
        events_list = []

    if len(events_list) == 0:
        print(f"    ⚠️ havetoeventdata，summarygenerate")
        return None

    try:
        # Use provided model or default
        if model is None:
            model = "gemini-2.5-pro-thinking-*"

        # initializeLLMTranslated comment
        llm_client = create_client(api_key=api_key, base_url=base_url, model=model)
        print("    ✅ Summary LLM client initialized")

        # initializeSummaryProcessor
        summary_processor = SummaryProcessor(
            llm_client=llm_client,
            checkpoint_dir="output/checkpoints/summaries"
        )
        print("    ✅ SummaryProcessor initialized")

        all_summaries = []

        # foreacheventgeneratesummary
        for i, target_event in enumerate(events_list):
            print(f"    Processing event {i+1}/{len(events_list)}: {target_event.get('event_name', 'Unknown')}")

            # Translated commentinputdata
            input_data = {
                "user_profile": person_data,
                "project_blueprint": blueprint,
                "full_event_log": events_list,
                "target_event": target_event
            }

            # generatesummary (Translated commentcheckTranslated commenteacheventTranslated commentbyprocess)
            result = summary_processor.process(input_data, use_checkpoint=False)

            if result.success:
                # Translated commentsessionsummarydata
                processor_output = result.data

                if isinstance(processor_output, list):
                    # Translated commentissessionsummarylist
                    session_summary = processor_output
                elif isinstance(processor_output, dict):
                    # Translated commentstructure，Translated commentsessions
                    if 'sessions' in processor_output and isinstance(processor_output['sessions'], list):
                        session_summary = processor_output['sessions']
                    else:
                        session_summary = []
                else:
                    session_summary = []

                all_summaries.extend(session_summary)
                print(f"      ✅ Generated {len(session_summary)} session summaries")
            else:
                print(f"      ❌ Failed to generate summary: {result.error_message}")

        if all_summaries:
            # Save summarydata
            summary_output = {
                "_metadata": {
                    "person_name": person_name,
                    "persona_role": person_data.get('role', 'Unknown'),
                    "blueprint_topic": blueprint.get('selected_topic', ''),
                    "total_events": len(events_list),
                    "total_sessions": len(all_summaries)
                },
                "sessions": all_summaries
            }

            # Generate project identifier for directory structure
            topic_name = blueprint.get('selected_topic', 'unknown_topic')
            task_id = blueprint.get('selected_task_id', 'unknown')
            safe_topic_name = convert_to_safe_filename(topic_name)
            safe_person_name = convert_to_safe_filename(person_name)
            project_identifier = f"{safe_topic_name}_{task_id}"

            # Save summary data to new directory structure
            summary_output_dir = Path(f"output/{safe_person_name}/{project_identifier}/session_summaries")
            summary_output_dir.mkdir(parents=True, exist_ok=True)

            summary_filename = f"{project_identifier}_summary.json"
            summary_filepath = summary_output_dir / summary_filename

            with open(summary_filepath, 'w', encoding='utf-8') as f:
                json.dump(summary_output, f, ensure_ascii=False, indent=2)

            print(f"    ✅ Session summaries saved to: {summary_filepath}")

            # Link summary to blueprint
            blueprint['summary_file'] = str(summary_filepath)

            return summary_output
        else:
            print(f"    ❌ No session summaries generated")
            return None

    except Exception as e:
        print(f"    ❌ Error generating session summaries: {str(e)}")
        return None


def generate_dialogues(person_data: Dict[str, Any], blueprint: Dict[str, Any], events_data: Dict[str, Any], summary_data: Dict[str, Any],
                       dialogue_model: str = None, evaluation_model: str = None, memory_model: str = None,
                       memory_retrieve_model: str = None, dedup_model: str = None, semantic_schedule_model: str = None,
                       max_turns: int = 12, sessions_to_regenerate=None, max_retries: int = 2, show_progress: bool = True,
                       current_time: str = "", current_plan_items: List[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """
    forsessionsummarygeneratedialogue，functionisdialoguegenerate，memoryforfunction
    useperson-basedConversationControlleritemsAgentcompletedialoguetask
    """
    person_name = person_data.get('name', 'unknown_person')
    sessions = summary_data.get('sessions', [])

    if not sessions:
        print(f"    ⚠️ havetosessionsummary，dialoguegenerate")
        return None

    try:
        # Use provided models or defaults
        if dialogue_model is None:
            dialogue_model = "gemini-2.5-pro-thinking-*"
        if evaluation_model is None:
            evaluation_model = "gpt-4o-mini"
        if memory_model is None:
            memory_model = "gpt-4o-mini"
        if memory_retrieve_model is None:
            memory_retrieve_model = "gemini-2.5-pro-thinking-*"
        if dedup_model is None:
            dedup_model = "gemini-2.5-flash-nothinking"
        if semantic_schedule_model is None:
            semantic_schedule_model = "gpt-4o-mini"

        # useTranslated commentMultiAgentDialogueProcessorTranslated commentrowTranslated commentdialoguegenerate
        dialogue_client = create_client(api_key=api_key, base_url=base_url, model=dialogue_model)
        evaluation_client = create_client(api_key=api_key, base_url=base_url, model=evaluation_model)
        memory_client = create_client(api_key=api_key, base_url=base_url, model=memory_model, reasoning_effort="none")
        dedup_client = create_client(api_key=api_key, base_url=base_url, model=dedup_model, reasoning_effort="none")
        memory_retrieve_client = create_client(api_key=api_key, base_url=base_url, model=memory_retrieve_model, reasoning_effort="none")
        semantic_schedule_client = create_client(api_key=api_key, base_url=base_url, model=semantic_schedule_model, reasoning_effort="none")

        processor = MultiAgentDialogueProcessor(dialogue_client)

        # configdialoguecontrolTranslated comment - Translated commentinitializeparameterTranslated comment
        if hasattr(processor, 'llm_client'):
            # createnewdialoguecontrolTranslated comment，Translated commentallTranslated commentneedTranslated comment
            from pipeline.multi_agent_dialogue_processor import ConversationController
            processor.conversation_controller = ConversationController(
                dialogue_client=dialogue_client,
                evaluation_client=evaluation_client,
                memory_client=memory_client,
                memory_retrieve_client=memory_retrieve_client,
                dedup_client=dedup_client,
                semantic_schedule_client=semantic_schedule_client,  # [NEW] addTranslated comment
                project_attributes_schema=blueprint.get('project_attributes_schema', '')
            )

        # Translated commenteventdata
        event_list = []

        if 'events' in events_data and isinstance(events_data['events'], list):
            # Translated commentuseeventsarrayTranslated commentforeventlist
            event_list = events_data['events']
        elif 'output_data' in events_data and isinstance(events_data['output_data'], list):
            event_list = events_data['output_data']
        else:
            event_list = []


        #print("-------------------------------------------------------------")
        #print(event_list)
        #print("-------------------------------------------------------------")
        
        # memorygetwillinTranslated commentinsideTranslated commentrow，Translated commentuseTranslated commentnewdata

        # processsessionlist
        processed_sessions = []
        successful_sessions = 0

        # processallsessions
        all_sessions = sessions

        # Translated commentcolumnTranslated commentlogic
        if sessions_to_regenerate:
            # Translated commentprocessneedwantTranslated commentnewgeneratesessions
            regenerate_set = set(sessions_to_regenerate)
            all_sessions = [s for s in all_sessions if s.get('session_id') in regenerate_set]
        else:
            print(f"    🎯 Processing {len(all_sessions)} sessions for {person_name}")

        all_dialogues = []

        # Generate project identifier for directory structure
        topic_name = blueprint.get('selected_topic', 'unknown_topic')
        task_id = blueprint.get('selected_task_id', 'unknown')
        safe_topic_name = convert_to_safe_filename(topic_name)
        safe_person_name = convert_to_safe_filename(person_name)
        project_identifier = f"{safe_topic_name}_{task_id}"

        # print(f"    ✅ Project identifier: {project_identifier}")

        # createoutputdirectory
        output_dir = Path(f"output/{safe_person_name}/{project_identifier}/dialogues")
        output_dir.mkdir(parents=True, exist_ok=True)

        # [NEW] readTranslated commenthaveTranslated commentdata
        person_schedule = get_person_schedule(person_name)
        loaded_current_date = person_schedule.get("current_date", "")
        loaded_plan_items = get_plan_items_full(person_schedule)

        # usefunctionparametervalueTranslated commentloadvalue（Translated commentparameter）
        final_current_date = current_time if current_time else loaded_current_date
        final_plan_items = current_plan_items if current_plan_items is not None else loaded_plan_items

        #print("------------------------------------------")
        #print("final_plan_items: " + str(final_plan_items))
        #print("-------------------------------------")

        if final_current_date or final_plan_items:
            print(f"    📅 Loaded schedule data: date='{final_current_date}', plan_items={len(final_plan_items)}")

        # initializeprojectmemoryfilevariable，Translated commentNameError
        project_memory_file = None

        # createTranslated commententry
        if show_progress:
            from tqdm import tqdm
            progress_bar = tqdm(all_sessions, desc="Generate dialogues", unit="session",
                               bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        else:
            progress_bar = all_sessions

            for session in progress_bar:
                session_id = session.get('session_id', f'S{len(processed_sessions)+1:03d}')

                # getcurrenteventallsessionsummary
                current_event_session_summary_list = get_event_session_summaries(
                    session.get('event_id', 0), all_sessions
                )

                # getTranslated commentdialogue - usenewTranslated commentdialoguegetlogic
                history_dialogue = get_history_dialogue(processed_sessions, str(output_dir), session_id)

                # getcurrentpersonmemorydata（useTranslated commentmemory）
                current_person_memory = get_person_memory(person_name)

                # Translated commenttimesTranslated comment，showmemorystatisticsinformation
                if len(processed_sessions) == 0:
                    existing_memory_points = current_person_memory.get("memory_points", [])
                    existing_session_count = current_person_memory.get("total_sessions", 0)
                    print(f"    🧠 Loading existing memory for {person_name}")
                    print(f"      - Existing memory points: {len(existing_memory_points)}")
                    print(f"      - Previous sessions: {existing_session_count}")

                    # statisticsTranslated commenthavememory
                    dynamic_count = len([p for p in existing_memory_points if p.get('type') == 'Dynamic'])
                    static_count = len([p for p in existing_memory_points if p.get('type') == 'Static'])
                    print(f"      - Memory composition: {dynamic_count} Dynamic, {static_count} Static")

                    def date_to_weekday(date_str: str) -> str:
                        """Translated docstring"""
                        if not date_str:
                            return ""

                        try:
                            from datetime import datetime
                            # attemptTranslated commentdateformat
                            for date_format in ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d"]:
                                try:
                                    date_obj = datetime.strptime(date_str, date_format)
                                    weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                                    return weekday_names[date_obj.weekday()]
                                except ValueError:
                                    continue
                            # Translated commentallformatTranslated commentfailed，returnTranslated commentstring
                            return date_str
                        except:
                            return date_str

                # buildinputdata - useMultiAgentDialogueProcessorTranslated commentformat，addtimeandTranslated comment
                input_data = {
                    "user_input_profile": person_data,
                    "full_event_log": event_list,
                    "current_event_session_summary_list": current_event_session_summary_list,
                    "target_session": {
                        "session_id": session_id,
                        "session_summary": session.get('session_summary', '')
                    },
                    "max_turns": max_turns,
                    "memory_context": current_person_memory,
                    "history_dialogue": history_dialogue,
                    # [NEW] timeandTranslated commentfield - useTranslated commentafterdata
                    "current_time": f"{final_current_date} ({date_to_weekday(final_current_date)})",  # Translated commentAPIfieldTranslated commentforcurrent_time
                    "current_plan_items": final_plan_items  # Translated commentprojectlist
                }


                #print("Input Data-------------------------------------------------------------------------------------------------------")
                #print(json.dumps(input_data, ensure_ascii=False, indent=2))
                #print("-----------------------------------------------------------------------------------------------------------------")

                # Processing session - useMultiAgentDialogueProcessorTranslated commentprocess，addretryTranslated comment
                last_error = None
                for attempt in range(max_retries + 1):  # Translated commentattempt + max_retriestimesretry
                    try:
                        if attempt >= 0:
                            print(f"      🔄  {attempt} timesretryprocess Session {session_id}...")

                        result = processor.process(input_data, use_checkpoint=False)

                        if result.success:
                            dialogue_output = result.data

                            # getTranslated commentmemorydata
                            retrieved_memory_data = dialogue_output.get('retrieved_memory', {})
                            retrieved_memory_points = retrieved_memory_data.get('all', [])

                            # initializenewmemoryTranslated commentlist
                            new_memory_points = []

                            # Translated commentdialoguedataTranslated commentmemoryTranslated comment
                            dialogue_data_for_memory = {
                                "session_id": session_id,
                                "dialogue_turns": dialogue_output.get('dialogue_turns', [])
                            }

                            # useMultiAgentDialogueProcessormemorymanagefunction
                            if hasattr(processor, 'conversation_controller') and processor.conversation_controller:
                                try:
                                    # Translated commentnewmemoryTranslated comment
                                    extracted_memory = processor.conversation_controller.extract_session_memory(dialogue_data_for_memory)
                                    new_memory_points = extracted_memory.get('memory_points', [])

                                    # formemoryTranslated commentIDaddproject_identifierbeforeTranslated comment
                                    for point in new_memory_points:
                                        current_index = point.get('index', '')
                                        if current_index and not current_index.startswith(project_identifier + '-'):
                                            point['index'] = f"{project_identifier}-{current_index}"

                                    # Translated comment：Translated commentmemory + newmemory
                                    combined_memory_points = retrieved_memory_points + new_memory_points
                                    if combined_memory_points:
                                        print(f"      🔄 begin: memory {len(retrieved_memory_points)} + newmemory {len(new_memory_points)}")

                                        # createTranslated commentdataTranslated commentrowTranslated comment
                                        temp_memory_data = {"memory_points": combined_memory_points}
                                        deduplicated_result = processor.conversation_controller.deduplicate_memory_points(temp_memory_data)
                                        final_memory_points = deduplicated_result.get("memory_points", [])

                                        # statisticsTranslated commentresult
                                        original_count = len(combined_memory_points)
                                        active_points_count = len([p for p in final_memory_points if not p.get('discard', False)])
                                        discarded_points_count = len([p for p in final_memory_points if p.get('discard', False)])
                                        print(f"      ✅ complete: {original_count} itemsmemory → {active_points_count} itemshave, {discarded_points_count} itemsbyfor")

                                        # showTranslated commenttonewmemoryTranslated comment（original）
                                        if new_memory_points:
                                            print(f"      ✅ to {len(new_memory_points)} itemsnewmemory")

                                            # shownewmemoryTranslated comment
                                            for i, point in enumerate(new_memory_points[:3]):
                                                print(f"      - {point.get('index', '')}: {point.get('content', '')[:60]}...")
                                            if len(new_memory_points) > 3:
                                                print(f"      ... have{len(new_memory_points)-3}itemsmemory")

                                        # Translated commentmemoryupdatelogic：Translated commentfromTranslated commentafterresultinTranslated commentmemoryTranslated commentupdateTranslated commentmemoryTranslated comment
                                        retrieved_indices = {m.get('index') for m in retrieved_memory_points}
                                        added_count = 0
                                        updated_count = 0

                                        for dedup_point in final_memory_points:
                                            point_index = dedup_point.get('index')
                                            if not point_index:
                                                continue

                                            # checkTranslated commentmemoryTranslated commentinisTranslated commentinthismemoryTranslated comment
                                            found_in_global = False
                                            for i, global_point in enumerate(current_person_memory['memory_points']):
                                                if global_point.get('index') == point_index:
                                                    # updateTranslated commenthavememoryTranslated commentcontentanddiscardstatus
                                                    current_person_memory['memory_points'][i]['content'] = dedup_point.get('content', global_point.get('content', ''))
                                                    current_person_memory['memory_points'][i]['discard'] = dedup_point.get('discard', False)
                                                    found_in_global = True
                                                    updated_count += 1
                                                    break

                                            # Translated commentinTranslated commentmemoryTranslated commentin，Translated commentaddnewmemoryTranslated comment
                                            if not found_in_global:
                                                current_person_memory['memory_points'].append(dedup_point)
                                                added_count += 1

                                        if updated_count > 0:
                                            print(f"      ✅ update {updated_count} itemshavememorycontentandstatus")
                                        if added_count > 0:
                                            print(f"      ✅ add {added_count} itemsnewmemorytomemory")

                                        # Translated commentsavecurrentmemorytofile（eachtimessessionprocessafter）
                                        current_person_memory_file = save_person_memory(person_name, f"output/{safe_person_name}")
                                        project_memory_file = save_project_memory(person_name, project_identifier, current_person_memory)
                                        print(f"      💾 memorysave: {current_person_memory_file}")
                                        print(f"      💾 projectmemorysave: {project_memory_file}")
                                    else:
                                        print(f"      ⚠️ havememoryneedwantprocess")

                                except Exception as e:
                                    print(f"      ⚠️ memoryprocessfailed: {str(e)}")

                            # [NEW] processTranslated commentupdateTranslated commentproject
                            updated_plan_items = dialogue_output.get('updated_plan_items', [])
                            if updated_plan_items and (final_current_date or final_plan_items):
                                print(f"      📅 projectupdate: {len(final_plan_items)} -> {len(updated_plan_items)} item")

                                # saveupdateafterTranslated commentprojecttofile - Translated commentformat
                                updated_schedule_data = {
                                    "plan_items": updated_plan_items,  # Translated commentupdateafterTranslated commentprojectlist
                                    "current_date": final_current_date,   # currentsessiondate
                                    "metadata": person_schedule.get("metadata", {})
                                }

                                schedule_file_path = save_person_schedule(person_name, updated_schedule_data, project_identifier, session_id)
                                if schedule_file_path:
                                    print(f"      💾 Updated plan items saved to: {schedule_file_path}")

                                # updateinsideTranslated commentinTranslated commentdata，Translated commentafterTranslated commentsession
                                final_plan_items = updated_plan_items

                            # builddialogueoutputformat，includesTranslated commentinput_data
                            dialogue_data = {
                                "session_id": session_id,
                                "input_data": input_data,
                                "dialogue_turns": dialogue_output.get('dialogue_turns', []),
                                "current_time": f"{final_current_date} ({date_to_weekday(final_current_date)})",
                                "goal_evaluation": dialogue_output.get('goal_evaluation', {}),
                                "new_memory_points": new_memory_points,  # currentsessiongeneratenewmemoryTranslated comment
                                "retrieved_memory_points": retrieved_memory_points,  # Translated commenttomemoryTranslated comment
                                "metadata": {
                                    "total_turns": len(dialogue_output.get('dialogue_turns', [])),
                                    "person_name": person_name,
                                    "processing_time": dialogue_output.get('metadata', {}).get('processing_time', 0),
                                    "processor_type": dialogue_output.get('metadata', {}).get('processor_type', 'multi_agent_dialogue'),
                                    "contains_input_data": True,  # Translated commentincludesinputdata
                                    "new_memory_count": len(new_memory_points),  # newmemoryTranslated commentquantity
                                    "retrieved_memory_count": len(retrieved_memory_points)  # Translated commentmemoryTranslated commentquantity
                                }
                            }

                            # Save dialogueoutput
                            output_file = output_dir / f"{session_id}.json"
                            with open(output_file, 'w', encoding='utf-8') as f:
                                json.dump(dialogue_data, f, ensure_ascii=False, indent=2)

                            # addtoTranslated commentlist
                            all_dialogues.append(dialogue_data)

                            processed_sessions.append(session_id)
                            successful_sessions += 1

                            dialogue_turns = len(dialogue_output.get('dialogue_turns', []))
                            goal_score = dialogue_output.get('goal_evaluation', {}).get('overall_score', 0)

                            # updateTranslated commententryafterTranslated commentinformation
                            if show_progress:
                                progress_bar.set_postfix({
                                    'success': f"{successful_sessions}/{len(all_sessions)}",
                                    '': dialogue_turns,
                                    '': f"{goal_score:.0f}",
                                    'memory': len(current_person_memory.get('memory_points', []))
                                })

                            # showTranslated commentinformation
                            if attempt > 0:
                                print(f"      ✅ Session {session_id} retrysuccess: {dialogue_turns}dialogue, goal: {goal_score} ({attempt}timesretry)")
                            else:
                                print(f"✅ Session {session_id} processsuccess: {dialogue_turns}dialogue, goal: {goal_score}")
                            print(f"💾 save: {output_file}")
                            print(f"📊 dialogue: {dialogue_turns}")
                            print(f"🎯 goalcomplete: {goal_score:.1f}")
                            print(f"💬 Sample dialogue:")
                            dialogue_turns_data = dialogue_output.get('dialogue_turns', [])
                            if dialogue_turns_data:
                                print(f"💬 Sample dialogue:")
                                for j, turn in enumerate(dialogue_turns_data[:4]):
                                    speaker = turn.get('speaker', 'Unknown')
                                    content = turn.get('content', '')[:100]
                                    phase = turn.get('phase', 'unknown')
                                    # print(f"  {j+1}. [{speaker}] ({phase}): {content}...")
                            #print()
                            break  # successTranslated comment，Translated commentretryTranslated comment

                        else:
                            last_error = result.error_message
                            if attempt < max_retries:
                                print(f"      ❌ Session {session_id}  {attempt + 1} timesattemptfailed: {last_error}")
                                print(f"         willin {attempt + 1} afterretry...")
                                import time
                                time.sleep(attempt + 1)  # Translated commentdelay
                            else:
                                print(f"      ❌ Session {session_id} allretryfailed: {last_error}")

                    except Exception as e:
                        last_error = str(e)
                        if attempt < max_retries:
                            print(f"      ❌ Session {session_id}  {attempt + 1} timesattemptexception: {last_error}")
                            print(f"         willin {attempt + 1} afterretry...")
                            import time
                            time.sleep(attempt + 1)  # Translated commentdelay
                        else:
                            print(f"      ❌ Session {session_id} allretryexception: {last_error}")

                # checkisTranslated commentallretryTranslated commentfailedTranslated comment
                if last_error is not None:
                    # allretryTranslated commentfailedTranslated comment，Translated commentwantreturn1，continueprocessunderTranslated commentitemssession
                    if show_progress:
                        progress_bar.set_postfix({
                            'success': f"{successful_sessions}/{len(all_sessions)}",
                            'status': 'failed'
                        })
                    print(f"💥 Session {session_id} processfailed，retry {max_retries} times，thissession")
                    continue  # Translated commentitemssession，continueprocessunderTranslated commentitems

        # getTranslated commentmemorystatusTranslated commentsave
        final_person_memory = get_person_memory(person_name)

        # Translated commentsaveTranslated commenttimesTranslated commentmemory（Translated commentstatusbysave）
        final_memory_file = save_person_memory(person_name, f"output/{safe_person_name}")
        print(f"💾 memorysavecomplete: {final_memory_file}")

        # descriptionprojectmemoryTranslated commentineachitemssessioninTranslated commentsave，noneedTranslated commentsave
        print(f"📝 projectmemoryineachitemssessionaftersave")

        if all_dialogues:
            # getTranslated commentpersonmemorydata
            final_person_memory = get_person_memory(person_name)
            final_memory_count = len(final_person_memory.get('memory_points', []))

            # calculateTranslated commentmemoryTranslated commentquantity（fromTranslated commentmemoryinTranslated commenttimesnewTranslated comment）
            total_new_memory = sum(session.get('_metadata', {}).get('memory', 0) for session in all_dialogues)
            initial_memory_count = max(0, final_memory_count - total_new_memory)

            print(f"✅ Successfully generated dialogues for {len(all_dialogues)} sessions")
            print(f"🧠 Final memory count for {person_name}: {final_memory_count} points")
            if project_memory_file:
                print(f"💾 Project memory saved to: {project_memory_file}")
            else:
                print(f"⚠️ Project memory was not saved (memory extraction may have failed)")
            print(f"💾 Final person memory saved to: {final_memory_file}")

            # savealldialogueTranslated commentfile
            try:
                # buildTranslated commentfilepath
                summary_file = output_dir / f"{project_identifier}_all_dialogues_summary.json"

                # savealldialogueTranslated comment
                save_all_dialogues(all_dialogues, str(summary_file))
                print(f"💾 All dialogues summary saved to: {summary_file}")

                # updatereturnTranslated commentdata
                summary_metadata = {
                    "all_dialogues_summary_file": str(summary_file)
                }
            except Exception as e:
                print(f"⚠️ Failed to save all dialogues summary: {str(e)}")
                summary_metadata = {
                    "all_dialogues_summary_file": None,
                    "summary_save_error": str(e)
                }

            return {
                "_metadata": {
                    "person_name": person_name,
                    "total_sessions_tested": len(all_sessions),
                    "total_sessions_available": len(sessions),
                    "successful_dialogues": len(all_dialogues),
                    "initial_memory_points": initial_memory_count,
                    "final_memory_points": final_memory_count,
                    "new_memory_points": total_new_memory,
                    "final_memory_file": final_memory_file,
                    "project_memory_file": project_memory_file,
                    **summary_metadata  # addTranslated commentfileTranslated commentinformation
                },
                "sessions": all_dialogues,
                "final_memory_data": final_person_memory
            }
        else:
            print(f"❌ No dialogues generated")
            return None

    except Exception as e:
        print(f"❌ Error generating dialogues: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def generate_dialogues_interleaved(person_data: Dict[str, Any], project_dialogue_data: Dict[str, Any], args=None) -> Dict[str, Any]:
    """
    projectdialoguegenerate

    Args:
        person_data: persondata
        project_dialogue_data: projectdialoguedatadict {project_identifier: project_data}
        args: parameterconfig

    Returns:
        dialogueresultdict
    """
    person_name = person_data.get('name', 'unknown_person')
    safe_person_name = convert_to_safe_filename(person_name)

    print(f"  🎯 projectdialoguegeneratebegin")

    # createTranslated commentdialogueTranslated commentcolumn，Translated commentprojectinsideTranslated commentsessionTranslated comment
    dialogue_queue = []
    import random
    import json
    import time

    # checkisTranslated commentinTranslated commentsavedialogueTranslated commentcolumnstatus
    safe_person_name = convert_to_safe_filename(person_name)
    queue_state_file = Path(f"output/{safe_person_name}/interleaved_dialogue_queue_state.json")

    # createdialogueTranslated commentcolumnstatusdirectory
    queue_state_file.parent.mkdir(parents=True, exist_ok=True)

    # attemptloadTranslated commentsaveTranslated commentcolumnstatus
    saved_queue_state = None

    if queue_state_file.exists():
        try:
            with open(queue_state_file, 'r', encoding='utf-8') as f:
                saved_queue_state = json.load(f)

            print(f"  📂 savedialoguecolumnstatus")
            print(f"      dialogue: {saved_queue_state.get('total_dialogues', 0)}")
            print(f"      process: {saved_queue_state.get('processed_count', 0)}")
            print(f"      process: {len(saved_queue_state.get('remaining_queue', []))}")

            # Translated commentusesaveTranslated commentcolumn
            dialogue_queue = saved_queue_state.get('remaining_queue', [])

            # Translated commentprocesssessions（useproject_identifier:session_idformat）
            processed_session_ids = set(saved_queue_state.get('processed_session_ids', []))
            if processed_session_ids:
                original_count = len(dialogue_queue)
                dialogue_queue = [item for item in dialogue_queue
                                 if f"{item['project_identifier']}:{item['session'].get('session_id')}" not in processed_session_ids]
                if dialogue_queue:
                    print(f"  🔄  {original_count - len(dialogue_queue)} itemsprocessdialoguesession")
                else:
                    print(f"  ✅ alldialoguesessionscomplete，dialoguegenerate")
                    return {}

        except Exception as e:
            print(f"  ⚠️ loadcolumnstatusfailed: {str(e)}，newgeneratecolumn")
            saved_queue_state = None

    if not saved_queue_state:
        # Translated commenthaveinTranslated commenthavesavestatusTranslated commentandprocessdata
        print(f"  🔄 generatenewdialoguecolumn...")

        # Translated commentallTranslated commentgeneratedialoguesession，Translated commentprojectTranslated commentinsideTranslated comment
        project_queues = {}
        for project_identifier, project_data in project_dialogue_data.items():
            sessions_to_generate = project_data["sessions_to_generate"]
            if sessions_to_generate:
                # getTranslated commentsessionslist
                summary_data = project_data["summary_data"]
                all_sessions = summary_data.get("sessions", [])

                # Translated commentneedwantgeneratesessions，Translated commenthaveTranslated comment
                generate_set = set(sessions_to_generate)
                project_session_queue = []
                for session in all_sessions:
                    if session.get("session_id") in generate_set:
                        project_session_queue.append({
                            "project_identifier": project_identifier,
                            "session": session,
                            "project_data": project_data
                        })

                if project_session_queue:
                    project_queues[project_identifier] = project_session_queue

        # Translated commenttopicbetweenTranslated commentsetting
        TOPIC_COOLDOWN_SESSIONS = 5  # Translated commenttopicneedwantbetweenTranslated commentsessionquantity

        # calculateTranslated commentdialoguequantityandstatisticstopicTranslated comment
        total_dialogues = sum(len(queue) for queue in project_queues.values())
        topic_distribution = {}
        for project_identifier, queue in project_queues.items():
            project_data = queue[0]["project_data"] if queue else None
            if project_data:
                task = project_data.get("task", {})
                topic_id = task.get('topic_id')
                if topic_id:
                    topic_distribution[topic_id] = topic_distribution.get(topic_id, 0) + len(queue)

        print(f"  📊 needwantgenerate {total_dialogues} itemsdialoguesessions， {len(project_queues)} itemsproject")
        print(f"  📈 Topic: {dict(topic_distribution)}")
        print(f"  ⏰ topicprojectendafter，needwantbetween {TOPIC_COOLDOWN_SESSIONS} itemsotherprojectsession")

        if not project_queues:
            print(f"  ⚠️ havegeneratedialoguesessions")
            return {}

        # usetopicTranslated commentcolumnlogic：eachitemstopicisTranslated commentitemsTranslated commentcolumn，Translated commentcolumninsideTranslated commentexecute，Translated commentcolumnbetweenTranslated comment
        # Translated commenttopicTranslated commentprojectTranslated commentcolumn
        topic_queues = {}  # {topic_id: [project1, project2, ...]}
        topic_active_projects = {}  # {topic_id: current_active_project_identifier}

        # buildtopicTranslated commentcolumn
        for project_identifier, project_sessions in project_queues.items():
            project_data = project_sessions[0]["project_data"] if project_sessions else None
            if project_data:
                task = project_data.get("task", {})
                topic_id = task.get('topic_id')

                if topic_id not in topic_queues:
                    topic_queues[topic_id] = []
                topic_queues[topic_id].append(project_identifier)

        # eachitemstopicTranslated commentprojectisTranslated commentcolumnTranslated commentitemsproject
        for topic_id, projects in topic_queues.items():
            topic_active_projects[topic_id] = projects[0]

        print(f"  📋 Topiccolumncomplete: {[(tid, len(projects)) for tid, projects in topic_queues.items()]}")
        print(f"  🎯 project: {topic_active_projects}")

        # Translated comment：Translated commenttopiccurrentprojectcompleteTranslated comment，thetopicTranslated comment
        topic_cooldown = {}  # {topic_id: cooldown_remaining_sessions}
        TOPIC_COOLDOWN_SESSIONS = 5  # Translated commenttopicprojectTranslated commentbetweenneedwantbetweenTranslated commentothertopicsessionquantity

        def get_topic_id_from_project(project_identifier, project_data):
            """fromproject_dataingettopic_id"""
            task = project_data.get("task", {})
            return task.get('topic_id')

        def update_topic_cooldown():
            """Translated docstring"""
            for topic_id in list(topic_cooldown.keys()):
                topic_cooldown[topic_id] -= 1
                if topic_cooldown[topic_id] <= 0:
                    # Translated commentcomplete，Translated commentthetopicunderTranslated commentitemsproject（Translated commenthaveTranslated comment）
                    del topic_cooldown[topic_id]
                    current_active = topic_active_projects[topic_id]

                    # checkcurrentTranslated commentprojectisTranslated commenthaveTranslated commentprocesssession
                    if current_active in project_queues and project_queues[current_active]:
                        # currentprojectTranslated commenthavesession，Translated commentneedwantTranslated comment
                        print(f"  ✅ Topic {topic_id} complete，continueprocesscurrentproject: {current_active}")
                        continue

                    # Translated commentunderTranslated commentitemsproject
                    if topic_id in topic_queues and len(topic_queues[topic_id]) > 1:
                        # haveunderTranslated commentitemsproject，Translated comment
                        try:
                            next_project_index = topic_queues[topic_id].index(current_active) + 1
                            if next_project_index < len(topic_queues[topic_id]):
                                next_project = topic_queues[topic_id][next_project_index]
                                # Translated commentunderTranslated commentitemsprojectTranslated commenthaveTranslated commentprocesssession
                                if next_project in project_queues and project_queues[next_project]:
                                    topic_active_projects[topic_id] = next_project
                                    print(f"  ✅ Topic {topic_id} complete，underitemsproject: {topic_active_projects[topic_id]}")
                                else:
                                    print(f"  ⚠️ Topic {topic_id} underitemsproject {next_project} nocansession，")
                                    # attemptTranslated commentafterTranslated commenthavesessionproject
                                    for i in range(next_project_index + 1, len(topic_queues[topic_id])):
                                        candidate = topic_queues[topic_id][i]
                                        if candidate in project_queues and project_queues[candidate]:
                                            topic_active_projects[topic_id] = candidate
                                            print(f"  ✅ Topic {topic_id} afterproject: {candidate}")
                                            break
                            else:
                                print(f"  ⚠️ Topic {topic_id} index")
                        except ValueError:
                            print(f"  ⚠️ Topic {topic_id} tocurrentproject {current_active}")
                    else:
                        # Translated commenthaveunderTranslated commentitemsprojectTranslated comment
                        print(f"  ✅ Topic {topic_id} complete，noproject")

        def mark_topic_project_completed(topic_id):
            """Translated docstring"""
            topic_cooldown[topic_id] = TOPIC_COOLDOWN_SESSIONS
            print(f"  ⏸️ Topic {topic_id} projectcomplete，begin {TOPIC_COOLDOWN_SESSIONS} itemssession")

        def get_available_active_projects():
            """Translated docstring"""
            available = []
            for topic_id, active_project in topic_active_projects.items():
                if (active_project in project_queues and
                    project_queues[active_project] and  # Translated commenthaveTranslated commentprocesssession
                    topic_id not in topic_cooldown):  # topicTranslated commentinTranslated comment

                    project_data = project_queues[active_project][0]["project_data"]
                    available.append((active_project, topic_id))
            return available

        # Translated comment：useTranslated comment，Translated commentintopicTranslated commentcolumnTranslated commentbetweenTranslated commentrow
        total_sessions_generated = 0
        max_possible_sessions = sum(len(queue) for queue in project_queues.values())

        while total_sessions_generated < max_possible_sessions:
            # getcurrentcanTranslated commentproject
            available_projects = get_available_active_projects()

            if not available_projects:
                # Translated commenthavecanTranslated commentproject（allTranslated commentprojectTranslated commentinTranslated commentorTranslated commentcomplete）
                print(f"  ⏳ havecanproject，cancanin")
                # updateTranslated commentstatus，Translated commentisTranslated commentcanTranslated commentnewproject
                update_topic_cooldown()
                available_projects = get_available_active_projects()

                if not available_projects:
                    print(f"  🔄 ，use：projectprocesssession")
                    # Translated comment：Translated commentprocessTranslated commentallsession
                    remaining_sessions = []
                    for project_identifier, queue in project_queues.items():
                        if queue:  # Translated commenttheprojectTranslated commenthavesession
                            for session_item in queue:
                                remaining_sessions.append(session_item)

                    if remaining_sessions:
                        print(f"  📋 ：to {len(remaining_sessions)} itemssession，projectprocess")
                        dialogue_queue.extend(remaining_sessions)
                        total_sessions_generated += len(remaining_sessions)
                        print(f"  ✅ complete，session: {total_sessions_generated}/{max_possible_sessions}")
                        break
                    else:
                        print(f"  ❌ nocontinueGenerate dialoguescolumn：nosession")
                        break

            # Translated commentitemscanTranslated commentproject - Translated commenthaveTranslated commentlogic
            selected_project, selected_topic_id = random.choice(available_projects)

            # fromTranslated commentinprojectinTranslated commentunderTranslated commentitemssession（Translated comment）
            next_session = project_queues[selected_project].pop(0)

            # [NEW] forsessionaddtimeinformation，Translated commentitemssessionforcurrentdate，afterTranslated commentsessionTranslated commentbetweenTranslated comment1-3Translated comment
            if len(dialogue_queue) == 0:
                # Translated commentitemssessionusecurrentdate
                current_date = datetime.now().strftime("%Y-%m-%d")
            else:
                # afterTranslated commentsessionTranslated commentbetweenTranslated comment1-3Translated comment
                last_date = datetime.strptime(dialogue_queue[-1].get("session_date", current_date), "%Y-%m-%d")
                random_days = random.randint(1, 3)
                current_date = (last_date + timedelta(days=random_days)).strftime("%Y-%m-%d")

            # willdateinformationaddtosessionin
            next_session["session_date"] = current_date

            dialogue_queue.append(next_session)
            total_sessions_generated += 1

            # updatealltopicTranslated commentstatus
            update_topic_cooldown()

            # checkTranslated commentinprojectisTranslated commentcomplete
            if not project_queues[selected_project]:
                # projectcomplete，beginthetopicTranslated comment
                mark_topic_project_completed(selected_topic_id)

                # fromtopicTranslated commentcolumninTranslated commenttheprojectTranslated commentcomplete（throughremoveorTranslated comment）
                # Translated commenttopic_active_projectsTranslated commentcurrentproject，Translated commenttoTranslated commentcompleteTranslated commenttounderTranslated commentitems

            # showcurrentstatus（debuginformation）
            if total_sessions_generated % 10 == 0:  # each10itemssessionshowTranslated commenttimesstatus
                print(f"  📊 generate {total_sessions_generated}/{max_possible_sessions} itemssession")
                if topic_cooldown:
                    print(f"  🌡️ status: {[(tid, remaining) for tid, remaining in topic_cooldown.items()]}")
                print(f"  🎯 currentproject: {[(tid, proj) for tid, proj in topic_active_projects.items() if proj in project_queues and project_queues[proj]]}")

        # saveTranslated commentcolumnstatus
        initial_queue_state = {
            "person_name": person_name,
            "total_dialogues": total_dialogues,
            "total_projects": len(project_queues),
            "remaining_queue": dialogue_queue.copy(),
            "processed_count": 0,
            "processed_session_ids": [],
            "created_at": time.time(),
            "metadata": {
                "project_identifiers": list(project_queues.keys()),
                "session_counts": {pid: len(queue) for pid, queue in project_queues.items()}
            }
        }

        try:
            with open(queue_state_file, 'w', encoding='utf-8') as f:
                json.dump(initial_queue_state, f, ensure_ascii=False, indent=2)
            print(f"  💾 dialoguecolumnstatussave: {queue_state_file}")

            # Translated commentkeyTranslated comment：willnewcreateTranslated commentcolumnstatusTranslated commentvalueTranslated commentsaved_queue_state
            saved_queue_state = initial_queue_state
        except Exception as e:
            print(f"  ⚠️ savecolumnstatusfailed: {str(e)}")

    # useTranslated commenthavegenerate_dialoguesfunctionprocess，Translated commenttimesTranslated commentprocessTranslated commentitemssession
    all_results = {}
    processed_count = 0

    print(f"  🔄 beginprocessdialogue...")


    #return

    # createTranslated commententry
    from tqdm import tqdm
    with tqdm(dialogue_queue, desc="🎯 dialoguegenerate", unit="session",
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
              dynamic_ncols=True) as pbar:

        for i, dialogue_item in enumerate(pbar):
            project_identifier = dialogue_item["project_identifier"]
            session = dialogue_item["session"]
            project_data = dialogue_item["project_data"]
            session_date = dialogue_item.get("session_date", datetime.now().strftime("%Y-%m-%d"))


            #print("-------------------------------------------------------------")
            #print(project_data["events_data"])
            #print("-------------------------------------------------------------")

            # updateTranslated commententryshow - inprocessbeginbeforeupdate
            current_project = project_identifier[:10] + '..' if len(project_identifier) > 10 else project_identifier
            current_session = session.get('session_id', 'Unknown')
            pbar.set_description(f"🎯 dialoguegenerate [{current_project}|{current_session}]")

            # callTranslated commenthavegenerate_dialoguesfunction，Translated commenttimesTranslated commentprocessTranslated commentitemssession
            try:
                # Translated commentsessiondata
                single_session_data = {
                    "sessions": [session],
                    "_metadata": project_data["summary_data"].get("_metadata", {})
                }

                # getTranslated commenteventlist
                all_events = project_data["events_data"].get("events", [])
                if not all_events and "output_data" in project_data["events_data"]:
                    if isinstance(project_data["events_data"]["output_data"], list):
                        all_events = project_data["events_data"]["output_data"]
                    elif isinstance(project_data["events_data"]["output_data"], dict):
                        all_events = project_data["events_data"]["output_data"].get("events", [])

                # forcurrentsessiongetTranslated commentevent（currentevent + beforeTranslated commentitemsevent）
                relevant_events = get_relevant_events_for_session(session, all_events)
                filtered_events_data = {
                    "events": relevant_events,
                    "_metadata": project_data["events_data"].get("_metadata", {})
                }

                #print("-------------------------------------------------------------")
                #print(filtered_events_data)
                #print("-------------------------------------------------------------")

            

                #print("----------------------------------------------")
                #print("session_date: " + session_date)
                #print("----------------------------------------------")
                # Translated commentnewcallTranslated commentprojectgenerate_dialogues，Translated commentusesession_to_regenerateparameterTranslated comment
                dialogue_result = generate_dialogues(
                    person_data=person_data,
                    blueprint=project_data["blueprint"],
                    events_data=filtered_events_data,
                    summary_data=single_session_data,
                    dialogue_model=args.dialogue_model if args else None,
                    evaluation_model=args.evaluation_model if args else None,
                    memory_model=args.memory_model if args else None,
                    memory_retrieve_model=args.memory_retrieve_model if args else None,
                    dedup_model=args.dedup_model if args else None,
                    semantic_schedule_model=getattr(args, 'semantic_schedule_model', None) if args else None,
                    max_turns=args.max_turns if args else 16,
                    sessions_to_regenerate=[session.get("session_id")],
                    max_retries=getattr(args, 'max_retries', 2) if args else 2,
                    show_progress=False,  # Translated commentinsideTranslated commententry，Translated commentwithoutsideTranslated commententryTranslated comment
                    current_time=session_date  # [NEW] Translated commentsessiondate
                )

                if dialogue_result:
                    # Translated commentresult
                    if project_identifier not in all_results:
                        all_results[project_identifier] = {"sessions": []}

                    all_results[project_identifier]["sessions"].extend(dialogue_result.get("sessions", []))
                    processed_count += 1
                    print(f"      ✅ successGenerate dialogues")

                    # updateTranslated commentcolumnstatus - addTranslated commentprocesssession ID
                    current_session_id = session.get("session_id")
                    print(f"      🔍 debugcolumnstatusupdate: saved_queue_state={saved_queue_state is not None}, current_session_id='{current_session_id}'")
                    if saved_queue_state and current_session_id:
                        # createTranslated commentIDTranslated comment：project_identifier + session_id
                        unique_session_id = f"{project_identifier}:{current_session_id}"
                        saved_queue_state["processed_session_ids"].append(unique_session_id)
                        saved_queue_state["processed_count"] = processed_count

                        # updateTranslated commentcolumn（removecurrentTranslated commentprocesssession）
                        # Translated commentusecurrentprocessTranslated commentafterallprojectTranslated commentfornewTranslated commentcolumn
                        saved_queue_state["remaining_queue"] = dialogue_queue[i+1:]

                        # saveupdateafterstatus
                        try:
                            saved_queue_state["last_updated"] = time.time()
                            with open(queue_state_file, 'w', encoding='utf-8') as f:
                                json.dump(saved_queue_state, f, ensure_ascii=False, indent=2)
                            print(f"      💾 columnstatusupdate: process {saved_queue_state['processed_count']}/{saved_queue_state['total_dialogues']},  {len(saved_queue_state.get('remaining_queue', []))}")
                        except Exception as e:
                            print(f"      ⚠️ updatecolumnstatusfailed: {str(e)}")

                else:
                    print(f"      ❌ Dialogue generation failed")

            except Exception as e:
                print(f"      💥 processexception: {str(e)}")
                continue

    print(f"  📈 successprocess {processed_count}/{len(dialogue_queue)} itemsdialoguesessions")

    # completeafterTranslated commentorTranslated commentupdatestatus
    if saved_queue_state:
        if processed_count == saved_queue_state.get("total_dialogues", 0):
            # alldialogueTranslated commentcomplete - Translated commentdeletefile，Translated commentisTranslated commentprocessTranslated comment
            try:
                # Translated commentwantinformation，Translated commentfield
                completed_state = {
                    "person_name": saved_queue_state.get("person_name"),
                    "total_dialogues": saved_queue_state.get("total_dialogues"),
                    "total_projects": saved_queue_state.get("total_projects"),
                    "processed_session_ids": saved_queue_state.get("processed_session_ids"),  # Translated commentprocessTranslated comment
                    "processed_count": saved_queue_state.get("processed_count"),
                    "completed_at": time.time(),
                    "status": "completed",
                    "metadata": saved_queue_state.get("metadata"),
                    "processing_order": "preserved"  # Translated commentprocessTranslated comment
                }

                with open(queue_state_file, 'w', encoding='utf-8') as f:
                    json.dump(completed_state, f, ensure_ascii=False, indent=2)
                print(f"  ✅ dialoguecomplete，processsavetostatusfile")

            except Exception as e:
                print(f"  ⚠️ savecompletestatusfailed: {str(e)}")
        else:
            # updateTranslated commentstatus
            try:
                saved_queue_state["completed_at"] = time.time()
                saved_queue_state["status"] = "partially_completed" if processed_count > 0 else "failed"
                with open(queue_state_file, 'w', encoding='utf-8') as f:
                    json.dump(saved_queue_state, f, ensure_ascii=False, indent=2)
                print(f"  💾 statussave")
            except Exception as e:
                print(f"  ⚠️ savestatusfailed: {str(e)}")

    return all_results


def get_relevant_events_for_session(session: Dict[str, Any], all_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    forsessiongetevent

    Args:
        session: currentprocesssession，includesevent_id
        all_events: eventlist

    Returns:
        eventlist：currentevent + beforeitemsevent（in）
    """
    current_event_id = session.get('event_id')
    if not current_event_id:
        return []

    # inTranslated commenteventlistinTranslated commenttocurrenteventindex
    current_event_index = None
    for i, event in enumerate(all_events):
        if event.get('event_id') == current_event_id or event.get('event_index') == current_event_id:
            current_event_index = i
            break

    if current_event_index is None:
        return []

    # getcurrenteventandbeforeTranslated commentitemsevent
    relevant_events = []
    if current_event_index > 0:
        relevant_events.append(all_events[current_event_index - 1])  # beforeTranslated commentitemsevent
    relevant_events.append(all_events[current_event_index])  # currentevent

    return relevant_events


def extract_session_sessions(session_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Translated docstring"""
    sessions = session_data.get('sessions', [])
    if isinstance(sessions, list):
        return sessions
    return []


def filter_session_by_id(sessions: List[Dict[str, Any]], session_id: str) -> List[Dict[str, Any]]:
    """Translated docstring"""
    if not session_id:
        return sessions

    filtered = [s for s in sessions if s.get('session_id') == session_id]
    return filtered if filtered else sessions


def get_event_session_summaries(event_id: int, sessions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Translated docstring"""
    return [s for s in sessions if s.get('event_id') == event_id]


def get_history_dialogue(processed_sessions: List[str], output_dir: str, target_session_id: str = None) -> List[Dict[str, Any]]:
    """
    getdialoguerecord - get

    Args:
        processed_sessions: currentrunprocesssessionlist
        output_dir: dialoguefiledirectory
        target_session_id: goalsession ID（get）
    """
    output_path = Path(output_dir)

    # Translated commentgoalsession，getthesessionTranslated commentbeforeTranslated commentdialogue
    if target_session_id:
        return get_history_dialogue_for_session(target_session_id, output_path)

    # Translated commenthavelogic：Translated commentprocessed_sessionsgetonTranslated commentitemssessionTranslated commentdialogue
    if not processed_sessions:
        return []

    # Translated commentgetonTranslated commentitemssession
    last_session_id = processed_sessions[-1]
    session_file = output_path / f"{last_session_id}.json"

    if not session_file.exists():
        return []

    try:
        session_data = json.load(open(session_file, 'r', encoding='utf-8'))

        # fromnewfilestructureingetdialogueTranslated commenttimes
        dialogue_output = session_data.get('dialogue_output', {})
        dialogue_turns = dialogue_output.get('dialogue_turns', [])

        # Translated commentafterTranslated commentdialogue
        last_two_turns = dialogue_turns[-2:] if len(dialogue_turns) >= 2 else dialogue_turns

        # Translated commentforTranslated commentformat
        history_dialogue = []
        for turn in last_two_turns:
            history_dialogue.append({
                'speaker': turn.get('speaker', 'Unknown'),
                'content': turn.get('content', ''),
                'phase': turn.get('phase', 'unknown'),
                'timestamp': turn.get('timestamp', 0)
            })

        return history_dialogue

    except Exception as e:
        print(f"⚠️ readsession {last_session_id} failed: {str(e)}")
        return []


def get_history_dialogue_for_session(target_session_id: str, output_dir: Path) -> List[Dict[str, Any]]:
    """
    forsessiongetdialogue

    Args:
        target_session_id: goalsession ID
        output_dir: dialoguefiledirectory

    Returns:
        goalsessionbeforeafteritemshavesessionafterdialogue
    """
    # getalldialoguefileTranslated commentsession IDTranslated comment
    dialogue_files = []
    for file_path in output_dir.glob("*.json"):
        try:
            session_id = file_path.stem
            # Translated commentsession_idishaveTranslated comment（Translated commentS1_01, S1_02, S3_01etc.）
            if session_id.startswith('S') and '_' in session_id:
                parts = session_id[1:].split('_')
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    dialogue_files.append((session_id, file_path))
        except:
            continue

    # Translated commentsession IDTranslated comment（Translated commenteventID，Translated comment：S1_01, S1_02, S2_01, S3_01...）
    dialogue_files.sort(key=lambda x: (int(x[0].split('_')[0][1:]), int(x[0].split('_')[1])))

    # Translated commenttogoalsessionTranslated commentbeforeTranslated commentafterTranslated commentitemshaveTranslated commentfile
    for session_id, file_path in reversed(dialogue_files):
        if session_id < target_session_id and validate_dialogue(file_path):
            # Translated commenttogoalsessionTranslated commentbeforeTranslated commentafterTranslated commentitemshaveTranslated commentfile
            return extract_last_two_turns(file_path)

    return []  # Translated commenthaveTranslated commenttoTranslated commentdialogue


def extract_last_two_turns(dialogue_file: Path) -> List[Dict[str, Any]]:
    """Translated docstring"""
    try:
        with open(dialogue_file, 'r', encoding='utf-8') as f:
            session_data = json.load(f)

        # Translated commentnewoldfilestructure
        if 'dialogue_turns' in session_data:
            # newstructure
            dialogue_turns = session_data['dialogue_turns']
        elif 'dialogue_output' in session_data and 'dialogue_turns' in session_data['dialogue_output']:
            # oldstructure
            dialogue_turns = session_data['dialogue_output']['dialogue_turns']
        else:
            return []

        # Translated commentafterTranslated commentdialogue
        last_two_turns = dialogue_turns[-2:] if len(dialogue_turns) >= 2 else dialogue_turns

        # Translated commentforTranslated commentformat
        history_dialogue = []
        for turn in last_two_turns:
            history_dialogue.append({
                'speaker': turn.get('speaker', 'Unknown'),
                'content': turn.get('content', ''),
                'phase': turn.get('phase', 'unknown'),
                'timestamp': turn.get('timestamp', 0)
            })

        return history_dialogue

    except Exception as e:
        print(f"⚠️ dialoguefailed {dialogue_file}: {str(e)}")
        return []


def save_dialogue_output(dialogue_data: Dict[str, Any], input_data: Dict[str, Any],
                         history_dialogue: List[Dict[str, Any]], output_file: str):
    """Translated docstring"""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # createTranslated commentdebugdatastructure
    debug_data = {
        "session_id": dialogue_data.get('session_id', 'unknown'),
        "input_data": {
            "user_input_profile": input_data.get('user_input_profile', {}),
            "target_session": input_data.get('target_session', {}),
            "current_event_session_summary_list": input_data.get('current_event_session_summary_list', []),
            "history_dialogue": history_dialogue
        },
        "dialogue_output": dialogue_data,
        "metadata": {
            "file_type": "debug",
            "contains_input_data": True,
            "processing_timestamp": time.time()
        }
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(debug_data, f, ensure_ascii=False, indent=2)


def save_all_dialogues(all_dialogues: List[Dict[str, Any]], output_file: str):
    """Translated docstring"""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Translated commentsession_idTranslated comment
    all_dialogues.sort(key=lambda x: x.get('session_id', ''))

    # createTranslated commentdatastructure - Translated commentincludesdialogueresult
    production_data = {
        "sessions": [
            {
                "session_id": dialogue.get('session_id', 'unknown'),
                "dialogue_turns": dialogue.get('dialogue_turns', []),
                "goal_evaluation": dialogue.get('goal_evaluation', {}),
                "metadata": {
                    "total_turns": len(dialogue.get('dialogue_turns', [])),
                    "processing_time": dialogue.get('metadata', {}).get('processing_time', 0),
                    "processor_type": dialogue.get('metadata', {}).get('processor_type', 'multi_agent_dialogue')
                }
            }
            for dialogue in all_dialogues
        ],
        "summary": {
            "total_sessions": len(all_dialogues),
            "total_dialogue_turns": sum(len(d.get('dialogue_turns', [])) for d in all_dialogues),
            "average_turns_per_session": sum(len(d.get('dialogue_turns', [])) for d in all_dialogues) / len(all_dialogues) if all_dialogues else 0,
            "average_goal_completion": sum(d.get('goal_evaluation', {}).get('overall_score', 0) for d in all_dialogues) / len(all_dialogues) if all_dialogues else 0
        },
        "metadata": {
            "file_type": "production",
            "generation_timestamp": time.time(),
            "generated_at": time.strftime('%Y-%m-%d %H:%M:%S'),
            "contains_input_data": False
        }
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(production_data, f, ensure_ascii=False, indent=2)


def generate_events(person_data: Dict[str, Any], blueprint: Dict[str, Any], project_attributes: List[str], model: str = None) -> Optional[Dict[str, Any]]:
    """
    forprojectblueprintgenerateeventcolumn

    Args:
        person_data: roledata
        blueprint: projectblueprint
        project_attributes: projectattributelist
        model: LLMname

    Returns:
        eventdataorNone
    """
    person_name = person_data.get('name', 'unknown_person')

    try:
        # Use provided model or default
        if model is None:
            model = "gemini-2.5-pro-thinking-*"

        # initializeLLMTranslated comment
        llm_client = create_client(api_key=api_key, base_url=base_url, model=model, 
        #reasoning_effort="disable"
        )
        print("✅ Event LLM client initialized")

        # initializeEventProcessor
        event_processor = EventProcessor(
            llm_client=llm_client,
            checkpoint_dir="output/checkpoints/events"
        )
        print("✅ EventProcessor initialized")

        # Translated commentinputdata
        processor_input = {
            "user_profile": person_data,
            "project_state": {
                "current_stage": "planning",
                "progress": 0,
                "milestones_completed": [],
                "current_focus": blueprint.get("project_goal", "")
            },
            "project_blueprint": blueprint
        }

        # processeventgenerate
        result = event_processor.process(
            data=processor_input,
            use_checkpoint=False
        )

        if result.success:
            print("✅ Events generated successfully!")
            # getLLMreturnoriginaldata
            raw_events_data = result.data

            # Translated commentformat
            if isinstance(raw_events_data, list):
                # Translated commentreturniseventarray，Translated commentincludeseventsfieldobject
                events_data = {
                    "events": raw_events_data,
                    "_metadata": {
                        "person_name": person_name,
                        "persona_role": person_data.get('role', 'Unknown'),
                        "blueprint_topic": blueprint.get('selected_topic', ''),
                        "total_attributes_used": len(project_attributes),
                        "event_count": len(raw_events_data)
                    }
                }
            elif isinstance(raw_events_data, dict):
                # Translated commentreturnisobject，addTranslated commentdata
                if "events" in raw_events_data:
                    events_data = raw_events_data
                else:
                    # Translated commentobjectinTranslated commenthaveeventsfield，Translated commentitemsobjectTranslated commentiseventdata
                    events_data = {"events": [raw_events_data]}

                events_data["_metadata"] = {
                    "person_name": person_name,
                    "persona_role": person_data.get('role', 'Unknown'),
                    "blueprint_topic": blueprint.get('selected_topic', ''),
                    "total_attributes_used": len(project_attributes),
                    "event_count": len(events_data.get('events', []))
                }
            else:
                print(f"❌ Unexpected events data format: {type(raw_events_data)}")
                return None

            # Generate project identifier for directory structure
            topic_name = blueprint.get('selected_topic', 'unknown_topic')
            task_id = blueprint.get('selected_task_id', 'unknown')
            safe_topic_name = convert_to_safe_filename(topic_name)
            safe_person_name = convert_to_safe_filename(person_name)
            project_identifier = f"{safe_topic_name}_{task_id}"

            # Save events data to new directory structure
            events_output_dir = Path(f"output/{safe_person_name}/{project_identifier}/project_events")
            events_output_dir.mkdir(parents=True, exist_ok=True)

            events_filename = f"{project_identifier}_events.json"
            events_filepath = events_output_dir / events_filename

            with open(events_filepath, 'w', encoding='utf-8') as f:
                json.dump(events_data, f, ensure_ascii=False, indent=2)

            print(f"✅ Events saved to: {events_filepath}")

            # Link events to blueprint
            blueprint['events_file'] = str(events_filepath)

            return events_data
        else:
            print(f"❌ Failed to generate events: {result.error_message}")
            return None

    except Exception as e:
        print(f"❌ Error generating events: {str(e)}")
        return None


def get_assigned_tasks(person_name: str) -> List[Dict[str, Any]]:
    """Translated docstring"""
    assigned_tasks = []
    for person_entry in PERSON_GOAL_DATA:
        if person_entry.get('persona_name') == person_name:
            assigned_tasks = person_entry.get('assigned_tasks', [])
            break
    return assigned_tasks


def check_person_progress(person_name: str) -> List[Dict[str, Any]]:
    """Translated docstring"""
    safe_person_name = convert_to_safe_filename(person_name)
    assigned_tasks = get_assigned_tasks(person_name)

    results = []

    for task in assigned_tasks:
        topic_id = task.get('topic_id')
        task_id = task.get('task_id')

        # gettopicinformationTranslated commentbuildproject_identifier
        topic_name = get_topic_name_by_id(topic_id)

        # Debug: checktask_idhaveTranslated comment
        if task_id is None or task_id == "":
            print(f"⚠️ Warning in check_person_progress: task_id is invalid for {person_name}")
            print(f"   Topic: {topic_id} ({topic_name})")
            print(f"   Task data: {task}")

        project_identifier = f"{convert_to_safe_filename(topic_name)}_{task_id}"

        project_dir = Path(f"output/{safe_person_name}/{project_identifier}")

        if not project_dir.exists():
            results.append({
                "task": task,
                "project_identifier": project_identifier,
                "status": "not_started",
                "progress": 0,
                "message": "projectbegin"
            })
        else:
            progress = check_project_progress(safe_person_name, project_identifier)
            progress["task"] = task
            progress["project_identifier"] = project_identifier

            # addstatusdescription
            status_messages = {
                "completed": "projectcomplete",
                "need_blueprint": "needwantproject blueprint",
                "blueprint_incomplete": "projectblueprint",
                "blueprint_corrupted": "projectblueprint",
                "need_events": "needwantGenerate events",
                "events_incomplete": "event",
                "events_corrupted": "event",
                "need_summaries": "needwantGenerate session summaries",
                "summaries_incomplete": "sessionsummary",
                "summaries_corrupted": "sessionsummary",
                "need_dialogues": "needwantGenerate dialogues",
                "dialogues_incomplete": "dialoguecomplete",
                "dialogue_corrupted": "dialoguefile",
                "need_project_memory": "needwantsaveprojectmemory"
            }
            progress["message"] = status_messages.get(progress["status"], "status")
            results.append(progress)

    return results


def smart_full_process(person, projects_to_generate=1, args=None):
    """Translated docstring"""
    person_data = json.loads(person) if isinstance(person, str) else person
    person_name = person_data.get('name', 'unknown_person')

    print(f"\n=== canprocess: {person_name} ===")

    # checkallprojectTranslated comment
    progress_results = check_person_progress(person_name)
    
    #print(progress_results)

    #return 

    print(f"\n📊 project:")
    for i, result in enumerate(progress_results):
        task = result["task"]
        status = result["status"]
        progress = result.get("progress", 0)
        message = result.get("message", "")
        topic_name = get_topic_name_by_id(task.get('topic_id'))
        task_id = task.get('task_id')

        print(f"  {i+1}. {topic_name} ({task_id}) - {progress}% - {message}")

    print(f"\n🔄 Starting processingcompleteproject...")

    for result in progress_results:
        task = result["task"]
        status = result["status"]
        project_identifier = result["project_identifier"]

        topic_name = get_topic_name_by_id(task.get('topic_id'))
        task_id = task.get('task_id')
        selected_goal = task.get('description') or task.get('title')

        print(f"\n=== processproject: {topic_name} ({task_id}) ===")
        print(f"status: {result.get('message', status)} (: {result.get('progress', 0)}%)")

        if status == "completed":
            print("✅ projectcomplete，")
            continue

        elif status == "not_started":
            print("🆕 beginnewproject")
            process_project_from_start(person_data, task, args)

        elif status in ["need_blueprint", "blueprint_incomplete", "blueprint_corrupted"]:
            print("🔄 fromblueprintsectionbegin")
            process_blueprint_stage(person_data, task, args)

        elif status in ["need_events", "events_incomplete", "events_corrupted"]:
            print("🔄 fromeventssectionbegin")
            process_events_stage(person_data, task, args)

        elif status in ["need_summaries", "summaries_incomplete", "summaries_corrupted"]:
            print("🔄 fromsummariessectionbegin")
            process_summaries_stage(person_data, task, args)
    
    # beforeTranslated commentdataTranslated comment - Translated commentdialoguegenerateproject
    print(f"\n🔍 validationallprojectprocesscompletestatus...")
    
    #return 

    ready_projects = []
    incomplete_projects = []

    for result in progress_results:
        status = result["status"]

        # Translated commentnewvalidationprojectstatus
        safe_person_name = convert_to_safe_filename(person_name)
        current_progress = check_project_progress(safe_person_name, result["project_identifier"])

        if current_progress["status"] in ["need_dialogues", "dialogues_incomplete", "dialogue_corrupted"]:
            ready_projects.append(result)
            print(f"  ✅ {result['project_identifier']}: dialoguegenerate")
        elif current_progress["status"] == "completed":
            print(f"  ✅ {result['project_identifier']}: complete")
        else:
            incomplete_projects.append(result)
            print(f"  ❌ {result['project_identifier']}: statusfor {current_progress['status']}，")

    if incomplete_projects:
        print(f"\n⚠️  {len(incomplete_projects)} itemsprojectcompleteprocess，dialoguegenerate")

    # processdialoguegenerate - Translated commentprojectTranslated comment
    if ready_projects:
        process_dialogues_stage(person_data, ready_projects, args)
    else:
        print(f"\n❌ havedialoguegenerateproject")


    print(f"\n✅ allprojectprocesscomplete！")


def process_project_from_start(person_data: Dict[str, Any], task: Dict[str, Any], args=None):
    """Translated docstring"""
    person_name = person_data.get('name', 'unknown_person')
    topic_id = task.get('topic_id')
    task_id = task.get('task_id')
    selected_goal = task.get('description') or task.get('title')

    # Get topic information
    topics = TOPIC_ATTR_DATA['topics']
    topic_map = {topic['topic_id']: topic for topic in topics}
    selected_topic = topic_map.get(topic_id)

    if not selected_topic:
        print(f"❌ Topic with ID '{topic_id}' not found")
        return

    topic_name = selected_topic['topic_name']
    project_attributes = selected_topic.get('project_attributes', [])

    print(f"    Processing new project: {topic_name} - {selected_goal[:50]}...")

    # Debug: checktask_idisTranslated commenthaveTranslated comment
    if not task_id:
        print(f"⚠️ Warning: task_id is empty or None: {task_id}")
        print(f"    Task data: {task}")

    # Process all stages sequentially
    # 1. Generate blueprint
    blueprint = generate_project_blueprint(
        person_data,
        project_attributes,
        selected_goal,
        model=args.blueprint_model if args else None,
        topic_name=topic_name,
        selected_task_id=task_id
    )

    if not blueprint:
        print(f"    ❌ Failed to generate blueprint")
        return

    # 2. Generate events
    events_data = generate_events(person_data, blueprint, project_attributes, model=args.event_model if args else None)
    if not events_data:
        print(f"    ❌ Failed to generate events")
        return

    # 3. Generate session summaries
    summary_data = generate_session_summaries(person_data, blueprint, events_data, model=args.summary_model if args else None)
    if not summary_data:
        print(f"    ❌ Failed to generate session summaries")
        return

    # 4. Generate dialogues
    # [NEW] loadTranslated commenthaveTranslated commentdata
    person_name = person_data.get('name', 'unknown_person')
    topic_name = selected_topic.get('topic_name', 'unknown_topic') if 'selected_topic' in blueprint else get_topic_name_by_id(topic_id)
    task_id = blueprint.get('selected_task_id', task_id)
    safe_topic_name = convert_to_safe_filename(topic_name)
    project_identifier = f"{safe_topic_name}_{task_id}"

    # readTranslated commenthaveTranslated commentdata
    person_schedule = get_person_schedule(person_name)
    current_time = person_schedule.get("current_date", "")  # usecurrent_datefield
    current_plan_items = extract_plan_items_content(person_schedule)

    if current_time or current_plan_items:
        print(f"    📅 Loaded existing schedule: date='{current_time}', plan_items={len(current_plan_items)}")

    dialogue_data = generate_dialogues(person_data, blueprint, events_data, summary_data,
                                      dialogue_model=args.dialogue_model if args else None,
                                      evaluation_model=args.evaluation_model if args else None,
                                      memory_model=args.memory_model if args else None,
                                      memory_retrieve_model=args.memory_retrieve_model if args else None,
                                      dedup_model=args.dedup_model if args else None,
                                      semantic_schedule_model=getattr(args, 'semantic_schedule_model', None) if args else None,
                                      max_turns=args.max_turns if args else 12,
                                      max_retries=getattr(args, 'max_retries', 2) if args else 2,
                                      current_time=current_time,
                                      current_plan_items=current_plan_items)

    if not dialogue_data:
        print(f"    ❌ Failed to generate dialogues")
        return

    print(f"    ✅ Project completed successfully!")


def process_blueprint_stage(person_data: Dict[str, Any], task: Dict[str, Any], args=None):
    """Translated docstring"""
    person_name = person_data.get('name', 'unknown_person')
    topic_id = task.get('topic_id')
    task_id = task.get('task_id')
    selected_goal = task.get('description') or task.get('title')

    # Get topic information
    topics = TOPIC_ATTR_DATA['topics']
    topic_map = {topic['topic_id']: topic for topic in topics}
    selected_topic = topic_map.get(topic_id)

    if not selected_topic:
        print(f"❌ Topic with ID '{topic_id}' not found")
        return

    topic_name = selected_topic['topic_name']
    project_attributes = selected_topic.get('project_attributes', [])

    print(f"    Regenerating blueprint for: {topic_name}")

    blueprint = generate_project_blueprint(
        person_data,
        project_attributes,
        selected_goal,
        model=args.blueprint_model if args else None,
        topic_name=topic_name,
        selected_task_id=task_id
    )

    if blueprint:
        print(f"    ✅ Blueprint generated successfully!")
        # Continue with remaining stages
        process_events_stage(person_data, task, args)
    else:
        print(f"    ❌ Failed to generate blueprint")


def process_events_stage(person_data: Dict[str, Any], task: Dict[str, Any], args=None):
    """Translated docstring"""
    person_name = person_data.get('name', 'unknown_person')
    topic_name = get_topic_name_by_id(task.get('topic_id'))
    task_id = task.get('task_id')
    project_identifier = f"{convert_to_safe_filename(topic_name)}_{task_id}"

    # Load existing blueprint
    safe_person_name = convert_to_safe_filename(person_name)
    blueprint_file = Path(f"output/{safe_person_name}/{project_identifier}/project_blueprints/{project_identifier}_blueprint.json")

    try:
        with open(blueprint_file, 'r', encoding='utf-8') as f:
            blueprint = json.load(f)
        print(f"    Loaded existing blueprint")
    except Exception as e:
        print(f"    ❌ Failed to load blueprint: {str(e)}")
        return

    # Generate events
    topic_id = task.get('topic_id')
    topics = TOPIC_ATTR_DATA['topics']
    topic_map = {topic['topic_id']: topic for topic in topics}
    selected_topic = topic_map.get(topic_id)
    project_attributes = selected_topic.get('project_attributes', []) if selected_topic else []

    print(f"    Generating events for: {topic_name}")

    events_data = generate_events(person_data, blueprint, project_attributes, model=args.event_model if args else None)

    if events_data:
        print(f"    ✅ Events generated successfully!")
        # Continue with remaining stages
        process_summaries_stage(person_data, task, args)
    else:
        print(f"    ❌ Failed to generate events")


def process_summaries_stage(person_data: Dict[str, Any], task: Dict[str, Any], args=None):
    """Translated docstring"""
    person_name = person_data.get('name', 'unknown_person')
    topic_name = get_topic_name_by_id(task.get('topic_id'))
    task_id = task.get('task_id')
    project_identifier = f"{convert_to_safe_filename(topic_name)}_{task_id}"

    # Load existing blueprint and events
    safe_person_name = convert_to_safe_filename(person_name)
    blueprint_file = Path(f"output/{safe_person_name}/{project_identifier}/project_blueprints/{project_identifier}_blueprint.json")
    events_file = Path(f"output/{safe_person_name}/{project_identifier}/project_events/{project_identifier}_events.json")

    try:
        with open(blueprint_file, 'r', encoding='utf-8') as f:
            blueprint = json.load(f)
        with open(events_file, 'r', encoding='utf-8') as f:
            events_data = json.load(f)
        print(f"    Loaded existing blueprint and events")
    except Exception as e:
        print(f"    ❌ Failed to load existing files: {str(e)}")
        return

    print(f"    Generating session summaries for: {topic_name}")

    summary_data = generate_session_summaries(person_data, blueprint, events_data, model=args.summary_model if args else None)

    if summary_data:
        print(f"    ✅ Session summaries generated successfully!")
        # Continue with remaining stages
        # process_dialogues_stage(person_data, task, args)
    else:
        print(f"    ❌ Failed to generate session summaries")


def process_dialogues_stage(person_data: Dict[str, Any], ready_projects: List[Dict[str, Any]], args=None):
    """
    processdialoguessection，projectdialoguegenerate

    Args:
        person_data: persondata
        ready_projects: dialoguegenerateprojectlist
        args: parameterconfig
    """
    person_name = person_data.get('name', 'unknown_person')
    safe_person_name = convert_to_safe_filename(person_name)

    print(f"\n🗣️ Starting to generate for {len(ready_projects)} itemsprojectgeneratedialogue...")

    # Translated commentallprojectdialoguedata
    project_dialogue_data = {}

    for project_result in ready_projects:
        task = project_result["task"]
        topic_name = get_topic_name_by_id(task.get('topic_id'))
        task_id = task.get('task_id')
        project_identifier = f"{convert_to_safe_filename(topic_name)}_{task_id}"

        print(f"  📁 loadprojectdata: {project_identifier}")

        # Load existing blueprint, events, and summaries
        blueprint_file = Path(f"output/{safe_person_name}/{project_identifier}/project_blueprints/{project_identifier}_blueprint.json")
        events_file = Path(f"output/{safe_person_name}/{project_identifier}/project_events/{project_identifier}_events.json")
        summaries_file = Path(f"output/{safe_person_name}/{project_identifier}/session_summaries/{project_identifier}_summary.json")

        try:
            with open(blueprint_file, 'r', encoding='utf-8') as f:
                blueprint = json.load(f)
            with open(events_file, 'r', encoding='utf-8') as f:
                events_data = json.load(f)
            with open(summaries_file, 'r', encoding='utf-8') as f:
                summary_data = json.load(f)

            # checkdialoguecompletestatus
            dialogues_dir = Path(f"output/{safe_person_name}/{project_identifier}/dialogues")
            dialogue_completeness = validate_dialogues_completeness(dialogues_dir, summary_data)

            project_dialogue_data[project_identifier] = {
                "task": task,
                "blueprint": blueprint,
                "events_data": events_data,
                "summary_data": summary_data,
                "dialogue_completeness": dialogue_completeness,
                "sessions_to_generate": dialogue_completeness.get("sequence_integrity", {}).get("sessions_to_regenerate", []),
                "completed_sessions": dialogue_completeness.get("existing_session_ids", [])
            }

            print(f"    ✅ {project_identifier}: {len(dialogue_completeness.get('sequence_integrity', {}).get('sessions_to_regenerate', []))} itemsdialoguegenerate")

        except Exception as e:
            print(f"    ❌ {project_identifier}: loadfailed - {str(e)}")
            continue

    if not project_dialogue_data:
        print(f"    ❌ havecanprojectdata")
        return

    # usemodifyaftergenerate_dialoguesfunctionTranslated commentrowTranslated commentprocess
    dialogue_results = generate_dialogues_interleaved(person_data, project_dialogue_data, args)

    if dialogue_results:
        total_generated = sum(len(result.get("sessions", [])) for result in dialogue_results.values())
        print(f"    ✅ dialoguegeneratecomplete！generate {total_generated} itemsdialogue")
    else:
        print(f"    ❌ Dialogue generation failed")


def process_project_memory_stage(person_data: Dict[str, Any], task: Dict[str, Any], args=None):
    """processproject memorysave"""
    person_name = person_data.get('name', 'unknown_person')
    topic_name = get_topic_name_by_id(task.get('topic_id'))
    task_id = task.get('task_id')
    project_identifier = f"{convert_to_safe_filename(topic_name)}_{task_id}"

    # Load current memory
    current_memory = get_person_memory(person_name)

    # Save project memory
    project_memory_file = save_project_memory(person_name, project_identifier, current_memory)

    # Save person memory
    safe_person_name = convert_to_safe_filename(person_name)
    person_memory_file = save_person_memory(person_name, f"output/{safe_person_name}")

    print(f"    ✅ Project memory saved to: {project_memory_file}")
    print(f"    ✅ Person memory saved to: {person_memory_file}")

def main():
    """Translated docstring"""

    read_persona_topic_files()
    args = parse_arguments()

    # PERSONA_DATA is a list, so we just get the length
    available_persons_count = len(PERSONA_DATA)
    print(f"Available persons: {available_persons_count}")

    # Translated commenthaveTranslated comment，Translated commentuseTranslated commentitems
    if args.names is None:
        name = PERSONA_DATA[0].get('name', 'Unknown')
        print(f"No names specified, using default: {name}")
    else:
        name = args.names  # Translated commentusestring

    # createnameto persona Translated comment
    persona_map = {persona.get('name'): persona for persona in PERSONA_DATA}

    # processTranslated commentuser
    if name not in persona_map:
        print(f"⚠️ Person '{name}' not found in persona data")
        return

    person = persona_map[name]
    print(f"\n=== Processing person: {name} ===")

    # Generate specified number of projects for this person
    if args.smart_recovery:
        print("🔄 canpattern")
        smart_full_process(person, projects_to_generate=args.projects, args=args)

    


if __name__ == "__main__":
    main()