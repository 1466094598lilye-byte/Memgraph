import json
from pathlib import Path
from typing import List, Dict, Any, Set
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()

def validate_dialogue_relevance(dialogue_file: str) -> List[Dict[str, Any]]:
    """
    Validate dialogue relevance between user queries and assistant memory usage.
    Returns a list of unreasonable turn details with reasoning.
    """
    prompt_template = """# Role
You are a strict Data Quality Auditor for a Retrieval-Augmented Generation (RAG) system. Your specific task is to evaluate the logical connection between a **User Query**, a **Retrieved Memory**, and the **Assistant Response**.

# Input Data
<user_query>
{user_content}
</user_query>

<assistant_response>
{assistant_content}
</assistant_response>

<memory_used>
{memory_used}
</memory_used>

# Evaluation Task
Determine if this interaction is **REASONABLE** (Keep) or **UNREASONABLE** (Filter Out).

# Rejection Criteria (The "Filter" Rules)
Mark the interaction as `is_reasonable: false` if ANY of the following are true:

1.  **Irrelevant Memory:** The content in `<memory_used>` is completely unrelated to the topic, intent, or context of the `<user_query>`. (e.g., User talks about schedule, Memory talks about diet restrictions).
2.  **Forced Association:** The `<memory_used>` is technically related to the topic but does not fit the specific context of the current turn, yet the Assistant tries to force it in awkwardly.
3.  **Memory Ignored/Hallucinated:** The Assistant claims to use the memory (it appears in the list), but the actual response contradicts the memory or completely ignores a critical instruction found in the memory.
4.  **Logic Break:** The Assistant's response follows the memory but makes no sense given the User's query.

# Acceptance Criteria
Mark as `is_reasonable: true` ONLY if:
1.  The memory is directly relevant to the user's specific input.
2.  The assistant naturally integrates the memory's insight to provide a helpful response.

# Output Format
Output ONLY a JSON object. Do not output markdown blocks.

{{
  "reasoning": "Brief explanation of the relationship between Query, Memory, and Response.",
  "error_type": "None" | "Irrelevant Memory" | "Forced Association" | "Contradiction",
  "is_reasonable": true | false
}}"""

    with open(dialogue_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_evaluations: List[Dict[str, Any]] = []

    if isinstance(data, dict):
        sessions = data.get('dialogues', [])
    elif isinstance(data, list):
        sessions = data
    else:
        return all_evaluations

    for session in tqdm(sessions, desc="Processing sessions"):
        dialogue_turns = session.get('dialogue_turns', [])

        i = 0
        while i < len(dialogue_turns) - 1:
            current = dialogue_turns[i]
            next_turn = dialogue_turns[i + 1]

            if (current.get('speaker') == 'User' and
                next_turn.get('speaker') == 'Assistant' and
                'memory_used' in next_turn):

                user_content = current.get('content', '')
                assistant_content = next_turn.get('content', '')
                memory_used = next_turn.get('memory_used', [])

                prompt = prompt_template.format(
                    user_content=user_content[:2000],
                    assistant_content=assistant_content[:3000],
                    memory_used=json.dumps(memory_used, ensure_ascii=False)
                )

                from openai import OpenAI
                client = OpenAI()

                try:
                    response = client.chat.completions.create(
                        model="gemini-2.5-flash",
                        messages=[
                            {"role": "system", "content": "You are a precise JSON evaluator. Always return valid JSON only."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.3,
                        response_format={"type": "json_object"}
                    )

                    result = json.loads(response.choices[0].message.content)

                    evaluation_record = {
                        "timestamp": current['timestamp'],
                        "session_identifier": session.get('session_identifier', 'unknown'),
                        "user_content": user_content[:500],
                        "assistant_content": assistant_content[:500],
                        "memory_used": memory_used,
                        "is_reasonable": result.get('is_reasonable', True),
                        "reasoning": result.get('reasoning', ''),
                        "error_type": result.get('error_type', 'None')
                    }
                    all_evaluations.append(evaluation_record)

                except Exception as e:
                    print(f"Error processing turn {i}: {e}")
                    continue

            i += 1

    return all_evaluations


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python dialogue_validator.py <dialogue_file.json>")
        sys.exit(1)

    dialogue_file = sys.argv[1]

    if not Path(dialogue_file).exists():
        print(f"Error: File {dialogue_file} not found")
        sys.exit(1)

    all_evaluations = validate_dialogue_relevance(dialogue_file)

    unreasonable_count = sum(1 for e in all_evaluations if not e['is_reasonable'])

    result = {
        "dialogue_file": dialogue_file,
        "total_evaluated_turns": len(all_evaluations),
        "total_unreasonable_turns": unreasonable_count,
        "total_reasonable_turns": len(all_evaluations) - unreasonable_count,
        "evaluations": all_evaluations
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))

    output_file = dialogue_file.replace('.json', '_validation.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nValidation results saved to: {output_file}")


if __name__ == "__main__":
    main()
