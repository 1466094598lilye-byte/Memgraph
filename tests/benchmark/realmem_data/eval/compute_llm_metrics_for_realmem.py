import json
import argparse
import numpy as np
import os
import sys
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

# Add project root to Python path
PROJECT_ROOT = os.path.expanduser("~/MemInsight")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def parse_args():
    parser = argparse.ArgumentParser(description="Compute RealMem metrics (LLM Judge)")
    parser.add_argument('--data_file', type=str, 
                        default=f'{PROJECT_ROOT}/data/realmem/input_data/Lin_Wanyu_dialogues_256k.json',
                        help='Input file containing dialogues (Ground Truth source)')
    parser.add_argument('--gen_file', type=str,
                        default=f'{PROJECT_ROOT}/data/realmem/Lin_Wanyu/graph_4omini-4omini_top20_entity_generation_results.json',
                        help='Input file containing generation results (JSON format)')
    parser.add_argument('--out_file', type=str, 
                        default=None,
                        help='Output file to save detailed metrics')
    
    # LLM Judge args
    parser.add_argument('--model_name', default='gpt-4o', help="Model name for judge.")
    parser.add_argument('--base_url', default=None, help="Base URL for the API.")
    parser.add_argument('--api_key', default=None, help="API Key for the client (set via env OPENAI_API_KEY).")
    parser.add_argument('--max_workers', type=int, default=10, help="Concurrency for LLM judge.")
    
    # New args for mem evaluation
    parser.add_argument('--eval_mem_type', default='chunk', help="Type of item to use for mem evaluation (default: chunk).")
    parser.add_argument('--top_k', type=int, default=5, help="Number of top items to use for mem evaluation.")
    parser.add_argument('--run_all', action='store_true', help='Run metrics computation for all datasets found.')
    parser.add_argument('--gen_model_part', default='4omini-4o', help='Model part of the generation filename (e.g., 4omini-4o) used for batch processing.')
    parser.add_argument('--retrieval_result_dir', type=str, default='eval/retrieval_result',
                        help='Directory containing retrieval result subdirectories to process in batch mode.')
    parser.add_argument('--process_retrieval_results', action='store_true',
                        help='Process all generation_results.json files from retrieval_result_dir.')
    parser.add_argument('--input_data_dir', type=str, default='datasets',
                        help='Directory containing input dialogue files (for batch mode).')

    return parser.parse_args()

QA_eval_prompt = """Your task is to evaluate the consistency between the [candidate answer] and the [user-related memory].

You will be given four pieces of information:
1. The user’s current query
2. The user-related memory, representing the latest valid user state
3. A reference answer based on the relevant memory
4. The candidate answer to be evaluated

Please follow these rules during evaluation:
- Focus only on whether “facts, constraints, preferences, and confirmed states” are correctly used
- Do NOT evaluate language style, tone, politeness, empathy, or fluency
- Do NOT give a high score just because the answer “sounds reasonable”
- The reference answer is only to help understand how relevant memory should ideally be used; a candidate answer does not need to exactly match the reference answer to receive a full score

Scoring criteria:
Score 0: Poor — the candidate answer conflicts with the user-related memory  
Score 1: Fair — the candidate answer does not conflict with the relevant memory but is generic and not based on user memory  
Score 2: Good — the candidate answer uses part of the user-related memory  
Score 3: Very good — the candidate answer (like the reference answer) uses all of the user-related memory

Output format:
```json
{
    "score": int,
    "reason": str  # Briefly explain the reason for the score
}
```
"""

Mem_eval_prompt = """Your task is to evaluate the consistency between the [retrieved memory] and the [ground-truth memory], and whether the retrieved memory is helpful.

### Input Data
• <question>: {question}
• <groundtruth_memory>: {groundtruth_memory}
• <retrieved_memory>: {retrieved_memory}

---

### Evaluation Dimensions
#### 1. Memory Recall
Mem_recall: Semantics-aware memory recall calculation (0–1)  
step1: For each groundtruth_memory, check in sequence whether its semantics are contained in any retrieved_memory.  
step2: Count how many groundtruth_memory items are covered (hits_cnt).  
step3: Compute the final recall score as hits_cnt / total number of groundtruth_memory items.

#### 2. Memory Helpfulness
Mem_helpful_score: The helpfulness of the retrieved memory for answering the question  
Score 0: retrieved_memory contains mutually conflicting or contradictory memories, which not only fail to help answer the question but may also cause confusion.  
Score 1: retrieved_memory is somewhat helpful for answering the question (can provide partial supporting evidence).  
Score 2: retrieved_memory is very helpful for answering the question (can provide comprehensive supporting evidence).

---

### Output Format
Please provide your evaluation results using the following structure:

```json
{{
  "Mem_recall": float,
  "Mem_helpful_score": int,
  "Mem_hits": list[str],  # List the matched groundtruth_memory items
  "Mem_helpful_reason": str  # Explain the reason for the score and list the retrieved_memory that helps answer the question
}}
```
"""

def extract_json_from_text(text):
    try:
        # Try finding JSON block
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        # Try finding brace enclosed JSON
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(text)
    except:
        return None

def construct_evidence_text(ranked_items, evidence_type='chunk', top_k=5):
    # Filter items
    filtered = [item for item in ranked_items if item.get('res_type') == evidence_type]
    
    selected = filtered[:top_k]
    
    evidence_texts = []
    for i, item in enumerate(selected):
        content = item.get('content', '').strip()
        # Add metadata if needed
        if item.get('res_type') == 'entity':
            content = f"{item['entity_name']}: {content}"
        evidence_texts.append(f"---- idx {i+1} ----\n{content}")
    
    return "\n\n".join(evidence_texts)

def load_ground_truth(file_path):
    """
    Load ground truth from dialogues file.
    Returns: dict { question_text: { 'answer': str, 'memory': str } }
    """
    print(f"Loading ground truth from {file_path}...")
    gt_map = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        dialogues = data.get('dialogues', [])
        for dialogue in dialogues:
            turns = dialogue.get('dialogue_turns', [])
            # Iterate through turns to find User -> Assistant pairs
            for i in range(len(turns)):
                turn = turns[i]
                if turn['speaker'] == 'User':
                    question = turn.get('content', '').strip()
                    if not question:
                        continue
                        
                    # Find next Assistant turn
                    j = i + 1
                    answer = ""
                    memory_str = ""
                    
                    while j < len(turns) and turns[j]['speaker'] != 'Assistant':
                        j += 1
                    
                    if j < len(turns):
                        asst_turn = turns[j]
                        answer = asst_turn.get('content', '').strip()
                        
                        # Extract memory from 'memory_used'
                        memory_used = asst_turn.get('memory_used', [])
                        if memory_used:
                            # Join the content of all memory items
                            mem_contents = [m.get('content', '') for m in memory_used]
                            memory_str = "\n".join(filter(None, mem_contents))
                    
                    if not memory_str:
                         memory_str = "No specific memory annotation found."

                    gt_map[question] = {
                        'answer': answer,
                        'memory': memory_str
                    }
    except Exception as e:
        print(f"Error loading ground truth: {e}")
        
    print(f"Loaded {len(gt_map)} QA pairs from dialogues.")
    return gt_map

def run_llm_judge(client, model, prompt):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        content = response.choices[0].message.content
        return extract_json_from_text(content)
    except Exception as e:
        print(f"Error in LLM judge: {e}")
        return None

def evaluate_single_item(qid, question, generated_answer, ranked_items, gt_info, client, model, eval_mem_type, top_k):
    if not gt_info:
        return qid, None
    
    gt_memory = gt_info.get('memory', '')
    gt_answer = gt_info.get('answer', '')
    
    results = {}

    # 1. QA Evaluation
    qa_formatted_prompt = f"""{QA_eval_prompt}

### Input Data
1. Query: {question}
2. User-related Memory: {gt_memory}
3. Reference Answer: {gt_answer}
4. Candidate Answer: {generated_answer}
"""
    qa_result = run_llm_judge(client, model, qa_formatted_prompt)
    if qa_result:
        results.update(qa_result)
        
    # 2. Mem Evaluation
    # Use ranked_items to construct evidence text for evaluation
    evidence_text = construct_evidence_text(ranked_items, eval_mem_type, top_k)
    
    # Only run if we have evidence_text and gt_memory (and gt_memory is not just the placeholder)
    if evidence_text and gt_memory and gt_memory != "No specific memory annotation found.":
        mem_formatted_prompt = Mem_eval_prompt.format(
            question=question,
            groundtruth_memory=gt_memory,
            retrieved_memory=evidence_text
        )
        mem_result = run_llm_judge(client, model, mem_formatted_prompt)
        if mem_result:
             results.update(mem_result)

    return qid, results

def main(args):
    # Load Ground Truth
    gt_map = load_ground_truth(args.data_file)
    
    # Load Generation Results
    print(f"Loading generation results from {args.gen_file}...")
    with open(args.gen_file, 'r', encoding='utf-8') as f:
        gen_data = json.load(f)
    
    # gen_data is { question_text: item_dict }
    items_to_judge = []
    
    for q_text, item in gen_data.items():
        # Match with GT
        # item has 'generated_answer', 'question' (should match q_text), 'evidence_used', 'ranked_items'
        
        # Try exact match
        match_q = None
        if q_text in gt_map:
            match_q = q_text
        else:
            # Maybe try stripped?
            q_stripped = q_text.strip()
            if q_stripped in gt_map:
                match_q = q_stripped
        
        if match_q:
            items_to_judge.append({
                'id': q_text, # Using question text as ID
                'question': match_q,
                'generated_answer': item.get('generated_answer', ''),
                'ranked_items': item.get('ranked_items', []),
                'gt_info': gt_map[match_q]
            })
    
    print(f"Found {len(items_to_judge)} items to judge out of {len(gen_data)} generation results.")
    
    if len(items_to_judge) == 0:
        print("No matching items found. Exiting.")
        return

    # LLM Judge
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
    client = OpenAI(api_key=api_key, base_url=args.base_url)
    
    detailed_results = {}
    
    # Metrics storage
    qa_scores = []
    mem_recalls = []
    mem_helpful_scores = []
    
    print(f"Running LLM Judge on {len(items_to_judge)} items with {args.max_workers} threads...")
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(evaluate_single_item, item['id'], item['question'], item['generated_answer'], item['ranked_items'], item['gt_info'], client, args.model_name, args.eval_mem_type, args.top_k): item['id']
            for item in items_to_judge
        }
        
        for future in tqdm(as_completed(futures), total=len(items_to_judge), desc="LLM Judge"):
            qid, res = future.result()
            if res:
                detailed_results[qid] = res
                
                # QA Score
                if 'score' in res and isinstance(res['score'], (int, float)):
                    qa_scores.append(res['score'])
                
                # Mem Metrics
                if 'Mem_recall' in res and isinstance(res['Mem_recall'], (int, float)):
                    mem_recalls.append(res['Mem_recall'])
                if 'Mem_helpful_score' in res and isinstance(res['Mem_helpful_score'], (int, float)):
                    mem_helpful_scores.append(res['Mem_helpful_score'])
    
    # Metrics Summary
    summary = {}
    
    # QA Summary
    if qa_scores:
        avg_score = np.mean(qa_scores)
        summary['average_qa_score'] = round(float(avg_score), 4)
        dist = {i: qa_scores.count(i) for i in range(4)}
        summary['qa_score_distribution'] = dist
        summary['qa_hallucination_rate'] = round(dist.get(0, 0) / len(qa_scores), 4)
        summary['qa_perfect_rate'] = round(dist.get(3, 0) / len(qa_scores), 4)
        
        print(f"\nAverage QA Score: {avg_score:.4f}")
        print(f"QA Score Distribution: {dist}")
    
    # Mem Summary
    if mem_recalls:
        avg_recall = np.mean(mem_recalls)
        summary['average_mem_recall'] = round(float(avg_recall), 4)
        print(f"Average Memory Recall: {avg_recall:.4f}")
        
    if mem_helpful_scores:
        avg_helpful = np.mean(mem_helpful_scores)
        summary['average_mem_helpful_score'] = round(float(avg_helpful), 4)
        print(f"Average Memory Helpful Score: {avg_helpful:.4f}")

    # Save results
    if not args.out_file:
        base, ext = os.path.splitext(args.gen_file)
        args.out_file = f"{base}_metrics.json"
        
    output_data = {
        "summary": summary,
        "detailed_results": detailed_results
    }
    
    with open(args.out_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    print(f"\nResults saved to {args.out_file}")


def process_retrieval_results_llm_metrics(args):
    """Process LLM metrics computation for all generation_results.json files"""
    retrieval_result_dir = args.retrieval_result_dir
    input_data_dir = args.input_data_dir

    if not os.path.exists(retrieval_result_dir):
        print(f"Error: Retrieval result directory not found at {retrieval_result_dir}")
        return

    if not os.path.exists(input_data_dir):
        print(f"Error: Input data directory not found at {input_data_dir}")
        return

    # Find all subdirectories with generation_results.json
    subdirs = []
    for item in os.listdir(retrieval_result_dir):
        item_path = os.path.join(retrieval_result_dir, item)
        if os.path.isdir(item_path):
            gen_file = os.path.join(item_path, 'generation_results.json')
            if os.path.exists(gen_file):
                subdirs.append((item, gen_file))

    subdirs.sort()

    print(f"Found {len(subdirs)} generation result files to process:")
    for name, _ in subdirs:
        print(f"  - {name}")
    print()

    for name, gen_file in subdirs:
        # Construct dialogues file path
        data_file = os.path.join(input_data_dir, f"{name}_dialogues_256k.json")

        # Construct output file path
        base, ext = os.path.splitext(gen_file)
        output_file = f"{base}_llm_metrics.json"

        # Check if dialogues file exists
        if not os.path.exists(data_file):
            print(f"⚠️  Skipping {name}: Dialogues file not found at {data_file}")
            continue

        # Check if output already exists
        if os.path.exists(output_file):
            print(f"⊙  Skipping {name}: Output file already exists at {output_file}")
            continue

        print(f"🔄 Processing {name}...")
        print(f"   Input: {gen_file}")
        print(f"   Dialogues: {data_file}")
        print(f"   Output: {output_file}")

        # Create a simple args object
        class RunArgs:
            def __init__(self, data_file, gen_file, out_file, model_name, base_url, api_key, max_workers, eval_mem_type, top_k):
                self.data_file = data_file
                self.gen_file = gen_file
                self.out_file = out_file
                self.model_name = model_name
                self.base_url = base_url
                self.api_key = api_key
                self.max_workers = max_workers
                self.eval_mem_type = eval_mem_type
                self.top_k = top_k

        run_args = RunArgs(
            data_file=data_file,
            gen_file=gen_file,
            out_file=output_file,
            model_name=args.model_name,
            base_url=args.base_url,
            api_key=args.api_key,
            max_workers=args.max_workers,
            eval_mem_type=args.eval_mem_type,
            top_k=args.top_k
        )

        try:
            main(run_args)
            print(f"✅ Finished {name}")
        except Exception as e:
            print(f"❌ Error processing {name}: {e}")
            import traceback
            traceback.print_exc()
        print()


if __name__ == "__main__":
    args = parse_args()

    if args.process_retrieval_results:
        process_retrieval_results_llm_metrics(args)
    elif args.run_all:
        # Detect project root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Try to find project root by looking for data directory
        # logic similar to run_all_llm_metrics.py but more robust
        project_root = os.path.dirname(script_dir) # eval
        
        # Check if data is in project_root/data (i.e. eval/data - unlikely)
        # or project_root/../data (real_bench/data)
        
        candidates = [
            os.path.join(project_root, "data/realmem/input_data"),
            os.path.join(os.path.dirname(project_root), "data/realmem/input_data"),
        ]
        
        if 'PROJECT_ROOT' in globals():
             candidates.append(os.path.join(PROJECT_ROOT, "data/realmem/input_data"))
             
        input_data_dir = None
        real_project_root = None
        
        for p in candidates:
            if os.path.exists(p):
                input_data_dir = p
                # Infer project root relative to input_data
                # p is root/data/realmem/input_data
                real_project_root = os.path.dirname(os.path.dirname(os.path.dirname(p)))
                break
        
        if not input_data_dir:
            print(f"Error: Input data directory not found. Searched in: {candidates}")
            sys.exit(1)

        files = os.listdir(input_data_dir)
        names = []
        for f in files:
            if f.endswith("_dialogues_256k.json"):
                name = f.replace("_dialogues_256k.json", "")
                names.append(name)
        
        names.sort()
        
        # Configuration
        filename_model_part = args.gen_model_part
        top_k = args.top_k
        evidence_type = args.eval_mem_type
        
        generation_filename = f"graph_{filename_model_part}_top{top_k}_{evidence_type}_generation_results.json"
        
        # We will let the main function decide output filename by default (which appends _metrics.json)
        # unless we want to enforce the specific naming from run_all script.
        # run_all script used: metrics_filename = f"graph_{filename_model_part}_top{top_k}_{evidence_type}_generation_results_metrics-rerun1.json"
        # We'll stick to default behavior of main() which is cleaner: gen_file + "_metrics.json"
        
        print(f"Found {len(names)} datasets. Model part: {filename_model_part}, Top K: {top_k}, Type: {evidence_type}")

        for name in names:
            base_dir = os.path.join(real_project_root, f"data/realmem/{name}")
            gen_file = os.path.join(base_dir, generation_filename)
            data_file = os.path.join(input_data_dir, f"{name}_dialogues_256k.json")
            
            if not os.path.exists(gen_file):
                print(f"Skipping {name}: Generation file not found at {gen_file}")
                continue
                
            if not os.path.exists(data_file):
                 print(f"Skipping {name}: Data file not found at {data_file}")
                 continue
            
            # Check if output already exists (using default naming convention of main)
            base, ext = os.path.splitext(gen_file)
            output_file = f"{base}_metrics.json"
            
            if os.path.exists(output_file):
                print(f"Skipping {name}: Output file already exists at {output_file}")
                continue

            print(f"Processing {name}...")
            
            # Create a new args object for this run
            class RunArgs:
                def __init__(self, data_file, gen_file, out_file, model_name, base_url, api_key, max_workers, eval_mem_type, top_k):
                    self.data_file = data_file
                    self.gen_file = gen_file
                    self.out_file = out_file # None allows main to generate it, or we pass explicit
                    self.model_name = model_name
                    self.base_url = base_url
                    self.api_key = api_key
                    self.max_workers = max_workers
                    self.eval_mem_type = eval_mem_type
                    self.top_k = top_k

            # We pass output_file explicit to be safe or None to let it auto-gen
            run_args = RunArgs(
                data_file=data_file,
                gen_file=gen_file,
                out_file=None, # Let main handle it (will be gen_file + _metrics.json)
                model_name=args.model_name,
                base_url=args.base_url,
                api_key=args.api_key,
                max_workers=args.max_workers,
                eval_mem_type=evidence_type,
                top_k=top_k
            )
            
            try:
                main(run_args)
                print(f"Finished {name}")
            except Exception as e:
                print(f"Error processing {name}: {e}")
                import traceback
                traceback.print_exc()

    else:
        main(args)
