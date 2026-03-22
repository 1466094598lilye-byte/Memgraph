import argparse
import json
import os
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def construct_evidence_text(ranked_items, evidence_type='chunk', top_k=5):
    # Filter items
    filtered = [item for item in ranked_items if item.get('res_type') == evidence_type]
    
    # If not enough items of specific type, maybe fallback or just take what we have
    # The requirement says "set allowed type (default chunk)", so we strictly filter.
    
    selected = filtered[:top_k]
    
    evidence_texts = []
    for i, item in enumerate(selected):
        content = item.get('content', '').strip()
        # Add metadata if needed
        if item.get('res_type') == 'entity':
            content = f"{item['entity_name']}: {content}"
        evidence_texts.append(f"---- idx {i+1} ----\n{content}")
    
    return "\n\n".join(evidence_texts)

def construct_answer_prompt(question, evidence_text):
    return f"""You are a personal AI assistant that helps the user with some long-term tasks. Please respond to the user’s latest message based on the reference memory.

Memories:
{evidence_text}

Query: {question}

Response:"""

def process_item(qid, item_data, args, client):
    question = item_data.get('question')
    ranked_items = item_data.get('ranked_items', [])
    
    evidence_text = construct_evidence_text(ranked_items, args.evidence_type, args.top_k)
    
    # Task 1: Generate Answer
    ans_prompt = construct_answer_prompt(question, evidence_text)
    gen_answer = ""
    try:
        ans_completion = client.chat.completions.create(
            model=args.model_name,
            messages=[{"role": "user", "content": ans_prompt}],
            temperature=0.7 
        )
        gen_answer = ans_completion.choices[0].message.content
    except Exception as e:
        print(f"Error generating answer for {qid}: {e}")

    return {
        "id": qid,
        "question": question,
        "generated_answer": gen_answer,
        "evidence_used": evidence_text
    }

import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Run generation for QA items using retrieval results.")
    parser.add_argument('--retrieval_file', default='data/realmem/Lin_Wanyu/graph_retrieval_results.json', help="Path to the retrieval results file (JSON).")
    parser.add_argument('--output_file', default='data/realmem/Lin_Wanyu/graph_4omini-4omini_top20_entity_generation_results.json', help="Path to save the output results.")
    parser.add_argument('--model_name', default='gpt-4o-mini', help="Model name to use (e.g., gpt-4o-mini).")
    parser.add_argument('--base_url', default=None, help="Base URL for the API (e.g., for local VLLM).")
    parser.add_argument('--api_key', default=None, help="API Key for the client.")
    parser.add_argument('--evidence_type', default='chunk', help="Type of evidence to use (default: chunk).")
    parser.add_argument('--top_k', type=int, default=5, help="Number of top evidence items to use.")
    parser.add_argument('--max_workers', type=int, default=10, help="Number of concurrent workers (default: 10).")
    
    # Batch processing args
    parser.add_argument('--run_all', action='store_true', help='Run generation for all datasets found.')
    parser.add_argument('--filename_model_part', default='4omini-4o', help='Model part string for output filename in batch mode (e.g., 4omini-4o).')
    parser.add_argument('--retrieval_result_dir', default='eval/retrieval_result', help='Directory containing retrieval result subdirectories to process in batch mode.')
    parser.add_argument('--process_retrieval_results', action='store_true', help='Process all example_retrieval_results.json files from retrieval_result_dir.')
    
    return parser.parse_args()

def run_generation(args):
    # Setup client
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
    
    print(f"Initializing OpenAI client with model={args.model_name}, base_url={args.base_url}")
    # OpenAI client is thread-safe
    client = OpenAI(api_key=api_key, base_url=args.base_url)
    
    # Load data
    print(f"Loading retrieval data from {args.retrieval_file}...")
    if not os.path.exists(args.retrieval_file):
        print(f"Error: Retrieval file not found at {args.retrieval_file}")
        return

    retrieval_data = load_data(args.retrieval_file)
    
    print(f"Found {len(retrieval_data)} items in retrieval file.")
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_item, qid, data, args, client): qid for qid, data in retrieval_data.items()}
        
        for future in tqdm(as_completed(futures), total=len(retrieval_data), desc="Processing Items"):
            try:
                result = future.result()
                qid = result['id']
                if qid not in retrieval_data:
                    retrieval_data[qid] = {}
                retrieval_data[qid].update(result)
            except Exception as e:
                qid = futures[future]
                print(f"Error processing item {qid}: {e}")
        
    # Save results (retrieval_data with updates)
    
    # Create directory if it doesn't exist
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(retrieval_data, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {args.output_file}")

def process_retrieval_results(args):
    """Process all example_retrieval_results.json files from retrieval_result_dir"""
    import copy

    retrieval_result_dir = args.retrieval_result_dir

    if not os.path.exists(retrieval_result_dir):
        print(f"Error: Retrieval result directory not found at {retrieval_result_dir}")
        return

    # Find all subdirectories with example_retrieval_results.json
    subdirs = []
    for item in os.listdir(retrieval_result_dir):
        item_path = os.path.join(retrieval_result_dir, item)
        if os.path.isdir(item_path):
            retrieval_file = os.path.join(item_path, 'example_retrieval_results.json')
            if os.path.exists(retrieval_file):
                subdirs.append((item, retrieval_file))

    subdirs.sort()

    print(f"Found {len(subdirs)} retrieval result files to process:")
    for name, _ in subdirs:
        print(f"  - {name}")
    print()

    for name, retrieval_file in subdirs:
        # Construct output file path
        output_filename = f"generation_results.json"
        output_file = os.path.join(retrieval_result_dir, name, output_filename)

        # Check if output already exists
        if os.path.exists(output_file):
            print(f"Skipping {name}: Output file already exists at {output_file}")
            continue

        print(f"Processing {name}...")
        print(f"  Input: {retrieval_file}")
        print(f"  Output: {output_file}")

        # Create a copy of args and update paths
        run_args = copy.deepcopy(args)
        run_args.retrieval_file = retrieval_file
        run_args.output_file = output_file

        try:
            run_generation(run_args)
            print(f"✅ Finished {name}")
        except Exception as e:
            print(f"❌ Error processing {name}: {e}")
            import traceback
            traceback.print_exc()
        print()

def main():
    args = parse_args()

    if args.process_retrieval_results:
        process_retrieval_results(args)
        return

    if args.run_all:
        # Detect project root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir) # eval
        
        # Try finding data dir
        candidates = [
            os.path.join(project_root, "data/realmem/input_data"),
            os.path.join(os.path.dirname(project_root), "data/realmem/input_data"),
        ]
        
        input_data_dir = None
        real_project_root = None
        
        for p in candidates:
            if os.path.exists(p):
                input_data_dir = p
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
        
        print(f"Found {len(names)} datasets to process: {names}")
        
        for name in names:
            base_dir = os.path.join(real_project_root, f"data/realmem/{name}")
            retrieval_file = os.path.join(base_dir, "graph_retrieval_results.json")
            
            # Construct output filename
            output_filename = f"graph_{args.filename_model_part}_top{args.top_k}_{args.evidence_type}_generation_results.json"
            output_file = os.path.join(base_dir, output_filename)
            
            if not os.path.exists(retrieval_file):
                print(f"Skipping {name}: Retrieval file not found at {retrieval_file}")
                continue
                
            if os.path.exists(output_file):
                print(f"Skipping {name}: Output file already exists at {output_file}")
                continue
            
            print(f"Processing {name}...")
            
            # Update args for this run
            # We create a new Namespace or modify existing args (copying recommended to avoid side effects)
            import copy
            run_args = copy.deepcopy(args)
            run_args.retrieval_file = retrieval_file
            run_args.output_file = output_file
            
            try:
                run_generation(run_args)
                print(f"Finished {name}")
            except Exception as e:
                print(f"Error processing {name}: {e}")
                import traceback
                traceback.print_exc()
    else:
        run_generation(args)

if __name__ == "__main__":
    main()
