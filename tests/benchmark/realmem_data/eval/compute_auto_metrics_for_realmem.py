import json
import argparse
import numpy as np
import os
import sys
import re
from tqdm import tqdm

# Add project root to Python path
PROJECT_ROOT = os.path.expanduser("~/MemInsight")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def dcg(relevances, k):
    """Discounted Cumulative Gain at k."""
    relevances = np.asfarray(relevances)[:k]
    if relevances.size:
        return relevances[0] + np.sum(relevances[1:] / np.log2(np.arange(2, relevances.size + 1)))
    return 0.

def ndcg(rankings, correct_docs, corpus_ids, k=10):
    """Normalized Discounted Cumulative Gain at k."""
    relevances = [1 if doc_id in correct_docs else 0 for doc_id in corpus_ids]
    # relevances map: index -> relevance (0 or 1)
    
    # ranked relevances
    sorted_relevances = []
    for idx in rankings[:k]:
        if idx < len(relevances):
            sorted_relevances.append(relevances[idx])
        else:
            sorted_relevances.append(0)
            
    ideal_relevance = sorted(relevances, reverse=True)
    ideal_dcg = dcg(ideal_relevance, k)
    actual_dcg = dcg(sorted_relevances, k)
    if ideal_dcg == 0:
        return 0.
    return actual_dcg / ideal_dcg

def evaluate_retrieval(rankings, correct_docs, corpus_ids, k=10):
    # rankings are indices in corpus_ids
    recalled_docs = set()
    for idx in rankings[:k]:
        if idx < len(corpus_ids):
            recalled_docs.add(corpus_ids[idx])
            
    recall_any = float(any(doc in recalled_docs for doc in correct_docs))
    recall_all = float(all(doc in recalled_docs for doc in correct_docs))
    ndcg_score = ndcg(rankings, correct_docs, corpus_ids, k)
    return recall_any, recall_all, ndcg_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str, 
                        default=f'{PROJECT_ROOT}/data/realmem/Kenta_Tanaka/graph_retrieval_results.json',
                        help='Input file containing retrieval results (JSON format)')
    parser.add_argument('--dialogues_file', type=str, 
                        default=f'{PROJECT_ROOT}/data/realmem/input_data/Kenta_Tanaka_dialogues_256k.json',
                        help='Original dialogues file containing ground truth memory usage')
    parser.add_argument('--out_file', type=str, default=f'{PROJECT_ROOT}/data/realmem/Kenta_Tanaka/graph_retrieval_results_metrics.json',
                        help='Output file to save detailed metrics (optional)')
    parser.add_argument('--run_all', action='store_true', help='Run metrics computation for all datasets found.')
    parser.add_argument('--retrieval_result_dir', type=str, default='eval/retrieval_result',
                        help='Directory containing retrieval result subdirectories to process in batch mode.')
    parser.add_argument('--process_retrieval_results', action='store_true',
                        help='Process all example_retrieval_results.json files from retrieval_result_dir.')
    parser.add_argument('--input_data_dir', type=str, default='datasets',
                        help='Directory containing input dialogue files (for batch mode).')
    return parser.parse_args()


def get_ground_truth(dialogues_data):
    """
    Extracts ground truth for each query.
    Returns: dict { query_text: [list of correct session_identifiers] }
    """
    query_to_gold = {}
    all_session_ids = set()
    uuid_to_sid = {}
    
    # First pass: collect all valid session identifiers and map uuid -> sid
    for dialogue in dialogues_data['dialogues']:
        sid = dialogue['session_identifier']
        suuid = dialogue.get('session_uuid')
        all_session_ids.add(sid)
        if suuid:
            uuid_to_sid[suuid] = sid

    for dialogue in dialogues_data['dialogues']:
        turns = dialogue.get('dialogue_turns', [])
        for i, turn in enumerate(turns):
            if turn.get('speaker') == 'User':
                query_text = turn.get('content', '').strip()
                if not query_text:
                    continue
                
                # Look for memory_used in the NEXT assistant turn
                if i + 1 < len(turns):
                    next_turn = turns[i+1]
                    if next_turn.get('speaker') == 'Assistant':
                        gold_sessions = set()

                        # Method 1: Check memory_session_uuids (Preferred)
                        mem_uuids = next_turn.get('memory_session_uuids', [])
                        if mem_uuids:
                            for uuid in mem_uuids:
                                if uuid in uuid_to_sid:
                                    gold_sessions.add(uuid_to_sid[uuid])
                        
                        # Method 2: Fallback to iterating memory_used items if no uuids found directly
                        if not gold_sessions:
                            memories = next_turn.get('memory_used', [])
                            for mem in memories:
                                if isinstance(mem, dict):
                                    # Try to get session_uuid from item
                                    muuid = mem.get('session_uuid')
                                    if muuid and muuid in uuid_to_sid:
                                        gold_sessions.add(uuid_to_sid[muuid])
                            
                        if gold_sessions:
                            query_to_gold[query_text] = list(gold_sessions)

    return query_to_gold, list(all_session_ids)

def main(args):
    print(f"Loading dialogues from {args.dialogues_file}...")
    with open(args.dialogues_file, 'r') as f:
        dialogues_data = json.load(f)
        
    query_to_gold, all_corpus_ids = get_ground_truth(dialogues_data)
    all_corpus_ids = sorted(list(all_corpus_ids)) # Ensure consistent order
    corpus_id_to_idx = {cid: idx for idx, cid in enumerate(all_corpus_ids)}
    
    print(f"Found {len(query_to_gold)} queries with ground truth.")
    print(f"Total corpus size (sessions): {len(all_corpus_ids)}")

    print(f"Loading retrieval results from {args.in_file}...")
    with open(args.in_file, 'r') as f:
        retrieval_results = json.load(f)

    # Metrics storage
    metrics = {
        'recall_all@5': [], 'recall_all@10': [], 'recall_all@20': [],
        'recall_any@5': [], 'recall_any@10': [], 'recall_any@20': [],
        'ndcg_any@5': [], 'ndcg_any@10': [], 'ndcg_any@20': []
    }
    
    ks = [5, 10, 20]

    count_evaluated = 0

    # In new format, retrieval_results is { query_text: { "question": ..., "ranked_items": [...] } }
    
    for query_text, result_obj in tqdm(retrieval_results.items(), desc="Evaluating"):
        # Match query text to ground truth
        # Try exact match
        gt_key = None
        if query_text in query_to_gold:
            gt_key = query_text
        else:
            # Try stripped
            stripped = query_text.strip()
            if stripped in query_to_gold:
                gt_key = stripped
        
        if not gt_key:
            print(f"Warning: No ground truth found for query: {query_text[:50]}...")
            continue
            
        correct_docs = query_to_gold[gt_key]
        
        ranked_items = result_obj.get('ranked_items', [])
        
        ranked_retrieved_ids = []
        for item in ranked_items:
            if isinstance(item, dict) and item.get('res_type') == 'chunk':
                rid = item.get('chunk_id')
                if rid:
                    ranked_retrieved_ids.append(rid)
        
        # Convert to indices for evaluation function
        rankings = []
        for rid in ranked_retrieved_ids:
            if rid in corpus_id_to_idx:
                rankings.append(corpus_id_to_idx[rid])
            # Else: retrieved doc not in known corpus (hallucinated ID or from excluded session), ignore or count as wrong?
            # Usually we just skip it in indices, effectively it's not a match.
        
        # We need rankings to be length of corpus? No, just the top K.
        # But for NDCG we assume we have a full ranking? 
        # The `evaluate_retrieval` takes `rankings` which is list of indices.
        
        # Calculate metrics
        for k in ks:
            recall_any, recall_all, ndcg_any = evaluate_retrieval(rankings, correct_docs, all_corpus_ids, k=k)
            metrics[f'recall_all@{k}'].append(recall_all)
            metrics[f'recall_any@{k}'].append(recall_any)
            metrics[f'ndcg_any@{k}'].append(ndcg_any)
            
        count_evaluated += 1

    print("-" * 30)
    print(f"Evaluated {count_evaluated} queries.")
    print("-" * 30)
    
    # Compute Averages
    print("Average Metrics:")
    for key, values in metrics.items():
        if values:
            avg_val = np.mean(values)
            print(f"{key}: {avg_val:.4f}")
        else:
            print(f"{key}: N/A")
            
    # Save if requested
    if args.out_file and metrics['recall_all@5']:
        # Construct summary dict
        summary = {k: round(float(np.mean(v)), 4) for k, v in metrics.items() if v}
        with open(args.out_file, 'w') as f:
            json.dump(summary, f, indent=4)
        print(f"Summary saved to {args.out_file}")


def process_retrieval_results_metrics(args):
    """Process metrics computation for all example_retrieval_results.json files"""
    retrieval_result_dir = args.retrieval_result_dir
    input_data_dir = args.input_data_dir

    if not os.path.exists(retrieval_result_dir):
        print(f"Error: Retrieval result directory not found at {retrieval_result_dir}")
        return

    if not os.path.exists(input_data_dir):
        print(f"Error: Input data directory not found at {input_data_dir}")
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
        # Construct dialogues file path
        dialogues_file = os.path.join(input_data_dir, f"{name}_dialogues_256k.json")

        # Construct output file path
        output_file = os.path.join(retrieval_result_dir, name, "metrics_results.json")

        # Check if dialogues file exists
        if not os.path.exists(dialogues_file):
            print(f"⚠️  Skipping {name}: Dialogues file not found at {dialogues_file}")
            continue

        # Check if output already exists
        if os.path.exists(output_file):
            print(f"⊙  Skipping {name}: Output file already exists at {output_file}")
            continue

        print(f"🔄 Processing {name}...")
        print(f"   Input: {retrieval_file}")
        print(f"   Dialogues: {dialogues_file}")
        print(f"   Output: {output_file}")

        # Create a simple args object
        class RunArgs:
            def __init__(self, in_file, dialogues_file, out_file):
                self.in_file = in_file
                self.dialogues_file = dialogues_file
                self.out_file = out_file

        run_args = RunArgs(retrieval_file, dialogues_file, output_file)

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
        process_retrieval_results_metrics(args)
    elif args.run_all:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        input_data_dir = os.path.join(project_root, "data/realmem/input_data")

        if not os.path.exists(input_data_dir):
            if 'PROJECT_ROOT' in globals():
                alt_dir = os.path.join(PROJECT_ROOT, "data/realmem/input_data")
                if os.path.exists(alt_dir):
                    input_data_dir = alt_dir
                    project_root = PROJECT_ROOT
        
        if not os.path.exists(input_data_dir):
            print(f"Error: Input data directory not found at {input_data_dir}")
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
            base_dir = os.path.join(project_root, f"data/realmem/{name}")
            retrieval_file = os.path.join(base_dir, "graph_retrieval_results.json")
            dialogues_file = os.path.join(input_data_dir, f"{name}_dialogues_256k.json")
            output_file = os.path.join(base_dir, "graph_retrieval_results_metrics.json")
            
            if not os.path.exists(retrieval_file):
                print(f"Skipping {name}: Retrieval file not found at {retrieval_file}")
                continue
            
            if not os.path.exists(dialogues_file):
                 print(f"Skipping {name}: Dialogues file not found at {dialogues_file}")
                 continue
            
            print(f"Processing {name}...")
            
            class RunArgs:
                def __init__(self, in_file, dialogues_file, out_file):
                    self.in_file = in_file
                    self.dialogues_file = dialogues_file
                    self.out_file = out_file
            
            run_args = RunArgs(retrieval_file, dialogues_file, output_file)
            
            try:
                main(run_args)
                print(f"Finished {name}")
            except Exception as e:
                print(f"Error processing {name}: {e}")
                import traceback
                traceback.print_exc()
    else:
        main(args)
