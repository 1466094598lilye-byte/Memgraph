import argparse
import json
import os
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description="Collect and aggregate LLM metrics across personas.")
    parser.add_argument(
        "--metrics_filename",
        default="graph_4omini-4o_top5_chunk_generation_results_metrics-rerun1.json",
        help="Metrics filename to look for inside each persona directory.",
    )
    parser.add_argument(
        "--out_file",
        default=None,
        help="Path to save the aggregated metrics JSON. Defaults to data/realmem/all_llm_metrics_summary.json",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console summary output.",
    )
    return parser.parse_args()


def find_metric_files(root_dir, filename):
    matches = []
    for dirpath, _, files in os.walk(root_dir):
        if filename in files:
            matches.append(os.path.join(dirpath, filename))
    return sorted(matches)


def load_summary(file_path):
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        summary = data.get("summary")
        if summary is None:
            print(f"Warning: No 'summary' section in {file_path}")
        return summary
    except Exception as e:
        print(f"Warning: Failed to load {file_path}: {e}")
        return None


def aggregate_summaries(results):
    numeric_totals = defaultdict(float)
    numeric_counts = defaultdict(int)
    dist_totals = defaultdict(int)

    for item in results:
        summary = item["summary"]
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                numeric_totals[key] += float(value)
                numeric_counts[key] += 1
            elif isinstance(value, dict):
                for sub_key, sub_val in value.items():
                    if isinstance(sub_val, (int, float)):
                        dist_totals[(key, sub_key)] += sub_val

    numeric_means = {
        key: numeric_totals[key] / numeric_counts[key]
        for key in numeric_totals
        if numeric_counts[key] > 0
    }

    distributions = defaultdict(dict)
    for (parent_key, sub_key), total in dist_totals.items():
        distributions[parent_key][sub_key] = total

    return {
        "mean": numeric_means,
        "distribution_totals": distributions,
    }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_root = os.path.join(project_root, "data", "realmem")

    args = parse_args()
    out_file = args.out_file or os.path.join(data_root, "all_llm_metrics_summary.json")

    metric_files = find_metric_files(data_root, args.metrics_filename)
    if not metric_files:
        print(f"No metric files named '{args.metrics_filename}' found under {data_root}")
        return

    results = []
    for path in metric_files:
        summary = load_summary(path)
        if summary is None:
            continue
        persona = os.path.basename(os.path.dirname(path))
        results.append(
            {
                "persona": persona,
                "file": path,
                "summary": summary,
            }
        )

    if not results:
        print("No valid summaries were loaded.")
        return

    aggregated = aggregate_summaries(results)
    output = {
        "metrics_filename": args.metrics_filename,
        "results": results,
        "overall": aggregated,
    }

    try:
        with open(out_file, "w") as f:
            json.dump(output, f, indent=2)
        if not args.quiet:
            print(f"Aggregated metrics saved to {out_file}")
    except Exception as e:
        print(f"Failed to write aggregated metrics to {out_file}: {e}")

    if not args.quiet:
        print(f"Found {len(results)} metric files.")
        for item in results:
            s = item["summary"]
            print(
                f"- {item['persona']}: "
                f"avg_qa_score={s.get('average_qa_score')}, "
                f"hallucination_rate={s.get('qa_hallucination_rate')}, "
                f"avg_mem_recall={s.get('average_mem_recall')}"
            )
        print("Overall mean metrics:")
        for k, v in aggregated["mean"].items():
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
