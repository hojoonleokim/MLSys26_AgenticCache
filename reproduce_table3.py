#!/usr/bin/env python3
"""
Reproduce Table 3 from the paper:
  "Cold-start results on standard tasks (3000 frame limit)"

Paper claim: AgenticCache (Ours / Ours+) improves over baseline in cold-start
scenarios with 10-object environments under the 3000-frame budget.

Requirements: Python 3, no GPU / API key / simulator needed.
Input: Raw JSON logs in results/table3/ directory.
Output: Printed table + table3-3000.tex LaTeX file.
"""

import json
import os
import re
import glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

BASE = os.path.join(SCRIPT_DIR, "results", "table3")

# New pricing per 1M tokens (from user's image)
PRICING = {
    "gpt-5": {"input": 1.25, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-5-nano": {"input": 0.05, "output": 0.40},
}

MODEL_KEYS = ["gpt-5", "gpt-5-mini", "gpt-5-nano"]

def calc_cost(input_tokens, output_tokens, model):
    """Calculate cost from token counts and model pricing."""
    p = PRICING[model]
    return (input_tokens / 1e6) * p["input"] + (output_tokens / 1e6) * p["output"]

def parse_token_file_inputoutput(path):
    """Parse token files with 'Input tokens:' / 'Output tokens:' format.
    Used for CoELA and COMBO."""
    inp, out = 0, 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            m = re.match(r"Input tokens:\s*(\d+)", line)
            if m:
                inp += int(m.group(1))
                continue
            m = re.match(r"Output tokens:\s*(\d+)", line)
            if m:
                out += int(m.group(1))
    return inp, out

def get_episode_time(ep):
    """Get time from episode dict, handling different key names."""
    for key in ["time_elapsed", "elapsed_times", "time"]:
        if key in ep:
            return ep[key]
    return 0

# ─────────────────────────────────────────────────────────────────────────────
# Table 2 & 3: Cold-start results from results/table3/
# ─────────────────────────────────────────────────────────────────────────────

def process_table23(method_dir, model, obj_prefix):
    """Process cold-start CoELA-style results.

    method_dir: 'Ours', 'Ours+', or 'baseline_large_maps'
    obj_prefix: '10objs' for 3000 frame, '30objs' for 6000 frame
    """
    configs = ["2a_food", "2a_stuff", "4a_food", "4a_stuff", "5a_food", "5a_stuff"]

    all_sr_vals = []
    total_latency = 0
    total_input = 0
    total_output = 0
    total_episodes = 0

    if method_dir == "baseline_large_maps":
        model_suffix_map = {
            "gpt-5": "gpt-5-2025-08-07",
            "gpt-5-mini": "gpt-5-mini-2025-08-07",
            "gpt-5-nano": "gpt-5-nano-2025-08-07",
        }
        model_suffix = model_suffix_map[model]

        for config in configs:
            dir_name = f"eval_{obj_prefix}_{config}_{model_suffix}"
            run_path = os.path.join(BASE, method_dir, dir_name, "run_eval")
            eval_path = os.path.join(run_path, "eval_result.json")

            if not os.path.exists(eval_path):
                print(f"  [MISSING baseline] {eval_path}")
                continue

            with open(eval_path) as f:
                data = json.load(f)

            episodes = data.get("episode_results", {})
            n_eps = len(episodes)
            total_episodes += n_eps

            for ep in episodes.values():
                sr_val = ep.get("finish", 0) / ep.get("total", 10)
                all_sr_vals.append(sr_val)
                total_latency += get_episode_time(ep)

            # Tokens at run_eval/LLM level
            for agent_file in glob.glob(os.path.join(run_path, "LLM", "token_usage_agent_*.txt")):
                i, o = parse_token_file_inputoutput(agent_file)
                total_input += i
                total_output += o
    else:
        model_dir_map = {
            "gpt-5": "LMs-gpt-5-2025-08-07",
            "gpt-5-mini": "LMs-gpt-5-mini-2025-08-07",
            "gpt-5-nano": "LMs-gpt-5-nano-2025-08-07",
        }
        model_dir = model_dir_map[model]

        for config in configs:
            dir_name = f"run_eval_eval_{obj_prefix}_{config}"
            run_path = os.path.join(BASE, method_dir, model_dir, dir_name)
            eval_path = os.path.join(run_path, "eval_result.json")

            if not os.path.exists(eval_path):
                print(f"  [MISSING {method_dir}] {eval_path}")
                continue

            with open(eval_path) as f:
                data = json.load(f)

            episodes = data.get("episode_results", {})
            n_eps = len(episodes)
            total_episodes += n_eps

            for ep in episodes.values():
                sr_val = ep.get("finish", 0) / ep.get("total", 10)
                all_sr_vals.append(sr_val)
                total_latency += get_episode_time(ep)

            for agent_file in glob.glob(os.path.join(run_path, "LLM", "token_usage_agent_*.txt")):
                i, o = parse_token_file_inputoutput(agent_file)
                total_input += i
                total_output += o

    if not all_sr_vals:
        return None

    sr = sum(all_sr_vals) / len(all_sr_vals)
    latency_hours = total_latency / 3600.0
    total_tokens = total_input + total_output
    cost = calc_cost(total_input, total_output, model)

    return {
        "sr": sr, "latency": latency_hours,
        "tokens": total_tokens, "input_tokens": total_input,
        "output_tokens": total_output, "cost": cost,
        "n_episodes": total_episodes,
    }


def generate_table23(obj_prefix, frame_limit, table_num):
    """Generate Table 2 or 3."""
    print(f"\n{'=' * 80}")
    print(f"TABLE {table_num}: Cold-start results ({frame_limit} frame limit, {obj_prefix})")
    print(f"{'=' * 80}")

    method_dirs = ["baseline_large_maps", "Ours", "Ours+"]
    method_names_map = {"baseline_large_maps": "Baseline", "Ours": "Ours", "Ours+": "Ours+"}

    for model in MODEL_KEYS:
        print(f"\n--- {model.upper()} ---")
        for md in method_dirs:
            result = process_table23(md, model, obj_prefix)
            if result:
                print(f"  {method_names_map[md]:<15} | SR={result['sr']*100:.1f}% | L={result['latency']:.2f}h | T={result['tokens']:,} (in={result['input_tokens']:,}, out={result['output_tokens']:,}) | C=${result['cost']:.2f} | #eps={result['n_episodes']}")
            else:
                print(f"  {method_names_map[md]:<15} | NO DATA")

    # Generate LaTeX
    if frame_limit == 3000:
        caption = r"\rev{Cold-start results on standard tasks. SR: Success Rate, L: Latency (hours), T: Token Usage, C: Cost.}"
        label = "tab:cold_start_standard"
    else:
        caption = r"\rev{Cold-start results on long-horizon tasks. SR: Success Rate, L: Latency (hours), T: Token Usage, C: Cost.}"
        label = "tab:cold_start_long"

    latex = []
    latex.append(r"\begin{table}[t]")
    latex.append(r"\centering")
    latex.append(r"\footnotesize")
    latex.append(r"\setlength{\tabcolsep}{4pt}")
    latex.append(r"\renewcommand{\arraystretch}{1.05}")
    latex.append(f"\\caption{{{caption}}}")
    latex.append(f"\\label{{{label}}}")
    latex.append(r"{\color{blue}")
    latex.append(r"\begin{tabular}{llcccc}")
    latex.append(r"\toprule")
    latex.append(r"Model & Method & SR & L & T & C \\")
    latex.append(r"\hline")

    for model in MODEL_KEYS:
        model_display = {"gpt-5": "GPT-5", "gpt-5-mini": "GPT-5-mini", "gpt-5-nano": "GPT-5-nano"}[model]

        for j, md in enumerate(method_dirs):
            result = process_table23(md, model, obj_prefix)
            name = method_names_map[md]

            if result:
                sr_str = f"{result['sr']*100:.1f}\\%"
                l_str = f"{result['latency']:.2f}"
                t = result['tokens']
                if t >= 1e6:
                    t_str = f"{t/1e6:.2f}M"
                elif t >= 1e3:
                    t_str = f"{t/1e3:.0f}K"
                else:
                    t_str = str(t)
                c_str = f"\\${result['cost']:.2f}"
            else:
                sr_str = l_str = t_str = c_str = "---"

            if j == 0:
                row = f"{model_display} & {name} & {sr_str} & {l_str} & {t_str} & {c_str} \\\\"
            else:
                row = f" & {name} & {sr_str} & {l_str} & {t_str} & {c_str} \\\\"
            latex.append(row)

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}}")
    latex.append(r"\end{table}")

    latex_str = "\n".join(latex)
    print(f"\n\n=== TABLE {table_num} LaTeX ===\n")
    print(latex_str)

    filename = f"table{table_num}.tex"
    with open(os.path.join(SCRIPT_DIR, filename), "w") as f:
        f.write(latex_str + "\n")
    print(f"\n[Saved to {filename}]")


if __name__ == "__main__":
    print("=" * 60)
    print("Reproducing Table 3: Cold-start results (3000 frame limit)")
    print("Paper: AgenticCache (MLSys 2026)")
    print("=" * 60)
    generate_table23("10objs", 3000, 3)
    print("\nDone. LaTeX output saved to table3.tex")
