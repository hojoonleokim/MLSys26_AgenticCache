#!/usr/bin/env python3
"""
Reproduce Table 2 from the paper:
  "Planning strategy performance across four benchmarks"
  (SR: Success Rate, L: Latency, T: Token Usage, C: Cost)

Paper claim: AgenticCache matches or outperforms baseline in success rate,
reduces latency, and reduces token usage across TDW-MAT, TDW-COOK, TDW-GAME,
and COHERENT benchmarks.

Requirements: Python 3, no GPU / API key / simulator needed.
Input: Raw JSON logs in results/ directory (download from Google Drive link in README).
Output: Printed table + table1.tex LaTeX file.
"""

import json
import os
import re
import glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

BASE = SCRIPT_DIR
RESULTS_BASE = os.path.join(BASE, "results", "table2")

# New pricing per 1M tokens (from user's image)
PRICING = {
    "gpt-5": {"input": 1.25, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-5-nano": {"input": 0.05, "output": 0.40},
}

MODEL_KEYS = ["gpt-5", "gpt-5-mini", "gpt-5-nano"]
METHODS = ["baseline", "parallel", "speculative", "agenticcache"]

def calc_cost(input_tokens, output_tokens, model):
    """Calculate cost from token counts and model pricing."""
    p = PRICING[model]
    return (input_tokens / 1e6) * p["input"] + (output_tokens / 1e6) * p["output"]

# ─────────────────────────────────────────────────────────────────────────────
# Token parsing helpers
# ─────────────────────────────────────────────────────────────────────────────

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

def parse_coherent_token_file(path):
    """Parse COHERENT *_token.txt files with 'INPUT: N, OUTPUT: N' format."""
    inp, out = 0, 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            m = re.match(r"INPUT:\s*(\d+),\s*OUTPUT:\s*(\d+)", line)
            if m:
                inp += int(m.group(1))
                out += int(m.group(2))
    return inp, out

def get_episode_time(ep):
    """Get time from episode dict, handling different key names."""
    for key in ["time_elapsed", "elapsed_times", "time"]:
        if key in ep:
            return ep[key]
    return 0

# ─────────────────────────────────────────────────────────────────────────────
# Table 1: Main results from /results/
# ─────────────────────────────────────────────────────────────────────────────

# Draft model used by speculative method
SPEC_DRAFT_MODEL = "gpt-5-nano"

def _coela_token_files(llm_dir, method, model):
    """Return list of (filepath, pricing_model) for CoELA token files.
    For speculative: primary model files + draft model files (priced at draft rate).
    For others: plain token_usage_agent_*.txt files."""
    model_suffix_map = {
        "gpt-5": "gpt-5-2025-08-07",
        "gpt-5-mini": "gpt-5-mini-2025-08-07",
        "gpt-5-nano": "gpt-5-nano-2025-08-07",
    }
    results = []
    if method == "speculative":
        # Primary model tokens
        suffix = model_suffix_map[model]
        for f in glob.glob(os.path.join(llm_dir, f"token_usage_agent_*_{suffix}.txt")):
            results.append((f, model))
        # Draft model tokens (gpt-5-nano without date)
        for f in glob.glob(os.path.join(llm_dir, "token_usage_agent_*_gpt-5-nano.txt")):
            results.append((f, SPEC_DRAFT_MODEL))
    else:
        for f in glob.glob(os.path.join(llm_dir, "token_usage_agent_*.txt")):
            results.append((f, model))
    return results


def process_coela(method, model):
    """Process CoELA results. Returns dict combining test_1 and test_2."""
    model_dir_map = {
        "gpt-5": "LMs-gpt-5-2025-08-07",
        "gpt-5-mini": "LMs-gpt-5-mini-2025-08-07",
        "gpt-5-nano": "LMs-gpt-5-nano-2025-08-07",
    }
    model_dir = model_dir_map[model]
    base_path = os.path.join(RESULTS_BASE, "CoELA", method, model_dir)

    all_sr_vals = []
    total_time = 0
    total_tokens = 0
    total_cost = 0
    total_input = 0
    total_output = 0
    total_episodes = 0

    for test_name in ["test_1", "test_2"]:
        test_path = os.path.join(base_path, test_name)
        eval_path = os.path.join(test_path, "eval_result.json")

        if not os.path.exists(eval_path):
            continue

        with open(eval_path) as f:
            data = json.load(f)

        episodes = data["episode_results"]
        total_episodes += len(episodes)
        for ep in episodes.values():
            all_sr_vals.append(ep["finish"] / ep["total"])
            total_time += get_episode_time(ep)

        # Tokens - handle speculative draft model separately
        llm_dir = os.path.join(test_path, "LLM")
        for fpath, pricing_model in _coela_token_files(llm_dir, method, model):
            i, o = parse_token_file_inputoutput(fpath)
            total_input += i
            total_output += o
            total_tokens += i + o
            total_cost += calc_cost(i, o, pricing_model)

    if not all_sr_vals:
        return None

    sr = sum(all_sr_vals) / len(all_sr_vals)
    latency_hours = total_time / 3600.0

    return {
        "sr": sr, "latency": latency_hours,
        "tokens": total_tokens, "input_tokens": total_input,
        "output_tokens": total_output, "cost": total_cost,
        "n_episodes": total_episodes,
    }


COMBO_CHATLOG_BASE = os.path.join(RESULTS_BASE, "COMBO", "chat_log")


def get_combo_agenticcache_episodes(task):
    """Return the set of episode IDs used by agenticcache for a COMBO task.
    These are identical across all 3 models."""
    ep = os.path.join(RESULTS_BASE, "COMBO", "agenticcache", "gpt5", task, "eval_result.json")
    if not os.path.exists(ep):
        return None
    with open(ep) as f:
        data = json.load(f)
    return set(data["episode_results"].keys())


def get_coherent_agenticcache_episodes(model_dir):
    """Return dict {env_name: set(episode_ids)} used by agenticcache for COHERENT.
    These are identical across all 3 models, but we accept model_dir for generality."""
    ac_path = os.path.join(RESULTS_BASE, "COHERENT", "agenticcache", model_dir)
    if not os.path.exists(ac_path):
        return None
    result = {}
    for env_dir in sorted(glob.glob(os.path.join(ac_path, "env*"))):
        eval_path = os.path.join(env_dir, "eval_result.json")
        if not os.path.exists(eval_path):
            continue
        with open(eval_path) as f:
            data = json.load(f)
        env_name = os.path.basename(env_dir)
        result[env_name] = set(data["episode_results"].keys())
    return result


def process_combo(method, model, task):
    """Process COMBO results for a specific task (cook/game).

    SR and time are computed only over the agenticcache episode set.
    Tokens are cumulative and kept unchanged.
    """
    model_dir_map = {
        "gpt-5": "gpt5",
        "gpt-5-mini": "gpt5-mini",
        "gpt-5-nano": "gpt5-nano",
    }
    model_suffix_map = {
        "gpt-5": "gpt-5-2025-08-07",
        "gpt-5-mini": "gpt-5-mini-2025-08-07",
        "gpt-5-nano": "gpt-5-nano-2025-08-07",
    }
    model_dir = model_dir_map[model]
    task_path = os.path.join(RESULTS_BASE, "COMBO", method, model_dir, task)
    eval_path = os.path.join(task_path, "eval_result.json")

    if not os.path.exists(eval_path):
        return None

    with open(eval_path) as f:
        data = json.load(f)

    all_episodes = data["episode_results"]

    # Filter to agenticcache episode set
    ac_eps = get_combo_agenticcache_episodes(task)
    if ac_eps is None:
        ac_eps = set(all_episodes.keys())

    succ_count = 0
    total_time = 0
    n_eps = 0
    for ep_id in ac_eps:
        if ep_id in all_episodes:
            ep = all_episodes[ep_id]
        else:
            # Fallback to per-episode result_episode.json in task_path/{ep_id}/
            ep_path = os.path.join(task_path, ep_id, "result_episode.json")
            if not os.path.exists(ep_path):
                continue
            with open(ep_path) as f:
                ep = json.load(f)
        n_eps += 1
        if ep.get("success", False):
            succ_count += 1
        total_time += get_episode_time(ep)

    if n_eps == 0:
        return None

    sr = succ_count / n_eps
    latency_hours = total_time / 3600.0

    total_input, total_output = 0, 0
    total_cost = 0

    # All token files live under chat_log/{method}/{model}/{task}/{model_suffix}/
    chatlog_base = os.path.join(COMBO_CHATLOG_BASE, method, model_dir, task)
    # Primary model tokens
    primary_dir = os.path.join(chatlog_base, model_suffix_map[model])
    for token_file in glob.glob(os.path.join(primary_dir, "*_token_usage.jsonl")):
        i, o = parse_token_file_inputoutput(token_file)
        total_input += i
        total_output += o
        total_cost += calc_cost(i, o, model)
    # For speculative: also count draft model (gpt-5-nano) tokens at nano prices
    if method == "speculative":
        draft_dir = os.path.join(chatlog_base, "gpt-5-nano")
        for token_file in glob.glob(os.path.join(draft_dir, "*_token_usage.jsonl")):
            i, o = parse_token_file_inputoutput(token_file)
            total_input += i
            total_output += o
            total_cost += calc_cost(i, o, SPEC_DRAFT_MODEL)

    total_tokens = total_input + total_output

    return {
        "sr": sr, "latency": latency_hours,
        "tokens": total_tokens, "input_tokens": total_input,
        "output_tokens": total_output, "cost": total_cost,
        "n_episodes": n_eps,
    }


def process_coherent(method, model):
    """Process COHERENT results across all envs.

    SR and time are computed only over the agenticcache episode set per env.
    Tokens are cumulative and kept unchanged.
    """
    model_dir_map = {
        "gpt-5": "gpt-5-2025-08-07",
        "gpt-5-mini": "gpt-5-mini-2025-08-07",
        "gpt-5-nano": "gpt-5-nano-2025-08-07",
    }
    model_dir = model_dir_map[model]
    model_suffix = model_dir  # e.g. "gpt-5-2025-08-07"
    model_path = os.path.join(RESULTS_BASE, "COHERENT", method, model_dir)

    if not os.path.exists(model_path):
        return None

    # Get agenticcache episode filter
    ac_eps_per_env = get_coherent_agenticcache_episodes(model_dir)

    total_sr_sum = 0
    total_sr_count = 0
    total_latency = 0
    total_input = 0
    total_output = 0
    total_episodes = 0
    total_cost = 0

    for env_dir in sorted(glob.glob(os.path.join(model_path, "env*"))):
        eval_path = os.path.join(env_dir, "eval_result.json")
        if not os.path.exists(eval_path):
            continue

        with open(eval_path) as f:
            data = json.load(f)

        all_episodes = data["episode_results"]
        env_name = os.path.basename(env_dir)

        # Filter to agenticcache episodes for this env
        if ac_eps_per_env and env_name in ac_eps_per_env:
            target_eps = ac_eps_per_env[env_name]
        else:
            target_eps = set(all_episodes.keys())

        for ep_id in target_eps:
            if ep_id not in all_episodes:
                continue
            ep = all_episodes[ep_id]
            total_episodes += 1
            total_sr_count += 1
            if ep.get("success", False):
                total_sr_sum += 1
            total_latency += get_episode_time(ep)

        # Primary model token files
        for token_file in glob.glob(os.path.join(env_dir, f"*{model_suffix}_token.txt")):
            i, o = parse_coherent_token_file(token_file)
            total_input += i
            total_output += o
            total_cost += calc_cost(i, o, model)

        # For speculative: also count draft model (gpt-5-nano) tokens at nano prices
        if method == "speculative":
            for token_file in glob.glob(os.path.join(env_dir, "*_gpt-5-nano_token.txt")):
                i, o = parse_coherent_token_file(token_file)
                total_input += i
                total_output += o
                total_cost += calc_cost(i, o, SPEC_DRAFT_MODEL)

    if total_sr_count == 0:
        return None

    sr = total_sr_sum / total_sr_count
    latency_hours = total_latency / 3600.0
    total_tokens = total_input + total_output
    # For non-speculative, compute cost normally
    if method != "speculative":
        total_cost = calc_cost(total_input, total_output, model)

    return {
        "sr": sr, "latency": latency_hours,
        "tokens": total_tokens, "input_tokens": total_input,
        "output_tokens": total_output, "cost": total_cost,
        "n_episodes": total_episodes,
    }


def fmt_sr(r):
    if r is None: return "---"
    v = r['sr'] * 100
    if v == int(v):
        return f"{int(v)}\\%"
    return f"{v:.2f}\\%"

def fmt_l(r):
    if r is None: return "---"
    return f"{r['latency']:.2f}"

def fmt_t(r):
    if r is None: return "---"
    t = r['tokens']
    if t >= 1e6:
        return f"{t/1e6:.1f}M"
    elif t >= 1e3:
        return f"{t/1e3:.0f}K"
    else:
        return str(t)

def fmt_c(r):
    if r is None: return "---"
    return f"\\${r['cost']:.1f}"


def generate_table2():
    """Generate Table 2: Planning strategy performance across benchmarks."""
    print("\n" + "=" * 80)
    print("TABLE 2: Planning strategy performance")
    print("=" * 80)

    method_names = {
        "baseline": "Baseline",
        "parallel": "Parallel",
        "speculative": "Speculative",
        "agenticcache": "AgenticCache(Ours)",
    }

    for model in MODEL_KEYS:
        print(f"\n--- {model.upper()} ---")
        hdr = f"{'Method':<25} | {'Bench':<10} | {'SR':>8} | {'L(hrs)':>10} | {'T(tokens)':>12} | {'C($)':>8} | {'#eps':>5}"
        print(hdr)
        print("-" * len(hdr))

        for method in METHODS:
            coela = process_coela(method, model)
            combo_cook = process_combo(method, model, "cook")
            combo_game = process_combo(method, model, "game")
            coherent = process_coherent(method, model)

            def prow(bench, r, extra=""):
                if r:
                    print(f"{method_names[method] if bench == 'TDW-MAT' else '':<25} | {bench:<10} | {r['sr']*100:>7.2f}% | {r['latency']:>10.2f} | {r['tokens']:>12,} | {r['cost']:>8.2f} | {r['n_episodes']:>5}{extra}")
                else:
                    print(f"{method_names[method] if bench == 'TDW-MAT' else '':<25} | {bench:<10} | NO DATA")

            prow("TDW-MAT", coela)
            prow("TDW-COOK", combo_cook)
            prow("TDW-GAME", combo_game)
            prow("COHERENT", coherent)
            print()

    # Generate LaTeX
    print("\n\n=== TABLE 2 LaTeX ===\n")

    latex = []
    latex.append(r"\begin{table*}[t]")
    latex.append(r"\centering")
    latex.append(r"\footnotesize")
    latex.append(r"\setlength{\tabcolsep}{3pt}")
    latex.append(r"\renewcommand{\arraystretch}{1.1}")
    latex.append(r"\caption{Planning strategy performance across four benchmarks. SR: Success Rate, L: Latency (hours), T: Token Usage, C: Cost.}")
    latex.append(r"")
    latex.append(r"\resizebox{\textwidth}{!}{%")
    latex.append(r"\begin{tabular}{llcccclcccclcccclcccc}")
    latex.append(r"\toprule")

    for model in MODEL_KEYS:
        model_display = {"gpt-5": "GPT-5", "gpt-5-mini": "GPT-5-mini", "gpt-5-nano": "GPT-5-nano"}[model]

        latex.append(r"\multicolumn{21}{c}{" + model_display + r"} \\ \hline")
        latex.append(r"\multicolumn{1}{c}{\multirow{2}{*}{Parallelism Method}} & \textbf{} & \multicolumn{4}{c}{TDW-MAT} & \textbf{} & \multicolumn{4}{c}{TDW-COOK} & \textbf{} & \multicolumn{4}{c}{TDW-GAME} & \textbf{} & \multicolumn{4}{c}{COHERENT} \\ \cline{3-6} \cline{8-11} \cline{13-16} \cline{18-21} ")
        latex.append(r"\multicolumn{1}{c}{} & \textbf{} & SR & L & T & C & \textbf{} & SR & L & T & C & \textbf{} & SR & L & T & C & \textbf{} & SR & L & T & C \\ \hline")

        active_methods = METHODS if model != "gpt-5-nano" else [m for m in METHODS if m != "speculative"]

        for method in active_methods:
            coela = process_coela(method, model)
            combo_cook = process_combo(method, model, "cook")
            combo_game = process_combo(method, model, "game")
            coherent = process_coherent(method, model)

            name = method_names[method]
            row = f"{name} &  & {fmt_sr(coela)} & {fmt_l(coela)} & {fmt_t(coela)} & {fmt_c(coela)}"
            row += f" &  & {fmt_sr(combo_cook)} & {fmt_l(combo_cook)} & {fmt_t(combo_cook)} & {fmt_c(combo_cook)}"
            row += f" &  & {fmt_sr(combo_game)} & {fmt_l(combo_game)} & {fmt_t(combo_game)} & {fmt_c(combo_game)}"
            row += f" &  & {fmt_sr(coherent)} & {fmt_l(coherent)} & {fmt_t(coherent)} & {fmt_c(coherent)} \\\\"

            latex.append(row)

        if model == "gpt-5-nano":
            latex.append(r"\bottomrule")
        else:
            latex.append(r"\hline")

    latex.append(r"\end{tabular}")
    latex.append(r"}")
    latex.append(r"\label{tab:planning_results}")
    latex.append(r"\end{table*}")

    latex_str = "\n".join(latex)
    print(latex_str)

    with open(os.path.join(BASE, "table2.tex"), "w") as f:
        f.write(latex_str + "\n")
    print("\n[Saved to table2.tex]")


if __name__ == "__main__":
    print("=" * 60)
    print("Reproducing Table 2: Planning strategy performance")
    print("Paper: AgenticCache (MLSys 2026)")
    print("=" * 60)
    generate_table2()
    print("\nDone. LaTeX output saved to table2.tex")
