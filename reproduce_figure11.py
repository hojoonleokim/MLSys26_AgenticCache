#!/usr/bin/env python3
"""
Reproduce Figure 11 from the paper:
  "Plan Execution Accuracy over time"

Paper claim: AgenticCache's cached plans maintain high execution accuracy
relative to ground-truth validation plans, and this accuracy correlates
with success rate improvements over baseline.

Requirements: Python 3, matplotlib, numpy. No GPU / API key / simulator needed.
Input: validation.jsonl and plan_tracking.txt files from result logs.
Output: cache_correctness.pdf/png
"""

import os
import sys
import json
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

FIG11_BASE = os.path.join(SCRIPT_DIR, "results", "fig11")
RESULT_LOG_BASE = os.path.join(FIG11_BASE, "result_log")

DIRS = [
    "run_validation_eval_10objs_2a_food",
    "run_validation_eval_10objs_2a_stuff",
    "run_validation_eval_10objs_4a_food",
    "run_validation_eval_10objs_4a_stuff",
    "run_validation_eval_10objs_5a_food",
    "run_validation_eval_10objs_5a_stuff",
]

MODELS = ["gpt-5-2025-08-07", "gpt-5-mini-2025-08-07", "gpt-5-nano-2025-08-07"]
MODEL_DISPLAY = {
    "gpt-5-2025-08-07": "gpt-5",
    "gpt-5-mini-2025-08-07": "gpt-5-mini",
    "gpt-5-nano-2025-08-07": "gpt-5-nano",
}

MAX_FRAME = 3000
OUTPUT_DIR = SCRIPT_DIR
VALIDATION_BASE = FIG11_BASE
BASELINE_BASE = os.path.join(FIG11_BASE, "baseline_large_maps")

MODEL_SUFFIX = {
    "gpt-5-2025-08-07": "gpt-5-2025-08-07",
    "gpt-5-mini-2025-08-07": "gpt-5-mini-2025-08-07",
    "gpt-5-nano-2025-08-07": "gpt-5-nano-2025-08-07",
}

CONFIGS = ["2a_food", "2a_stuff", "4a_food", "4a_stuff", "5a_food", "5a_stuff"]


# ─────────────────────────────────────────────────────────────────────────────
# Parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_validation(filepath):
    """Parse validation.jsonl → list of dicts sorted by frame."""
    entries = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    entries.sort(key=lambda e: e['frame'])
    return entries


def extract_plans_from_action_history(val_entries):
    """
    Extract all (step, plan_name) pairs from action_history across all entries.
    Skips canceled plans (suffix " - canceled").
    Returns a sorted list of (step, plan_name).
    """
    all_plans = set()
    for entry in val_entries:
        for action_str in entry.get('action_history', []):
            action_str = action_str.strip()

            if action_str.endswith(" - canceled"):
                continue

            step = None
            plan = None

            if " at initial step" in action_str:
                plan = action_str.split(" at initial step")[0].strip()
                step = 0
            elif " at step " in action_str:
                idx = action_str.rfind(" at step ")
                plan = action_str[:idx].strip()
                rest = action_str[idx + len(" at step "):].strip()
                step = int(rest.split()[0])

            if step is not None and plan is not None:
                all_plans.add((step, plan))

    return sorted(all_plans, key=lambda x: x[0])


def parse_plan_tracking(filepath):
    """Parse plan_tracking.txt → all_plans list.

    all_plans: (frame, plan) for both "Plan started" and "Plan replaced" targets.
    """
    all_plans = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.match(r'\[Frame (\d+)\]', line)
            if not m:
                continue
            frame = int(m.group(1))

            if 'Cache lines:' in line:
                continue

            started = re.search(r'Plan started: (.+)$', line)
            if started:
                all_plans.append((frame, started.group(1).strip()))
                continue

            replaced = re.search(r"Plan replaced: '.+' -> '(.+)' by", line)
            if replaced:
                all_plans.append((frame, replaced.group(1).strip()))
                continue

    return all_plans


# ─────────────────────────────────────────────────────────────────────────────
# Compute event-based cumulative accuracy
# ─────────────────────────────────────────────────────────────────────────────

def compute_event_accuracy(val_entries, all_plans, max_frame):
    """
    Per-validation-event cumulative accuracy (step function).

    For each validation interval [f_{i-1}, f_i]:
      - Collect all non-canceled plans executed in that range.
      - If validation_plan appears among them → numerator += 1, denominator += 1
      - Otherwise                             → denominator += 1
      - accuracy = numerator / denominator

    Returns a dense array of length (max_frame + 1) for frames [0 .. max_frame].
    Accuracy holds constant between validation events.
    """
    acc_array = np.zeros(max_frame + 1, dtype=np.float64)

    if len(val_entries) < 2:
        return acc_array

    numerator = 0
    denominator = 0

    for i in range(1, len(val_entries)):
        f_prev = val_entries[i - 1]['frame']
        f_curr = val_entries[i]['frame']
        vplan = val_entries[i]['validation_plan']

        # All non-canceled plans executed in [f_prev, f_curr] (inclusive)
        executed = set(
            plan for step, plan in all_plans if f_prev <= step <= f_curr
        )

        if vplan in executed:
            numerator += 1
        denominator += 1

        acc_val = numerator / denominator if denominator > 0 else 0.0
        start = min(f_curr + 1, max_frame + 1)
        acc_array[start:] = acc_val

    # # If last validation frame < max_frame, treat remaining as correct
    # if val_entries and val_entries[-1]['frame'] < max_frame and denominator > 0:
    #     numerator += 1
    #     denominator += 1
    #     acc_val = numerator / denominator
    #     start = min(val_entries[-1]['frame'] + 1, max_frame + 1)
    #     acc_array[start:] = acc_val

    return acc_array


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate across scenarios
# ─────────────────────────────────────────────────────────────────────────────

def collect_accuracy_curves(model, max_frame=MAX_FRAME, dirs=None):
    """Collect all accuracy curves for a given model across all dirs & agents."""
    if dirs is None:
        dirs = DIRS
    curves = []
    for d in dirs:
        dirpath = os.path.join(RESULT_LOG_BASE, d)
        if not os.path.isdir(dirpath):
            continue
        pattern = os.path.join(dirpath, f"agent_*_lm_{model}_validation.jsonl")
        for vfile in sorted(glob.glob(pattern)):
            val_entries = parse_validation(vfile)
            if not val_entries:
                continue

            # Plans from action_history
            ah_plans = extract_plans_from_action_history(val_entries)

            # Plans from plan_tracking
            pt_file = vfile.replace("_validation.jsonl", "_plan_tracking.txt")
            pt_all = []
            if os.path.exists(pt_file):
                pt_all = parse_plan_tracking(pt_file)

            # Merge both sources
            merged = sorted(set(ah_plans) | set(pt_all), key=lambda x: x[0])

            acc = compute_event_accuracy(val_entries, merged, max_frame)
            curves.append(acc)
    return curves


# ─────────────────────────────────────────────────────────────────────────────
# Success rate from table3-validation
# ─────────────────────────────────────────────────────────────────────────────

def compute_success_rate_oursplus(model):
    """Compute average SR for Ours+ from table3-validation eval_result.json (same experiment)."""
    model_dir = f"LMs-{model}"
    sr_vals = []
    for d in DIRS:
        eval_path = os.path.join(VALIDATION_BASE, model_dir, d, "eval_result.json")
        if not os.path.exists(eval_path):
            continue
        with open(eval_path) as f:
            data = json.load(f)
        for ep in data.get("episode_results", {}).values():
            sr_vals.append(ep["finish"] / ep["total"])
    if not sr_vals:
        return None
    return sum(sr_vals) / len(sr_vals) * 100



def compute_success_rate_baseline(model):
    """Compute average SR for Baseline from baseline_large_maps eval_result.json."""
    suffix = MODEL_SUFFIX[model]
    sr_vals = []
    for c in CONFIGS:
        d = f"eval_10objs_{c}_{suffix}"
        eval_path = os.path.join(BASELINE_BASE, d, "run_eval", "eval_result.json")
        if not os.path.exists(eval_path):
            continue
        with open(eval_path) as f:
            data = json.load(f)
        for ep in data.get("episode_results", {}).values():
            sr_vals.append(ep["finish"] / ep["total"])
    if not sr_vals:
        return None
    return sum(sr_vals) / len(sr_vals) * 100


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_plan_execution_accuracy():
    frames = np.arange(0, MAX_FRAME + 1)

    fig, ax_left = plt.subplots(figsize=(4.0, 2.6))
    ax_right = ax_left.twinx()

    colors = {
        "gpt-5-2025-08-07": "#3A86FF",
        "gpt-5-mini-2025-08-07": "#E63946",
        "gpt-5-nano-2025-08-07": "#6A994E",
    }

    # ── Left axis: Plan Execution Accuracy curves with confidence bands ──
    for model in MODELS:
        curves = collect_accuracy_curves(model, MAX_FRAME)
        if not curves:
            print(f"  [WARN] No curves for {model}")
            continue
        arr = np.array(curves)
        mean_acc = np.mean(arr, axis=0)
        se_acc = np.std(arr, axis=0) / np.sqrt(len(curves))

        c = colors[model]
        sr_ours = compute_success_rate_oursplus(model)
        sr_base = compute_success_rate_baseline(model)
        dname = MODEL_DISPLAY[model]
        print(f"  {dname}: {len(curves)} curves, final acc={mean_acc[-1]:.3f}, "
              f"Ours+ SR={sr_ours}, Baseline SR={sr_base}")

        label = f"{dname} (Ours+:{sr_ours:.0f}% / BL:{sr_base:.0f}%)"
        ax_left.plot(frames, mean_acc, color=c, linewidth=1.3, label=label, zorder=3)
        ax_left.fill_between(frames, mean_acc - se_acc, mean_acc + se_acc,
                             color=c, alpha=0.12, zorder=2)

    # ── Right axis: SR reference lines (no text annotations, info is in legend) ──
    for model in MODELS:
        c = colors[model]
        sr_ours = compute_success_rate_oursplus(model)
        sr_base = compute_success_rate_baseline(model)
        if sr_ours is not None:
            ax_right.axhline(y=sr_ours, color=c, linestyle='-', linewidth=1.2,
                             alpha=0.5, zorder=1)
        if sr_base is not None:
            ax_right.axhline(y=sr_base, color=c, linestyle='--', linewidth=1.2,
                             alpha=0.5, zorder=1)

    # Dummy lines for SR legend
    ax_right.plot([], [], color='gray', linestyle='-', linewidth=0.8, label='Ours+ SR')
    ax_right.plot([], [], color='gray', linestyle='--', linewidth=0.8, label='Baseline SR')

    # ── Left axis formatting ──
    ax_left.set_xlabel("Frame", fontsize=8)
    ax_left.set_ylabel("Plan Execution Accuracy", fontsize=8)
    ax_left.set_xlim(0, MAX_FRAME)
    ax_left.set_ylim(0.0, 1.0)
    ax_left.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax_left.set_xticks(np.arange(0, MAX_FRAME + 1, 500))
    ax_left.tick_params(axis='both', labelsize=6, length=2, width=0.5, pad=2)
    ax_left.grid(True, alpha=0.2, linewidth=0.4, zorder=0)

    # ── Right axis formatting ──
    ax_right.set_ylabel("Success Rate (%)", fontsize=8)
    ax_right.set_ylim(0, 100)
    ax_right.set_yticks(np.arange(0, 101, 20))
    ax_right.tick_params(axis='y', labelsize=6, length=2, width=0.5, pad=2)

    # ── Combined legend ──
    h1, l1 = ax_left.get_legend_handles_labels()
    h2, l2 = ax_right.get_legend_handles_labels()
    leg = ax_left.legend(h1 + h2, l1 + l2, loc='lower right', fontsize=5.5,
                         frameon=True, edgecolor='black', fancybox=False,
                         handlelength=1.8, borderpad=0.4, labelspacing=0.3,
                         framealpha=0.5)
    leg.get_frame().set_linewidth(0.6)

    for spine in ax_left.spines.values():
        spine.set_linewidth(0.5)
    for spine in ax_right.spines.values():
        spine.set_linewidth(0.5)

    plt.tight_layout(pad=0.4)
    plt.subplots_adjust(right=0.82)

    # Save
    out_pdf = os.path.join(OUTPUT_DIR, "reproduce_figure11.pdf")
    out_png = os.path.join(OUTPUT_DIR, "reproduce_figure11.png")
    fig.savefig(out_pdf, dpi=300, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(out_png, dpi=300, bbox_inches='tight', pad_inches=0.02)
    print(f"\nSaved: {out_pdf}")
    print(f"Saved: {out_png}")
    plt.close(fig)


if __name__ == "__main__":
    print("=" * 60)
    print("Reproducing Figure 11: Plan Execution Accuracy")
    print("Paper: AgenticCache (MLSys 2026)")
    print("=" * 60)
    plot_plan_execution_accuracy()
    print("\nDone. Figures saved to reproduce_figure11.pdf/png")
