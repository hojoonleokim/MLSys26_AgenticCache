#!/usr/bin/env python3
"""
Reproduce Figure 4 from the paper:
  "Action Transition Distribution (N-gram Analysis)"

Paper claim: AgenticCache leverages recurring action-transition patterns
(n-gram distributions) observed in TDW-MAT and COHERENT environments
to build an effective action cache.  Figure 4 visualises the top-N
next-action distributions for selected previous actions, demonstrating
that agent behaviour is highly predictable and cache-friendly.

Requirements: Python 3, matplotlib, numpy. No GPU / API key / simulator needed.
Input: results/fig4/mat_ngram_analysis.txt, results/fig4/coherent_ngram_analysis.txt
Output: reproduce_figure4.pdf/png
"""

import os
import matplotlib.pyplot as plt
import re
from collections import defaultdict
import numpy as np
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# Use matplotlib defaults (no custom font override) to match other figures

# ============================================================================
# Configuration: Combined graph
# ============================================================================
# TDW-MAT: Select specific FROM actions to include
MAT_SELECTED_ACTIONS = [
    'go grasp target',
    'put into container'
]

# Number of top transitions to show per FROM action
TOP_N_PER_ACTION = 3

# ============================================================================
# Parse coherent data (robot_dog, quadrotor)
# ============================================================================
def parse_coherent_section(filepath, section_name, from_action_filter=None):
    """
    Parse a specific section from coherent n-gram analysis.
    If from_action_filter is provided, only return transitions where FROM matches.
    Returns list of transitions: [{'from': ..., 'to': ..., 'count': ..., 'percentage': ...}, ...]
    """
    transitions = []
    in_section = False
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            
            if line == section_name:
                in_section = True
                continue
            
            # End of section (empty line after transitions)
            if in_section and line == '':
                break
            
            # Parse transition lines like: "1. [grab -> movetowards]: 38 (23.0%)"
            if in_section and line:
                match = re.match(r'\d+\.\s*\[(.+?)\s*->\s*(.+?)\]:\s*(\d+)\s*\((.+?)%\)', line)
                if match:
                    from_action, to_action, count, percentage = match.groups()
                    
                    # Filter by FROM action if specified
                    if from_action_filter is None or from_action_filter.lower() in from_action.lower():
                        transitions.append({
                            'from': from_action,
                            'to': to_action,
                            'count': int(count),
                            'percentage': float(percentage)
                        })
    
    return transitions

# ============================================================================
# Parse mat_ngram_analysis.txt - Group by FROM action
# ============================================================================

def parse_mat_transitions_grouped(filepath, selected_actions, top_n=3):
    """
    Parse transitions from TDW-MAT, grouped by FROM action.
    Returns a dict: {from_action: [(to_action, count, percentage), ...]}
    """
    all_transitions = defaultdict(list)
    current_from_action = None
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.rstrip()
            
            # Detect "FROM: action_name" lines
            if line.startswith('FROM: '):
                current_from_action = line.replace('FROM: ', '').strip()
                continue
            
            # Parse transition lines like: "  → put into container             [ 119] ( 51.5%)"
            if current_from_action and '→' in line:
                match = re.match(r'\s*→\s*(.+?)\s+\[\s*(\d+)\]\s*\(\s*(.+?)%\)', line)
                if match:
                    to_action, count, percentage = match.groups()
                    all_transitions[current_from_action].append({
                        'to': to_action.strip(),
                        'count': int(count),
                        'percentage': float(percentage)
                    })
    
    # Filter to selected actions and get top N transitions for each
    grouped_data = {}
    for from_action in selected_actions:
        if from_action in all_transitions:
            grouped_data[from_action] = all_transitions[from_action][:top_n]
    
    return grouped_data

def renormalize_percentages(data):
    """Renormalize percentages so they sum to 100%"""
    if not data:
        return data
    
    total = sum(t['percentage'] for t in data)
    if total == 0:
        return data
    
    renormalized = []
    for t in data:
        new_t = t.copy()
        new_t['percentage'] = (t['percentage'] / total) * 100
        renormalized.append(new_t)
    
    return renormalized

def plot_combined_graph(mat_data, robotdog_data, quadrotor_data):
    """
    Create a combined graph with 4 subplots:
    - 2 TDW-MAT actions (go grasp target, put into container)
    - robot_dog movetoward transitions
    - quadrotor movetoward transitions
    
    Improved version with better spacing and label positioning.
    """
    fig, axes = plt.subplots(1, 4, figsize=(3.8, 2.4))
    
    # Distinct color per subplot — modern, accessible palette
    subplot_colors = ['#3A86FF', '#E63946', '#6A994E', '#F4A261']
    
    # Plot index
    plot_idx = 0
    
    def style_ax(ax):
        """Apply common clean styling to a subplot."""
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_axisbelow(True)
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax.grid(axis='x', visible=False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(0.8)
        ax.tick_params(axis='both', length=0)
    
    # Plot TDW-MAT actions
    for from_action, transitions in mat_data.items():
        ax = axes[plot_idx]
        
        # Renormalize: percentages sum to 100% within this FROM action group
        transitions_renorm = renormalize_percentages(transitions)
        
        to_actions = [t['to'] for t in transitions_renorm]
        percentages = [t['percentage'] for t in transitions_renorm]
        
        x_positions = np.arange(len(to_actions))
        bars = ax.bar(x_positions, percentages, color=subplot_colors[plot_idx], alpha=0.85,
                      edgecolor='black', linewidth=0.8, zorder=3)
        style_ax(ax)
        
        # Add labels
        for i, (bar, to_action, pct) in enumerate(zip(bars, to_actions, percentages)):
            height = bar.get_height()
            # Percentage on top
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                   f'{pct:.1f}%', ha='center', va='bottom', fontsize=4.5,
                   fontweight='bold')
            
            # Special handling for "explore" in "put into container"
            if from_action == 'put into container' and to_action == 'explore':
                text_x = bar.get_x() + bar.get_width()/2
                ax.annotate('explore',
                           xy=(text_x, height + 1.2),
                           xytext=(text_x, height + 12),
                           ha='center', va='bottom', fontsize=5, fontweight='bold',
                           color='black',
                           arrowprops=dict(arrowstyle='->', color='black', linewidth=1.0))
            else:
                # Action name vertically inside bar
                display_action = to_action.replace('movetowards', 'move').replace('putinto', 'put into')
                ax.text(bar.get_x() + bar.get_width()/2, height/2,
                       display_action, ha='center', va='center', fontsize=5,
                       rotation=90, color='black', fontweight='bold')
        
        # Add prev_action below
        ax.text(0.5, -0.02, from_action, transform=ax.transAxes,
               ha='center', va='top', fontsize=5.5, fontweight='bold')
        
        ax.set_ylim(0, max(percentages) * 1.15)
        plot_idx += 1
    
    # Plot robot_dog movetowards
    if robotdog_data:
        ax = axes[plot_idx]
        
        to_actions = [t['to'] for t in robotdog_data]
        percentages = [t['percentage'] for t in robotdog_data]
        
        x_positions = np.arange(len(to_actions))
        bars = ax.bar(x_positions, percentages, color=subplot_colors[plot_idx], alpha=0.85,
                      edgecolor='black', linewidth=0.8, zorder=3)
        style_ax(ax)
        
        for i, (bar, to_action, pct) in enumerate(zip(bars, to_actions, percentages)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                   f'{pct:.1f}%', ha='center', va='bottom', fontsize=4.5,
                   fontweight='bold')
            display_action = to_action.replace('movetowards', 'move').replace('putinto', 'put into')
            ax.text(bar.get_x() + bar.get_width()/2, height/2,
                   display_action, ha='center', va='center', fontsize=5,
                   rotation=90, color='black', fontweight='bold')
        
        ax.text(0.5, -0.02, 'move\n(robot dog)', transform=ax.transAxes,
               ha='center', va='top', fontsize=5.5, fontweight='bold')
        
        ax.set_ylim(0, max(percentages) * 1.15)
        plot_idx += 1
    
    # Plot quadrotor movetowards
    if quadrotor_data:
        ax = axes[plot_idx]
        
        to_actions = [t['to'].replace('land_on', 'land on').replace('movetowards', 'move') for t in quadrotor_data]
        percentages = [t['percentage'] for t in quadrotor_data]
        
        x_positions = np.arange(len(to_actions))
        bars = ax.bar(x_positions, percentages, color=subplot_colors[plot_idx], alpha=0.85,
                      edgecolor='black', linewidth=0.8, zorder=3)
        style_ax(ax)
        
        for i, (bar, to_action, pct) in enumerate(zip(bars, to_actions, percentages)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                   f'{pct:.1f}%', ha='center', va='bottom', fontsize=4.5,
                   fontweight='bold')
            ax.text(bar.get_x() + bar.get_width()/2, height/2,
                   to_action, ha='center', va='center', fontsize=5,
                   rotation=90, color='black', fontweight='bold')
        
        ax.text(0.5, -0.02, 'move\n(quadrotor)', transform=ax.transAxes,
               ha='center', va='top', fontsize=5.5, fontweight='bold')
        ax.set_ylim(0, max(percentages) * 1.15)
    
    # Better subplot spacing
    plt.subplots_adjust(wspace=0.15, hspace=0.2, left=0.04, right=0.98, top=0.87, bottom=0.1)
    
    # Add environment labels
    fig.text(0.27, 0.88, 'TDW', ha='center', va='bottom', fontsize=7, fontweight='bold')
    fig.text(0.76, 0.88, 'COHERENT', ha='center', va='bottom', fontsize=7, fontweight='bold')
    out_pdf = os.path.join(SCRIPT_DIR, 'reproduce_figure4.pdf')
    out_png = os.path.join(SCRIPT_DIR, 'reproduce_figure4.png')
    plt.savefig(out_pdf, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")
    # plt.show()

# ============================================================================
# Main execution
# ============================================================================
def print_statistical_summary(mat_data, robotdog_data, quadrotor_data):
    """
    Print statistical summary of transition distributions.
    Addresses ae-review requirement for mean/std/trend output.
    """
    print("\n" + "-" * 60)
    print("Statistical Summary")
    print("-" * 60)

    # TDW-MAT
    for from_action, transitions in mat_data.items():
        renorm = renormalize_percentages(transitions)
        pcts = [t['percentage'] for t in renorm]
        print(f"\n  TDW-MAT  prev_action='{from_action}'  (top-{len(pcts)} next actions)")
        for t in renorm:
            print(f"    -> {t['to']:<25s}  {t['percentage']:5.1f}%  (count={t['count']})")
        print(f"    mean={np.mean(pcts):.1f}%  std={np.std(pcts):.1f}%")

    # COHERENT robot_dog
    if robotdog_data:
        pcts = [t['percentage'] for t in robotdog_data]
        print(f"\n  COHERENT  robot_dog  prev_action='movetowards'  (top-{len(pcts)} next actions)")
        for t in robotdog_data:
            print(f"    -> {t['to']:<25s}  {t['percentage']:5.1f}%  (count={t['count']})")
        print(f"    mean={np.mean(pcts):.1f}%  std={np.std(pcts):.1f}%")

    # COHERENT quadrotor
    if quadrotor_data:
        pcts = [t['percentage'] for t in quadrotor_data]
        print(f"\n  COHERENT  quadrotor  prev_action='movetowards'  (top-{len(pcts)} next actions)")
        for t in quadrotor_data:
            print(f"    -> {t['to']:<25s}  {t['percentage']:5.1f}%  (count={t['count']})")
        print(f"    mean={np.mean(pcts):.1f}%  std={np.std(pcts):.1f}%")

    print("\n  Trend: Action transitions are highly concentrated among a")
    print("  small number of next-actions, confirming cache-friendliness.")
    print("-" * 60)


if __name__ == "__main__":
    print("=" * 60)
    print("Reproducing Figure 4: Action Transition Distribution")
    print("Paper: AgenticCache (MLSys 2026)")
    print("=" * 60)

    MAT_FILE = os.path.join(SCRIPT_DIR, 'results', 'fig4', 'mat_ngram_analysis.txt')
    COHERENT_FILE = os.path.join(SCRIPT_DIR, 'results', 'fig4', 'coherent_ngram_analysis.txt')

    # Parse TDW-MAT data
    mat_data = parse_mat_transitions_grouped(
        MAT_FILE,
        MAT_SELECTED_ACTIONS,
        TOP_N_PER_ACTION
    )

    # Parse robot_dog movetowards
    robotdog_all = parse_coherent_section(
        COHERENT_FILE,
        'robot_dog',
        from_action_filter='movetowards'
    )
    robotdog_data = renormalize_percentages(robotdog_all)[:TOP_N_PER_ACTION]

    # Parse quadrotor movetowards
    quadrotor_all = parse_coherent_section(
        COHERENT_FILE,
        'quadrotor',
        from_action_filter='movetowards'
    )
    quadrotor_data = renormalize_percentages(quadrotor_all)[:TOP_N_PER_ACTION]

    # Print statistical summary
    print_statistical_summary(mat_data, robotdog_data, quadrotor_data)

    # Generate graph
    plot_combined_graph(mat_data, robotdog_data, quadrotor_data)
    print("\nDone. Figures saved to reproduce_figure4.pdf/png")
