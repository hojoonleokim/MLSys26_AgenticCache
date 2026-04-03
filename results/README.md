# Experiment Results

This directory contains pre-run evaluation logs used by the reproduction scripts.

**Download:** The raw logs are hosted on Google Drive as `results.tar.gz` (~347 MB). See the [main README](../README.md#results--reproducing-tables-and-figures) for the download link and extraction instructions.

All evaluation logs are in JSON format (one file per episode per method).

## Structure

```
results/
├── table2/                # Table 2: Planning strategy performance
│   ├── CoELA/             #   Per-method eval_result.json + token logs
│   ├── COMBO/             #   Per-method eval_result.json + token logs + chat_log
│   └── COHERENT/          #   Per-method eval_result.json + token logs
├── table3/                # Table 3: Cold-start results (3000 frame, 10objs)
│   ├── baseline_large_maps/
│   ├── Ours/
│   └── Ours+/
├── table4/                # Table 4: Cold-start results (6000 frame, 30objs)
│   ├── baseline_large_maps/
│   ├── Ours/
│   └── Ours+/
├── fig4/                  # Figure 4: Plan Transition Distribution
│   ├── mat_ngram_analysis.txt
│   └── coherent_ngram_analysis.txt
└── fig11/                 # Figure 11: Plan Execution Accuracy
    ├── result_log/        #   validation.jsonl + plan_tracking.txt
    ├── LMs-gpt-5-*/       #   Ours+ eval_result.json (SR calculation), e.g. LMs-gpt-5-cook/
    └── baseline_large_maps/ # Baseline eval_result.json (SR calculation)
```

## Which script uses what

| Script | Data directory |
|--------|---------------|
| `reproduce_table2.py` | `results/table2/` |
| `reproduce_table3.py` | `results/table3/` |
| `reproduce_table4.py` | `results/table4/` |
| `reproduce_figure4.py` | `results/fig4/` |
| `reproduce_figure11.py` | `results/fig11/` |
