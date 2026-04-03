# MLSys'26 Artifact Evaluation — AgenticCache

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19396846.svg)](https://doi.org/10.5281/zenodo.19396846)

This repository contains the artifacts for the MLSys 2026 paper **AgenticCache**.

> **License:** [MIT](./LICENSE) (submodules retain their original licenses; see [Licenses](#licenses))  
> **Persistent Archive:** [Zenodo DOI 10.5281/zenodo.19396846](https://doi.org/10.5281/zenodo.19396846)

## Repository Structure

```
├── MLSys26_AgenticCache-COHERENT/   # COHERENT agent (BEHAVIOR-1K benchmark)
├── MLSys26_AgenticCache-CoELA/      # CoELA agent (TDW-MAT benchmark)
├── MLSys26_AgenticCache-COMBO/      # COMBO agent (TDW-COOK, TDW-GAME benchmarks)
├── scripts/                          # Automated run scripts for all benchmarks
├── envs/                             # Pre-configured conda environment files
├── results/                          # Pre-run evaluation logs
│   ├── table2/                       #   Table 2: Planning strategy performance
│   ├── table3/                       #   Table 3: Cold-start (3000 frame, 10objs)
│   ├── table4/                       #   Table 4: Cold-start (6000 frame, 30objs)
│   ├── fig4/                         #   Figure 4: Plan Transition Distribution
│   └── fig11/                        #   Figure 11: Plan Execution Accuracy
├── reproduce_table2.py               # Reproduce Table 2
├── reproduce_table3.py               # Reproduce Table 3
├── reproduce_table4.py               # Reproduce Table 4
├── reproduce_figure4.py              # Reproduce Figure 4
└── reproduce_figure11.py             # Reproduce Figure 11
```

## Quick Start

### 1. Clone Repository

```bash
git clone --recursive https://github.com/hojoonleokim/MLSys26_AgenticCache.git
cd MLSys26_AgenticCache
```

If you already cloned without `--recursive`:
```bash
git submodule update --init --recursive
```

### 2. Set OpenAI API Key

All benchmarks require an OpenAI API key for LLM inference (GPT-5 `gpt-5-2025-08-07` / GPT-5-mini `gpt-5-mini-2025-08-07` / GPT-5-nano `gpt-5-nano-2025-08-07`).

```bash
export OPENAI_API_KEY="sk-..."
```

You can also add this to your shell profile (`~/.bashrc` / `~/.zshrc`) for persistence.

### 3. Setup Conda Environments

Create conda environments using the pre-configured environment files:

```bash
# Create all three environments
conda env create -f envs/coherent.yml
conda env create -f envs/coela.yml
conda env create -f envs/combo.yml
```

**Note**: Each submodule also ships its own `environment.yml` (available on every branch) if you prefer manual setup.

> **CUDA version mismatch?** The environment YAMLs pin specific CUDA/cuDNN versions. If your system has a different CUDA driver, you may need to adjust the `cudatoolkit` / `pytorch-cuda` version in the YAML before creating the environment. Run `nvidia-smi` to check your driver version.

### 4. Setup X Server (for CoELA & COMBO)

CoELA and COMBO use [TDW (ThreeDWorld)](https://www.threedworld.org/), which requires an active X server. **Skip this step if you're only running COHERENT.**

See the [TDW server setup guide](https://github.com/threedworld-mit/tdw/blob/master/Documentation/lessons/setup/server.md) for full details.

> **Note:** The instructions below assume a **headless server** (no desktop environment). If you are on a machine with an active desktop session, you can skip step (a) and directly start a new X server on an unused display (e.g., `:1`).

#### a) (Headless only) Kill existing X server processes

If stale Xorg processes are occupying the GPU, remove them first:

```bash
nvidia-smi  # Look for Xorg / gnome-shell processes
sudo kill -9 <PID_of_Xorg>
```

#### b) Start a new X server on display `:1`

```bash
sudo nohup Xorg :1 -config /etc/X11/xorg-1.conf &
```

Verify with `nvidia-smi` — you should see an `Xorg` process on the target GPU.

### 5. Smoke Test (Minimal Validation)

To quickly verify that your environment, API key, and dependencies are working correctly, run one of the smoke test scripts. Each runs **1 branch × 1 model × 1 episode** (~5–10 min):

```bash
# COHERENT (no Xorg needed)
./scripts/smoke_test_coherent.sh

# CoELA (requires Xorg on :1)
./scripts/smoke_test_coela.sh

# COMBO (requires Xorg on :1)
./scripts/smoke_test_combo.sh
```

| Script | Agent | Benchmark | Branch | Model | Scope | Xorg? |
|--------|-------|-----------|--------|-------|-------|:-----:|
| `smoke_test_coherent.sh` | COHERENT | BEHAVIOR-1K | agenticcache | gpt-5 | env0, task 2 | No |
| `smoke_test_coela.sh` | CoELA | TDW-MAT | agenticcache | gpt-5 | episode 5 | **Yes** |
| `smoke_test_combo.sh` | COMBO | TDW-COOK | agenticcache | gpt-5 | cook episode 2 | **Yes** |

If a smoke test completes without errors, your setup is ready for full benchmark runs.

## Pre-trained COMBO Checkpoint

COMBO evaluation requires a finetuned inpainting VDM checkpoint (`modl-100.pt`). You can either download the pre-trained checkpoint or train from scratch.

**Download pre-trained checkpoint (recommended):**

> **Google Drive:** [modl-100.pt.tar.gz](https://drive.google.com/file/d/1zXoVxRLNwet4PZVMvk9OWDO1BquO0ReX/view?usp=drive_link)

After downloading, extract and place:
```bash
tar xzf modl-100.pt.tar.gz
cp modl-100.pt MLSys26_AgenticCache-COMBO/tdw_maco/modl-100.pt
```

## Execution Workflow

Each wrapper script in `scripts/` automatically checks out all 4 branches (`baseline`, `agenticcache`, `parallel`, `speculative`), runs the evaluation with the appropriate conda environment, and restores the original branch.

| Script | Submodule Script (relative to submodule root) | Environment | Description |
|--------|--------------------------------------------------|-------------|-------------|
| `run_coherent.sh` | `src/experiment/PEFA/scripts/run_all.sh` | `coherent` | Runs all 3 models × 5 envs (no Xorg) |
| `run_coela.sh` | `tdw_mat/scripts/test_2_LMs-gpt-5.sh` | `coela` | Runs all 3 models on test_2 split (requires Xorg) |
| `run_combo.sh` | `tdw_maco/scripts/run_gpt5_all.sh` | `combo` | Runs all 3 models on cook+game tasks (requires Xorg) |

```bash
# 1. COHERENT Evaluation (no Xorg needed)
./scripts/run_coherent.sh                 # all models × all envs

# 2. CoELA Evaluation (requires Xorg on :1)
./scripts/run_coela.sh                    # all models on test_2 split

# 3. COMBO Evaluation (requires Xorg on :1)
./scripts/run_combo.sh                    # all tasks (cook + game)
./scripts/run_combo.sh cook               # cook only
./scripts/run_combo.sh game               # game only
```

**Note**: The wrapper scripts activate the appropriate conda environment automatically. If you run a submodule script manually (outside the wrapper), activate the environment first with `conda activate <env>`.

For benchmark-specific details, see each submodule's README:
[COHERENT](https://github.com/hojoonleokim/MLSys26_AgenticCache-COHERENT) · [CoELA](https://github.com/hojoonleokim/MLSys26_AgenticCache-CoELA) · [COMBO](https://github.com/hojoonleokim/MLSys26_AgenticCache-COMBO)

### Cache Episodes (excluded from evaluation)

The following episodes are used to warm up the AgenticCache and are **excluded** from evaluation runs:

| Benchmark | Cache Episodes |
|-----------|---------------|
| **COHERENT** | `env0/task_15`, `env1/task_10`, `env2/task_11`, `env3/task_16` |
| **CoELA** | test_2 episodes `1 2 3 4` |
| **COMBO** | cook `0 1`, game `0` |

## Results & Reproducing Tables and Figures

All pre-run result logs are available for download. These scripts require **no GPU, no API key, and no simulator** — the table scripts use only the Python standard library; the figure scripts additionally require `matplotlib` and `numpy`.

### 1. Download and extract pre-run logs

> **Google Drive:** [results.tar.gz](https://drive.google.com/file/d/1f8dwVKWVlNicKy0SL61jfaQ_kzberH-D/view?usp=drive_link) (~347 MB)

```bash
cd results/
tar xzf results.tar.gz
cd ..
```

The extracted structure:

```
results/
├── table2/        # Table 2: CoELA, COMBO, COHERENT eval logs (4 methods × 3 models)
├── table3/        # Table 3: Cold-start 10objs/3000-frame (Baseline, Ours, Ours+)
├── table4/        # Table 4: Cold-start 30objs/6000-frame (Baseline, Ours, Ours+)
├── fig4/          # Figure 4: N-gram transition data (TDW-MAT + COHERENT)
└── fig11/         # Figure 11: Validation logs, Ours+ & Baseline SR data
```

See [`results/README.md`](./results/README.md) for detailed structure.

### 2. Run reproduction scripts

Each table script prints a formatted table to stdout; each figure script saves both PDF and PNG files in the current directory.

| Script | Paper Claim | Description |
|--------|-------------|-------------|
| `reproduce_table2.py` | Table 2 | Main planning strategy performance across benchmarks |
| `reproduce_table3.py` | Table 3 | Cold-start results (3000 frame limit) |
| `reproduce_table4.py` | Table 4 | Cold-start results (6000 frame limit) |
| `reproduce_figure4.py` | Figure 4 | Plan transition distribution (n-gram analysis) |
| `reproduce_figure11.py` | Figure 11 | Plan execution accuracy over time |

```bash
# Run all reproduction scripts (Python 3.9+, no GPU needed)
pip install matplotlib numpy   # only needed for figure scripts

python reproduce_table2.py
python reproduce_table3.py
python reproduce_table4.py
python reproduce_figure4.py
python reproduce_figure11.py
```

## Licenses

This repository is released under the [MIT License](./LICENSE). Each submodule is an independent project and retains the license of its original authors:

| Submodule | Original Project | License | Copyright |
|-----------|-----------------|---------|-----------|
| `MLSys26_AgenticCache-COHERENT` | [OmniGibson](https://github.com/StanfordVL/OmniGibson) | MIT | 2023 Stanford Vision and Learning Group |
| `MLSys26_AgenticCache-CoELA` | [CoELA (tdw_mat)](https://github.com/UMass-Foundation-Model/Co-LLM-Agents) | MIT | 2023 Esther Alter |
| `MLSys26_AgenticCache-COMBO` | [AVDC](https://github.com/flow-diffusion/AVDC) | MIT | 2023 flow-diffusion |

## External Dependencies & Licenses

The following external dependencies require separate downloads or have their own license obligations:

| Dependency | Required For | License | Download |
|------------|-------------|---------|----------|
| [TDW (ThreeDWorld)](https://www.threedworld.org/) | CoELA, COMBO | BSD-2-Clause | Auto-downloaded via `pip install tdw` |
| [OmniGibson](https://behavior.stanford.edu/omnigibson/) | Not required (only for full BEHAVIOR-1K 3D sim) | Apache-2.0 | See [OmniGibson install guide](https://behavior.stanford.edu/omnigibson/getting_started/installation.html) |
| OpenAI API (GPT-5) | All benchmarks | Commercial API | Requires `OPENAI_API_KEY` |
| NVIDIA GPU (≥24GB VRAM) | All benchmarks | — | Hardware requirement |
| X11 / Xorg | CoELA, COMBO | — | System package |
