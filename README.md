# MLSys'26 Artifact Evaluation — AgenticCache

[![DOI](https://zenodo.org/badge/DOI/TODO.svg)](https://doi.org/TODO)

This repository contains the artifacts for the MLSys 2026 paper **AgenticCache**.

> **License:** [MIT](./LICENSE) (submodules retain their original licenses; see [Licenses](#licenses))  
> **Persistent Archive:** A Zenodo deposit with a permanent DOI will be linked here once available.

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
├── reproduce_figure4.py               # Reproduce Figure 4
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

All benchmarks require an OpenAI API key for LLM inference (GPT-5 / GPT-5-mini / GPT-5-nano).

```bash
export OPENAI_API_KEY="sk-..."
```

You can also add this to your `~/.bashrc` or `~/.zshrc` for persistence.

### 3. Setup Conda Environments

Create conda environments in your home directory using pre-configured environment files:

```bash
# Create all three environments
conda env create -f envs/coherent.yml
conda env create -f envs/coela.yml
conda env create -f envs/combo.yml
```

**Note**: Each submodule has its own `environment.yml` on the `baseline` branch if you prefer manual setup.

> **CUDA version mismatch?** The environment YAMLs pin specific CUDA/cuDNN versions. If your system has a different CUDA driver, you may need to adjust the `cudatoolkit` / `pytorch-cuda` version in the YAML before creating the environment. Run `nvidia-smi` to check your driver version.

### 4. Smoke Test (Minimal Validation)

To quickly verify that your environment, API key, and dependencies are working correctly, run one of the smoke test scripts. Each runs **1 branch × 1 model × 1 episode** (~5–10 min):

```bash
# COHERENT only (no Xorg needed)
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

### 5. Setup X Server (for CoELA & COMBO)

CoELA and COMBO use [TDW (ThreeDWorld)](https://www.threedworld.org/), which requires an active X server. **Skip this if you're only running COHERENT.**

See the [TDW server setup guide](https://github.com/threedworld-mit/tdw/blob/master/Documentation/lessons/setup/server.md) for full details.

#### a) Kill existing X server processes

```bash
# Check what's running
nvidia-smi  # Look for Xorg / gnome-shell processes

# Kill them
sudo kill -9 <PID_of_Xorg>
sudo kill -9 <PID_of_gnome-shell>
```

#### b) Start a new X server

We use display `:1` with a custom xorg config:

```bash
sudo nohup Xorg :1 -config /etc/X11/xorg-1.conf &
```

Verify with `nvidia-smi` — you should see an `Xorg` process on the target GPU.

## Pre-trained COMBO Checkpoint

COMBO evaluation requires a finetuned inpainting VDM checkpoint (`modl-100.pt`). You can either download the pre-trained checkpoint or train from scratch.

**Download pre-trained checkpoint (recommended):**

> **Google Drive:** [modl-100.pt.tar.gz](https://drive.google.com/file/d/1zXoVxRLNwet4PZVMvk9OWDO1BquO0ReX/view?usp=drive_link)

After downloading, extract and place:
```bash
tar xzf modl-100.pt.tar.gz
cp modl-100.pt MLSys26_AgenticCache-COMBO/tdw_maco/modl-100.pt
```

If you prefer to train from scratch, see the section below.

## COMBO: Finetuning (before evaluation)

COMBO requires a finetuned inpainting VDM model before running experiments. The full pipeline is on the `training-code` branch and **also requires Xorg** for TDW data collection.

### Automated training script

```bash
./scripts/run_combo_train.sh              # full pipeline (env setup → data gen → preprocess → train)
./scripts/run_combo_train.sh --step 1     # skip env setup (env already exists)
./scripts/run_combo_train.sh --step 2     # skip data generation
./scripts/run_combo_train.sh --step 3     # skip data gen + preprocess (train only)
```

### Manual training (alternative)

```bash
cd MLSys26_AgenticCache-COMBO
git checkout training-code
cd AVDC/flowdiffusion
bash train_all.sh              # full pipeline (env setup → data gen → preprocess → train)
bash train_all.sh --step 1     # skip env setup (env already exists)
bash train_all.sh --step 2     # skip data generation
bash train_all.sh --step 3     # skip data gen + preprocess (train only)
```

| Step | Description | Xorg needed? |
|------|-------------|:------------:|
| 0 | Conda env setup | No |
| 1 | Generate train/test data via TDW (`DISPLAY=:1`) | **Yes** |
| 2 | Preprocess text embeddings (T5-XXL) | No |
| 3 | Train inpainting diffusion model (100K steps) | No |

The final checkpoint (`modl-100.pt`) is used by all experiment branches.

## Execution Workflow

### Overview

1. **COMBO Training** (optional, if checkpoint not available): `./scripts/run_combo_train.sh`
2. **Run Evaluations**: Use the automated scripts below to run all branches

All scripts automatically:
- Checkout each branch (`baseline`, `agenticcache`, `parallel`, `speculative`)
- Execute the per-branch evaluation script using the appropriate conda environment
- Restore the original branch after completion

### What Each Script Does

| Script | Submodule Script | Environment | Description |
|--------|------------------|-------------|-------------|
| `run_coherent.sh` | `scripts/run_all.sh` | `coherent` | Runs all 3 models × 5 envs (no Xorg) |
| `run_coela.sh` | `scripts/test_2_LMs-gpt-5.sh` | `coela` | Runs all 3 models on test_2 split (requires Xorg) |
| `run_combo.sh` | `scripts/run_gpt5_all.sh` | `combo` | Runs all 3 models on cook+game tasks (requires Xorg) |
| `run_combo_train.sh` | `train_all.sh` | `combo` | Trains inpainting VDM model (requires Xorg) |

**Note**: All submodule scripts use `python3` or `python` directly. The wrapper scripts (`scripts/run_*.sh`) execute them via `conda run -n <env>` to ensure the correct environment is activated.

## Evaluation

Please refer to the README in each submodule for benchmark-specific setup and reproduction instructions:

- [COHERENT](./MLSys26_AgenticCache-COHERENT/)
- [CoELA](./MLSys26_AgenticCache-CoELA/)
- [COMBO](./MLSys26_AgenticCache-COMBO/)

### Cache Episodes (excluded from evaluation)

The following episodes are used to build the AgenticCache and are **excluded** from evaluation runs:

| Benchmark | Cache Episodes |
|-----------|---------------|
| **COHERENT** | `env0/task_15`, `env1/task_10`, `env2/task_11`, `env3/task_16` |
| **CoELA** | test_2 episodes `1 2 3 4` |
| **COMBO** | cook `0 1`, game `0` |

### Run All Branches Automatically

The following scripts automatically iterate over all 4 branches (`baseline`, `agenticcache`, `parallel`, `speculative`), check out each one, run the experiments, and restore the original branch:

```bash
# 1. COMBO Training (run once before evaluation)
./scripts/run_combo_train.sh              # full pipeline
./scripts/run_combo_train.sh --step 3     # train only (if data already exists)

# 2. COHERENT Evaluation (no Xorg needed)
./scripts/run_coherent.sh                 # all models × all envs

# 3. CoELA Evaluation (requires Xorg on :1)
./scripts/run_coela.sh                    # all models on test_2 split

# 4. COMBO Evaluation (requires Xorg on :1)
./scripts/run_combo.sh                    # all tasks (cook + game)
./scripts/run_combo.sh cook               # cook only
./scripts/run_combo.sh game               # game only
```

## Results

All pre-run result logs are available for download. After downloading, extract them into the `results/` directory:

> **Google Drive:** [results.tar.gz](https://drive.google.com/file/d/1f8dwVKWVlNicKy0SL61jfaQ_kzberH-D/view?usp=drive_link)

```bash
cd results/
tar xzf results.tar.gz
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

### Reproducing Tables and Figures

We provide analysis scripts that parse the raw JSON logs and reproduce the tables and figures in the paper. These scripts require **no GPU, no API key, and no simulator** — only Python with `pandas` and `matplotlib`.

| Script | Paper Claim | Description |
|--------|-------------|-------------|
| `reproduce_table2.py` | Table 2 | Main planning strategy performance across benchmarks |
| `reproduce_table3.py` | Table 3 | Cold-start results (3000 frame limit) |
| `reproduce_table4.py` | Table 4 | Cold-start results (6000 frame limit) |
| `reproduce_figure4.py` | Figure 4 | Plan transition distribution (n-gram analysis) |
| `reproduce_figure11.py` | Figure 11 | Plan execution accuracy over time |

```bash
# Run all reproduction scripts (no GPU needed)
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
