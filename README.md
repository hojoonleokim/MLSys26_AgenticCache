# MLSys'26 Artifact Evaluation — AgenticCache

This repository contains the artifacts for the MLSys 2026 paper **AgenticCache**.

## Repository Structure

```
├── MLSys26_AgenticCache-COHERENT/   # Evaluation on COHERENT benchmark
├── MLSys26_AgenticCache-CoELA/      # Evaluation on CoELA benchmark
├── MLSys26_AgenticCache-COMBO/      # Evaluation on COMBO benchmark
├── scripts/                          # Automated run scripts for all benchmarks
├── envs/                             # Pre-configured conda environment files
```

## Quick Start

### 1. Clone Repository

```bash
git clone --recursive git@github.com:hojoonleokim/MLSys26_AgenticCache.git
cd MLSys26_AgenticCache
```

If you already cloned without `--recursive`:
```bash
git submodule update --init --recursive
```

### 2. Setup Conda Environments

Create conda environments in your home directory using pre-configured environment files:

```bash
# Create all three environments
conda env create -f envs/coherent.yml
conda env create -f envs/coela.yml
conda env create -f envs/combo.yml
```

**Note**: Each submodule has its own `environment.yml` on the `baseline` branch if you prefer manual setup.

### 3. Setup X Server (for CoELA & COMBO)

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

Pre-run result logs for all benchmarks and methods are available for download:

**Google Drive:** https://drive.google.com/file/d/1L33wrEXIl2mOA3OrRLqkJ7zRVDpD5Qtm/view?usp=sharing
