# MLSys'26 Artifact Evaluation — AgenticCache

This repository contains the artifacts for the MLSys 2026 paper **AgenticCache**.

## Repository Structure

```
├── MLSys26_AgenticCache-COHERENT/   # Evaluation on COHERENT benchmark
├── MLSys26_AgenticCache-CoELA/      # Evaluation on CoELA benchmark
├── MLSys26_AgenticCache-COMBO/      # Evaluation on COMBO benchmark
```

## Getting Started

### Clone with all submodules

```bash
git clone --recursive git@github.com:hojoonleokim/MLSys26_AgenticCache.git
```

### If you already cloned without `--recursive`

```bash
git submodule update --init --recursive
```

### Updating submodules to the latest commit

```bash
git submodule update --remote --merge
```

## Environment Setup

Pre-configured environment files are available in the `envs/` directory. Create the conda environments:

```bash
# COHERENT
conda env create -f envs/coherent.yml

# CoELA
conda env create -f envs/coela.yml

# COMBO
conda env create -f envs/combo.yml
```

Alternatively, each submodule has its own `environment.yml` on the `baseline` branch.

## X Server Setup (CoELA & COMBO only)

CoELA and COMBO use [TDW (ThreeDWorld)](https://www.threedworld.org/), which requires an active X server. See the [TDW server setup guide](https://github.com/threedworld-mit/tdw/blob/master/Documentation/lessons/setup/server.md) for full details.

### 1. Kill existing X server processes

```bash
# Check what's running
nvidia-smi  # Look for Xorg / gnome-shell processes

# Kill them
sudo kill -9 <PID_of_Xorg>
sudo kill -9 <PID_of_gnome-shell>
```

### 2. Start a new X server

We use display `:1` with a custom xorg config:

```bash
sudo nohup Xorg :1 -config /etc/X11/xorg-1.conf &
```

Verify with `nvidia-smi` — you should see an `Xorg` process on the target GPU.

## COMBO: Finetuning (before evaluation)

COMBO requires a finetuned inpainting VDM model before running experiments. The full pipeline is on the `training-code` branch and **also requires Xorg** for TDW data collection.

```bash
cd MLSys26_AgenticCache-COMBO
git checkout training-code
```

### End-to-end training pipeline

The script `AVDC/flowdiffusion/train_all.sh` runs all steps:

```bash
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

Each benchmark has 4 branches: `baseline`, `agenticcache`, `parallel`, `speculative`. The scripts below automatically iterate over all branches, check out each one, and run the experiments.

```bash
# COHERENT — no Xorg needed
./scripts/run_coherent.sh              # all envs, gpt-5-2025-08-07

# CoELA — requires Xorg on :1
./scripts/run_coela.sh

# COMBO — requires Xorg on :1
./scripts/run_combo.sh                 # all tasks (cook + game)
```

## Results

Pre-run result logs for all benchmarks and methods are available for download:

**Google Drive:** https://drive.google.com/file/d/1L33wrEXIl2mOA3OrRLqkJ7zRVDpD5Qtm/view?usp=sharing
