# DRIFTS

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
  - [Option 1: Docker (Recommended)](#option-1-docker-recommended)
  - [Option 2: Local Installation](#option-2-local-installation)
- [Usage](#usage)
  - [Aeon dataset initialisation](#aeon-dataset-initialisation)
- [Quick Start Tutorial](#quick-start-tutorial)
  - [Step 1: Test with a Single Dataset (5 minutes)](#step-1-test-with-a-single-dataset-5-minutes)
  - [Step 2: Start the Worker Algorithm](#step-2-start-the-worker-algorithm)
  - [Step 3: Analyze Results in Jupyter Notebook](#step-3-analyze-results-in-jupyter-notebook)
- [Experimental Results](#experimental-results)
- [Troubleshooting](#troubleshooting)
  - [Common Issues](#common-issues)
  - [Performance Optimization](#performance-optimization)
- [License](#license)
  - [Academic Use License](#academic-use-license)
- [Support](#support)

## Overview

DRIFTS (Distributed Reason Intervals for Time Series) is a computational framework for analyzing continuous anti-reasons in time series classification using Random Forest ensembles. The system implements a distributed algorithm for computing interpretability constraints (ICF - Interval Constraint Functions) across multiple UCR Archive time series datasets.
This repository contains utilities for converting scikit-learn tree ensembles
into the internal ICF representation and for preparing time-series datasets
from the [Aeon](https://www.aeon-toolkit.org/) collection. The tooling spans
from dataset initialisation scripts to helpers that persist forests and samples
into Redis-backed caches.

## Installation


### Option 1: Docker (Recommended)

Docker provides an isolated environment with Redis pre-configured.

**Requirements:**

- Docker Desktop installed and running
  - Windows: https://www.docker.com/products/docker-desktop
  - Linux/macOS: https://docs.docker.com/get-docker/

**Quick Start:**

```bash
# Windows
run.bat

# Linux/macOS
chmod +x run.sh  # First time only
./run.sh
```

This will build the Docker image and start a container with Redis on `localhost:6379`.

**Available Commands:**

| Command | Windows                          | Linux/macOS                        | Description                  |
| ------- | -------------------------------- | ---------------------------------- | ---------------------------- |
| Start   | `run.bat` or `run.bat start` | `./run.sh` or `./run.sh start` | Build and start container    |
| Stop    | `run.bat stop`                 | `./run.sh stop`                  | Stop container               |
| Shell   | `run.bat shell`                | `./run.sh shell`                 | Open bash shell in container |
| Logs    | `run.bat logs`                 | `./run.sh logs`                  | View container logs          |
| Restart | `run.bat restart`              | `./run.sh restart`               | Restart container            |
| Help    | `run.bat help`                 | `./run.sh help`                  | Show help                    |

**Using the container:**

```bash
# Open shell in container
run.bat shell  # Windows
./run.sh shell # Linux/macOS

# Inside the container, run any script:
python init_aeon_univariate.py Coffee --class-label 0 --optimize
python enhanced_launch_workers.py start --profile development
```

The following directories are automatically mounted and accessible from your host:

- `./logs` - Application logs
- `./workers` - Workers configuration
- `./results` - Experiment results
- `./fig` - Plots and visualizations

### Option 2: Local Installation

**Prerequisites**
- Python 3.12+
- Redis Server
- Virtual environment (recommended)

1. **Create a virtual environment (recommended)**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. **Install the Python dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   The `requirements.txt` file includes all necessary dependencies.
3. **Start Redis**

   ```bash
   redis-server
   ```

## Usage

### Aeon dataset initialisation

`init_aeon_univariate.py` exposes a command-line utility for initialising the
Redis caches with samples and optimised forests for a single dataset. Examples:

```bash
# List supported datasets
python init_aeon_univariate.py --list-datasets

# Optimise a Random Forest for the ECG200 dataset using Bayesian search
python init_aeon_univariate.py ECG200 --class-label "1" --optimize
```

#### Core arguments

- `dataset_name` — Dataset to load
- `--class-label` — Class label whose samples will be processed
- `--list-datasets` — Print the curated catalogue of supported Aeon datasets and exit.
- `--info` — Display dataset metadata without performing any processing.
- `--optimize` — Enable Bayesian optimisation via `scikit-optimize` to tune the Random Forest hyper-parameters.
- `--redis-port` *(int, default: 6379)* — Port of the Redis or KeyDB instance used by the workers.

Run `python init_aeon_univariate.py --help` to view the auto-generated help message with the latest defaults.

---

## Quick Start Tutorial

> **Note:** If using Docker, run `run.bat shell` (Windows) or `./run.sh shell` (Linux/macOS) first to enter the container, then execute the commands below.

### Step 1: Test with a Single Dataset (5 minutes)

```bash
# Initialize Coffee dataset with Bayesian optimization
python3 init_aeon_univariate.py Coffee --class-label "0" --optimize
```

### Step 2: Start the Worker Algorithm

```bash
# Launch workers to process the initialized dataset (default: 1 worker)
python3 enhanced_launch_workers.py start

# Or use development profile (4 workers with logging)
python3 enhanced_launch_workers.py start --profile development

# Or use production profile (4 workers with logging)
python3 enhanced_launch_workers.py start --profile production
```

Edit the file `worker_config.yaml` to customize worker settings, e.g., increase the number of workers.

**Key parameters:**

- `start` — Start workers using configuration
- `--profile {default|development|production}` — Use predefined worker profiles
  - `default`: 1 worker with `worker_cache.py`
  - `development`: 4 workers with `worker_cache_logged.py`
  - `production`: 4 workers with `worker_cache_logged.py`
- `--config FILE` — Use custom YAML configuration file

**Worker Management:**

```bash
# Check worker status
python3 enhanced_launch_workers.py status

# View logs for a specific worker
python3 enhanced_launch_workers.py logs 1

# Stop all workers
python3 enhanced_launch_workers.py stop

# Clean restart (stop + clean + start fresh)
python3 enhanced_launch_workers.py clean-restart
```

**Expected output:**

```
Starting 1 worker processes...
Worker 1 started (PID: 12345)
Workers running. Press Ctrl+C to monitor or stop.
```

**Monitor progress:**

```bash
# Check Redis databases for candidate reasons and confirmed reasons
redis-cli -n 1 DBSIZE  # CAN database (candidates)
redis-cli -n 5 DBSIZE  # Anti-reasons

# Or use the status command
python3 enhanced_launch_workers.py status
```

### Step 3: Analyze Results in Jupyter Notebook

```bash
# Open the analysis notebook
jupyter notebook models_analysis.ipynb
jupyter notebook reasons_analysis.ipynb
```
1. **`reasons_analysis.ipynb`**: Main analysis notebook
   - Robustness calculations
   - Statistical summaries

2. **`models_analysis.ipynb`**: Model  analysis
   - Dataset complexity ranking


### Experimental Results
Experimental results of the distributed algorithm  DRIFTS for computing continuous anti-reasons across 25 time series datasets from the UCR Archive (2019).  
It reports dataset characteristics, endpoint universe stats (mean/std of features in EU_RF), computational metrics, early stopping effectiveness, and pruning efficiency.

| **Dataset** | Train<br>Size | Test<br>Size | Series<br>Length | N<br>Estimators | Test<br>Accuracy | CV<br>Score | ICF<br>Checks | Anti-Reason<br>Check<br>Iter. | N<br>Features | EU<br>Complexity | EU<br>Min | EU<br>Max | Mean EU<br>(± Std) | Robustness<br>(± Std) | Robust.<br>Min | Robust.<br>Max | Cand.<br>Anti-Reason | Anti<br>Reason | Total<br>Time (ms) |
|--------------|---------------|--------------|------------------|-----------------|------------------|--------------|----------------|-------------------------------|----------------|------------------|------------|------------|----------------------|----------------------|----------------|----------------|----------------------|----------------|--------------------|
| Wine | 57 | 54 | 234 | 10 | 0.759 | 0.705 | 2164 | 235916 | 86 | 297 | 3 | 6 | 3.453 (±0.710) | 0.707 (±0.015) | 0.671 | 0.725 | 49057 | 55 | 580118 |
| MiddlePhalanx<br>OutlineCorrect | 600 | 291 | 80 | 10 | 0.821 | 0.777 | 976 | 228277 | 80 | 1440 | 7 | 39 | 18.000 (±5.725) | 0.792 (±0.039) | 0.652 | 0.881 | 71352 | 39 | 1469590 |
| SonyAIBO<br>RobotSurface1 | 20 | 601 | 70 | 17 | 0.577 | 0.800 | 32738 | 37299 | 32 | 106 | 3 | 5 | 3.312 (±0.583) | 0.705 (±0.006) | 0.687 | 0.719 | 303484 | 12237 | 93066 |
| Beetle<br>Fly | 20 | 20 | 512 | 26 | 0.850 | 0.700 | 24975 | 124132 | 42 | 129 | 3 | 4 | 3.071 (±0.258) | 0.820 (±0.012) | 0.796 | 0.850 | 460178 | 2066 | 4947272 |
| TwoLead<br>ECG | 23 | 1139 | 82 | 54 | 0.775 | 0.820 | 56944 | 20677 | 33 | 116 | 3 | 6 | 3.515 (±0.925) | 0.709 (±0.002) | 0.702 | 0.717 | 484385 | 22215 | 33251 |
| Lightning<br>2 | 60 | 61 | 637 | 65 | 0.738 | 0.883 | 2042 | 144482 | 88 | 273 | 3 | 5 | 3.102 (±0.370) | 0.709 (±0.002) | 0.447 | 0.555 | 315023 | 49 | 1482647 |
| Face<br>Four | 24 | 88 | 350 | 84 | 0.739 | 0.880 | 1700 | 2901 | 63 | 197 | 3 | 5 | 3.127 (±0.418) | 0.622 (±0.027) | 0.539 | 0.646 | 144617 | 36 | 15835 |
| ToeSegmentation<br>2 | 36 | 130 | 343 | 98 | 0.731 | 0.804 | 4013 | 1002539 | 80 | 251 | 3 | 5 | 3.138 (±0.379) | 0.717 (±0.006) | 0.702 | 0.730 | 96095 | 412 | 2710865 |
| ECG<br>200 | 100 | 100 | 96 | 101 | 0.810 | 0.880 | 3507 | 13825002 | 72 | 291 | 3 | 7 | 4.042 (±1.148) | 0.509 (±0.033) | 0.393 | 0.551 | 105182 | 36 | 45937485 |
| ItalyPower<br>Demand | 67 | 1029 | 24 | 169 | 0.959 | 0.986 | 98010 | 12849 | 24 | 132 | 3 | 11 | 5.500 (±2.082) | 0.432 (±0.006) | 0.413 | 0.444 | 0 | 4942 | 4850 |
| Meat | 60 | 60 | 448 | 193 | 0.933 | 1.000 | 3142 | 7017067 | 39 | 120 | 3 | 4 | 3.077 (±0.266) | 0.760 (±0.004) | 0.753 | 0.768 | 118759 | 36 | 30077438 |
| SonyAIBO<br>RobotSurface2 | 27 | 953 | 65 | 217 | 0.794 | 0.893 | 2051 | 2054688 | 33 | 110 | 3 | 5 | 3.333 (±0.532) | 0.528 (±0.036) | 0.384 | 0.600 | 78248 | 33 | 4952624 |
| Coffee | 28 | 28 | 286 | 233 | 1.000 | 1.000 | 5382 | 6255421 | 27 | 84 | 3 | 4 | 3.111 (±0.314) | 0.758 (±0.030) | 0.688 | 0.794 | 124507 | 98 | 26156013 |
| Bird<br>Chicken | 20 | 20 | 512 | 233 | 0.500 | 0.850 | 8494 | 249253 | 42 | 127 | 3 | 4 | 3.024 (±0.152) | 0.836 (±0.014) | 0.800 | 0.844 | 370182 | 1353 | 4285450 |
| Gun<br>Point | 50 | 150 | 150 | 233 | 0.880 | 0.960 | 15932 | 9676 | 58 | 190 | 3 | 5 | 3.276 (±0.484) | 0.854 (±0.002) | 0.850 | 0.859 | 147517 | 6083 | 6119 |
| CinC<br>ECGTorso | 40 | 1380 | 1639 | 245 | 0.714 | 0.725 | 1128 | 2032222 | 118 | 366 | 3 | 5 | 3.102 (±0.354) | 0.823 (±0.002) | 0.817 | 0.829 | 86683 | 73 | 8650001 |
| Mote<br>Strain | 20 | 1252 | 84 | 300 | 0.884 | 0.850 | 3218 | 657010 | 38 | 124 | 3 | 6 | 3.263 (±0.714) | 0.609 (±0.030) | 0.462 | 0.690 | 114422 | 39 | 2205104 |

## Troubleshooting

### Common Issues

**Redis Connection Issues:**
```bash
# Check Redis is running
redis-cli ping

# Check port availability
netstat -an | grep 6379
```

**Worker Process Issues:**
```bash
# Check worker logs
python enhanced_launch_workers.py logs <worker_id>

# Clean restart
python enhanced_launch_workers.py clean-restart
```

**Memory Issues:**
- Reduce worker count in configuration
- Use `--batch-size` parameter to limit memory usage
- Monitor Redis memory usage: `redis-cli info memory`


### Performance Optimization

1. **Increase Worker Count**: For CPU-bound workloads
2. **Optimize Redis**: Configure appropriate memory limits
3. **Batch Processing**: Use smaller batch sizes for large datasets


## License

This project is developed for research purposes as part of the IEEE Conference on Artificial Intelligence (CAI) 2026. 

### Academic Use License

**Permission is hereby granted for academic and research use only**, subject to the following conditions:

1. **Attribution Required**: Any use of this code in academic work must include proper citation of the original research
2. **Research Only**: This software is intended for academic research and educational purposes
3. **No Commercial Use**: Commercial use is prohibited without explicit written permission
4. **Share Improvements**: Derivative works should be made available to the research community
5. **No Warranty**: This software is provided "as is" without any warranty


## Support

For questions or issues:
1. Check existing issues in the repository
2. Review the troubleshooting section
3. Examine worker logs for error details
4. Create a new issue with detailed error information

---

*Last updated: November 2025*