# NEURON-TIS-HPC

## Overview

`NEURON-TIS-HPC` is a high-performance computing framework that combines **NEURON-based biophysical neuron simulations** with a **hybrid MPI/OpenMP Master–Worker architecture**. The project is designed to address the prohibitive runtime of large-scale parameter sweeps (e.g., 46,656 electrode orientation configurations) when executed sequentially on a single machine.

By distributing individual simulation jobs across multiple MPI ranks and CPU cores, the framework enables efficient, scalable **high-throughput extracellular stimulation (TIS) simulations** on HPC clusters.

---

## Key Features

* **Large-scale parameter scanning**

  * Systematic sweep over 3D electrode orientations (Theta, Ro, Roll)
  * Automatically generates up to 46,656 unique stimulation configurations

* **Biophysical neuron simulation**

  * Uses **NEURON** and **LFPy**
  * Ball-and-stick neuron model
  * Extracellular electrical stimulation

* **Hybrid parallel execution model**

  * **C++ / MPI control layer**

    * Rank 0 acts as a dedicated Master
    * Remaining ranks act as Workers
  * **OpenMP threading layer**

    * Each Worker rank fully utilizes node-level CPU cores

* **Robust data storage**

  * Centralized **SQLite database**
  * WAL (Write-Ahead Logging) mode enabled
  * Safe concurrent writes from hundreds of parallel jobs

---

## Project Structure

```
neuron-tis-hpc/
├── DB/                     # Path of database
├── model/                  # Path of Neuron model
├── create_db.py            # Initialize SQLite database and tables
├── set_test_parameter.py   # Pre-generate all scan parameters (angles, frequencies, etc.)
├── main.py                 # Single NEURON simulation task and result recording
├── job_launcher.cpp        # C++ MPI Master–Worker job dispatcher
├── job.sh                  # Slurm submission script
├── pyproject.toml          # Project dependencies (uv-managed)
└── README.md
```

---

## Environment Management

This project uses **uv** for Python environment and dependency management.

All Python commands should be executed via `uv run` to ensure reproducibility across nodes and clusters.

---

## Requirements

### System Requirements

* Python ≥ 3.12 (recommended)
* uv
* C++ compiler with MPI and OpenMP support (e.g., `mpicxx`)
* MPI implementation (OpenMPI or MPICH)
* NEURON
* LFPy
* Slurm (for HPC execution)

---

## Installation & Setup

### 1. Create Python Environment

```bash
uv venv
source .venv/bin/activate
uv sync
```

### 2. Compile MPI Job Launcher

```bash
mpicxx -o job_launcher job_launcher.cpp -fopenmp
```

---

## Experiment Initialization

Before launching simulations, initialize the database and generate all test parameters:

```bash
# Create database schema
uv run create_db.py

# Insert all 46,656 parameter combinations
uv run set_test_parameter.py
```

---

## Running Simulations (HPC / Slurm)

This project is designed to run under a Slurm scheduler. Adjust resource requests in `job.sh` as needed, then submit the job:

```bash
sbatch job.sh
```

### Core Execution Logic (job.sh)

```bash
# Load MPI module
ml openmpi/5.0.4

# Launch MPI Master–Worker framework (example: 4 MPI ranks)
mpirun -np 4 ./job_launcher 46656
```

Each MPI Worker dynamically receives simulation tasks until all parameter sets are exhausted.

---

## Core Technical Details

### Rotation Matrix

Electrode orientations are generated using a 3D rotation matrix:

```
R = Rz(ρ) × Ry(θ) × Rx(φ)
```

This formulation ensures orthogonal axes and consistent electrode geometry during spatial scanning.

---

### Database Concurrency Protection

* SQLite is configured in **WAL mode**
* `main.py` implements a `safe_execute()` retry mechanism
* Automatic delayed retries handle transient `database is locked` errors

---

### MPI Scheduling Strategy

* **Rank 0 (Master)**

  * Handles all `MPI_Recv` / `MPI_Send` operations
  * Assigns simulation IDs
  * Does not perform computation

* **Worker Ranks**

  * Execute multiple Python simulation tasks in parallel via OpenMP threads
  * Terminate cleanly when no jobs remain

This design minimizes scheduling overhead and MPI contention.

---

## Performance Motivation

On a single machine, a full 46,656-configuration sweep may take **~12 hours** when executed sequentially.

By scaling across multiple nodes and CPU cores, total runtime decreases approximately linearly with available resources, making exhaustive parameter studies practical.

---

## Limitations

* SQLite may become a bottleneck at very large node counts
* No checkpointing or failed-job recovery
* Requires NEURON-compatible HPC environments

---

## Future Improvements

* Distributed databases (PostgreSQL / Redis)
* Checkpointing and job resubmission
* GPU acceleration (if supported by NEURON models)
* Support for multiple neuron morphologies

---

## Note

This project was developed to address the limitations of single-node execution for high-throughput scientific simulations, demonstrating how parallel architectures can dramatically reduce time-to-solution in computational neuroscience.
