# Dataset: NASA C-MAPSS (`FD002`)

This directory holds the **NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) `FD002`** turbofan engine degradation dataset used by the pipeline.

## Expected Files

Place the following whitespace-delimited files directly in `data/`:

- `train_FD002.txt` — Full run-to-failure training trajectories across 260 engines (53,759 total cycle rows; 26 columns: `engine_id`, `cycle`, 3 operating settings, and 21 sensor measurements).
- `test_FD002.txt` — Truncated test engine trajectories prior to failure across 259 engines.
- `RUL_FD002.txt` — Ground-truth Remaining Useful Life (RUL) values corresponding to the final cycle of each engine in `test_FD002.txt`.

## Source & Access

The C-MAPSS dataset is publicly available through NASA's Prognostics Center of Excellence (PCoE) data repository (commonly distributed as `CMAPSSData` via NASA PCoE / Kaggle / IEEE PHM '08 prognostic benchmarks). Because dataset terms and file sizes vary, download the official `FD002` files yourself and place them in `data/` rather than relying on an external URL inside the code.

## How the Pipeline Uses This Folder

- **Data Ingestion (`src/data_loader.py`):** `load_fd002_data("data")` reads `train_FD002.txt`, `test_FD002.txt`, and `RUL_FD002.txt` using whitespace parsing (`sep=r"\s+"`) and drops any empty trailing columns.
- **Sensor & RUL Handling (`src/preprocessing.py`):** `get_useful_sensors` filters out `sensor_20`, `sensor_21`, and near-constant low-variance sensors (`sensor_16`), while `compute_rul_full` calculates `max(cycle) - cycle` strictly for full run-to-failure training data.
