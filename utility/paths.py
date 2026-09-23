"""
This file stores all file pathways that will be referenced in other modules.
"""

from pathlib import Path

# Root
ROOT                        = Path(__file__).parent.parent

# Root Level Misc
CACHE_PATH                  = ROOT / "X_100hz_12x1000.npy"
STATS_PATH                  = ROOT / "norm_stats.npz"

# Runs CSV
RUNS_CSV                    = ROOT / "runs.csv"
FINAL_RUNS_CSV              = ROOT / "final_runs.csv"

# Full dataset
FULL_DATASET_PATH           = ROOT / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"

# Subfiles of full dataset
PTBXL_DB_PATH               = FULL_DATASET_PATH / "ptbxl_database.csv"
SCP_STATEMENTS_PATH         = FULL_DATASET_PATH / "scp_statements.csv"

# Logging Paths
CKPT_DIR                    = ROOT / "checkpoints"
LOG_DIR                     = ROOT / "logs"

# Visuals
VIS_PATH                    = ROOT / "visuals"
ANALYSIS_PATH               = VIS_PATH / "analysis"

# Prediction
PREDS_DIR                   = ROOT / "predictions"


# Assert Check
for p in (PTBXL_DB_PATH, SCP_STATEMENTS_PATH):
    assert p.exists(), f"ROOT resolved to {ROOT}; missing {p.name}"