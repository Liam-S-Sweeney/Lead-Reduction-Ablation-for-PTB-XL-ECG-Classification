import csv
import logging
from itertools import product
from tqdm import tqdm
from utility.paths import RUNS_CSV
from utility.startup import initialize_logger
from train import run_experiment

# 12 Lead ECG DF
# i      0  1    2    3    4    5   6   7   8   9   10  11
# Leads  I  II  III  AVR  AVL  AVF  V1  V2  V3  V4  V5  V6

CONFIGS = [
    # (category,     name,       leads)
    ("baseline",     "full12",   list(range(12))),      # all leads

    ("wall_region",  "anterior", [7, 8, 9]),            # V2, V3, V4
    ("wall_region",  "septal",   [6, 7, 8]),            # V1, V2, V3
    ("wall_region",  "lateral",  [0, 4, 10]),           # I, aVL, V5
    ("wall_region",  "inferior", [1, 2, 5]),            # II, III, aVF

    ("lead_group",   "limb6",    list(range(6))),       # I, II, III, AVR, AVL, AVF  <- KardiaMobile (6L)
    ("lead_group",   "chest6",   list(range(6, 12))),   # V1, V2, V3, V4, V5, V6

    ("device",       "leadI",    [0]),                  # I                          <- KardiaMobile (1L)
    ("device",       "I_II",     [0, 1]),               # I, II — rank-2 minimum
]

SEEDS = (0, 1, 2)

HPARAMS = dict(n_epochs=10, max_lr=3e-3, weight_decay=0.0)


def append_row(category, result, classes):
    write_header = not RUNS_CSV.exists()
    with open(RUNS_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["category", "lead_set", "n_leads", "seed", "split", "macro",
                        *[f"auc_{c}" for c in classes],
                        *[f"ap_{c}" for c in classes]])
        w.writerow([
            category, result["lead_set"], result["n_leads"],
            result["seed"], result["split"], f"{result['macro']:.4f}",
            *[f"{result['per_class'][c]:.4f}" for c in classes],
            *[f"{result['per_class_ap'][c]:.4f}" for c in classes],
        ])


if __name__ == "__main__":
    initialize_logger()
    logger = logging.getLogger(__name__)

    runs = list(product(CONFIGS, SEEDS))

    for (category, name, leads), seed in tqdm(runs, colour="yellow", desc="ablation"):
        logger.info(f"starting {name}, seed {seed}")
        try:
            result = run_experiment(leads=leads, lead_set_name=name,
                                    seed=seed, final_eval=False, **HPARAMS)
            append_row(category, result, result["classes"])
            logger.info(f"{name} seed {seed}: macro {result['macro']:.4f}")
        except Exception:
            logger.exception(f"{name} seed {seed} FAILED")