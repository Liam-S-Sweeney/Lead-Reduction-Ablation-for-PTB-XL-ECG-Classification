import csv
from datetime import datetime
import logging
from itertools import product
from tqdm import tqdm
from utility.paths import RUNS_CSV, FINAL_RUNS_CSV
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

    ("lead_group",   "limb6",    list(range(6))),       # I, II, III, aVR, aVL, aVF  <- KardiaMobile (6L)
    ("lead_group",   "chest6",   list(range(6, 12))),   # V1, V2, V3, V4, V5, V6

    ("device",       "leadI",    [0]),                  # I                          <- KardiaMobile (1L)
    ("device",       "I_II",     [0, 1]),               # I, II — rank-2 minimum
]

RUN_ID = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
SEEDS = (0, 1, 2)
HPARAMS = dict(n_epochs=10, max_lr=3e-3, weight_decay=0.0)

def append_row(category: str, result: dict, final_eval: bool):
    """
    Append one experiment's results as a row to the validation or holdout CSV.

    Writes a header if the file is new; raises if an existing file's header
    doesn't match, so rows can never silently misalign with their columns.
    """
    csv_path = FINAL_RUNS_CSV if final_eval else RUNS_CSV
    classes = result["classes"]
    header = ["run_id", "category", "lead_set", "n_leads", "seed", "split", "macro",
              *[f"auc_{c}" for c in classes],
              *[f"ap_{c}" for c in classes]]

    write_header = not csv_path.exists()
    if not write_header:
        with open(csv_path, newline="") as f:
            if next(csv.reader(f)) != header:
                raise ValueError(f"{csv_path.name} has a different schema; archive it before appending")

    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow([
            RUN_ID, category, result["lead_set"], result["n_leads"],
            result["seed"], result["split"], f"{result['macro']:.4f}",
            *[f"{result['per_class'][c]:.4f}" for c in classes],
            *[f"{result['per_class_ap'][c]:.4f}" for c in classes],
        ])


def run_ablation(final_eval: bool = False):
    """
    Run every configuration in CONFIGS at every seed in SEEDS.

    Each run trains a fresh model and appends one row to the validation or
    holdout CSV as soon as it finishes. A failed run is logged with its
    traceback and skipped, so the rest of the grid still completes.
    """
    initialize_logger()
    logger = logging.getLogger(__name__)

    runs = list(product(CONFIGS, SEEDS))
    failed = []

    for (category, name, leads), seed in tqdm(runs, colour="yellow", desc="ablation"):
        logger.info(f"starting {name}, seed {seed}")
        try:
            result = run_experiment(leads=leads, lead_set_name=name, seed=seed,
                                    final_eval=final_eval, **HPARAMS)
            append_row(category, result, final_eval=final_eval)
            logger.info(f"{name} seed {seed}: macro {result['macro']:.4f}")
        except Exception:
            logger.exception(f"{name} seed {seed} FAILED")
            failed.append(f"{name} s{seed}")

    if failed:
        logger.error(f"{len(failed)}/{len(runs)} runs failed: {failed}")
        print(f"\n{len(failed)} of {len(runs)} runs FAILED — see details.log: {failed}")
    else:
        print(f"\nAll {len(runs)} runs completed.")


if __name__ == "__main__":
    run_ablation(final_eval=False) 
    run_ablation(final_eval=True) 