import csv
import logging
from tqdm import tqdm
from utility.paths import RUNS_CSV
from utility.startup import initialize_logger
from train import run_experiment

LEAD_SETS = {
    "full12":   list(range(12)),
    "limb6":    [0, 1, 2, 3, 4, 5],
    "leadI_II": [0, 1],
    "leadI":    [0],
}

SEEDS = (0, 1, 2)

HPARAMS = dict(n_epochs=10, max_lr=3e-3, weight_decay=0.0)


def append_row(result, classes):
    write_header = not RUNS_CSV.exists()
    with open(RUNS_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["lead_set", "n_leads", "seed", "split", "macro",
            *[f"auc_{c}" for c in classes],
            *[f"ap_{c}" for c in classes]])
        w.writerow([
            result["lead_set"], result["n_leads"], result["seed"], result["split"],
            f"{result['macro']:.4f}",
            *[f"{result['per_class'][c]:.4f}" for c in classes],
        ])


if __name__ == "__main__":
    initialize_logger()
    logger = logging.getLogger(__name__)

    for name, leads in tqdm(LEAD_SETS.items(), 
                            colour="yellow", desc="Lead Sets", leave=False
                            ):
        for seed in tqdm(SEEDS,
                         colour="cyan", desc=f"{name} seeds", leave=False
                         ):
            logger.info(f"starting {name} seed {seed}")
            try:
                result = run_experiment(
                    leads=leads, lead_set_name=name, seed=seed,
                    final_eval=False, **HPARAMS,
                )
                append_row(result, result["classes"])
                logger.info(f"{name} seed {seed}: macro {result['macro']:.4f}")
            except Exception:
                logger.exception(f"{name} seed {seed} FAILED")