import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator, ScalarFormatter
from utility.paths import FINAL_RUNS_CSV, RUNS_CSV, ANALYSIS_PATH
from run_ablation import CONFIGS, SEEDS
from data import prepare_labels

DEVICE_LADDER = ["leadI", "I_II", "limb6", "full12"]
REGIONS = ["anterior", "septal", "lateral", "inferior"]
LEAD_NAMES = {"anterior": "V2,V3,V4", "septal": "V1,V2,V3",
              "lateral": "I,aVL,V5", "inferior": "II,III,aVF"}
DISPLAY = {"leadI": "Lead I", "I_II": "Leads I + II",
           "limb6": "6 limb", "full12": "12-lead"}


def _save(fig, name, final_eval):
    """Add the seed-SD note, lay out, save to ANALYSIS_PATH, and close."""
    fig.text(0.99, 0.01, f"Error bars: ±1 SD across {len(SEEDS)} seeds (not a CI)",
             ha="right", va="bottom", fontsize=7, color="gray")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    ANALYSIS_PATH.mkdir(parents=True, exist_ok=True)
    prefix = "FINAL_" if final_eval else ""
    fig.savefig(ANALYSIS_PATH / f"{prefix}{name}.png", dpi=150)
    plt.close(fig)


def load_results(final_eval=False):
    """
    Loads the rows from the most recent run in runs.csv (val) or final_runs.csv (test).
    "Most recent" is the largest run_id; this works because run IDs are zero-padded
    timestamps, which sort alphabetically in time order.

    Returns:
        DataFrame of that run's rows (27 for a complete run: 9 configs * 3 seeds)
    """
    run = FINAL_RUNS_CSV if final_eval else RUNS_CSV
    if not run.exists():
        raise FileNotFoundError(f"{run} not found - run the ablation first")
    
    df = pd.read_csv(run)

    latest = df[df.run_id == df.run_id.max()]
    expected = len(CONFIGS) * len(SEEDS)

    if len(latest) != expected:
        raise ValueError(
            f"Latest run {latest.run_id.iloc[0]} has {len(latest)} rows, expected {expected} "
            f"({len(CONFIGS)} configs * {len(SEEDS)} seeds); it may be incomplete")
    return latest


def macro_auc_vs_lead_count(final_eval=False):
    """
    Plot macro AUC (mean ± seed SD) against lead count for the device-ladder lead sets.
    """
    split = "test" if final_eval else "val"
    df = load_results(final_eval)
    df = df[df.lead_set.isin(DEVICE_LADDER)]    # regional (3-lead) sets are plotted separately

    # Collapses the number of seeds per lead into one mean and one std; carries n_leads
    agg = df.groupby(["lead_set", "n_leads"]).macro.agg(["mean", "std"]).reset_index()  # reset_index() turns
                                                                                        # group keys into cols

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.errorbar(
        agg.n_leads, 
        agg["mean"], 
        yerr=agg["std"],
        fmt="o", 
        color="tab:blue", 
        ecolor="gray", 
        capsize=3,)
    
    for _, r in agg.iterrows():
        ax.annotate(DISPLAY[r.lead_set], (r.n_leads, r["mean"]),
                    textcoords="offset points", xytext=(8, -3), fontsize=9)

    ax.set_xscale("log", base=2)    # Space between 1-2 same as 6-12
    ax.set_xticks([1, 2, 6, 12])   
    ax.get_xaxis().set_major_formatter(ScalarFormatter())   # prints as "1..." instead of "2⁰..."
    ax.set_xlim(0.8, 18)
    ax.set_xlabel("number of leads")
    ax.set_ylabel(f"macro AUC ({split})")
    ax.set_title("Performance vs. lead count")
    ax.xaxis.set_minor_locator(NullLocator())

    _save(fig, "macro_auc_vs_lead_count", final_eval)



def regions_at_fixed_count(final_eval=False):
    """
    Plot macro AUC (mean ± seed SD) for the four 3-lead regional subsets against the 12-lead baseline.
    """
    split = "test" if final_eval else "val"
    df = load_results(final_eval)

    baseline = df[df.lead_set == "full12"].macro.mean()
    agg = (df[df.lead_set.isin(REGIONS)]
           .groupby("lead_set").macro.agg(["mean", "std"])
           .sort_values("mean"))

    labels = [f"{s}\n({LEAD_NAMES[s]})" for s in agg.index]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axvline(baseline, color="crimson", ls="--", lw=1,
               label=f"full 12-lead ({baseline:.3f})")
    y = range(len(agg))
    ax.errorbar(agg["mean"], y, xerr=agg["std"],
                fmt="o", color="tab:blue", ecolor="gray", capsize=3)
    ax.set_yticks(y, labels)
    ax.set_xlim(0.80, 0.95)
    ax.set_xlabel(f"macro AUC ({split})")
    ax.set_title("Three-lead regional subsets")
    ax.legend(fontsize=8)

    _save(fig, "regions_3lead", final_eval)


def ap_by_class(final_eval=False):
    """
    Plot per-class average precision (mean ± seed SD) across the device-ladder lead sets.
    Dashed lines mark each class's prevalence in the split, which is AP's chance level.
    """
    split = "test" if final_eval else "val"
    df = load_results(final_eval)
    df = df[df.lead_set.isin(DEVICE_LADDER)]
    classes = ["CD", "HYP", "MI", "NORM", "STTC"]
    _, Y, label_classes, masks = prepare_labels()
    prevalence = dict(zip(label_classes, Y[masks[split].to_numpy()].mean(axis=0)))

    fig, axes = plt.subplots(1, 5, figsize=(14, 3.2), sharex=True)
    for ax, c in zip(axes, classes):
        agg = (df.groupby(["lead_set", "n_leads"])[f"ap_{c}"]
                 .agg(["mean", "std"]).reset_index()
                 .sort_values("n_leads"))
        x = range(len(agg))
        ax.errorbar(x, agg["mean"], yerr=agg["std"], fmt="o-",
                    color="tab:blue", ecolor="gray", capsize=3)
        ax.set_xticks(list(x))
        ax.set_xticklabels([DISPLAY[s] for s in agg.lead_set], rotation=45, ha="right", fontsize=8)
        ax.axhline(prevalence[c], color="crimson", ls="--", lw=1,
                   label="chance (prevalence)")
        ax.set_ylim(0, 1)
        ax.set_title(c, fontsize=10)
    axes[0].set_ylabel(f"average precision ({split})")
    axes[0].legend(fontsize=7, loc="lower right")

    _save(fig, "ap_by_class", final_eval)

if __name__ == "__main__":
    for final_eval in (False, True):
        macro_auc_vs_lead_count(final_eval)
        regions_at_fixed_count(final_eval)
        ap_by_class(final_eval)