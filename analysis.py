import pandas as pd
import matplotlib.pyplot as plt
from utility.paths import FINAL_RUNS_CSV, RUNS_CSV, ANALYSIS_PATH

DEVICE_LADDER = ["leadI", "I_II", "limb6", "full12"]
REGIONS = ["anterior", "septal", "lateral", "inferior"]
LEAD_NAMES = {"anterior": "V2,V3,V4", "septal": "V1,V2,V3",
              "lateral": "I,aVL,V5", "inferior": "II,III,aVF"}


def load_results(final_eval=False):
    run = FINAL_RUNS_CSV if final_eval else RUNS_CSV
    if not run.exists():
        raise FileNotFoundError(f"{run} not found - run the ablation first")
    df = pd.read_csv(run)
    return df[df.run_id == df.run_id.max()]


def macro_auc_vs_lead_count(final_eval=False):
    split = "test" if final_eval else "val"
    prefix = "FINAL_" if final_eval else ""
    df = load_results(final_eval)
    df = df[df.lead_set.isin(DEVICE_LADDER)]

    agg = df.groupby(["lead_set", "n_leads"]).macro.agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.errorbar(agg.n_leads, agg["mean"], yerr=agg["std"],
                fmt="o", color="tab:blue", ecolor="gray", capsize=3)
    for _, r in agg.iterrows():
        ax.annotate(r.lead_set, (r.n_leads, r["mean"]),
                    textcoords="offset points", xytext=(8, -3), fontsize=9)

    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 6, 12])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    
    ax.set_xlim(0.8, 18)
    ax.set_xlabel("number of leads")
    ax.set_ylabel(f"macro AUC ({split})")
    ax.set_title("Performance vs. lead count")
    fig.tight_layout()
    ANALYSIS_PATH.mkdir(parents=True, exist_ok=True)
    fig.savefig(ANALYSIS_PATH / f"{prefix}macro_auc_vs_lead_count.png", dpi=150)


def regions_at_fixed_count(final_eval=False):
    split = "test" if final_eval else "val"
    prefix = "FINAL_" if final_eval else ""
    df = load_results(final_eval)

    baseline = df[df.lead_set == "full12"].macro.mean()
    agg = (df[df.lead_set.isin(REGIONS)]
           .groupby("lead_set").macro.agg(["mean", "std"])
           .sort_values("mean"))

    labels = [f"{s}\n({LEAD_NAMES[s]})" for s in agg.index]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(labels, agg["mean"], xerr=agg["std"], color="tab:blue", capsize=3)
    ax.axvline(baseline, color="crimson", ls="--", lw=1,
               label=f"full 12-lead ({baseline:.3f})")
    ax.set_xlim(0.80, 0.95)
    ax.set_xlabel(f"macro AUC ({split})")
    ax.set_title("Three-lead regional subsets")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ANALYSIS_PATH / f"{prefix}regions_3lead.png", dpi=150)


def ap_by_class(final_eval=False):
    split = "test" if final_eval else "val"
    prefix = "FINAL_" if final_eval else ""
    df = load_results(final_eval)
    df = df[df.lead_set.isin(DEVICE_LADDER)]
    classes = ["CD", "HYP", "MI", "NORM", "STTC"]

    fig, axes = plt.subplots(1, 5, figsize=(14, 3.2), sharex=True)
    for ax, c in zip(axes, classes):
        agg = (df.groupby(["lead_set", "n_leads"])[f"ap_{c}"]
                 .agg(["mean", "std"]).reset_index()
                 .sort_values("n_leads"))
        x = range(len(agg))
        ax.errorbar(x, agg["mean"], yerr=agg["std"], fmt="o-",
                    color="tab:blue", ecolor="gray", capsize=3)
        ax.set_xticks(list(x))
        ax.set_xticklabels(agg.lead_set, rotation=45, ha="right", fontsize=8)
        ax.set_title(c, fontsize=10)
        ax.set_ylim(0.3, 1.0)
    axes[0].set_ylabel(f"average precision ({split})")
    fig.tight_layout()
    fig.savefig(ANALYSIS_PATH / f"{prefix}ap_by_class.png", dpi=150)

if __name__ == "__main__":
    macro_auc_vs_lead_count(final_eval=False)
    regions_at_fixed_count(final_eval=False)
    ap_by_class(final_eval=False)