import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap

# --- Apply publication-wide style ---
sns.set_theme(
    context="talk",  # larger, presentation-ready
    style="white",  # light, clean background
    font="Arial",
    font_scale=1.2
)

# --- Define and register your custom palette ---
CUSTOM_PALETTE = [
    "#6b3f69", "#ddc3c3", "#81c088", "#f08b51",
    "#bcd2dc", "#1a4052", "#f4eea9", "#9b4444",
    "#136a53", "#afafaf"
]

PALETTE_MAP_PRETTY = {
    "CV set": "#88ccee",
    "OOD test set": "#661100",
    "Duran (2021)": "#44aa99",
    "Finkel (2019)": "#999933",
    "Finkel (2020)": "#ddcc77",
    "Hou (2021)": "#6699cc",
    "Lebeis (2015)": "#332288",
    "Wolinska (2021)": "#117733"
}

DISPLAY_NAMES = {
    "inhouse_easy": "CV set",
    "inhouse_hard": "OOD test set",
    "Duran_2021": "Duran (2021)",
    "Finkel_2019": "Finkel (2019)",
    "Finkel_2020": "Finkel (2020)",
    "Hou_2021": "Hou (2021)",
    "Lebeis_2015": "Lebeis (2015)",
    "Wolinska_2021": "Wolinska (2021)",
}

sns.set_palette(CUSTOM_PALETTE)
rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.2,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "figure.dpi": 300,
})

cmap_rel = LinearSegmentedColormap.from_list("rel_cmap", ["#6b3f69", "#81c088"])


# --------------------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------------------
RANKS = ["phylum", "class", "order", "family", "genus"]

# Paths (adjust to your system)
inhouse_tax_path = ".../data/SSC_data/SSC_taxonomy_GTDB.tsv"
external_tax_path = ".../data/Bacillus_functional_state_files/taxonomy.tsv"
split_file = ".../data/hard_easy_tax_split.tsv"

external_study_files = [
    ".../data/Bacillus_functional_state_files/isolate_list/Duran_2021.txt",
    ".../data/Bacillus_functional_state_files/isolate_list/Finkel_2019.txt",
    ".../data/Bacillus_functional_state_files/isolate_list/Finkel_2020.txt",
    ".../data/Bacillus_functional_state_files/isolate_list/Hou_2021.txt",
    ".../data/Bacillus_functional_state_files/isolate_list/Lebeis_2015.txt",
    ".../data/Bacillus_functional_state_files/isolate_list/Wolinska_2021.txt"
]

# --------------------------------------------------------------
# 2. HELPER FUNCTIONS
# --------------------------------------------------------------
def normalize_name(x):
    return str(x).strip()

def shared_taxa_percent(df, rank="genus"):
    """Jaccard similarity (presence/absence) between datasets."""
    taxa = {d: set(df.loc[df["dataset"] == d, rank].dropna()) for d in df["dataset"].unique()}
    datasets = sorted(taxa.keys())
    shared = pd.DataFrame(index=datasets, columns=datasets, dtype=float)
    for d1 in datasets:
        for d2 in datasets:
            inter = len(taxa[d1] & taxa[d2])
            union = len(taxa[d1] | taxa[d2])
            shared.loc[d1, d2] = inter / union if union > 0 else 0
    return shared

def bray_curtis(df, rank="genus"):
    """Bray–Curtis dissimilarity on counts of taxa per dataset."""
    counts = (
        df.groupby(["dataset", rank])["isolate"]
        .count()
        .unstack(fill_value=0)
        .astype(float)
    )
    bc = squareform(pdist(counts, metric="braycurtis"))
    return pd.DataFrame(bc, index=counts.index, columns=counts.index)

# --------------------------------------------------------------
# 3. LOAD TAXONOMY DATA
# --------------------------------------------------------------
print("Loading in-house taxonomy...")
inhouse = pd.read_csv(inhouse_tax_path, sep="\t")
inhouse["isolate"] = inhouse["isolate"].apply(normalize_name)

split_df = pd.read_csv(split_file, sep="\t")
split_df["isolate"] = split_df["isolate"].apply(normalize_name)
easy_ids = set(split_df[split_df["split"]=="easy"]["isolate"])
hard_ids = set(split_df[split_df["split"]=="hard"]["isolate"])

inhouse_easy = inhouse[inhouse["isolate"].isin(easy_ids)].copy()
inhouse_hard = inhouse[inhouse["isolate"].isin(hard_ids)].copy()
inhouse_easy["dataset"] = "inhouse_easy"
inhouse_hard["dataset"] = "inhouse_hard"

print("Loading external taxonomy...")
external = pd.read_csv(external_tax_path, sep="\t")
external["isolate"] = external["isolate"].apply(normalize_name)

# --------------------------------------------------------------
# 4. BUILD EXTERNAL DATASETS
# --------------------------------------------------------------
print("Building dataset tables...")
inhouse_isolates = set(inhouse["isolate"])
external_dfs = []

for f in external_study_files:
    ds = Path(f).stem
    with open(f) as fh:
        isolates = [normalize_name(line) for line in fh if line.strip()]
    sub = external[external["isolate"].isin(isolates)].copy()
    sub = sub[~sub["isolate"].isin(inhouse_isolates)]  # exclude overlap with in-house
    sub["dataset"] = ds
    external_dfs.append(sub)
    print(f"{ds}: {len(sub)} isolates retained")

taxonomy_all = pd.concat([inhouse_easy, inhouse_hard] + external_dfs, ignore_index=True)
datasets = taxonomy_all["dataset"].unique()

# --------------------------------------------------------------
# 5. MULTI-RANK METRICS
# --------------------------------------------------------------
metrics_records = []

for rank in RANKS:
    shared = shared_taxa_percent(taxonomy_all, rank=rank)
    bray = bray_curtis(taxonomy_all, rank=rank)

    for ds in datasets:
        if ds == "inhouse_easy":
            continue
        rec = {
            "dataset": ds,
            "rank": rank,
            "shared_with_easy": shared.loc[ds, "inhouse_easy"] if "inhouse_easy" in shared else np.nan,
            "shared_with_hard": shared.loc[ds, "inhouse_hard"] if "inhouse_hard" in shared else np.nan,
            "bray_to_easy": bray.loc[ds, "inhouse_easy"] if "inhouse_easy" in bray else np.nan,
            "bray_to_hard": bray.loc[ds, "inhouse_hard"] if "inhouse_hard" in bray else np.nan,
        }
        rec["novelty_to_easy"] = 1 - rec["shared_with_easy"]
        rec["novelty_to_hard"] = 1 - rec["shared_with_hard"]
        metrics_records.append(rec)

metrics_df = pd.DataFrame(metrics_records)
metrics_df.to_csv(".../results/taxonomy_similarity_summary.csv", index=False)
print("Saved taxonomy_similarity_summary.csv")

# --------------------------------------------------------------
# 6. PLOTS — INDIVIDUAL FIGURES
# --------------------------------------------------------------
metrics_df["dataset_pretty"] = metrics_df["dataset"].map(DISPLAY_NAMES)

sns.set(style="whitegrid", font_scale=1.2)

# (A) Shared taxa vs rank (to in-house easy)
fig1, ax = plt.subplots(figsize=(10,6))
sns.lineplot(
    data=metrics_df, 
    x="rank", y="shared_with_easy",
    hue="dataset_pretty", marker="o",
    palette=PALETTE_MAP_PRETTY,
    ax=ax, 
    linewidth=2
)
leg = ax.get_legend()
if leg is not None:
    leg.set_title("Datasets")
    leg.set_bbox_to_anchor((0.97, 0.97))
    leg._loc = 1          # upper left
    leg.set_frame_on(False)
ax.grid(False)
ax.set_title("Taxonomic Overlap Across Ranks with the Training Dataset", pad=12, fontweight="bold")
ax.set_ylim(-0.05, 1)
ax.set_ylabel("Proportion of Shared Taxa (Jaccard)")
ax.set_xlabel("Taxonomic Rank")
plt.tight_layout()

fig1.savefig(".../results/taxonomic_dataset_overlap.pdf", format="pdf", bbox_inches="tight")

print("Done")