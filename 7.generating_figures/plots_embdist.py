import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Patch
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist


# --- Apply publication-wide style ---
sns.set_theme(
    context="talk",  # larger, presentation-ready
    style="white",  # light, clean background
    font="Arial",
    font_scale=1.2
)

# --- Define and register your custom palette ---
CUSTOM_PALETTE = [
    "#88ccee", "#661100", "#117733",  "#332288",
    "#ddcc77", "#6699cc", "#44aa99", "#999933",
    "#cc6677", "#aa4499"
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

cmap_rel = LinearSegmentedColormap.from_list("rel_cmap", ["#6699cc", "#661100"])


DISTANCE_METRIC = "cosine"  # or "euclidean"
UMAP_METRIC = "cosine"

base_path = ".../results/datasets_analysis"

mpd_df = pd.read_csv(f"{base_path}/bacformer_embedding_mpd_matrix.csv", index_col=0)
mmd_df = pd.read_csv(f"{base_path}/bacformer_embedding_mmd_matrix.csv", index_col=0)
rel_mpd = pd.read_csv(f"{base_path}/bacformer_embedding_mpd_relative_shift.csv", index_col=0)
rel_mmd = pd.read_csv(f"{base_path}/bacformer_embedding_mmd_relative_shift.csv", index_col=0)

# UMAP data
embedding_2d = np.load(f"{base_path}/bacformer_umap_coords.npy")
all_datasets = np.load(f"{base_path}/bacformer_umap_labels.npy")

# centroid data
df_dist = pd.read_csv(f"{base_path}/bacformer_embedding_distance_to_inhouse.csv")
centroid_dists = pd.read_csv(f"{base_path}/bacformer_embedding_centroid_distance.csv")

# --------------------------------------------------------------
# 7. VISUALIZATION
# --------------------------------------------------------------

rel_mpd_pretty = rel_mpd.rename(index=DISPLAY_NAMES, columns=DISPLAY_NAMES)
rel_mmd_pretty = rel_mmd.rename(index=DISPLAY_NAMES, columns=DISPLAY_NAMES)
mpd_df_pretty = mpd_df.rename(index=DISPLAY_NAMES, columns=DISPLAY_NAMES)
mmd_df_pretty = mmd_df.rename(index=DISPLAY_NAMES, columns=DISPLAY_NAMES)

sns.set(style="whitegrid", font_scale=1.0)

fig10, ax = plt.subplots(figsize=(10, 6))
sns.kdeplot(
    data=df_dist,
    x="distance",
    hue="dataset_pretty",
    fill=True,
    alpha=0.1,
    lw=2,
    common_norm=False,
    palette=PALETTE_MAP_PRETTY,
    ax=ax
)
leg = ax.get_legend()
if leg is not None:
    leg.set_title("Datasets")
    leg.set_bbox_to_anchor((0.97, 0.97))
    leg._loc = 1          # upper left
    leg.set_frame_on(False)
ax.grid(False)
ax.set_title("Bacformer based Embedding Cosine Distance from the Training Dataset Centroid", pad=12, fontweight="bold")
ax.set_xlabel("Cosine Distance")
ax.set_ylabel("Density")
plt.tight_layout()

fig10.savefig(".../results/dataset_bacformerbased_distance_from_centroid.pdf", format="pdf", bbox_inches="tight")

print("Done")