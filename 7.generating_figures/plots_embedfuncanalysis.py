import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap

# ---------- STYLE ----------
sns.set_theme(
    context="talk", style="white", font="Arial", font_scale=1.2
)
CUSTOM_PALETTE = [
    "#88ccee", "#661100", "#117733",  "#ddcc77",
    "#332288", "#6699cc", "#44aa99", "#999933",
    "#cc6677", "#aa4499"
]
sns.set_palette(CUSTOM_PALETTE)
rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 1.2, "xtick.direction": "out",
    "ytick.direction": "out", "legend.frameon": False,
    "figure.dpi": 300,
})
cmap_rel = LinearSegmentedColormap.from_list("rel_cmap", ["#6699cc", "#661100"])

# ------- CONFIG UMAP -------
RUN_FOLDER = ".../results/esm2/esm2_embanalysis/embedding_funccheck/cosine/run5/"

# -------- MAIN UMAP --------
coords_file = [f for f in os.listdir(RUN_FOLDER) if f.endswith("_coords.tsv")]
if not coords_file:
    raise FileNotFoundError("No *_coords.tsv file found in selected run folder.")
coords_path = os.path.join(RUN_FOLDER, coords_file[0])

coords = pd.read_csv(coords_path, sep="\t")

fig, ax = plt.subplots(figsize=(12, 6))

sns.kdeplot(
    data=coords,
    x="UMAP_1", y="UMAP_2",
    hue="OG",
    fill=True, levels=2, thresh=0.05,
    linewidths=1.0,
    alpha=0.3,
    common_norm=False,
    legend=False,
    ax=ax
)

sns.scatterplot(
    data=coords,
    x="UMAP_1", y="UMAP_2",
    hue="OG",
    palette=CUSTOM_PALETTE,
    s=80, alpha=0.7, edgecolor="none",
    ax=ax
)

ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
ax.set_title("Cosine Distance–Based UMAP of ESM2 Embeddings vs. Protein Families", pad=15, weight="bold", loc='left', x=-0.1)

# Legend cleanup
ax.legend(
    title="Protein families (OGs)",
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    borderaxespad=0.,
    handletextpad=0.3
)

sns.despine()
plt.tight_layout()
fig.savefig(".../results/cosinedist_esm2based_OGumap_run5.pdf", format="pdf", bbox_inches="tight")

# -------- CONFIG VIOLIN --------
esm2_summary = pd.read_csv(".../results/esm2/esm2_embanalysis/embedding_funccheck/metrics_summary_mean_sd.tsv", sep="\t")
esm2_cosine  = pd.read_csv(".../results/esm2/esm2_embanalysis/embedding_funccheck/all_runs_metrics_cosine.tsv", sep="\t")

bac_summary = pd.read_csv(".../results/bacformer/bacformer_embanalysis/metrics_summary_mean_sd.tsv", sep="\t")
bac_cosine  = pd.read_csv(".../results/bacformer/bacformer_embanalysis/all_runs_metrics_cosine.tsv", sep="\t")

metrics = ["ARI", "NMI", "Homogeneity", "Completeness"]

# Prepare per-run data
esm2_data = [esm2_cosine[m].dropna().values for m in metrics]
bac_data  = [bac_cosine[m].dropna().values for m in metrics]

# Filter summaries
esm2_summary_cosine = (
    esm2_summary.query("Method == 'cosine'")
                .set_index("Metric")
                .loc[metrics]
                .reset_index()
)
bac_summary_cosine = (
    bac_summary.query("Method == 'cosine'")
               .set_index("Metric")
               .loc[metrics]
               .reset_index()
)

# Define x positions for grouped violins
xpos = np.linspace(0, len(metrics) - 1, len(metrics)) * 0.8
offset = 0.1

esm2_summary_cosine["xpos"] = xpos - offset
bac_summary_cosine["xpos"]  = xpos + offset

# -------- MAIN VIOLIN --------
fig2, ax = plt.subplots(figsize=(12, 5))  # smaller figure now

# Violin plots
parts1 = ax.violinplot(
    esm2_data, positions=xpos - offset,
    showmeans=False, showmedians=True, showextrema=False,
    widths=0.25
)
for pc in parts1['bodies']:
    pc.set_facecolor("#88ccee")
    pc.set_alpha(0.5)
    
parts1['cmedians'].set_color("#88ccee")

parts2 = ax.violinplot(
    bac_data, positions=xpos + offset,
    showmeans=False, showmedians=True, showextrema=False,
    widths=0.25
)
for pc in parts2['bodies']:
    pc.set_facecolor("#661100")
    pc.set_alpha(0.5)
    
parts2['cmedians'].set_color("#661100")

# Means
ax.errorbar(esm2_summary_cosine["xpos"], esm2_summary_cosine["Mean"],
            yerr=esm2_summary_cosine["SD"], fmt='o',
            color='#88ccee', capsize=3, label="ESM2 emb.")
ax.errorbar(bac_summary_cosine["xpos"], bac_summary_cosine["Mean"],
            yerr=bac_summary_cosine["SD"], fmt='o',
            color='#661100', capsize=3, label="Bacformer emb.")

# Baseline
ax.axhline(0.0, linestyle='--', color='black', linewidth=1.2, label='Random baseline')

# Formatting
ax.set_xticks(xpos)
ax.set_xticklabels(metrics)
ax.set_xlabel("Metric")
ax.set_ylabel("Score")
ax.set_ylim(-0.05, 1)
ax.set_title("Cosine-Based Clustering of Protein Embeddings vs. Protein Families",
             pad=15, fontweight="bold", loc='center', x=0.65)
ax.legend(
    loc="upper left", bbox_to_anchor=(1.02, 1.0),
    frameon=False, borderaxespad=0
)

plt.tight_layout()
fig2.savefig(".../results/cosinedist_OGmetrics_violinplot.pdf", format="pdf", bbox_inches="tight")


# # ------- CONFIG TAXONOMY -------
esm2_df = pd.read_csv(".../results/esm2/esm2_embanalysis/embedding_taxcheck/taxonomy_metrics.tsv", sep="\t")
bac_df  = pd.read_csv(".../results/bacformer/bacformer_embanalysis/taxonomy_metrics.tsv", sep="\t")
dna_df  = pd.read_csv(".../results/dnabert/dna_embanalysis/taxonomy_metrics.tsv", sep="\t")

esm2_df["Model"] = "ESM2"
bac_df["Model"]  = "Bacformer"
dna_df["Model"]  = "DNABERT-S"

# === Keep only ARI, NMI, Vmeasure ===
metrics_to_keep = ["ARI", "Vmeasure"]

# # ------- MAIN TAXONOMY -------
# === Combine all three ===
df_all = pd.concat([esm2_df, bac_df, dna_df], ignore_index=True)

# === Melt to long format ===
metrics_df = df_all.melt(
    id_vars=["Rank", "Model"],
    value_vars=metrics_to_keep,
    var_name="Metric",
    value_name="Score"
)

# === Ensure correct taxonomic order ===
rank_order = ["phylum", "class", "order", "family", "genus"]
metrics_df["Rank"] = pd.Categorical(metrics_df["Rank"], categories=rank_order, ordered=True)

# === Plot ===
fig3, ax = plt.subplots(figsize=(12, 5))

sns.lineplot(
    data=metrics_df,
    x="Rank", y="Score",
    hue="Model",
    style="Metric",
    dashes=True,
    markers=True,
    linewidth=2,
    ax=ax,
)

# === Formatting ===
ax.set_title(
    "Cosine-Based Clustering of Embeddings Across Taxonomic Levels",
    pad=15, fontweight="bold", loc='left'
)
ax.set_ylim(-0.05, 1)
ax.set_ylabel("Score")
ax.set_xlabel("Taxonomic Rank")

ax.legend(
    title="",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    frameon=False
)

plt.tight_layout()

fig3.savefig(".../results/cosinedist_LLMs_taxonomymetrics.pdf", format="pdf", bbox_inches="tight")

print("Done")
