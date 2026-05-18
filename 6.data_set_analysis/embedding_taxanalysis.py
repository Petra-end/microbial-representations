"""
Taxonomy-level embedding analysis.

This script:
- Loads isolate-level embeddings (mean across embeddings per isolate)
- Loads taxonomy annotations
- Computes cosine distance matrix
- Evaluates how well embeddings capture taxonomic structure
- Saves metrics per taxonomic rank for later plotting
"""

import os
import h5py
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    pairwise_distances,
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_completeness_v_measure,
    silhouette_score,
)
from sklearn.cluster import AgglomerativeClustering


# ---------- CONFIG ---------- 
EMBED_DIR = ".../embeddings/SSC/bacformer"
TAXONOMY_FILE = ".../data/SSC_data/SSC_taxonomy_GTDB.tsv"  # taxonomy table
OUTPUT_DIR = ".../results/bacformer_embanalysis"
LINKAGE = "average"
protein=True

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------- HELPERS ----------
def ci95(values):
    """Compute mean, std, and 95% confidence interval."""
    arr = np.asarray(values, float)
    m, s = np.mean(arr), np.std(arr, ddof=1)
    half = 1.96 * s / math.sqrt(len(arr))
    return m, s, (m - half, m + half)


def load_mean_embeddings(folder, protein=True):
    """
    Compute mean embedding for each isolate file.

    Args:
        folder (str): Path to the folder containing .h5 files.
        protein (bool): Whether to treat files as protein embeddings or DNA embeddings.

    Returns:
        dict: Mapping isolate_id -> mean embedding vector (np.ndarray)
    """
    embeddings = {}
    files = [f for f in os.listdir(folder) if f.endswith(".h5")]

    for file in tqdm(files, desc="Loading isolate embeddings"):
        isolate_id = (
            file.replace("_embeddings.h5", "").replace(".h5", "")
        )
        filepath = os.path.join(folder, file)

        with h5py.File(filepath, "r") as h5f:
            all_vecs = []

            if protein:
                # --- Protein case: each protein has its own mean_embedding
                for key in h5f.keys():
                    try:
                        vec = h5f[key]["embedding"][:]
                        all_vecs.append(vec)
                    except Exception:
                        continue
            else:
                # --- DNA case: single dataset called 'mean_embedding' (matrix or vector)
                try:
                    mean_emb = h5f["mean_embedding"][:]
                    if mean_emb.ndim == 1:
                        mean_emb = mean_emb[np.newaxis, :]  # handle single vector
                    all_vecs.append(mean_emb)
                except Exception:
                    continue

            if all_vecs:
                # Stack and average across all embeddings for isolate
                all_vecs = np.vstack(all_vecs)
                embeddings[isolate_id] = np.mean(all_vecs, axis=0)

    return embeddings


def clustering_metrics(D, labels):
    """Compute clustering performance metrics."""
    n_clusters = len(np.unique(labels))
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters, metric="precomputed", linkage=LINKAGE
    )
    pred = clustering.fit_predict(D)
    ari = adjusted_rand_score(labels, pred)
    nmi = normalized_mutual_info_score(labels, pred)
    h, c, v = homogeneity_completeness_v_measure(labels, pred)
    try:
        sil = silhouette_score(D, labels, metric="precomputed")
    except Exception:
        sil = np.nan
    return dict(ARI=ari, NMI=nmi, Homogeneity=h, Completeness=c, Vmeasure=v, Silhouette=sil)


# ---------- MAIN ----------
def main():
    print("=== TAXONOMY EMBEDDING ANALYSIS ===")

    # Load mean embeddings per isolate
    emb_dict = load_mean_embeddings(EMBED_DIR, protein=protein)
    isolates = list(emb_dict.keys())
    X = np.vstack([emb_dict[i] for i in isolates])

    # Load and clean taxonomy table
    tax = pd.read_csv(TAXONOMY_FILE, sep="\t", dtype=str)
    tax = tax.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

    # Drop non-taxonomic columns
    drop_cols = {"kingdom", "SynCom"}
    tax = tax[[c for c in tax.columns if c not in drop_cols]]

    # Keep only isolates for which we have embeddings
    iso_df = pd.DataFrame({"isolate": isolates})
    tax = tax.merge(iso_df, on="isolate", how="inner")

    # Match embedding order
    tax = tax.set_index("isolate").loc[iso_df["isolate"]].reset_index()

    # Clean string labels globally
    for col in tax.columns:
        tax[col] = tax[col].astype(str).replace(
            {"nan": "Unknown", "NA": "Unknown", "NaN": "Unknown", "": "Unknown"}
        )

    # Compute cosine distance matrix
    print("\nComputing cosine distance matrix...")
    D = pairwise_distances(X, metric="cosine")
    np.savez(os.path.join(OUTPUT_DIR, "taxonomy_pairwise_cosine.npz"), D=D, isolates=isolates)

    # Evaluate clustering per rank
    ranks = [c for c in tax.columns if c != "isolate"]
    results = []
    for rank in ranks:
        labels = tax[rank].values
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            print(f"Skipping {rank}: fewer than 2 valid classes ({unique_labels})")
            continue

        print(f"\n→ Evaluating taxonomy level: {rank} ({len(unique_labels)} classes)")
        metrics = clustering_metrics(D, labels)
        metrics["Rank"] = rank
        results.append(metrics)

    # 5️⃣ Save metrics
    df = pd.DataFrame(results)
    out_path = os.path.join(OUTPUT_DIR, "taxonomy_metrics.tsv")
    df.to_csv(out_path, sep="\t", index=False)

    print(f"\n Saved metrics to {out_path}\n")
    print(df.round(3).to_string(index=False))
    return df


if __name__ == "__main__":
    df = main()

