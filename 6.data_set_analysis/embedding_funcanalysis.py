"""
Compute and save pairwise distances and UMAP coordinates
for sampled OGs (protein families) using protein embeddings.

Workflow:
  1. Load gene→OG mapping.
  2. Build or load cached protein→isolate→h5_path map (from eggNOG annotations).
  3. Sample N OGs, collect their proteins.
  4. Load each protein's mean embedding from relevant HDF5 isolate files.
  5. Compute cosine-based pairwise distances:
  6. Run UMAP on precomputed distances for visualization.
  7. Save numeric results (UMAP coords, distance matrices, centroids, metadata).
"""

import os
import json
import time
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
import traceback

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


# ==============================
# --- USER CONFIGURATION ---
# ==============================
GENE_TO_OG_PATH = ".../embeddings/embedding_sanitycheck/gene_to_OG.tsv"
H5_FOLDER = ".../embeddings/SSC/bacformer/"
ANNOTATION_FOLDER = ".../data/SSC_data/eggNOG_annotations"
OUTPUT_FOLDER = ".../results/bacformer_embanalysis/cosine/run20/"

CACHE_MAP_PATH = os.path.join(OUTPUT_FOLDER, "protein_to_isolate_map.tsv")

N_OGS = 10
RANDOM_SEED = 7
DISTANCE_TYPE = "cosine"   

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ==============================
# --- Alignment distance ---
# ==============================
from scipy.spatial.distance import cdist


def compute_similarity_matrix(A, B, metric="cosine"):
    D = cdist(A, B, metric=metric)
    return np.exp(-D)


# ==============================
# --- Loading and Mapping ---
# ==============================
def build_protein_to_isolate_map(annotation_folder, h5_folder, cache_path):
    if os.path.exists(cache_path):
        print(f"Loading cached protein-to-isolate map from {cache_path}")
        return pd.read_csv(cache_path, sep="\t")

    print("Building protein-to-isolate map (first run, will cache it)...")
    records = []
    annots = [f for f in os.listdir(annotation_folder) if f.endswith(".annotations")]

    for f_annot in tqdm(annots, desc="Scanning annotation files"):
        isolate_id = f_annot.replace(".annotations", "")
        h5_path = os.path.join(h5_folder, f"{isolate_id}_embeddings.h5")
        if not os.path.exists(h5_path):
            continue

        with open(os.path.join(annotation_folder, f_annot), "r") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                protein = line.split("\t")[0].strip()
                records.append((protein, isolate_id, h5_path))

    df = pd.DataFrame(records, columns=["Protein_ID", "Isolate_ID", "h5_path"])
    df.to_csv(cache_path, sep="\t", index=False)
    print(f"Saved mapping cache to {cache_path} ({len(df)} entries)")
    return df


def load_gene_to_og(path):
    df = pd.read_csv(path, sep="\t")
    if not {"gene", "OG"}.issubset(df.columns):
        raise ValueError("gene_to_OG.tsv must have columns 'gene' and 'OG'")
    return df


def sample_ogs(
    gene_to_og: pd.DataFrame,
    protein_map: pd.DataFrame,
    n_ogs: int = 10,
    seed: int | None = None,
    max_iter: int = 500,
):
    """
    Randomly sample OGs one by one until n_ogs have at least one embedding.
    - gene_to_og: dataframe with columns ['gene', 'OG']
    - protein_map: dataframe with 'Protein_ID' column (genes that have embeddings)
    - n_ogs: how many valid OGs you want
    - seed: optional global seed for reproducibility
    - max_iter: safety cap to avoid infinite loop
    """
    rng = np.random.default_rng(seed)
    all_ogs = gene_to_og["OG"].unique()
    valid_ogs = []
    tried_ogs = set()
    n_total = len(all_ogs)

    print(f"Trying to collect {n_ogs} OGs with embeddings out of {n_total:,} total...")

    for _ in range(max_iter):
        og = rng.choice(all_ogs)
        if og in tried_ogs:
            continue
        tried_ogs.add(og)

        genes = gene_to_og.loc[gene_to_og["OG"] == og, "gene"]
        has_emb = protein_map["Protein_ID"].isin(genes).any()
        if has_emb:
            valid_ogs.append(og)
            print(f"Found {og} ({len(valid_ogs)}/{n_ogs})")
        if len(valid_ogs) >= n_ogs:
            break

    if len(valid_ogs) < n_ogs:
        print(f"Only found {len(valid_ogs)} valid OGs after {max_iter} iterations.")
    else:
        print("Finished sampling successfully.")

    sampled_ogs = pd.Series(valid_ogs, name="OG")
    sampled_genes = gene_to_og[gene_to_og["OG"].isin(valid_ogs)]
    return sampled_ogs, sampled_genes


def get_embeddings_h5(sampled_genes, protein_to_isolate_map, verbose=False):
    merged = sampled_genes.merge(protein_to_isolate_map, left_on="gene", right_on="Protein_ID", how="left")
    embeddings = []
    not_found = []

    for h5_path, sub in merged.groupby("h5_path"):
        if pd.isna(h5_path) or not os.path.exists(h5_path):
            if verbose:
                print(f"Missing h5_path for {len(sub)} genes")
            not_found.extend(sub["gene"].tolist())
            continue
        
        if not os.path.exists(h5_path):
            if verbose:
                print(f"File not found: {h5_path}")
            not_found.extend(sub["gene"].tolist())
            continue

        if verbose:
            print(f"Processing {h5_path} with {len(sub)} proteins")

        try:
            with h5py.File(h5_path, "r") as h5f:
                for _, row in sub.iterrows():
                    prot, og = row["gene"], row["OG"]
                    if prot not in h5f:
                        if verbose:
                            print(f"  ⚠️ {prot} not found in {h5_path}")
                        not_found.append(prot)
                        continue
                    try:
                        emb = h5f[prot]["embedding"][:]
                        embeddings.append({"Protein_ID": prot, "OG": og, "embedding": emb})
                    except KeyError:
                        if verbose:
                            print(f"No 'embedding' dataset for {prot} in {h5_path}")
                        not_found.append(prot)
                        continue
        except Exception as e:
            print(f"Error reading {h5_path}: {e}")
            continue

    if not_found:
        print(f"Warning: {len(not_found)} proteins missing in embeddings.")
    return embeddings


# ==============================
# --- Distance and UMAP ---
# ==============================
def compute_distance_matrix(embeddings, method="cosine"):
    proteins = [e["Protein_ID"] for e in embeddings]
    labels = [e["OG"] for e in embeddings]
    data = [e["embedding"] for e in embeddings]
    n = len(data)
    D = np.zeros((n, n))

    if method == "cosine":
        D = pairwise_distances(np.vstack(data), metric="cosine")
    else:
        raise ValueError(f"Unsupported method: {method}")

    return D, np.array(proteins), np.array(labels)


def run_umap_from_distances(D, random_state=7):
    if not HAS_UMAP:
        raise ImportError("umap-learn not installed. Install it with `pip install umap-learn`.")
    reducer = umap.UMAP(n_components=2, metric="precomputed", random_state=random_state)
    coords = reducer.fit_transform(D)
    return coords


# ==============================
# --- Centroid computation ---
# ==============================
def compute_centroid_distances(embeddings, labels, method="cosine", dist_matrix=None):
    """
    Compute representative distances between OGs:
    - For cosine/euclidean: centroid (mean vector)
    - For alignment/custom: medoid (most central real embedding)
    """
    unique_labels = np.unique(labels)
    print(f"Computing group distances using method={method} for {len(unique_labels)} groups...")

    # Extract raw embeddings as an array for convenience
    X = np.array([e['embedding'] if isinstance(e, dict) else e for e in embeddings])

    if method in ["cosine", "euclidean"]:
        # --- centroid-based distances ---
        centroids = []
        for label in unique_labels:
            cluster_vecs = X[np.array(labels) == label]
            if len(cluster_vecs) == 0:
                continue
            centroid = np.mean(cluster_vecs, axis=0)
            centroids.append(centroid)
        centroids = np.vstack(centroids)
        cent_dist = pairwise_distances(centroids, metric=method)
        return unique_labels, cent_dist

    elif method == "alignment":
        # --- medoid-based distances ---
        if dist_matrix is None:
            raise ValueError("dist_matrix must be provided for alignment-based distances")

        cent_dist = np.zeros((len(unique_labels), len(unique_labels)))
        medoid_indices = []

        for label in unique_labels:
            idxs = np.where(np.array(labels) == label)[0]
            sub_d = dist_matrix[np.ix_(idxs, idxs)]
            medoid_idx = idxs[np.argmin(sub_d.mean(axis=0))]
            medoid_indices.append(medoid_idx)

        # distances between medoids
        medoid_dists = dist_matrix[np.ix_(medoid_indices, medoid_indices)]
        return unique_labels, medoid_dists

    else:
        raise ValueError(f"Unknown distance method: {method}")


# ==============================
# --- Save outputs ---
# ==============================
def save_outputs(D, coords, proteins, labels, centroid_labels, centroid_dist, outdir, method):
    ts = time.strftime("%Y%m%d_%H%M%S")
    prefix = os.path.join(outdir, f"OGsample_{method}_{ts}")

    coords_df = pd.DataFrame({
        "Protein_ID": proteins,
        "OG": labels,
        "UMAP_1": coords[:, 0],
        "UMAP_2": coords[:, 1],
    })
    coords_path = prefix + "_coords.tsv"
    coords_df.to_csv(coords_path, sep="\t", index=False)

    np.savez_compressed(prefix + "_pairwise_distances.npz", D=D, proteins=proteins, labels=labels)

    cent_df = pd.DataFrame(centroid_dist, index=centroid_labels, columns=centroid_labels)
    cent_df.to_csv(prefix + "_centroid_distances.tsv", sep="\t")

    meta = {
        "timestamp": ts,
        "distance_method": method,
        "N_points": len(proteins),
        "N_OGs": len(np.unique(labels)),
        "random_seed": RANDOM_SEED,
        "umap_used": True,
    }
    with open(prefix + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\n Saved:")
    print(" -", coords_path)
    print(" -", prefix + "_pairwise_distances.npz")
    print(" -", prefix + "_centroid_distances.tsv")
    print(" -", prefix + "_meta.json")


# ==============================
# --- Main ---
# ==============================
def main():
    gene_to_og = load_gene_to_og(GENE_TO_OG_PATH)
    protein_map = build_protein_to_isolate_map(ANNOTATION_FOLDER, H5_FOLDER, cache_path="/media/wolf2/TeamDeJonge/projects/AI4Life/Petra_Matyskova/attention_version/results/bacformer_embanalysis/protein_to_isolate_map.tsv")
    sampled_ogs, sampled_genes = sample_ogs(gene_to_og, protein_map, N_OGS)
    print(f"Selected {len(sampled_ogs)} OGs, {len(sampled_genes)} proteins total.")

    embeddings = get_embeddings_h5(sampled_genes, protein_map)
    print(f"Loaded embeddings for {len(embeddings)} proteins.")

    D, proteins, labels = compute_distance_matrix(embeddings, method=DISTANCE_TYPE)
    coords = run_umap_from_distances(D, random_state=RANDOM_SEED)
    centroid_labels, centroid_dist = compute_centroid_distances(embeddings, labels, method=DISTANCE_TYPE, 
                                                                dist_matrix=D if DISTANCE_TYPE == "alignment" else None)

    save_outputs(D, coords, proteins, labels, centroid_labels, centroid_dist, OUTPUT_FOLDER, DISTANCE_TYPE)


if __name__ == "__main__":
    main()
    
    
print("Break")
