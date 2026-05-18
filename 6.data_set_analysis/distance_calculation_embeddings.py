import os
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt
import seaborn as sns
import umap

# --------------------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------------------

# Paths
inhouse_embedding_folder = ".../embeddings/SSC/bacformer"
external_embedding_folder = ".../embeddings/metaanalysis/bacformer"
protein = True

# In-house split for easy/hard
inhouse_split_file = ".../data/hard_easy_tax_split.tsv"

# External study isolate lists
external_study_files = [
    ".../data/Bacillus_functional_state_files/isolate_list/Duran_2021.txt",
    ".../data/Bacillus_functional_state_files/isolate_list/Finkel_2019.txt",
    ".../data/Bacillus_functional_state_files/isolate_list/Finkel_2020.txt",
    ".../data/Bacillus_functional_state_files/isolate_list/Hou_2021.txt",
    ".../data/Bacillus_functional_state_files/isolate_list/Lebeis_2015.txt",
    ".../data/Bacillus_functional_state_files/isolate_list/Wolinska_2021.txt"
]

DISTANCE_METRIC = "cosine"  # or "euclidean"
MMD_GAMMA = None  # None -> 1/dim heuristic
UMAP_METRIC = "cosine"

# --------------------------------------------------------------
# 2. HELPER FUNCTIONS
# --------------------------------------------------------------

def normalize_name(x):
    return str(x).strip()

def load_embeddings(folder_path, isolate_list=None, protein=False, limit=None):
    """
    Load embeddings and return a dictionary of isolate -> mean_embedding_vector.
    Similar logic as your working load_data_and_targets, but only returns mean embeddings.
    """
    embeddings_dict = {}
    files = [f for f in os.listdir(folder_path) if f.endswith('.h5')]
    
    if limit:
        files = files[:limit]
        
    if isolate_list is not None:
        isolate_list = set(str(x).strip() for x in isolate_list)

    for file in files:
        isolate_id = file.replace("_embeddings.h5", "") if protein else file.replace(".h5", "")
        isolate_id = normalize_name(isolate_id)
        if isolate_list and isolate_id not in isolate_list:
            continue

        filepath = os.path.join(folder_path, file)
        if not os.path.isfile(filepath):
            print("Skipping invalid path:", filepath)
            continue
        
        embeddings = []
        try:
            with h5py.File(filepath, 'r') as h5f:
                if protein:
                    for protein_id in h5f.keys():
                        try:
                            mean_emb = h5f[protein_id]['embedding'][:]
                            embeddings.append(mean_emb)
                        except KeyError:
                            continue
                else:
                    try:
                        mean_emb = h5f['mean_embedding'][:]
                        if mean_emb.ndim == 1:
                            mean_emb = mean_emb[np.newaxis, :]
                        embeddings.append(mean_emb)
                    except KeyError:
                        continue
        except (OSError, ValueError) as e:
            print(f"Skipping {file}: {e}")
            continue  # skip corrupted/unreadable files

        if not embeddings:
            continue

        # take mean across all embeddings for this isolate
        embeddings_dict[isolate_id] = np.mean(np.vstack(embeddings), axis=0)

    return embeddings_dict

def dataset_pairwise_distance(d1, d2, metric="cosine"):
    X = np.vstack(list(d1.values()))
    Y = np.vstack(list(d2.values()))
    return cdist(X, Y, metric=metric).mean()

def compute_mmd(X, Y, gamma=None):
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    K_XX = rbf_kernel(X, X, gamma=gamma)
    K_YY = rbf_kernel(Y, Y, gamma=gamma)
    K_XY = rbf_kernel(X, Y, gamma=gamma)
    m = X.shape[0]
    n = Y.shape[0]
    mmd2 = K_XX.sum() / (m*m) + K_YY.sum() / (n*n) - 2*K_XY.sum() / (m*n)
    return np.sqrt(mmd2)

# --------------------------------------------------------------
# 3. LOAD IN-HOUSE EMBEDDINGS
# --------------------------------------------------------------

split_df = pd.read_csv(inhouse_split_file, sep="\t")
split_df["isolate"] = split_df["isolate"].apply(normalize_name)

inhouse_easy_ids = set(split_df[split_df["split"]=="easy"]["isolate"])
inhouse_hard_ids = set(split_df[split_df["split"]=="hard"]["isolate"])

print("Loading in-house easy embeddings...")
inhouse_easy_embeddings = load_embeddings(inhouse_embedding_folder, isolate_list=inhouse_easy_ids, protein=protein)
print(len(inhouse_easy_embeddings))
print("Loading in-house hard embeddings...")
inhouse_hard_embeddings = load_embeddings(inhouse_embedding_folder, isolate_list=inhouse_hard_ids, protein=protein)
print(len(inhouse_hard_embeddings))

# --------------------------------------------------------------
# 4. LOAD EXTERNAL EMBEDDINGS
# --------------------------------------------------------------

external_embeddings_dict = {}  # {study_name: {isolate_id: vector}}
all_inhouse_ids = inhouse_easy_ids | inhouse_hard_ids

for f in external_study_files:
    study_name = Path(f).stem
    with open(f) as fh:
        isolates = [normalize_name(line.strip()) for line in fh if line.strip()]
    isolates = [i for i in isolates if i not in all_inhouse_ids]  # exclude in-house overlap
    emb = load_embeddings(external_embedding_folder, isolate_list=isolates, protein=protein)
    external_embeddings_dict[study_name] = emb
    print(f"{study_name}: {len(emb)} isolates loaded")

# --------------------------------------------------------------
# 5. COMBINE DATASETS
# --------------------------------------------------------------

datasets = {
    "inhouse_easy": inhouse_easy_embeddings,
    "inhouse_hard": inhouse_hard_embeddings
}
datasets.update(external_embeddings_dict)

dataset_names = list(datasets.keys())

# --------------------------------------------------------------
# 6. COMPUTE DISTANCE AND MMD MATRICES
# --------------------------------------------------------------

n = len(dataset_names)
mpd_matrix = np.zeros((n,n))
mmd_matrix = np.zeros((n,n))

for i, d1 in enumerate(dataset_names):
    for j, d2 in enumerate(dataset_names):
        mpd_matrix[i,j] = dataset_pairwise_distance(datasets[d1], datasets[d2], metric=DISTANCE_METRIC)
        X = np.vstack(list(datasets[d1].values()))
        Y = np.vstack(list(datasets[d2].values()))
        mmd_matrix[i,j] = compute_mmd(X, Y, gamma=MMD_GAMMA)

mpd_df = pd.DataFrame(mpd_matrix, index=dataset_names, columns=dataset_names)
mmd_df = pd.DataFrame(mmd_matrix, index=dataset_names, columns=dataset_names)

# Relative shift to in-house easy
rel_mpd = mpd_df / mpd_df.loc["inhouse_easy", "inhouse_easy"]
rel_mmd = mmd_df / mmd_df.loc["inhouse_easy", "inhouse_easy"]

# Compute KDE distance from cetroid
inhouse_centroid = np.mean(np.vstack(list(datasets["inhouse_easy"].values())), axis=0)

# Compute distances for each dataset
distances = []
labels = []

for ds, emb_dict in datasets.items():
    if len(emb_dict) == 0:
        continue
    vecs = np.vstack(list(emb_dict.values()))
    d = cdist(vecs, [inhouse_centroid], metric="cosine").flatten()
    distances.extend(d)
    labels.extend([ds] * len(d))

df_dist = pd.DataFrame({"distance": distances, "dataset": labels})

centroid_dists = {
    ds: cdist([np.mean(np.vstack(list(emb_dict.values())), axis=0)],
              [inhouse_centroid], metric="cosine")[0,0]
    for ds, emb_dict in datasets.items() if len(emb_dict) > 0
}

# --------------------------------------------------------------
# 7. VISUALIZATION
# --------------------------------------------------------------

# UMAP of all isolates colored by dataset
all_embeddings = []
all_labels = []
all_datasets = []
for dataset_name, emb_dict in datasets.items():
    for isolate, vec in emb_dict.items():
        all_embeddings.append(vec)
        all_labels.append(isolate)
        all_datasets.append(dataset_name)

all_embeddings = np.vstack(all_embeddings)
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric=UMAP_METRIC, random_state=42)
embedding_2d = reducer.fit_transform(all_embeddings)

# --------------------------------------------------------------
# 8. SAVE MATRICES
# --------------------------------------------------------------

mpd_df.to_csv("/media/wolf2/TeamDeJonge/projects/AI4Life/Petra_Matyskova/attention_version/results/bacformer_embedding_mpd_matrix.csv")
mmd_df.to_csv("/media/wolf2/TeamDeJonge/projects/AI4Life/Petra_Matyskova/attention_version/results/bacformer_embedding_mmd_matrix.csv")
rel_mpd.to_csv("/media/wolf2/TeamDeJonge/projects/AI4Life/Petra_Matyskova/attention_version/results/bacformer_embedding_mpd_relative_shift.csv")
rel_mmd.to_csv("/media/wolf2/TeamDeJonge/projects/AI4Life/Petra_Matyskova/attention_version/results/bacformer_embedding_mmd_relative_shift.csv")
np.save("/media/wolf2/TeamDeJonge/projects/AI4Life/Petra_Matyskova/attention_version/results/bacformer_umap_coords.npy", embedding_2d)
np.save("/media/wolf2/TeamDeJonge/projects/AI4Life/Petra_Matyskova/attention_version/results/bacformer_umap_labels.npy", np.array(all_datasets))

# Save for later
df_dist.to_csv(
    "/media/wolf2/TeamDeJonge/projects/AI4Life/Petra_Matyskova/attention_version/results/bacformer_embedding_distance_to_inhouse.csv",
    index=False
)
pd.Series(centroid_dists).to_csv(
    "/media/wolf2/TeamDeJonge/projects/AI4Life/Petra_Matyskova/attention_version/results/bacformer_embedding_centroid_distance.csv"
)

print("Done. MPD and MMD matrices saved.")