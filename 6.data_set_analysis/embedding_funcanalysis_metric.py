"""
Compute clustering metrics (ARI, NMI, Homogeneity, Completeness, V-measure, Silhouette)
for cosine and alignment runs stored in separate directories.

Directory structure:
BASE_DIR/
├── cosine/
│   ├── seed1/
│   ├── seed2/
│   └── ...
└── alignment/
    ├── seed1/
    ├── seed2/
    └── ...
"""

import os
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_completeness_v_measure,
    silhouette_score,
)

# ---------- CONFIG ----------
BASE_DIR = ".../results/bacformer_embanalysis"  # contains cosine/ and alignment/
OUT_DIR = BASE_DIR
LINKAGE = "average"
N_CLUSTERS = None  # None = use number of unique OGs per run

# ---------- HELPERS ----------
def ci95(values):
    """Compute 95% confidence interval and std."""
    arr = np.asarray(values, float)
    m, s = np.mean(arr), np.std(arr, ddof=1)
    half = 1.96 * s / math.sqrt(len(arr))
    return m, s, (m - half, m + half)

def load_run(folder, method):
    """Load one run’s pairwise distance matrix and labels."""
    npz_files = [f for f in os.listdir(folder) if f.endswith("_pairwise_distances.npz")]
    if not npz_files:
        return None
    npz = np.load(os.path.join(folder, npz_files[0]))
    D = npz["D"]
    labels = npz["labels"]

    # Normalize alignment distances to [0, 1]
    if method == "alignment":
        D = (D - D.min()) / (D.max() - D.min() + 1e-8)

    return {"folder": os.path.basename(folder), "D": D, "labels": labels, "method": method}

def clustering_metrics(D, labels, n_clusters=None, linkage="average"):
    """Compute all clustering metrics from a precomputed distance matrix."""
    n_clusters = n_clusters or len(np.unique(labels))
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters, metric="precomputed", linkage=linkage
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

def summarize(df, method):
    """Compute mean, std, and 95% CI per metric for one method."""
    summary = []
    for m in ["ARI", "NMI", "Homogeneity", "Completeness", "Vmeasure", "Silhouette"]:
        if m in df.columns:
            mean, std, (lo, hi) = ci95(df[m])
            summary.append(
                dict(
                    Method=method,
                    Metric=m,
                    Mean=mean,
                    SD=std,
                    CI95_low=lo,
                    CI95_high=hi,
                    N=len(df),
                )
            )
    return pd.DataFrame(summary)

# ---------- MAIN ----------
def process_method(method_dir, method_name):
    """Process all runs in one method folder (cosine or alignment)."""
    seed_dirs = [
        os.path.join(method_dir, d)
        for d in os.listdir(method_dir)
        if os.path.isdir(os.path.join(method_dir, d))
    ]

    all_results = []
    for folder in tqdm(seed_dirs, desc=f"Processing {method_name} runs"):
        run = load_run(folder, method_name)
        if run is None:
            continue
        D, labels = run["D"], run["labels"]
        metrics = clustering_metrics(D, labels, N_CLUSTERS, LINKAGE)
        metrics["Run"] = run["folder"]
        metrics["Method"] = method_name
        all_results.append(metrics)

    if not all_results:
        print(f"No valid runs found for {method_name}.")
        return None, None

    df = pd.DataFrame(all_results)
    out_path = os.path.join(OUT_DIR, f"all_runs_metrics_{method_name}.tsv")
    df.to_csv(out_path, sep="\t", index=False)
    print(f"Saved per-run metrics for {method_name} to {out_path}")
    summary_df = summarize(df, method_name)
    return df, summary_df

def main():
    cosine_dir = os.path.join(BASE_DIR, "cosine")

    # Process cosine
    cos_df, cos_summary = process_method(cosine_dir, "cosine")

    summary_path = os.path.join(OUT_DIR, "metrics_summary_mean_sd.tsv")
    cos_df.to_csv(summary_path, sep="\t", index=False)

    print(f"\n Summary table saved to {summary_path}\n")

    return cos_df

if __name__ == "__main__":
    cos_df = main()
    
print("Done")
