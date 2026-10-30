"""
Integrated Gradients per isolate and per protein

Description:
------------
Computes Integrated Gradients (IG) for a trained MLP model
to estimate per-protein contribution to isolate-level predictions.

For each isolate:
  - Computes IG w.r.t. baseline (mean embedding of NRC proteins)
  - Outputs per-protein attributions + functional annotations (KEGG KO, OG)
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from captum.attr import IntegratedGradients

# add parent directory (attention_version/) to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MLP_classif_keggog import MLPClassifier
from data_loader import load_data_and_targets

# -------------------- CONFIG --------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Paths
model_dir = ".../results/bacformer/bacformer_480_GT0_3l"
config_file = os.path.join(model_dir, "fold_2_config.json")
checkpoint = os.path.join(model_dir, "model_2.pth")

embeddings_folder = ".../embeddings/SSC/bacformer"
label_file = ".../data/GT_SSC_s1.txt"
split_file = ".../data/hard_easy_tax_split.tsv"
eggnog_dir = ".../data/eggNOG_annotations"
gene_to_og_file = "/data/gene_to_OG.tsv"
prokka_root = ".../data/prokka_output"

output_dir = ".../results/bacformer/feature_importance_bf"
os.makedirs(output_dir, exist_ok=True)

MAX_ISOLATES = None  
IG_N_STEPS = 100     # integration steps (50–200)
INTERNAL_BS  = 64
protein = True

# -------------------- LOAD MODEL --------------------
print("Loading trained model...")
with open(config_file, "r") as f:
    config = json.load(f)
params = config["fixed_params"]
best_params = config["optuna_best_params"]

model = MLPClassifier(
    input_size=params["input_size"],
    hidden_sizes=params["hidden_size"],
    output_size=params["output_size"],
    dropout_rate=best_params["dropout_rate"],
    loss_type=params["loss_type"]
).to(device)

state = torch.load(checkpoint, map_location=device)
model.load_state_dict(state)
model.eval()
print("Model loaded.")

# -------------------- LOAD DATA --------------------
print("Loading embeddings and metadata...")
X_all, y_all, IDs_all = load_data_and_targets(
    folder_path=embeddings_folder,
    targets_file=label_file,
    meta_split_file=split_file,
    # split_type="easy",
    limit=MAX_ISOLATES,
    protein=protein
)

X_all = np.array(X_all, dtype=object)
y_all = np.array(y_all)
IDs_all = np.array(IDs_all, dtype=object)
print(f"Loaded {len(X_all)} isolates.")

# -------------------- HELPER FUNCTIONS --------------------
def forward_func(x_2d):
    """Forward wrapper for Captum. Takes [k, d] tensor → scalar probability."""
    out = model(x_2d.unsqueeze(0))
    if isinstance(out, tuple):
        out = out[0]
    if out.ndim == 1:
        out = out.unsqueeze(0)
    if out.shape[1] == 1:
        out = torch.sigmoid(out)
    return out

def load_eggnog_map(isolate_id):
    file_path = os.path.join(eggnog_dir, f"{isolate_id}.annotations")
    if not os.path.exists(file_path):
        return {}

    # Read file while skipping only lines starting with '##'
    with open(file_path, "r") as f:
        lines = [line for line in f if not line.startswith("##")]
    from io import StringIO
    df = pd.read_csv(StringIO("".join(lines)), sep="\t", low_memory=False)

    # normalize column name
    if "#query" in df.columns:
        df.rename(columns={"#query": "Protein_ID"}, inplace=True)
    elif "query" in df.columns:
        df.rename(columns={"query": "Protein_ID"}, inplace=True)

    # If KEGG_ko column is missing, create it as "Unknown"
    if "KEGG_ko" not in df.columns:
        df["KEGG_ko"] = "Unknown"

    return dict(zip(df["Protein_ID"].astype(str), df["KEGG_ko"].astype(str)))

def load_gene_to_og_map():
    """Load protein → OG mapping."""
    if not os.path.exists(gene_to_og_file):
        return {}
    df = pd.read_csv(gene_to_og_file, sep="\t")
    df.columns = [c.strip().lower() for c in df.columns]
    df["gene"] = df["gene"].astype(str)
    df["og"] = df["og"].astype(str)
    return dict(zip(df["gene"], df["og"]))

def clean_annotation_fields(df):
    # --- KEGG_ko cleaning ---
    df["KEGG_ko"] = (
        df["KEGG_ko"]
        .astype(str)
        .str.replace('"', '', regex=False)     # remove quotes
        .str.replace("'", '', regex=False)     # remove single quotes
        .str.strip()                           # remove leading/trailing spaces
        .replace({"-": "Unknown", "": "Unknown", "nan": "Unknown"})
    )

    # --- OG cleaning ---
    if "OG" in df.columns:
        df["OG"] = (
            df["OG"]
            .astype(str)
            .str.replace('"', '', regex=False)
            .str.replace("'", '', regex=False)
            .str.strip()
            .replace({"-": "Unknown", "": "Unknown", "nan": "Unknown"})
        )

    return df

def parse_prokka_tbl(isolate_id):
    """
    Parse Prokka .tbl file for one isolate.
    Returns a dict {scaffold_id: [protein_ids]}.
    """
    folder = os.path.join(prokka_root, f"prokka_{isolate_id}")
    if not os.path.exists(folder):
        print(f"⚠️ Folder not found for {isolate_id}: {folder}")
        return {}

    # Find .tbl file (handles weird naming)
    tbl_files = [f for f in os.listdir(folder) if f.lower().endswith(".tbl")]
    if len(tbl_files) == 0:
        print(f"⚠️ No .tbl file found in {folder}")
        return {}

    # Usually only one .tbl — take the first
    path = os.path.join(folder, tbl_files[0])
    mapping = {}
    current_scaf = None
    prot_count = 0

    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith(">Feature"):
                    # e.g., ">Feature scaffold_0"
                    current_scaf = line.split(" ", 1)[1].strip()
                    mapping.setdefault(current_scaf, [])
                elif line.startswith("locus_tag"):
                    parts = line.split()
                    if len(parts) >= 2 and current_scaf:
                        prot_id = parts[1]
                        mapping[current_scaf].append(prot_id)
                        prot_count += 1
    except Exception as e:
        print(f"⚠️ Error reading {path}: {e}")
        return {}

    print(f"Parsed {len(mapping)} scaffolds, {prot_count} proteins for {isolate_id}")
    return mapping

def build_metadata_table(split_file, label_file):
    """
    Build a DataFrame linking each isolate ID to its Phenotype (RC/NRC)
    and Difficulty (Easy/Hard) based on the label and split files.
    """
    # --- Load phenotype (RC/NRC) ---
    phenos = {}
    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                isolate = parts[0]
                label = int(float(parts[1]))
                phenos[isolate] = "RC" if label == 1 else "NRC"

    # --- Load difficulty (Easy/Hard) ---
    difficulty = {}
    if split_file and os.path.exists(split_file):
        df_split = pd.read_csv(split_file, sep="\t")
        for _, row in df_split.iterrows():
            difficulty[str(row["isolate"])] = row["split"].capitalize()  # Easy / Hard

    # --- Merge into one DataFrame ---
    meta = pd.DataFrame({
        "Isolate": list(phenos.keys()),
        "Phenotype": list(phenos.values()),
        "Difficulty": [difficulty.get(k, "Unknown") for k in phenos.keys()]
    })
    return meta

# ---------------------- GET METADATA -----------------------
meta_table = build_metadata_table(split_file, label_file)
meta_lookup = dict(zip(meta_table["Isolate"], zip(meta_table["Phenotype"], meta_table["Difficulty"])))
print(f"Metadata loaded for {len(meta_lookup)} isolates.")

# -------------------- BUILD NRC BASELINE --------------------
print("Computing NRC baseline embedding...")
nrc_indices = np.where(y_all == 0)[0]
if len(nrc_indices) == 0:
    raise RuntimeError("No NRC isolates found!")

sum_vec, count = None, 0
for i in nrc_indices:
    emb = X_all[i]
    if emb is None or len(emb) == 0:
        continue
    if sum_vec is None:
        sum_vec = emb.sum(axis=0).astype(np.float64)
    else:
        sum_vec += emb.sum(axis=0)
    count += emb.shape[0]

b = (sum_vec / count).astype(np.float32)
b_torch = torch.tensor(b, device=device)

print(f"NRC baseline computed: dim={len(b)}, proteins={count}")

# -------------------- INTEGRATED GRADIENTS --------------------
ig = IntegratedGradients(forward_func)
prot_to_og = load_gene_to_og_map()
kegg_cache = {}

rc_indices = np.arange(len(X_all))
if len(rc_indices) == 0:
    print("No RC isolates found; running all.")
    rc_indices = np.arange(len(X_all))

print(f"Running IG for {len(rc_indices)} isolates...")
all_summaries = []

for idx in tqdm(rc_indices, desc="IG per isolate"):
    # unpack ID / meta
    id_entry = IDs_all[idx]
    if protein:
        isolate_id, protein_ids = id_entry
    else:
        isolate_id = id_entry
        protein_ids = None

    # skip if exists (protein mode writes protein file)
    out_scaf = os.path.join(output_dir, f"{isolate_id}_scaffold_ig.csv")
    out_prot = os.path.join(output_dir, f"{isolate_id}_protein_ig.csv")
    if protein:
        if os.path.exists(out_prot):
            print(f"⏩ Skipping {isolate_id} (already computed)")
            continue

    # tensorize isolate
    iso_np = X_all[idx]
    if iso_np is None or len(iso_np) == 0:
        continue
    iso_tensor = torch.tensor(iso_np, dtype=torch.float32, device=device)
    k, d = iso_tensor.shape
    baseline = b_torch.repeat(k, 1)

    # run IG
    attributions, delta = ig.attribute(
        inputs=iso_tensor,
        baselines=baseline,
        n_steps=IG_N_STEPS,
        internal_batch_size=INTERNAL_BS,
        return_convergence_delta=True
    )

    ig_meanabs = attributions.abs().mean(dim=1).detach().cpu().numpy()
    ig_signed  = attributions.sum(dim=1).detach().cpu().numpy()
    delta_val  = float(delta.detach().cpu().mean().item())

    # --------- PROTEIN MODE: save per-protein table ---------
    if protein:
        # use protein_ids from loader
        rows = []
        # preload KEGG per isolate once
        kegg_map = kegg_cache.get(isolate_id)
        if kegg_map is None:
            kegg_map = load_eggnog_map(isolate_id)
            kegg_cache[isolate_id] = kegg_map
        phenotype, difficulty = meta_lookup.get(isolate_id, ("Unknown", "Unknown"))

        for i, pid in enumerate(protein_ids):
            kegg = kegg_map.get(pid, np.nan)
            og   = prot_to_og.get(pid, np.nan)
            rows.append({
                "Isolate": isolate_id,
                "Phenotype": phenotype,
                "Dataset": difficulty,
                "Protein_Index": i,
                "Protein_ID": pid,
                "IG_MeanAbs": float(ig_meanabs[i]),
                "IG_Signed": float(ig_signed[i]),
                "KEGG_ko": kegg,
                "OG": og
            })
        df = pd.DataFrame(rows)
        df = clean_annotation_fields(df)
        df.to_csv(out_prot, index=False)
        continue

print("Done. Per-isolate IG CSVs saved to:", output_dir)



