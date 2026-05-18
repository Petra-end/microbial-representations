import pandas as pd
import numpy as np
import glob
import os
from io import StringIO
from tqdm import tqdm

# ================= USER CONFIG =================
ig_folder = ".../results/bacformer/feature_importance_withpred"
eggnog_dir = ".../data/SSC_data/eggNOG_annotations"
output_folder = ".../results/bacformer"

os.makedirs(output_folder, exist_ok=True)
TOP_FRAC = 0.50  # Cumulative importance threshold

# ================= HELPER FUNCTIONS =================

def classify_annotation_tier(row):
    """
    Classify protein into annotation tiers:
      - 'strong'  : KEGG KO present
      - 'partial' : any non-DUF PFAM, or GO, or some descriptive text, or EC but not full
      - 'none'    : only DUF/hypothetical/no useful info
    """
    def is_missing(x):
        x = str(x).strip()
        return (x == "") or (x == "-") or (x.lower() == "nan") or (x == "Unknown") or (x == " ")

    # KEGG KO
    ko = row.get("KEGG_ko", "")
    has_ko = (not is_missing(ko))

    # EC
    ec = str(row.get("EC", "")).strip()
    has_any_ec = (not is_missing(ec))

    # PFAMs
    pf = str(row.get("PFAMs", "")).strip()
    pf_tokens = [p.strip() for p in pf.split(",") if not is_missing(p) and p.strip() not in ["", "-"]]
    has_non_duf_pf = any("DUF" not in tok.upper() for tok in pf_tokens)

    # GO
    gos = row.get("GOs", "")
    has_go = not is_missing(gos)

    # Description / Preferred_name
    desc = str(row.get("Description", "")).strip()
    pref = str(row.get("Preferred_name", "")).strip()

    def has_meaningful_text(text):
        if is_missing(text):
            return False
        t = str(text).strip().lower()
        bad_substrings = [
            "hypothetical protein",
            "uncharacterized protein",
            "protein of unknown function",
            "unknown function",
            "unknown protein",
            "putative protein",
        ]
        if any(b in t for b in bad_substrings):
            return False
        return True

    has_text = has_meaningful_text(desc) or has_meaningful_text(pref)

    # Strong annotation
    if has_ko: #or has_full_ec: if ec considered strong annotation
        return "strong"

    # Partial annotation
    if has_non_duf_pf or has_go or has_text or has_any_ec:
        return "partial"

    # No real annotation
    return "none"

eggnog_cache = {}
missing = {"-", "nan", "NA", "None", "",  " "}

def load_eggnog_for_isolate(isolate_id):
    """
    Load eggNOG annotation file for a given isolate, return DataFrame with:
    Isolate, Protein_ID, KEGG_ko, GOs, EC, COG_category, Description,
    Preferred_name, PFAMs (missing columns are filled with '-').
    """
    if isolate_id in eggnog_cache:
        return eggnog_cache[isolate_id]

    path = os.path.join(eggnog_dir, f"{isolate_id}.annotations")
    if not os.path.exists(path):
        eggnog_cache[isolate_id] = pd.DataFrame(columns=[
            "Isolate", "Protein_ID", "KEGG_ko", "GOs", "EC",
            "COG_category", "Description", "Preferred_name", "PFAMs"
        ])
        return eggnog_cache[isolate_id]

    with open(path, "r") as f:
        lines = [ln for ln in f if not ln.startswith("##")]

    df = pd.read_csv(StringIO("".join(lines)), sep="\t", low_memory=False)

    # Normalize query column
    if "#query" in df.columns:
        df.rename(columns={"#query": "Protein_ID"}, inplace=True)
    elif "query" in df.columns:
        df.rename(columns={"query": "Protein_ID"}, inplace=True)
    else:
        # weird file, return empty
        eggnog_cache[isolate_id] = pd.DataFrame(columns=[
            "Isolate", "Protein_ID", "KEGG_ko", "GOs", "EC",
            "COG_category", "Description", "Preferred_name", "PFAMs"
        ])
        return eggnog_cache[isolate_id]

    # Ensure columns exist
    for col in ["KEGG_ko", "GOs", "EC", "COG_category",
                "Description", "Preferred_name", "PFAMs"]:
        if col not in df.columns:
            df[col] = "-"

    df["Protein_ID"] = df["Protein_ID"].astype(str)
    df["Isolate"] = isolate_id

    eggnog_cache[isolate_id] = df[[
        "Isolate", "Protein_ID", "KEGG_ko", "GOs", "EC",
        "COG_category", "Description", "Preferred_name", "PFAMs"
    ]]
    return eggnog_cache[isolate_id]

# ================= MAIN LOGIC =================

all_dfs = []
files = glob.glob(os.path.join(ig_folder, "*_protein_ig_withpred.csv"))

print(f"🔹 Processing {len(files)} isolate files...")

for path in tqdm(files):
    df = pd.read_csv(path)
    
    # Basic Validation
    if "Correct" not in df.columns or "CumAbs" not in df.columns:
        continue
        
    # 1. Filter: Correct Predictions Only
    if not bool(df["Correct"].iloc[0]):
        continue
        
    # 2. Filter: Top 50% Importance
    df_top = df[df["CumAbs"] <= TOP_FRAC].copy()
    
    if df_top.empty:
        continue

    isolate_id = df_top["Isolate"].iloc[0]
    
    # 3. Load Annotations
    annot_df = load_eggnog_for_isolate(isolate_id)
    
    if not annot_df.empty:
        annot_df["Protein_ID"] = annot_df["Protein_ID"].astype(str)
        df_top["Protein_ID"] = df_top["Protein_ID"].astype(str)
        
        # Merge annotations (Left join to keep protein even if no annotation)
        # We drop 'KEGG_ko' from df_top before merge to avoid duplication if they exist in IG file
        cols_drop = [c for c in ["KEGG_ko"] if c in df_top.columns]
        df_top = df_top.drop(columns=cols_drop)
        
        df_top = df_top.merge(annot_df.drop(columns=["Isolate"]), on="Protein_ID", how="left")

    # 4. Add Annotation Tier
    df_top["AnnotTier"] = df_top.apply(classify_annotation_tier, axis=1)
    
    all_dfs.append(df_top)

# Combine all
master_df = pd.concat(all_dfs, ignore_index=True)

# Fill NaNs
fillna_cols = ["KEGG_ko", "OG", "Description", "Preferred_name"]
for c in fillna_cols:
    if c in master_df.columns:
        master_df[c] = master_df[c].fillna("Unknown")

# Save
out_path = os.path.join(output_folder, "master_top50_annotated.csv")
master_df.to_csv(out_path, index=False)

print("\n Master List Created")
print(f"   Output: {out_path}")
print(f"   Total Proteins: {len(master_df)}")
print(f"   Unique Isolates: {master_df['Isolate'].nunique()}")
print(f"   RC Isolates: {master_df[master_df['Phenotype']=='RC']['Isolate'].nunique()}")
print(f"   NRC Isolates: {master_df[master_df['Phenotype']=='NRC']['Isolate'].nunique()}")