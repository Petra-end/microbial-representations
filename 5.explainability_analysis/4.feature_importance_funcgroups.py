import pandas as pd
import numpy as np
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
import os

# ================= USER CONFIG =================
output_folder = ".../results/bacformer"
master_file = os.path.join(output_folder, "master_top50_annotated.csv")

# Annotation Maps (for KOs)
ko_to_pathway_file = ".../data/Bacillus_functional_state_files/annotations/KO_to_pathway.txt"
pathway_desc_file = ".../data/Bacillus_functional_state_files/annotations/pathway_to_descrip.txt"

# ================= HELPER FUNCTIONS =================

def load_pathway_maps():
    """Loads KO->Pathway and Pathway->Description maps."""
    if not os.path.exists(ko_to_pathway_file): return None, None
    
    k2p = pd.read_csv(ko_to_pathway_file, sep="\t", header=None, names=["KO", "Pathway"])
    k2p["KO"] = k2p["KO"].str.strip()
    
    p2d = pd.read_csv(pathway_desc_file, sep="\t", header=None, names=["Pathway", "Description"])
    p2d["Pathway"] = p2d["Pathway"].str.strip()
    
    return k2p, p2d

def calculate_fisher_enrichment(row, total_rc, total_nrc):
    """
    Contingency Table:
              | In Top 50% | Not in Top 50%
        ------+------------+---------------
        RC    |     a      |   Total_RC - a
        NRC   |     c      |   Total_NRC - c
    """
    a = row["RC_Isolate_Count"]
    c = row["NRC_Isolate_Count"]
    
    b = total_rc - a
    d = total_nrc - c
    
    # Two-sided Fisher's exact test
    odds, p_value = stats.fisher_exact([[a, b], [c, d]], alternative='two-sided')
    return odds, p_value

def assign_biolabels(df):
    """Add Frac_Diff, IG_Diff and the BioLabel columns"""

    # relative metrics
    df["Frac_Diff"] = df["RC_Frac"] - df["NRC_Frac"]
    df["IG_Diff"] = df["RC_Mean_Abs"] - df["NRC_Mean_Abs"]

    # ClassSpec
    df["BioLabel_RC_ClassSpec"] = np.where(
        df["RC_Mean_Signed"] > 0, "RC_bio", "RC_spurious"
    )
    df["BioLabel_NRC_ClassSpec"] = np.where(
        df["NRC_Mean_Signed"] < 0, "NRC_bio", "NRC_spurious"
    )

    # RelL1
    df["BioLabel_RC_RelL1"] = np.where(
        (df["IG_Diff"] > 0) & (df["RC_Mean_Signed"] > 0),
        "RC_bio",
        "RC_spurious",
    )
    df["BioLabel_NRC_RelL1"] = np.where(
        (df["IG_Diff"] < 0) & (df["NRC_Mean_Signed"] < 0),
        "NRC_bio",
        "NRC_spurious",
    )

    # RelL2
    df["BioLabel_RC_RelL2"] = np.where(
        (df["Frac_Diff"] > 0) & (df["IG_Diff"] > 0) & (df["RC_Mean_Signed"] > 0),
        "RC_bio",
        "RC_spurious",
    )
    df["BioLabel_NRC_RelL2"] = np.where(
        (df["Frac_Diff"] < 0) & (df["IG_Diff"] < 0) & (df["NRC_Mean_Signed"] < 0),
        "NRC_bio",
        "NRC_spurious",
    )

    return df

# ================= ANALYSIS ENGINE =================

def analyze_features(master_df, group_col, mode="OG"):
    print(f"🔹 Analyzing by {group_col}...")
    
    # 1. Prepare Data
    if mode == "KO":
        # Explode comma-separated KOs
        df_exp = master_df.assign(KEGG_ko=master_df["KEGG_ko"].str.split(",")).explode("KEGG_ko")
        df_exp["KEGG_ko"] = df_exp["KEGG_ko"].str.replace("ko:", "").str.strip()
    elif mode == "Pathway":
        # First, get KOs
        df_exp = master_df.assign(KEGG_ko=master_df["KEGG_ko"].str.split(",")).explode("KEGG_ko")
        df_exp["KEGG_ko"] = df_exp["KEGG_ko"].str.replace("ko:", "").str.strip()
        # Load maps and merge to get Pathway IDs
        k2p, p2d = load_pathway_maps()
        if k2p is None:
            print("Pathway files not found. Skipping.")
            return
        # Merge Proteins to Pathways
        df_exp = df_exp.merge(k2p, left_on="KEGG_ko", right_on="KO", how="inner")
        # group_col is now 'Pathway'
    else:
        df_exp = master_df.copy()
        df_exp[group_col] = df_exp[group_col].astype(str).str.strip()

    # Remove Unknowns
    df_exp = df_exp[~df_exp[group_col].isin(["Unknown", "nan", "None", "-"])]

    # 2. Get Total Isolate Counts (For Fisher's Test)
    # We count unique isolates present in the Master List
    total_rc = master_df[master_df["Phenotype"] == "RC"]["Isolate"].nunique()
    total_nrc = master_df[master_df["Phenotype"] == "NRC"]["Isolate"].nunique()

    # 3. Group and Aggregate
    # Step 1: compute per-isolate group metrics (each isolate contributes once)
    per_iso = df_exp.groupby(["Isolate", "Phenotype", group_col]).agg(
        IsoMean_Abs=("NormAbs", "sum"),
        IsoMean_Signed=("IG_Signed", "mean")
    ).reset_index()

    # Step 2: average across isolates within phenotype (isolate-weighted)
    grouped = per_iso.groupby([group_col, "Phenotype"]).agg(
        Isolate_Count=("Isolate", "nunique"),
        Mean_Abs=("IsoMean_Abs", "mean"),
        Mean_Signed=("IsoMean_Signed", "mean")
    ).reset_index()

    # Pivot to get RC vs NRC columns side-by-side
    pivot = grouped.pivot(index=group_col, columns='Phenotype', values=['Isolate_Count', 'Mean_Abs', 'Mean_Signed'])
    pivot.columns = [f"{pheno}_{metric}" for metric, pheno in pivot.columns]
    pivot = pivot.reset_index()
    
    cols_to_zero = [c for c in pivot.columns if ("_Count" in c) or ("_Mean" in c)]
    pivot[cols_to_zero] = pivot[cols_to_zero].fillna(0)
    
    # 4. Calculate Statistics (Fisher + Fractions)
    pivot["RC_Frac"] = pivot["RC_Isolate_Count"] / total_rc
    pivot["NRC_Frac"] = pivot["NRC_Isolate_Count"] / total_nrc
    
    results = pivot.apply(lambda r: calculate_fisher_enrichment(r, total_rc, total_nrc), axis=1)
    pivot["odds_ratio_Fisher"], pivot["p_value_Fisher"] = zip(*results)
    
    # 5. FDR CORRECTION (Benjamini-Hochberg)
    if len(pivot) > 0:
        reject, fdr, _, _ = multipletests(pivot["p_value_Fisher"], alpha=0.05, method='fdr_bh')
        pivot["FDR"] = fdr
        pivot["Significant_Biomarker"] = reject
        pivot["-log10FDR"] = -np.log10(pivot["FDR"] + 1e-300)

    # 5. Biolabels
    pivot = assign_biolabels(pivot)

    # 6. Add Pathway Info (Only for KO)
    if mode == "KO":
        k2p, p2d = load_pathway_maps()
        if k2p is not None:
            pivot = pivot.merge(k2p, left_on=group_col, right_on="KO", how="left")
            pivot = pivot.merge(p2d, on="Pathway", how="left")
            # Drop duplicate KO column from merge if exists
            if "KO" in pivot.columns and group_col == "KEGG_ko": pivot.drop(columns=["KO"], inplace=True)
            
    if mode == "Pathway":
        # Add Number of unique KOs that contributed to this pathway
        ko_counts = df_exp.groupby("Pathway")["KEGG_ko"].nunique().reset_index(name="Num_KOs_in_Pathway")
        pivot = pivot.merge(ko_counts, on="Pathway", how="left")
        
        # Add Descriptions
        _, p2d = load_pathway_maps()
        pivot = pivot.merge(p2d, on="Pathway", how="left")

    # 7. Add Annotation Tier Stats (Only for OG - as requested in your previous script)
    if mode == "OG":
        # Calculate % Strong/Partial/None for this OG across all proteins in Master List
        tier_counts = df_exp.groupby([group_col, "AnnotTier"]).size().unstack(fill_value=0)
        tier_counts["Total"] = tier_counts.sum(axis=1)
        
        pivot = pivot.merge(tier_counts, on=group_col, how="left")

    # Save
    out_name = f"Summary_{mode}_Fisher_BioAnnotated.csv"
    pivot.to_csv(os.path.join(output_folder, out_name), index=False)
    print(f"Saved: {out_name}")

# ================= EXECUTION =================

print("Loading Master List...")
master = pd.read_csv(master_file)

# Run for OGs
analyze_features(master, group_col="OG", mode="OG")

# Run for KOs
analyze_features(master, group_col="KEGG_ko", mode="KO")

# Run for Pathways
analyze_features(master, group_col="Pathway", mode="Pathway")