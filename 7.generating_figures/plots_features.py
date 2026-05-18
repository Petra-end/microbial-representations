import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
import re
import requests

# ---------- STYLE ----------
sns.set_theme(
    context="talk", style="white", font="Arial", font_scale=1.2
)
CUSTOM_PALETTE = [
    "#88ccee", "#117733", "#661100",  "#ddcc77",
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
cmap_rel = LinearSegmentedColormap.from_list("rel_cmap", ["#6b3f69", "#81c088"])

power = -5
factor = 10**-power   # = 1e5
def sci_e_formatter(x, pos):
    if x == 0:
        return "0"
    return f"{x * factor:g}e{power}"


# KO
# Assign KEGG category based on first two numbers of pathway code
BRITE_PATH = ".../data/br08901.keg"

def strip_html(s: str) -> str:
    # remove any HTML tags like <b>...</b>
    return re.sub(r"<[^>]+>", "", s).strip()

def strip_brite_code(label: str) -> str:
    # remove leading numeric code if present: "09100 Metabolism" -> "Metabolism"
    return re.sub(r"^\d+\s*", "", label).strip()

def parse_brite_keg(filepath: str):
    L1 = None
    mapping = {}

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            s = raw.rstrip("\n").lstrip()  # lstrip handles indentation

            if s.startswith("A"):
                L1_raw = s[1:].strip()
                L1 = strip_html(strip_brite_code(L1_raw))  # <-- key change

            elif s.startswith("C"):
                m = re.search(r"\b(\d{5})\b", s)
                if m:
                    map_id = f"map{m.group(1)}"
                    mapping[map_id] = L1

    return mapping

brite_L1 = parse_brite_keg(BRITE_PATH)

def normalize_map_id(x: str):
    m = re.search(r"(\d{5})", str(x))
    return f"map{m.group(1)}" if m else None

def kegg_L1(pathway: str) -> str:
    mid = normalize_map_id(pathway)
    return brite_L1.get(mid, "Other")

KEGG_ORDER = [
    "Metabolism",
    "Genetic Information Processing",
    "Environmental Information Processing",
    "Cellular Processes",
    "Organismal Systems"#,
    # "Human Diseases",
    # "Drug Development"
]

# -------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------

df_pathway = pd.read_csv(
    "/media/wolf2/TeamDeJonge/projects/AI4Life/Petra_Matyskova/attention_version/results/bacformer/Summary_Pathway_Fisher_BioAnnotated.csv"
)

df_rel = df_pathway[df_pathway["BioLabel_NRC_RelL2"] == "NRC_bio"].copy()
df_rel = df_rel[df_rel["Significant_Biomarker"] == True].copy()

df_rc = df_pathway[df_pathway["BioLabel_NRC_ClassSpec"] == "NRC_bio"].copy()


df_rel["KEGG_Category"] = df_rel["Pathway"].apply(kegg_L1)
df_rc["KEGG_Category"] = df_rc["Pathway"].apply(kegg_L1)
df_rel["KEGG_Category"] = pd.Categorical(df_rel["KEGG_Category"], categories=KEGG_ORDER, ordered=True)
df_rc["KEGG_Category"] = pd.Categorical(df_rc["KEGG_Category"], categories=KEGG_ORDER, ordered=True)
print(df_rc["KEGG_Category"])

#dot plot relative
topN = 20
df_top = df_rel.reindex(df_rel["IG_Diff"].abs().sort_values(ascending=False).index).head(topN)

fig2=plt.figure(figsize=(18,10))
ax=sns.scatterplot(
    data=df_top,
    x="IG_Diff",
    y=pd.Categorical(df_top["Description"], categories=df_top["Description"], ordered=True),
    size=df_top["Frac_Diff"].abs().round(1),
    hue="KEGG_Category",
    palette=CUSTOM_PALETTE,
    sizes=(100,800),
    alpha=1,
    size_norm=(0, 0.4), 
    legend=True
)

for t in ax.legend_.texts:
    if t.get_text().isdigit() or t.get_text().replace('.', '', 1).isdigit():
        # Convert the displayed positive tick back to the correct negative value
        orig = float(t.get_text())
        t.set_text(f"{-orig:.1f}")
    if "Frac_Diff" in t.get_text():
        t.set_text("Δ Fraction enriched")
    elif "KEGG_Category" in t.get_text():
        t.set_text("KEGG category")

plt.title("Top 20 KEGG Pathways Distinguishing NRC from RC Isolates", pad=12, fontweight="bold")
plt.xlabel("Δ Mean Attribution Mass (RC − NRC)") 
plt.ylabel("KEGG Pathway")
plt.tight_layout()
fig2.savefig(".../results/featureimp_KEGG_nrcrelativel2.pdf", format="pdf", bbox_inches="tight")


# dot plot rc only
topN = 20
df_top_rc = df_rc.sort_values("NRC_Mean_Abs", ascending=False).head(topN)

fig3=plt.figure(figsize=(18,10))
ax=sns.scatterplot(
    data=df_top_rc,
    x="NRC_Mean_Abs",
    y=pd.Categorical(df_top_rc["Description"], categories=df_top_rc["Description"], ordered=True),
    size=df_top_rc["NRC_Frac"].round(1),
    hue="KEGG_Category",
    palette=CUSTOM_PALETTE,
    sizes=(900,1600),
    alpha=1,
    size_norm=(0.7, 1),
    legend=True
)

for t in ax.legend_.texts:
    if "NRC_Frac" in t.get_text():
        t.set_text("Fraction of NRC isolates")
    elif "KEGG_Category" in t.get_text():
        t.set_text("KEGG category")

plt.title("Top 20 KEGG Pathways Driving NRC Classification", pad=12, fontweight="bold")
plt.xlabel("Mean Attribution Mass (NRC)")
plt.ylabel("KEGG Pathway")
plt.tight_layout()
fig3.savefig(".../results/featureimp_KEGG_nrconly.pdf", format="pdf", bbox_inches="tight")
print("Done KEGG")

# OG
# Load both OG tables
df_funcog = pd.read_csv(
    "/media/wolf2/TeamDeJonge/projects/AI4Life/Petra_Matyskova/attention_version/results/bacformer/Summary_OG_Fisher_BioAnnotated.csv"
)

# Filter only biologically meaningful RC-associated OGs
df_rcbio = df_funcog[df_funcog["BioLabel_RC_ClassSpec"] == "RC_bio"].copy()
df_relbio = df_funcog[df_funcog["BioLabel_RC_RelL2"] == "RC_bio"].copy()
df_relbio = df_relbio[df_relbio["Significant_Biomarker"] == True].copy()
df_nrcrelbio = df_funcog[df_funcog["BioLabel_NRC_RelL2"] == "NRC_bio"].copy()
df_nrcrelbio = df_nrcrelbio[df_nrcrelbio["Significant_Biomarker"] == True].copy()
print(df_rcbio.shape, df_relbio.shape)

def og_annotation_class(row, strong_threshold=3, partial_threshold=5):
    Ns = row["strong"]
    Np = row["partial"]
    Nn = row["none"]
    N = Ns + Np + Nn

    if N == 0:
        return "none"

    # 1) Strong OG
    if Ns >= strong_threshold:
        return "strong"

    # 2) Partial OG
    if (0 < Ns < strong_threshold) or (Np >= partial_threshold):
        return "partial"

    # 3) None
    return "none"

df_relbio["Annot_Class"] = df_relbio.apply(og_annotation_class, axis=1)
df_nrcrelbio["Annot_Class"] = df_nrcrelbio.apply(og_annotation_class, axis=1)
df_rcbio["Annot_Class"]  = df_rcbio.apply(og_annotation_class, axis=1)

# rc/nrc only
topN = 20
df_top_rc = df_rcbio.sort_values("RC_Mean_Abs", ascending=False).head(topN)
df_top_rc["RC_Frac_Display"] = df_top_rc["RC_Frac"].round(3)

# # Ensure AnnotTier has a consistent order
order = ["strong", "partial", "none"]
df_top_rc["Annot_Class"] = pd.Categorical(df_top_rc["Annot_Class"], categories=order, ordered=True)

fig13=plt.figure(figsize=(14,10))
ax =sns.scatterplot(
    data=df_top_rc,
    x="RC_Mean_Abs",
    y=pd.Categorical(df_top_rc["OG"], categories=df_top_rc["OG"], ordered=True),
    size="RC_Frac_Display",
    hue="Annot_Class",
    palette=CUSTOM_PALETTE,
    sizes=(200,1500),
    legend=True
)

for t in ax.legend_.texts:
    if "RC_Frac_Display" in t.get_text():
        t.set_text("Fraction of RC isolates")
    elif "Annot_Class" in t.get_text():
        t.set_text("Annotation category")

sns.move_legend(ax, "lower right", bbox_to_anchor=(1, 0), frameon=False)

plt.title("Top 20 Orthogroups Driving RC Classification")
plt.xlabel("Mean Attribution Mass (RC)")
plt.ylabel("Orthogroup (OG)")
plt.tight_layout()
fig13.savefig(".../results/featureimp_OGs_nrconly_orderIG.pdf", format="pdf", bbox_inches="tight")

# relative
topN = 20
df_top = df_relbio.reindex(df_relbio["IG_Diff"].abs().sort_values(ascending=False).index).head(topN).copy()

label_map = {
    "OG0000279": "TonB-dependent SusC-like receptor",
    "OG0002646": "SusD-like glycan-binding outer membrane proteins",
    "OG0000290": "TonB-dependent iron/siderophore receptor",
    "OG0000197": "TonB-dependent heme-like iron receptor",
    "OG0001088": "Tryptophan 7-halogenase",
    "OG0002030": "TonB-dependent outer membrane receptor",
    "OG0000087": "ABC transporter permease",
    "OG0000116": "TRAP transporter permease",
    "OG0000102": "Polyamine ABC transporter permease",
    "OG0000088": "ABC sugar-binding periplasmic proteins",
    "OG0000171": "TRAP transporter solute-binding protein",
    "OG0000872": "RTX-family secreted calcium-binding protein",
    "OG0000048": "ABC nitrate/sulfonate/taurine-family permease",
    "OG0000051": "ABC nitrate/sulfonate/taurine-family ATPase",
    "OG0000019": "GntR-family transcriptional regulator",
    "OG0004336": "SusD-family glycan-binding outer membrane protein",
    "OG0000025": "Two-component sensor histidine kinase",
    "OG0006452": "Rhs-family repeat-associated protein (OG0006452)",
    "OG0012002": "Rhs-family repeat-associated protein (OG0012002)",
    "OG0005757": "Uncharacterized protein family (OG0005757)"
}

palette_sub = {
    "strong": CUSTOM_PALETTE[6],
    "partial": CUSTOM_PALETTE[5],
    "none": CUSTOM_PALETTE[7],
}

# # Ensure AnnotTier has a consistent order
df_top["Annot_Class"] = pd.Categorical(df_top["Annot_Class"], categories=order, ordered=True)

fig12=plt.figure(figsize=(18,10))
ax = sns.scatterplot(
    data=df_top,
    x="IG_Diff",
    y=pd.Categorical(df_top["OG"], categories=df_top["OG"], ordered=True),
    size=df_top["Frac_Diff"].round(1),
    hue="Annot_Class",
    palette=palette_sub,
    sizes=(100,800),
    alpha=1,
    size_norm=(0, 0.4),
    legend=True
)

for t in ax.legend_.texts:
    if "Frac_Diff" in t.get_text():
        t.set_text("Δ Fraction enriched")
    elif "Annot_Class" in t.get_text():
        t.set_text("Annotation category")

ax.set_yticklabels([label_map.get(og, og) for og in df_top["OG"]])

plt.title("Top 20 Orthogroups Distinguishing RC from NRC Isolates", pad=12, fontweight="bold")
plt.xlabel("Δ Mean Attribution Mass (RC − NRC)")
plt.ylabel("Orthogroup (OG)")
plt.tight_layout()
fig12.savefig(".../results/featureimp_OGs_rcrel_orderIG.pdf", format="pdf", bbox_inches="tight")

# nrc relative
df_nrctop = df_nrcrelbio.reindex(df_nrcrelbio["IG_Diff"].abs().sort_values(ascending=False).index).head(topN).copy()

nrclabel_map = {
    "OG0001591": "Ger family (A-subunit) spore germination receptor",
    "OG0004023": "Ag85-family esterase / mycolyltransferase",
    "OG0005133": "Rap-family aspartate phosphatase group",
    "OG0005389": "lg/PKD/PA-family extracellular repeat-rich proteins",
    "OG0004179": "APS/PAPS sulfur metabolism kinase",
    "OG0002492": "Ger family (B-subunit) spore germination receptor",
    "OG0002597": "Ger family (C-subunit) spore germination receptor",
    "OG0000013": "LuxR/REC family response regulators",
    "OG0003237": "LuxR-family transcriptional regulators",
    "OG0000763": "OprD-family outer membrane porins",
    "OG0001382": "ANTAR-family RNA-binding antiterminator proteins",
    "OG0006915": "Uncharacterized protein family (OG0006915)",
    "OG0007049": "Uncharacterized protein family (OG0007049)",
    "OG0009232": "XRE-family transcriptional regulators",
    "OG0003401": "Extracellular repeat-rich FN3/Ig-like proteins",
    "OG0004750": "Extracellular YD-repeat β-solenoid surface proteins",
    "OG0002153": "Uncharacterized protein family (OG0002153)",
    "OG0008946": "Extracellular α/β-hydrolase–like proteins",
    "OG0003107": "SpoIIAA-like domain proteins",
    "OG0004305": "CpaB/RcpC-family Flp/Tad pilus assembly protein"
}

palette_sub = {
    "strong": CUSTOM_PALETTE[4],
    "partial": CUSTOM_PALETTE[6],
    "none": CUSTOM_PALETTE[7],
}

# # Ensure AnnotTier has a consistent order
df_nrctop["Annot_Class"] = pd.Categorical(df_nrctop["Annot_Class"], categories=order, ordered=True)

fig15=plt.figure(figsize=(18,10))
ax2 = sns.scatterplot(
    data=df_nrctop,
    x="IG_Diff",
    y=pd.Categorical(df_nrctop["OG"], categories=df_nrctop["OG"], ordered=True),
    size=df_nrctop["Frac_Diff"].abs().round(1),
    hue="Annot_Class",
    palette=palette_sub,
    sizes=(100,800),
    alpha=1,
    size_norm=(0, 0.4),
    legend=True
)

for t in ax2.legend_.texts:
    if t.get_text().isdigit() or t.get_text().replace('.', '', 1).isdigit():
        # Convert the displayed positive tick back to the correct negative value
        orig = float(t.get_text())
        t.set_text(f"{-orig:.1f}")
    if "Frac_Diff" in t.get_text():
        t.set_text("Δ Fraction enriched")
    elif "Annot_Class" in t.get_text():
        t.set_text("Annotation category")

sns.move_legend(ax2, "lower left", frameon=False)
ax2.set_yticklabels([nrclabel_map.get(og, og) for og in df_nrctop["OG"]])

plt.title("Top 20 Orthogroups Distinguishing NRC from RC Isolates", pad=12, fontweight="bold")
plt.xlabel("Δ Mean Attribution Mass (RC − NRC)")
plt.ylabel("Orthogroup (OG)")
plt.tight_layout()
fig15.savefig(".../results/featureimp_OGs_nrcrel_orderIG.pdf", format="pdf", bbox_inches="tight")

print("Done")