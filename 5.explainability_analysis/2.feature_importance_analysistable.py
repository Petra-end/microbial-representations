import pandas as pd
import numpy as np
import ast
import torch
import glob
import os
import re

#------------ FUNCTIONS -------------
def extract_first_id(s):
    s = str(s)
    # Look for pattern like ['XXXX'
    m = re.search(r"\['([^']+)'", s)
    if m:
        return m.group(1)
    else:
        return None

def load_pred_file(path, dataset_label, annot_extended):
    df = pd.read_csv(path)
    if annot_extended:
        df["Isolate"] = df["Sample_ID"].apply(extract_first_id)
    else:
        df["Isolate"] = df["Sample_ID"]

    # convert logits to probability + predicted class
    logits = torch.tensor(df["Prediction_Class_0"].values, dtype=torch.float32)
    df["Prediction_Prob"] = torch.sigmoid(logits).numpy()
    df["Prediction_Label"] = (df["Prediction_Prob"] > 0.5).astype(int)

    df["Ground_Truth"] = df["Ground_Truth"].astype(int)
    df["Correct"] = df["Prediction_Label"] == df["Ground_Truth"]
    df["Dataset"] = dataset_label
    return df[["Isolate","Ground_Truth","Prediction_Prob","Prediction_Label","Correct","Dataset"]]

#------------ IMPORTS -------------
easy_train = load_pred_file(
    ".../results/bacformer/bacformer_480_GT0_3l/predictions_epoch_2_58_final_train.csv",
    dataset_label="Easy_Train",
    annot_extended=True,
)
easy_test = load_pred_file(
    ".../results/bacformer/bacformer_480_GT0_3l/predictions_epoch_2_58_final_test.csv",
    dataset_label="Easy_Test",
    annot_extended=True,
)
hard = load_pred_file(
    ".../results/bacformer/bacformer_480_GT0_3l/predictions_2_external_hard.csv",
    dataset_label="Hard",
    annot_extended=False,
)

ig_folder = ".../results/bacformer/feature_importance_bf"  # your IG folder
new_folder = ".../results/bacformer/feature_importance_withpred/"  # your IG folder)
os.makedirs(new_folder, exist_ok=True)

#----------ANALYSIS----------

pred_master = pd.concat([easy_train, easy_test, hard], ignore_index=True)
pred_master = pred_master.drop_duplicates(subset="Isolate")
pred = pred_master.set_index("Isolate")
pred_master.to_csv("all_isolates_predictions.csv", index=False)

for path in glob.glob(os.path.join(ig_folder, "*_protein_ig.csv")):
    iso = os.path.basename(path).replace("_protein_ig.csv", "")
    if iso not in pred.index:
        print(f"No prediction for {iso}, skipping")
        continue

    meta = pred.loc[iso]  # Ground_Truth, Prediction_Label, Correct, Dataset, etc.
    df = pd.read_csv(path)

    df["Pred_Label"] = meta["Prediction_Label"]
    df["Correct"] = meta["Correct"]
    df["Dataset_Pred"] = meta["Dataset"]
    
    #Normalise and sort
    df["NormAbs"] = df["IG_MeanAbs"] / df["IG_MeanAbs"].sum()
    df = df.sort_values("NormAbs", ascending=False)
    df["CumAbs"] = df["NormAbs"].cumsum()

    out_path = os.path.join(new_folder, f"{iso}_protein_ig_withpred.csv")
    df.to_csv(out_path, index=False)
