import pandas as pd
import numpy as np
import torch
import ast
import re
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

def extract_first_id(s):
    s = str(s)
    # Look for pattern like ['XXXX'
    m = re.search(r"\['([^']+)'", s)
    if m:
        return m.group(1)
    else:
        return None

# List your test CSV files (one per fold)
test_files = [
    ".../results/kos/binary_filt0_GT0_3l/predictions_0_external_durans1.csv",
    ".../results/kos/binary_filt0_GT0_3l/predictions_1_external_durans1.csv",
    ".../results/kos/binary_filt0_GT0_3l/predictions_2_external_durans1.csv",
]

annot_extended = False

def evaluate_subset(df):
    # Convert model output to probabilities and predictions
    predictions_tensor = torch.tensor(df["Prediction_Class_0"].values, dtype=torch.float32)
    df["Prediction_Prob"] = torch.sigmoid(predictions_tensor).numpy()
    df["Prediction"] = np.where(df["Prediction_Prob"] > 0.5, 1, 0)

    y_true = df["Ground_Truth"]
    y_pred = df["Prediction"]
    y_prob = df["Prediction_Prob"]

    # Compute metrics
    acc = np.mean(y_pred == y_true)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    return acc, prec, rec, f1, auc, pr_auc

metrics = []

for i, file in enumerate(test_files, start=1):
    df = pd.read_csv(file)
    if annot_extended:
        df["Sample_ID"] = df["Sample_ID"].apply(extract_first_id)
    acc, prec, rec, f1, auc, pr_auc = evaluate_subset(df)
    print(f"\nFold {i} results:")
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f} | PR-AUC: {pr_auc:.4f}")
    metrics.append([acc, prec, rec, f1, auc, pr_auc])

# Aggregate across folds
metrics_df = pd.DataFrame(metrics, columns=["Accuracy", "Precision", "Recall", "F1", "AUC", "PR_AUC"])
print("\n==== Cross-validation results (mean ± SD) ====")
print(metrics_df.describe().loc[["mean", "std"]].T)

all_results = pd.concat([pd.read_csv(f) for f in test_files], ignore_index=True)
all_results["Prediction_Prob"] = torch.sigmoid(
    torch.tensor(all_results["Prediction_Class_0"].values, dtype=torch.float32)
).numpy()
all_results["Prediction"] = np.where(all_results["Prediction_Prob"] > 0.5, 1, 0)

print("\nOverall (combined folds) classification report:")
print(classification_report(all_results["Ground_Truth"], all_results["Prediction"]))