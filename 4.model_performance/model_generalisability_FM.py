import os
import numpy as np
import torch
from torch.autograd import Variable
import sys, importlib

if "MLP_classif" in sys.modules:
    importlib.reload(sys.modules["MLP_classif"])
    
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MLP_classif import MLPClassifier
from data_loader import create_dataloader, load_data_and_targets
from utils import save_predictions_to_csv 

def load_and_validate_model(model_dir, config_file, X, y, IDs, save_dir, phase, cfold, device="cpu", num_cpus=1):
    """
    Load a saved model, perform validation on external data, and save predictions.
    """
    import json
    with open(config_file, "r") as f:
        config = json.load(f)
    params = config["fixed_params"]
    best_params = config["optuna_best_params"]

    model = MLPClassifier(
        input_size=params["input_size"],
        hidden_sizes=params["hidden_size"],
        output_size=params["output_size"],
        dropout_rate=best_params["dropout_rate"],
        loss_type=params["loss_type"],
        use_attention=False,  #model does not use attention
    )
    model = model.to(device)

    checkpoint = os.path.join(model_dir, f"model_{cfold}.pth")
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    
    test_loader = create_dataloader(X, y, IDs, params["batch_size"], params["bucket_size"], num_workers=num_cpus)
    
    all_predictions = []
    all_targets = []
    all_ids = []
    
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(test_loader):
            if len(batch) == 3:
                data, target, ids = batch
            else:
                raise ValueError("Unexpected batch structure.")

            data, target = Variable(data), Variable(target.float())
            target = target.view(-1) #not before
            data, target = data.to(device), target.to(device)
            
            if model.use_attention:
                output, attn_weights = model(data)
            else:
                attn_weights = None
                output = model(data)

            all_predictions.append(output.cpu().detach())
            all_targets.append(target.cpu())
            all_ids.append(ids)
            
    save_predictions_to_csv(torch.cat(all_predictions), torch.cat(all_targets), all_ids, save_dir, phase=phase, cfold=cfold)

# Run script
num_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
print(f"Using {num_cpus} CPUs")
torch.set_num_threads(num_cpus)
CUDA = torch.cuda.is_available()
if CUDA:
    device = "cuda"

phase = "OOD_testset"    
phase = "external_durans1"
cfold = 1
protein = False

model_dir = ".../results/bacformer/480_GT0_3l"
config_file = os.path.join(model_dir, f"fold_{cfold}_config.json")

embeddings_folder = ".../embeddings/SSC/bacformer"
embeddings_folder = ".../embeddings/metaanalysis/bacformer"

label_file = ".../data/GT_SSC_s1.txt"
label_file = ".../data/GT_Duran_s1.txt"
split_file = ".../data/hard_easy_tax_split.tsv"

#EMBEDDINGS
# OOD TEST SET
X_ood, y_ood, IDs_ood = load_data_and_targets(
    folder_path=embeddings_folder,
    targets_file=label_file,
    meta_split_file=split_file,
    split_type='hard',
    protein=protein
)

# EXTERNAL DATA
X_external, y_external, IDs_external = load_data_and_targets(
    folder_path=embeddings_folder,
    targets_file=label_file,
    protein=protein
)

X = np.array(X_external, dtype=object)
y = np.array(y_external)
IDs = np.array(IDs_external)

load_and_validate_model(model_dir, config_file, X, y, IDs, save_dir=model_dir, phase=phase, cfold=cfold, device=device, num_cpus=num_cpus)