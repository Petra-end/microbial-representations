import os
import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
import optuna
from torch.optim import Adam
import sys, importlib

if "MLP_classif" in sys.modules:
    importlib.reload(sys.modules["MLP_classif"])
    
from MLP_classif import MLPClassifier
from data_loader import load_data_and_targets, create_dataloader, label_hard_taxonomies
from optuna_training import objective
from optuna.samplers import TPESampler
from training import trainf_model
import json

if __name__ == "__main__":
    # DEFINE PARAMETERS
    input_size = 768 #dna: 768, protein: 320
    hidden_size = [256, 128, 32]
    output_size = 1
    batch_size = 6
    bucket_size = 6
    epochs = 100
    loss_type="classic_binary"
    use_attention=False
    learning_rate = 1e-4
    warm_up=5
    early_stopping_patience=10
    early_stopping_delta=0.005
    cv_outer_folds = 3
    cv_inner_folds = 3
    optuna_n_trials = 60
    optuna_startup = 10
    optuna_epoch_override = 60
    optuna_pruner_threshold = 0.7
    optuna_patience = 15
    optuna_early_stopping_delta = 1e-5
    limit_samples = None
    use_mixup = False #data augmentation with alpha 0.2 (in utils)
    use_class_weights = False #handle class imbalance
    protein=False #True of protein embeddings
    
    # Define optimizer
    optimizer_class = Adam
    optimizer_params = {'lr': learning_rate}
    
    num_cpus = 16  #on a server: int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
    torch.set_num_threads(num_cpus)
    CUDA = torch.cuda.is_available()
    if CUDA:
        device = "cuda"

    # PATHS
    # Paths to save model and predictions
    save_dir = ".../dnabert/768sc_GT0_3l"

    # Path to data folders
    folder=".../DNA_embeddings/embeddings_dnabert" 
    # or folder=".../protein_embeddings/embeddings_esm2"
    label_file=".../data/GT_SSC_s1.txt"
    meta_file=".../data/SSC_taxonomy_GTDB.tsv"

    # SPLIT TAXONOMIES
    # function only ran once to save the splits
    splits = label_hard_taxonomies(
        meta_file=meta_file,
        label_file=label_file,
        hard_classes=['Bacilli', 'Bacteroidia', 'Acidobacteriae'],
        output_file=".../data/hard_easy_tax_split.tsv",
        stratify_col='class')
    
    # path to the saved splits
    split_annotation_file = ".../data/hard_easy_tax_split.tsv"

    # LOAD DATA
    # Load only 'easy' samples — for cross-validation
    X_easy, y_easy, IDs_easy = load_data_and_targets(
        folder_path=folder,
        targets_file=label_file,
        meta_split_file=split_annotation_file,
        split_type='easy',
        limit = limit_samples,
        protein = protein
    )
    
    def standardize_protein(x):
        x_arr = np.array(x)
        if x_arr.ndim == 1:
            x_arr = x_arr[np.newaxis, :]
        return (x_arr - x_arr.mean(axis=1, keepdims=True)) / (x_arr.std(axis=1, keepdims=True) + 1e-8)

    # X_easy_std = Parallel(n_jobs=num_cpus)(delayed(standardize_protein)(x) for x in X_easy)
    
    X_easy = np.array(X_easy, dtype=object)
    y_easy = np.array(y_easy)
    IDs_easy = np.array(IDs_easy)
    
    #class weights
    if use_class_weights:
        num_pos = int((y_easy == 1).sum())
        num_neg = int((y_easy == 0).sum())
        pos_weight = num_neg / num_pos

        # Convert to a tensor for PyTorch
        class_weights = torch.tensor([pos_weight], dtype=torch.float).to(device)
        print("Positive Class Weights:", class_weights, type(class_weights))
    else:
        class_weights = None

    # Load only 'hard' samples — for OOD evaluation
    X_hard, y_hard, IDs_hard = load_data_and_targets(
        folder_path=folder,
        targets_file=label_file,
        meta_split_file=split_annotation_file,
        split_type='hard',
        protein=protein
    )
    
    # OUTER CV LOOP
    # SPLIT to test vs trainval
    kfold_outer = KFold(n_splits=cv_outer_folds, shuffle=True, random_state=42)
    for fold, (trainval_idx, test_idx) in enumerate(kfold_outer.split(X_easy)):
        print(f"\nFold {fold+1}:")
        
        X_trainval, X_test = X_easy[trainval_idx], X_easy[test_idx]
        y_trainval, y_test = y_easy[trainval_idx], y_easy[test_idx]
        IDs_trainval, IDs_test = IDs_easy[trainval_idx], IDs_easy[test_idx]

        trainval_loader = create_dataloader(X_trainval, y_trainval, IDs_trainval, batch_size, bucket_size, num_workers=num_cpus)
        test_loader = create_dataloader(X_test, y_test, IDs_test, batch_size, bucket_size, num_workers=num_cpus)
    
        # INNER CV LOOP (runs inside optuna each trial is run on x fold)
        # SPLIT to tran vs val
        
        # RUN OPTUNA
        # Define hyperparameter ranges
        dropout_range = (0.1, 0.4)
        weight_decay_range = (1e-6, 1e-3) 
        l1_lambda_range = (1e-6, 1e-3) 
        
        # compute folds: https://stackoverflow.com/questions/63224426/how-can-i-cross-validate-by-pytorch-and-optuna
        # start loop across folds
        #load fold data
        def wrapped_objective(trial):
            kfold_inner = KFold(n_splits=cv_inner_folds, shuffle=True, random_state=0)
            scores = []
            
            for fold_idx, (train_idx, valid_idx) in enumerate(kfold_inner.split(range(len(X_trainval)))):
                X_train, X_val = X_trainval[train_idx], X_trainval[valid_idx]
                y_train, y_val = y_trainval[train_idx], y_trainval[valid_idx]
                IDs_train, IDs_val = IDs_trainval[train_idx], IDs_trainval[valid_idx]
                
                train_loader = create_dataloader(X_train, y_train, IDs_train, batch_size, bucket_size, num_workers=num_cpus)
                val_loader = create_dataloader(X_val, y_val, IDs_val, batch_size, bucket_size, num_workers=num_cpus)
            
                score = objective(
                    trial=trial,
                    model_class=MLPClassifier,
                    input_size=input_size,
                    hidden_sizes=hidden_size,
                    output_size=output_size,
                    loss_type=loss_type,
                    use_attention=use_attention,
                    optimizer_class=optimizer_class,
                    optimizer_params=optimizer_params,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    warm_up=warm_up,
                    epochs=optuna_epoch_override,
                    device=device,
                    dropout_range=dropout_range,
                    weight_decay_range=weight_decay_range,
                    l1_lambda_range=l1_lambda_range,
                    patience=optuna_patience,
                    early_stopping_delta=optuna_early_stopping_delta,
                    class_weights=class_weights,
                    use_mixup=use_mixup
                )
                
                scores.append(score)
            
            return np.mean(scores) #return mean score
        
        study = optuna.create_study(direction="minimize",
                                    sampler=TPESampler(n_startup_trials=optuna_startup,seed=62),
                                    pruner=optuna.pruners.ThresholdPruner(upper=optuna_pruner_threshold, n_warmup_steps=20))
        study.optimize(wrapped_objective, n_trials=optuna_n_trials)
        print("Best trial:")
        print(study.best_trial)
        
        best_params = study.best_params

        weight_decay = best_params["weight_decay"] 
        dropout_rate = best_params["dropout_rate"] 
        l1_lambda = best_params["l1_lambda"] 
        batch_norm = best_params["batch_norm"] 
        
        # REDEFINE MODEL
        model = MLPClassifier(input_size, hidden_size, output_size, dropout_rate, loss_type="classic_binary", use_attention=use_attention, class_weights=class_weights)
        model = model.to(device)
        
        optimizer = optimizer_class(
            model.parameters(),
            weight_decay=weight_decay,
            **optimizer_params)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1, threshold=0.001)
        
        # TRAIN FINAL MLP ON TRAIN+VAL DATA
        trainf_model(model, optimizer, trainval_loader, test_loader, epochs, save_dir, warm_up=warm_up, device=device, scheduler=scheduler, l1_lambda=l1_lambda, early_stopping_patience=early_stopping_patience, early_stopping_delta=early_stopping_delta, cfold=fold, use_mixup=use_mixup)

        config = {
            "fold": fold,
            "optuna_best_params": best_params,  # from Optuna
            "fixed_params": {
                "input_size": input_size,
                "hidden_size": hidden_size,
                "output_size": output_size,
                "batch_size": batch_size,
                "bucket_size": bucket_size,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "loss_type": loss_type,
                "use_attention": use_attention,
                "optimizer_class": optimizer_class.__name__,
                "warm_up": warm_up,
                "early_stopping_patience": early_stopping_patience,
                "early_stopping_delta": early_stopping_delta,
                "cv_outer_folds": cv_outer_folds,
                "cv_inner_folds": cv_inner_folds,
                "device": device,
                "use_mixup": use_mixup,
                "class_weights": class_weights.item() if class_weights is not None else None
            }
        }

        # Save as JSON
        config_path = os.path.join(save_dir, f"fold_{fold}_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

        print(f"Saved config for fold {fold} → {config_path}")