import optuna
from training import train_val_test
import torch

def trainval_optuna(model, optimizer, train_loader, val_loader, warm_up, epochs, device, patience=15, early_stopping_delta=0.005, l1_lambda=0.0, trial=None, use_mixup=False):
    val_losses = []
    best_loss = float('inf')
    epochs_no_improve = 0
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.3, threshold=0.01
    )

    for epoch in range(1, epochs + 1): 
        # Training phase
        _, _, _, _ = train_val_test(model, optimizer, train_loader, warm_up=warm_up, epoch=epoch, max_epochs=epochs, device=device, phase="train", l1_lambda=l1_lambda, use_mixup=use_mixup)

        # Validation phase
        val_loss, _, _, _ = train_val_test(model, optimizer, val_loader, warm_up=warm_up, epoch=epoch, max_epochs=epochs, device=device, phase="validation", l1_lambda=l1_lambda)

        val_losses.append(val_loss)
        scheduler.step(val_loss) #added recently

        if val_loss < best_loss - early_stopping_delta:
            best_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if trial is not None:
            # Report intermediate validation loss to Optuna
            trial.report(val_loss, step=epoch)

            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    return min(val_losses), val_losses.index(min(val_losses)) + 1 #before last val_loss: val_losses[-1]

def objective(trial, model_class, input_size, hidden_sizes, output_size, loss_type,
              optimizer_class, optimizer_params, train_loader, val_loader, warm_up, epochs, device,
              dropout_range=(0.1, 0.7), weight_decay_range=(1e-8, 1e-2),
              l1_lambda_range=(1e-8, 1e-2), patience=15, early_stopping_delta=0.005, class_weights=None, use_mixup=False):
    
    # Define hyperparameter search space
    dropout_rate = trial.suggest_float("dropout_rate", dropout_range[0], dropout_range[1])
    weight_decay = trial.suggest_float("weight_decay", weight_decay_range[0], weight_decay_range[1], log=True)
    l1_lambda = trial.suggest_float("l1_lambda", l1_lambda_range[0], l1_lambda_range[1], log=True)
    batch_norm = trial.suggest_categorical("batch_norm", [True, False])
    
    # Update model with trial-specific parameters
    model = model_class(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        dropout_rate=dropout_rate,
        use_batchnorm=batch_norm,
        loss_type=loss_type,
        class_weights=class_weights
    ).to(device)

    # Create optimizer with trial-specific parameters
    optimizer = optimizer_class(model.parameters(), weight_decay=weight_decay, **optimizer_params)
    
    # Call the traintest_mlp_model function with these hyperparameters
    val_loss, best_epoch = trainval_optuna(
        model = model,
        optimizer = optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        warm_up=warm_up,
        epochs=epochs,
        device=device,
        patience=patience,
        early_stopping_delta=early_stopping_delta,
        l1_lambda=l1_lambda,
        trial=trial,
        use_mixup=use_mixup
    )

    return val_loss, best_epoch