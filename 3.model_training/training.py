import torch
from torch.autograd import Variable
from utils import save_predictions_to_csv, save_model, mixup_data
import matplotlib.pyplot as plt
import numpy as np
import os

def train_val_test(model, optimizer, loader, warm_up, epoch, max_epochs, device="cpu", phase="train", l1_lambda=0.0, use_mixup=False):
    device = torch.device(device)
    model = model.to(device)
    
    all_predictions = []
    all_targets = []
    all_ids = []
    loss = 0
    phase_loss = 0

    if phase == "train":
        model.train()
        for _, batch in enumerate(loader):
            if len(batch) == 4:  # This means the batch includes mask
                data, mask, target, ids = batch
                mask = mask.to(device)
            elif len(batch) == 3:  # No mask in this batch
                data, target, ids = batch
                mask = None  # Set mask to None if not present
            else:
                raise ValueError("Unexpected batch structure.")

            data, target = Variable(data), Variable(target.float())
            target = target.view(-1)
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            if use_mixup:
                data, targets_a, targets_b, lam = mixup_data(data, target, alpha=0.2)
            
            if model.use_attention:
                if mask is not None:
                    output, attn_weights = model(data, mask)
                else:
                    output, attn_weights = model(data)
                
                batch_attn_var = attn_weights.var(dim=1).mean().item()
            else:
                attn_weights = None
                if mask is not None:
                    output = model(data, mask)
                else:
                    output = model(data)
            
            if use_mixup:
                loss_a = model.loss_function(output, targets_a, warm_up, epoch, max_epochs, l1_lambda, attn_weights)
                loss_b = model.loss_function(output, targets_b, warm_up, epoch, max_epochs, l1_lambda, attn_weights)
                loss = lam * loss_a + (1 - lam) * loss_b
            else: 
                loss = model.loss_function(output, target, warm_up, epoch, max_epochs, l1_lambda, attn_weights)

            loss.backward()
            optimizer.step()
            phase_loss += loss.item() * data.size(0)

            all_predictions.append(output.cpu().detach())
            all_targets.append(target.cpu())
            all_ids.append(ids)

    else:
        model.eval()
        with torch.no_grad():
            for _, batch in enumerate(loader):
                if len(batch) == 4:  # This means the batch includes mask
                    data, mask, target, ids = batch
                    mask = mask.to(device)
                elif len(batch) == 3:  # No mask in this batch
                    data, target, ids = batch
                    mask = None  # Set mask to None if not present
                else:
                    raise ValueError("Unexpected batch structure.")

                data, target = Variable(data), Variable(target.float())
                target = target.view(-1)
                data, target = data.to(device), target.to(device)

                # Pass the mask if it exists, otherwise pass data alone
                if model.use_attention:
                    if mask is not None:
                        output, attn_weights = model(data, mask)
                    else:
                        output, attn_weights = model(data)
                else:
                    if mask is not None:
                        output = model(data, mask)
                        attn_weights = None
                    else:
                        output = model(data)
                        attn_weights = None
                    
                loss = model.loss_function(output, target, warm_up, epoch, max_epochs, l1_lambda, attn_weights)
                phase_loss += loss.item() * data.size(0)

                all_predictions.append(output.cpu().detach())
                all_targets.append(target.cpu())
                all_ids.append(ids)
    
    phase_loss = phase_loss / len(loader.dataset)
    return phase_loss, all_predictions, all_targets, all_ids

def traintest_model(model, optimizer, train_loader, val_loader, test_loader, epochs, save_dir, warm_up=5, device = "cpu", scheduler = None, l1_lambda=0.0, early_stopping_patience=5, early_stopping_delta=0.005, use_mixup=False):
    train_losses = []  # List to store training losses
    val_losses = []   # List to store validation losses

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(1, epochs + 1):
        # Training phase
        train_loss, all_train_predictions, all_train_targets, all_train_ids = train_val_test(model, optimizer, train_loader, warm_up, epoch=epoch, max_epochs=epochs, device=device, phase="train", l1_lambda=l1_lambda, use_mixup=use_mixup)
        
        train_losses.append(train_loss)
        print(f'====> Epoch: {epoch} Average train loss: {train_loss}')
        
        first_preds = torch.cat(all_train_predictions[:5], dim=0)
        print("Predictions (logits):", first_preds.view(-1).detach().cpu().numpy())
        first_targets = torch.cat(all_train_targets[:5], dim=0)
        print("Labels:", first_targets.view(-1).detach().cpu().numpy())

        # Validation phase
        val_loss, all_val_predictions, all_val_targets, all_val_ids = train_val_test(model, optimizer, val_loader, warm_up, epoch=epoch, max_epochs=epochs, device=device, phase="validation", l1_lambda=l1_lambda)

        val_losses.append(val_loss)
        print(f'====> Epoch: {epoch} Average validation loss: {val_loss}')

        if scheduler is not None:
                scheduler.step(val_loss)
        
        # Early stopping logic:
        if val_loss < best_val_loss - early_stopping_delta:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

        if epoch == epochs:  # If last epoch and didn't early stop
            best_model_state = model.state_dict()

    # Load best model weights before test
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Save train and validation predictions and targets
    save_predictions_to_csv(torch.cat(all_train_predictions), torch.cat(all_train_targets), all_train_ids, save_dir, epoch, phase='train')
    save_predictions_to_csv(torch.cat(all_val_predictions), torch.cat(all_val_targets), all_val_ids, save_dir, epoch, phase='validation')
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', marker='o')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', marker='o')
    plt.title('Training and Validation Loss per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig(f"{save_dir}/loss_plot_{epochs}.png")

    # Testing phase
    _, all_test_predictions, all_test_targets, all_test_ids = train_val_test(model, optimizer, test_loader, warm_up, epoch=epoch, max_epochs=epochs, device=device, phase="test", l1_lambda=0.0)

    # Save test predictions and targets
    save_predictions_to_csv(torch.cat(all_test_predictions), torch.cat(all_test_targets), all_test_ids, save_dir, phase='test')

def trainf_model(model, optimizer, trainval_loader, test_loader, epochs, save_dir, warm_up=5, device = "cpu", scheduler=None, l1_lambda=0.0, early_stopping_patience=15, early_stopping_delta=0.005, cfold=None, use_mixup=False):
    trainval_losses = []  # List to store training losses
    test_losses = []   # List to store validation losses

    best_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(1, epochs + 1):
        # Training phase
        trainval_loss, all_trainval_predictions, all_trainval_targets, all_trainval_ids = train_val_test(model, optimizer, trainval_loader, warm_up, epoch=epoch, max_epochs=epochs, device=device, phase="train", l1_lambda=l1_lambda, use_mixup=use_mixup)
        
        trainval_losses.append(trainval_loss)
        print(f'====> Epoch: {epoch} Average train loss: {trainval_loss}')

        # Testing phase
        test_loss, all_test_predictions, all_test_targets, all_test_ids = train_val_test(model, optimizer, test_loader, warm_up, epoch=epoch, max_epochs=epochs, device=device, phase="test", l1_lambda=l1_lambda)

        test_losses.append(test_loss)
        print(f'====> Epoch: {epoch} Average test loss: {test_loss}')
        
        if scheduler is not None:
                scheduler.step(test_loss)

        # Early stopping logic:
        if test_loss < best_loss - early_stopping_delta:
            best_loss = test_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

        if epoch == epochs:  # If last epoch and didn't early stop
            best_model_state = model.state_dict()

    # Load best model weights before test
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Save train and validation predictions and targets
    save_predictions_to_csv(torch.cat(all_trainval_predictions), torch.cat(all_trainval_targets), all_trainval_ids, save_dir, epoch, phase='final_train', cfold=cfold)
    save_predictions_to_csv(torch.cat(all_test_predictions), torch.cat(all_test_targets), all_test_ids, save_dir, epoch, phase='final_test', cfold=cfold)
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(trainval_losses) + 1), trainval_losses, label='Training Loss', marker='o')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig(f"{save_dir}/loss_plot_{cfold}_{epochs}.png")

    # Save the trained MLP model
    save_model(model, save_dir, cfold=cfold)
    
    #save the query vector
    if model.use_attention:
        query_vec = model.attention_layer.query.detach().cpu()
        query_path = os.path.join(save_dir, "query_vector.npy")
        np.save(query_path, query_vec.numpy())