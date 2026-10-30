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

            if mask is not None:
                output = model(data, mask)
            else:
                output = model(data)
            
            if use_mixup:
                loss_a = model.loss_function(output, targets_a, warm_up, epoch, max_epochs, l1_lambda)
                loss_b = model.loss_function(output, targets_b, warm_up, epoch, max_epochs, l1_lambda)
                loss = lam * loss_a + (1 - lam) * loss_b
            else: 
                loss = model.loss_function(output, target, warm_up, epoch, max_epochs, l1_lambda)

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
                target = target.view(-1) #not before
                data, target = data.to(device), target.to(device)

                # Pass the mask if it exists, otherwise pass data alone
                if mask is not None:
                    output = model(data, mask)
                else:
                    output = model(data)
                    
                loss = model.loss_function(output, target, warm_up, epoch, max_epochs, l1_lambda)
                phase_loss += loss.item() * data.size(0)

                all_predictions.append(output.cpu().detach())
                all_targets.append(target.cpu())
                all_ids.append(ids)
    
    phase_loss = phase_loss / len(loader.dataset)
    return phase_loss, all_predictions, all_targets, all_ids


def trainf_model(model, optimizer, trainval_loader, test_loader, epochs, save_dir, warm_up=5, device = "cpu", l1_lambda=0.0, cfold=None, use_mixup=False):
    trainval_losses = []  # List to store training losses

    best_model_state = None

    for epoch in range(1, epochs + 1):
        # Training phase
        trainval_loss, all_trainval_predictions, all_trainval_targets, all_trainval_ids = train_val_test(model, optimizer, trainval_loader, warm_up, epoch=epoch, max_epochs=epochs, device=device, phase="train", l1_lambda=l1_lambda)
        
        trainval_losses.append(trainval_loss)
        print(f'====> Epoch: {epoch} Average train loss: {trainval_loss}')
        
        best_model_state = model.state_dict()

    # Load best model weights before test
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    _, all_trainval_predictions, all_trainval_targets, all_trainval_ids = train_val_test(
        model, optimizer, trainval_loader, 
        warm_up,epoch=epochs, max_epochs=epochs,
        device=device, phase="test", l1_lambda=0.0)
    
    test_loss, all_test_predictions, all_test_targets, all_test_ids = train_val_test(
        model, optimizer, test_loader, warm_up,
        epoch=epochs, max_epochs=epochs,
        device=device, phase="test", l1_lambda=l1_lambda)
    
    print(f'====> Final test loss: {test_loss}')

    # Save train and validation predictions and targets
    save_predictions_to_csv(torch.cat(all_trainval_predictions), torch.cat(all_trainval_targets), all_trainval_ids, save_dir, epoch, phase='final_train', cfold=cfold)
    save_predictions_to_csv(torch.cat(all_test_predictions), torch.cat(all_test_targets), all_test_ids, save_dir, epoch, phase='final_test', cfold=cfold)
    
    # Save learning curve
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