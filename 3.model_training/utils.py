import numpy as np
import pandas as pd
import os
import torch
from torch.autograd import Variable

def extract_latent(model, data_loader, CUDA=False):
    model.eval()
    latent_space = []

    with torch.no_grad():
        for data, _ in data_loader:
            data = Variable(data)
            if CUDA:
                data = data.cuda()
            mu, _ = model.encode(data.view(-1, 9696))
            latent_space.append(mu.cpu().numpy())

    return np.concatenate(latent_space, axis=0)

def save_model(model, save_dir, cfold=None):
    file_path = os.path.join(save_dir, f'model_{cfold}.pth')
    torch.save(model.state_dict(), file_path)

def load_model(model, path, input_size, zdims, layer_sizes):
    model = model(input_size, zdims, layer_sizes)
    model.load_state_dict(torch.load(path))
    return model

def save_predictions_to_csv(predictions, targets, ids, save_dir, epoch=None, phase='train', cfold=None):
    num_classes = predictions.size(1)
    prediction_columns = [f'Prediction_Class_{i}' for i in range(num_classes)]
    target_columns = ['Ground_Truth']
    df_predictions = pd.DataFrame(predictions.numpy(), columns=prediction_columns)
    df_targets = pd.DataFrame(targets.numpy(), columns=target_columns)
    df_ids = pd.DataFrame({"Sample_ID": [item for sublist in ids for item in sublist]})
    df = pd.concat([df_predictions, df_targets, df_ids], axis=1)

    if epoch == None:
        file_path = os.path.join(save_dir, f'predictions_{cfold}_{phase}.csv')
    else:
        file_path = os.path.join(save_dir, f'predictions_epoch_{cfold}_{epoch}_{phase}.csv')
    df.to_csv(file_path, index=False)

def mixup_data(x, y, alpha=0.2):
    """Returns mixed inputs, pairs of targets, and lambda."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam