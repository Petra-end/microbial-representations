### ---------- !!! CATION PRE RUNNING !!! ----------
# Whe using the data_loader in the feature analysis,
# some lines in the load_data_and_targets function
# need to be adjusted.
#
# Further instructions bellow.
### ------------------------------------------------

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.utils.data.sampler import BatchSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import math
import random
import pandas as pd
import h5py

class ProteinDataset(Dataset):
    def __init__(self, isolates, targets, ids):
        self.isolates = isolates
        self.targets = targets
        self.ids = list(ids)

    def __len__(self):
        return len(self.isolates)
    
    def __getitem__(self, idx):
        return self.isolates[idx], self.targets[idx], self.ids[idx]

def load_data_and_targets(folder_path, targets_file, meta_split_file=None, 
                          split_type=None, limit=None, protein=False):
    data = []
    labels = []
    ids = []

    # Load label dict
    target_dict = {}
    with open(targets_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            isolate_name = parts[0]
            label = int(float(parts[1]))
            target_dict[isolate_name] = label

    # Load split info
    id_filter = None
    if meta_split_file and split_type:
        split_df = pd.read_csv(meta_split_file, sep="\t")
        id_filter = set(split_df[split_df['split'] == split_type]['isolate'])

    # Load HDF5 embeddings
    files = [f for f in os.listdir(folder_path) if f.endswith('.h5')]
    for file in files:
        if protein:
            isolate_id = file.replace("_embeddings.h5", "")
        else: 
            isolate_id = file.replace(".h5", "")
            
        if isolate_id not in target_dict:
            continue
        if id_filter and isolate_id not in id_filter:
            continue

        filepath = os.path.join(folder_path, file)
        embeddings = []
        protein_ids = []

        with h5py.File(filepath, 'r') as h5f:
            # PROTEIN CASE
            if protein:
                for protein_id in h5f.keys():
                    try:
                        mean_emb = h5f[protein_id]['embedding'][:]
                        embeddings.append(mean_emb)
                        protein_ids.append(protein_id)
                    except KeyError:
                        continue

            # DNA CASE
            else:
                try:
                    mean_emb = h5f['mean_embedding'][:]
                    # mean_emb is expected to be (num_scaffolds, embedding_dim)
                    if mean_emb.ndim == 1:
                        mean_emb = mean_emb[np.newaxis, :]  # handle vector edge case
                    embeddings.append(mean_emb)
                except KeyError:
                        continue

        if not embeddings:
            continue
        
        if protein:
            embedding_matrix = np.array(embeddings)
        else:
            embedding_matrix = np.vstack(embeddings)
        data.append(embedding_matrix)
        labels.append(target_dict[isolate_id])
        
        ### ----- !!! FOR IG feature analysis comment this section: -----
        ids.append(isolate_id)
        ### ---------------------------------------------------------------
        
        ### ----- !!! FOR IG feature analysis uncomment this section: -----
        # if protein:
        #     ids.append((isolate_id, protein_ids))
        # else:
        #     ids.append(isolate_id)

        # if limit and len(data) >= limit:
        #     break
        ### ---------------------------------------------------------------

    return data, labels, ids

def label_hard_taxonomies(meta_file, label_file, hard_classes, output_file, stratify_col='class'):

    # Load label file
    label_dict = {}
    with open(label_file, 'r') as f:
        for line in f:
            isolate, label = line.strip().split()
            label_dict[isolate] = int(float(label))

    # Load metadata
    meta = pd.read_csv(meta_file, sep="\t")
    meta = meta[meta['isolate'].isin(label_dict)].copy()
    meta['label'] = meta['isolate'].map(label_dict)

    # Assign 'hard' or 'easy'
    meta['split'] = meta[stratify_col].apply(lambda x: 'hard' if x in hard_classes else 'easy')

    # Save to file
    meta.to_csv(output_file, sep="\t", index=False)
    print(f"Saved easy/hard annotations to {output_file}")


def collate_fn(batch):
    isolates, targets, ids = zip(*batch)
    isolates = [torch.tensor(matrix, dtype=torch.float32) for matrix in isolates]
    max_len = max(matrix.size(0) for matrix in isolates)

    padded_isolate = [torch.nn.functional.pad(matrix, (0, 0, 0, max_len - matrix.size(0))) for matrix in isolates]
    mask = [torch.cat([torch.ones(matrix.size(0), dtype=torch.bool), torch.zeros(max_len - matrix.size(0), dtype=torch.bool)]) for matrix in isolates]
    
    padded_isolates = torch.stack(padded_isolate)
    masks = torch.stack(mask)
    targets = torch.tensor(targets, dtype=torch.float32)
    return padded_isolates, masks, targets, ids

def identity(x):
    return x

class BucketBatchSampler(BatchSampler):
    """
    BucketBatchSampler groups samples into sorted buckets based on their length,
    and divides those buckets into random batches without repetition within an epoch.
    This is not currently necessary but was developed when testing attention pooling.

    Args:
        sampler: Base sampler for sampling the indices.
        batch_size: Number of samples in each batch.
        drop_last: Whether to drop the last incomplete batch.
        sort_key: Key for sorting the samples within a bucket.
        bucket_size_multiplier: Determines the bucket size as a multiple of the batch size.
    """

    def __init__(self,
                 sampler,
                 batch_size,
                 drop_last,
                 sort_key=identity,
                 bucket_size_multiplier=100):
        super().__init__(sampler, batch_size, drop_last)
        self.sort_key = sort_key
        self.batch_size = batch_size
        _bucket_size = batch_size * bucket_size_multiplier
        if hasattr(sampler, "__len__"):
            _bucket_size = min(_bucket_size, len(sampler))
        self.bucket_size = _bucket_size
        self.sampler = sampler  # Store sampler to use later

    def __iter__(self):
        # Get all indices from the sampler
        indices = list(self.sampler)
        
        # Sort indices based on sequence length (using sort_key)
        sorted_indices = sorted(indices, key=self.sort_key)

        # Divide the sorted indices into buckets
        buckets = [sorted_indices[i:i + self.bucket_size] for i in range(0, len(sorted_indices), self.bucket_size)]
        
        # Shuffle each bucket to create randomness
        for bucket in buckets:
            random.shuffle(bucket)

            # Divide the shuffled bucket into batches
            batches = [bucket[i:i + self.batch_size] for i in range(0, len(bucket), self.batch_size)]

            # If the last batch is smaller than batch_size and drop_last is True, drop it
            if self.drop_last and len(batches[-1]) < self.batch_size:
                batches = batches[:-1]

            for batch in batches:
                yield batch

    def __len__(self):
        # Return the number of batches, accounting for drop_last
        total_samples = len(self.sampler)
        return (total_samples // self.batch_size) if self.drop_last else math.ceil(total_samples / self.batch_size)

def split_data(isolates, targets, train_size, random_state):
    if len(isolates) == 1:
        # Use the single isolate for training, validation, and testing.
        # This was used for code debugging purposes.
        return isolates, isolates, isolates, targets, targets, targets
    
    train_data, temp_data, train_targets, temp_targets = train_test_split(
        isolates, targets, train_size=train_size, random_state=random_state
    )

    val_data, test_data, val_targets, test_targets = train_test_split(
        temp_data, temp_targets, test_size=0.5, random_state=random_state
    )

    return train_data, val_data, test_data, train_targets, val_targets, test_targets

def create_dataloader(isolates, targets, ids, batch_size, bucket_size_multiplier, 
                      num_workers=4, pin_memory=True):
    dataset = ProteinDataset(isolates, targets, ids)

    def sort_key(index):
        matrix, _, _ = dataset[index]  # Retrieve the matrix using the index
        return matrix.shape[0]
    
    if len(isolates) == 1:
        sampler = RandomSampler(dataset)
        return DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
    
    sampler = RandomSampler(dataset)
    bucket_batch_sampler = BucketBatchSampler(
        sampler,
        batch_size=batch_size,
        drop_last=False,  # Do not drop last small batch; handle it by grouping into the last bucket.
        sort_key=sort_key,
        bucket_size_multiplier=bucket_size_multiplier
    )
    
    return DataLoader(
        dataset,
        batch_sampler=bucket_batch_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
#FUNCTIONAL SECTION
class FunctionalAnnotationDataset(Dataset):
    """
    Dataset for tabular features such as KO functional annotations.
    Each isolate has a fixed-length feature vector.
    """
    def __init__(self, features, targets, ids):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.ids = list(ids)

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx], self.ids[idx]
    
def load_functional_annotations(kos_file, targets_file, meta_split_file=None, split_type=None, non_binary_input=False):
    kos = pd.read_csv(kos_file)
    ssc_target_df = pd.read_csv(targets_file, sep=' ', header=None, names=['sequence', 'classification'])
    
    kos['sequence'] = kos['sequence'].astype(str)
    ssc_target_df['sequence'] = ssc_target_df['sequence'].astype(str)
    
    merged_df = pd.merge(kos, ssc_target_df, on='sequence')

    # Optional filtering based on easy/hard (or train/test/val)
    if meta_split_file and split_type:
        split_df = pd.read_csv(meta_split_file, sep="\t")
        id_col = 'isolate' if 'isolate' in split_df.columns else 'sequence'
        id_filter = set(split_df[split_df['split'] == split_type][id_col])
        merged_df = merged_df[merged_df['sequence'].isin(id_filter)]

    X = merged_df.drop(['sequence', 'classification'], axis=1)
    y = merged_df['classification']
    IDs = merged_df['sequence']

    if non_binary_input:
        # 1. Log transform to reduce skewness
        X_log = np.log1p(X)  # log(1 + count)

        # 2. Standardize each feature (mean 0, std 1)
        scaler = StandardScaler()
        X = scaler.fit_transform(X_log)

    return X, y, IDs

def create_functional_dataloader(X, y, IDs, batch_size=64, num_workers=4, pin_memory=True, shuffle=True):
    """
    Create a simple DataLoader for KO-based tabular data.
    """
    dataset = FunctionalAnnotationDataset(X, y, IDs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
