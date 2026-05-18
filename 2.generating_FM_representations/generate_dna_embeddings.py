import os
import h5py
import numpy as np
from Bio import SeqIO
import torch
from transformers import AutoTokenizer, AutoModel
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load DNABERT-S
model_name = "zhihan1996/DNABERT-S"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
model.eval()


# Utility functions
def chunk_tokens(tokens, max_tokens=512, stride=256):
    """Yield (start_idx, chunk_tokens) for a tokenized sequence."""
    for start in range(0, len(tokens), stride):
        chunk = tokens[start:start + max_tokens]
        if chunk:
            yield start, chunk

def embed_gene(seq, max_tokens=512, stride=256):
    """Return mean and max embeddings for a gene sequence."""
    """Overlapping tokens are averaged before computing the final pooling"""
    # Tokenize sequence
    tokens = tokenizer(seq, add_special_tokens=False)["input_ids"]

    # Prepare arrays to accumulate embeddings
    hidden_dim = model.config.hidden_size
    token_embeddings = np.zeros((len(tokens), hidden_dim), dtype=np.float32)
    token_counts = np.zeros(len(tokens), dtype=np.int32)

    # Process each chunk
    for start, chunk_ids in chunk_tokens(tokens, max_tokens=max_tokens, stride=stride):
        encoded_input = torch.tensor([chunk_ids]).to(device)  # [1, chunk_len]
        with torch.no_grad():
            outputs = model(encoded_input)
        hs = outputs[0].squeeze(0).cpu().numpy()  # [chunk_len, hidden_dim]

        # Map embeddings back to original token positions
        for i, token_idx in enumerate(range(start, start + len(chunk_ids))):
            token_embeddings[token_idx] += hs[i]
            token_counts[token_idx] += 1

        torch.cuda.empty_cache()

    # Average overlapping tokens
    token_embeddings = token_embeddings / token_counts[:, None]

    # Compute final embeddings for the gene
    mean_emb = token_embeddings.mean(axis=0)
    max_emb = token_embeddings.max(axis=0)

    return mean_emb, max_emb, token_embeddings

def process_fna_file(fna_path, output_path):
    gene_ids = []
    contig_ids = []
    mean_embs = []
    max_embs = []

    for record in SeqIO.parse(fna_path, "fasta"):
        header = record.id
        seq = str(record.seq)
        contig_id = header.split()[0] if header else "NA"
        
        mean_emb, max_emb, _ = embed_gene(seq)
        if mean_emb is None:
            continue

        gene_ids.append(header)
        contig_ids.append(contig_id)
        mean_embs.append(mean_emb)
        max_embs.append(max_emb)

    # Save to HDF5
    with h5py.File(output_path, "w") as f:
        f.create_dataset("gene_id", data=np.array(gene_ids, dtype='S'), compression="gzip")
        f.create_dataset("contig_id", data=np.array(contig_ids, dtype='S'), compression="gzip")
        f.create_dataset("mean_embedding", data=np.stack(mean_embs), compression="gzip")
        f.create_dataset("max_embedding", data=np.stack(max_embs), compression="gzip")

# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, required=True, help="Folder with .fna files OR prokka_* subfolders")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save .h5 files")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    # Case 1: input_folder contains .fna files directly
    fna_files = [f for f in os.listdir(args.input_folder) if f.endswith(".fna")]

    if fna_files:  
        # Flat folder mode
        print("Detected flat folder with .fna files.")
        for fna_file in fna_files:
            input_path = os.path.join(args.input_folder, fna_file)
            isolate_id = os.path.splitext(fna_file)[0]  # filename without extension
            output_path = os.path.join(args.output_folder, f"{isolate_id}.h5")
            
            if os.path.exists(output_path):
                print(f"Skipping {fna_file}, embeddings already exist.")
                continue

            print(f"Processing {fna_file} ...")
            process_fna_file(input_path, output_path)
            print(f"Saved to {output_path}")

    else:
        # Nested folder mode
        print("Detected prokka_* subfolders.")
        for subfolder in os.listdir(args.input_folder):
            subfolder_path = os.path.join(args.input_folder, subfolder)
            if not os.path.isdir(subfolder_path):
                continue
            if not subfolder.startswith("prokka_"):
                continue

            # Find .fna file inside this subfolder
            fna_files = [f for f in os.listdir(subfolder_path) if f.endswith(".fna")]
            if len(fna_files) == 0:
                print(f"⚠️ No .fna file found in {subfolder_path}, skipping.")
                continue
            if len(fna_files) > 1:
                print(f"⚠️ Multiple .fna files in {subfolder_path}, using the first one.")

            fna_file = fna_files[0]
            input_path = os.path.join(subfolder_path, fna_file)

            # Strip "prokka_" from folder name
            isolate_id = subfolder.replace("prokka_", "")
            output_path = os.path.join(args.output_folder, f"{isolate_id}.h5")
            
            if os.path.exists(output_path):
                print(f"Skipping {isolate_id}, embeddings already exist.")
                continue

            print(f"Processing {input_path} (isolate {isolate_id}) ...")
            process_fna_file(input_path, output_path)
            print(f"Saved to {output_path}")
