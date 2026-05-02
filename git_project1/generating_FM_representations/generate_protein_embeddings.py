import os
import torch
from transformers import AutoModel, AutoTokenizer
import re
import pandas as pd
import numpy as np
import h5py
from bacformer.pp import protein_seqs_to_bacformer_inputs

# Check if a GPU is available and use it if so
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_fasta(file_path):
    """Load sequences from a FASTA file."""
    sequences = []
    protein_ids = []
    with open(file_path, 'r') as file:
        sequence = ''
        protein_id = ''
        for line in file:
            if line.startswith('>'):
                if sequence:
                    sequences.append(sequence)
                    protein_ids.append(protein_id)
                sequence = ''
                protein_id = line.split()[0][1:]  # Get protein ID (ignore '>' symbol)
            else:
                sequence += line.strip()
        if sequence:
            sequences.append(sequence)
            protein_ids.append(protein_id)
    return protein_ids, sequences

def get_model(model_name, esm_attn_backend="sdpa"):
    """Load the appropriate model based on the model_name."""
    print(f"Loading model: {model_name}")
    if model_name.lower() == 'esm2':
        model_name = "facebook/esm2_t6_8M_UR50D"  # You can adjust the version as needed
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model = model.to(device)  # Move model to GPU if available
        print("ESM-2 model loaded.")
        return model, tokenizer, model_name
    elif model_name.lower() == 'bacformer':
        model = AutoModel.from_pretrained(
            "macwiatrak/bacformer-large-masked-complete-genomes", trust_remote_code=True
        ).to(device).eval().to(torch.float32)
        tokenizer = None  # Bacformer doesn’t use a HuggingFace tokenizer
        print("Bacformer model loaded.")
        return model, tokenizer, model_name
    else:
        raise ValueError("Model not recognized. Use 'esm2' or 'bacformer'.")

def find_gff_file(isolate_dir):
    """Find the first .gff file in the isolate's directory."""
    if not os.path.exists(isolate_dir):
        print(f"Directory not found: {isolate_dir}")
        return 0  # Return None if the directory does not exist
    
    for file in os.listdir(isolate_dir):
        if file.endswith(".gff"):
            return os.path.join(isolate_dir, file)
    
    print(f"No .gff file found in directory: {isolate_dir}")
    return 0

def extract_contig_info_gff(gff_file):
    """Extract contig info for each protein from a GFF file."""
    contig_map = {}
    with open(gff_file, 'r') as f:
        for line in f:
            if line.startswith("#"):
                continue  # Skip comments
            columns = line.strip().split('\t')
            if len(columns) > 8 and columns[2] == "CDS":  # Check if it's a CDS (coding sequence)
                attributes = columns[8]  # Attributes column
                protein_id_match = re.search(r'ID=([^;]+)', attributes)
                if protein_id_match:
                    protein_id = protein_id_match.group(1)
                    contig_id = columns[0]  # Contig is in the first column
                    contig_map[protein_id] = contig_id
    return contig_map

def extract_contig_info_tabular(contig_file):
    """Extract contig info for each protein from a tabular file with Protein_ID and Contig_ID."""
    contig_map = {}
    df = pd.read_csv(contig_file, sep='\t', compression='gzip', header=None)
    for _, row in df.iterrows():
        protein_id = row[0]
        contig_id = row[1]
        contig_map[protein_id] = contig_id
    return contig_map

def split_sequence(seq, chunk_size):
    """Split a sequence into chunks."""
    return [seq[i:i + chunk_size] for i in range(0, len(seq), chunk_size)]

def generate_embeddings(protein_ids, sequences, contig_map, model, tokenizer, model_name, chunk_sequences, chunk_size):
    """Generate embeddings for sequences and save them."""
    embeddings = []
    
    if model_name == "bacformer":
        # Prepare inputs (all protein sequences together)
        inputs = protein_seqs_to_bacformer_inputs(
            sequences,
            device=device,
            batch_size=128,
            max_n_proteins=9000
        )

        with torch.no_grad():
            outputs = model(**inputs, return_dict=True)

        # contextualized protein embeddings
        protein_embeddings = outputs["last_hidden_state"].cpu().numpy()  # shape: (1, n_prots, hidden_dim)

        # collect per-protein embeddings
        for i, protein_id in enumerate(protein_ids):
            contig_id = contig_map.get(protein_id, "unknown")
            prot_emb = protein_embeddings[0, i, :]  # embedding for protein i
            embeddings.append((protein_id, contig_id, prot_emb))

        print(f"Generated {len(embeddings)} Bacformer protein embeddings.")
        return embeddings

    else: 
        for i, seq in enumerate(sequences):
            # Determine whether to chunk the sequence or not
            if chunk_sequences:
                seq_chunks = split_sequence(seq, chunk_size)
            else:
                seq_chunks = [seq]
            
            embedding_accumulator = []

            for chunk in seq_chunks:
                encoded_input = tokenizer(chunk, return_tensors="pt").to(device)

                with torch.no_grad():
                    outputs = model(**encoded_input)

                embedding = outputs.last_hidden_state.squeeze().cpu().numpy()
                embedding = embedding[1:-1, :] #remove cls and eos token, first and last

                embedding_accumulator.append(embedding)

                # Clear GPU memory after processing each chunk
                torch.cuda.empty_cache()

            # Accumulate the embeddings from all chunks
            final_embedding = np.vstack(embedding_accumulator)
            
            # Also creates average embedding, median and max embedding
            mean_embedding = np.mean(final_embedding, axis=0)
            median_embedding = np.median(final_embedding, axis=0)
            max_embedding = final_embedding.max(axis=0)

            # Get contig ID for the protein (if available)
            protein_id = protein_ids[i]
            contig_id = contig_map.get(protein_id, "unknown")

            embeddings.append((protein_id, contig_id, mean_embedding, max_embedding, median_embedding)) #, final_embedding - for full per AA embedding to be included

        print("Embeddings generated.")
        return embeddings

def save_embeddings_h5(embeddings, output_file, model_name):
    """Save embeddings to an HDF5 (.h5) file."""
    with h5py.File(output_file, 'w') as h5f:
        if model_name == "bacformer":
            for protein_id, contig_id, mean_embedding in embeddings:
                grp = h5f.create_group(protein_id)
                grp.attrs['contig_id'] = contig_id
                grp.create_dataset('embedding', data=mean_embedding)
        else:
            for protein_id, contig_id, mean_embedding, max_embedding, median_embedding in embeddings: #, embedding_matrix - for full per AA embedding to be included
                grp = h5f.create_group(protein_id)
                grp.attrs['contig_id'] = contig_id
                grp.create_dataset('mean_embedding', data=mean_embedding)
                grp.create_dataset('median_embedding', data=median_embedding)
                grp.create_dataset('max_embedding', data=max_embedding)
                #grp.create_dataset('embedding_matrix', data=embedding_matrix) - for full per AA embedding to be included
    print(f"Embeddings saved to {output_file}.")

def process_faa_files(faa_directory, model_name, output_dir, chunk_sequences=True, chunk_size=512, contig_file=False, contig_format=None, gff_directory=None):
    """Process all .faa or .fasta files in the directory."""
    model, tokenizer, model_name = get_model(model_name)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    contig_map = {}
    
    for filename in os.listdir(faa_directory):
        if filename.endswith(".faa") or filename.endswith(".fasta"):
            output_file = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_embeddings.h5")
            
            # Skip processing if the embedding already exists
            if os.path.exists(output_file):
                print(f"Embeddings for {filename} already exist. Skipping...")
                continue

            print(f"\nProcessing file: {filename}")
            input_file = os.path.join(faa_directory, filename)

            try:
                protein_ids, sequences = load_fasta(input_file)
                if not sequences:  # Handle empty files
                    print(f"No sequences found in {filename}. Skipping...")
                    continue

                print(f"Loaded {len(sequences)} sequences from {filename}.")

                # Extract contig info from the corresponding files either .gff from prokka or tabular data
                if contig_file:
                    if contig_format == "gff":
                        isolate_name = os.path.splitext(filename)[0]

                        # Locate the corresponding .gff file in the gff_directory
                        isolate_dir = os.path.join(gff_directory, f"prokka_{isolate_name}")
                        gff_file_path = find_gff_file(isolate_dir)

                        if not os.path.exists(gff_file_path):
                            print(f"GFF file not found for isolate: {isolate_name}. Skipping...")
                            continue
                        
                        contig_map = extract_contig_info_gff(gff_file_path)
                    elif contig_format == "tabular":
                        contig_map = extract_contig_info_tabular(contig_file)
                    else:
                        raise ValueError("Invalid contig format. Use 'gff' or 'tabular'.")
                else:
                    contig_map = {}

                # Generate embeddings and save them
                embeddings = generate_embeddings(protein_ids, sequences, contig_map, model, tokenizer, model_name, chunk_sequences, chunk_size)
                save_embeddings_h5(embeddings, output_file, model_name)
                print(f"Completed processing for {filename}.")

            except Exception as e:
                print(f"An error occurred while processing {filename}: {str(e)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate embeddings for all .faa files in a directory, associating with .gff files for contig information.")
    parser.add_argument("faa_directory", type=str, help="Path to the directory containing .faa files or .fasta files. Default checks both can be adjusted to only look for one of the two.")
    parser.add_argument("model_name", type=str, help="Name of the model to use ('esm2' or 'prot_bert').")
    parser.add_argument("output_dir", type=str, help="Directory to save the output embeddings.")
    parser.add_argument("--chunk_sequences", action='store_true', help="Specify to chunk sequences. Default is True.")
    parser.add_argument("--chunk_size", type=int, default=512, help="Size of each chunk when chunking sequences. Default is 512.")
    parser.add_argument("--contig_file", action="store_true", help="Specify if contig info is available or needed. Default is False.")
    parser.add_argument("--contig_format", type=str, default=None, help="Format of the contig file (e.g., 'gff'). Required if using --gff_directory.")
    parser.add_argument("--gff_directory", type=str, default=None, help="Path to the directory containing isolate folders with .gff files. Required if --contig_format is 'gff'.")

    args = parser.parse_args()
    if args.contig_format == "gff" and args.gff_directory is None:
        parser.error("--gff_directory is required when --contig_format is 'gff'.")
    
    process_faa_files(args.faa_directory, args.model_name, args.output_dir, args.chunk_sequences, args.chunk_size, args.contig_file, args.contig_format, args.gff_directory)


