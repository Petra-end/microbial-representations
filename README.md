# Beyond the annotated: protein foundation models enable robust prediction of microbial root competence

Code accompanying *Beyond the annotated: protein foundation models enable robust prediction of microbial root competence* (Matyskova et al.). The study compares KEGG Orthology (KO), OrthoFinder orthogroup (OG), ESM-2, Bacformer, and DNABERT-S representations for predicting bacterial root competence, with a focus on generalisation across taxa and interpretation of model predictions.

## Repository structure

| Directory | Contents |
| --- | --- |
| `1.data/` | Phenotype labels, taxonomic split, and reference files |
| `2.generating_FM_representations/` | ESM-2, Bacformer, and DNABERT-S embedding generation |
| `3.model_training/` | MLP training and nested cross-validation |
| `4.model_performance/` | In-distribution, out-of-distribution, and external dataset evaluation |
| `5.explainability_analysis/` | Integrated Gradients and functional attribution analysis |
| `6.data_set_analysis/` | Taxonomic and representation-space analyses |
| `7.generating_figures/` | Figure-generation scripts |
| `suplementary_hyperparameters/` | Model hyperparameters for each cross-validation fold |
| `suplementary_results/` | Supplementary functional analysis tables |

## Running the analyses

The scripts follow the numbered workflow above. Obtain the source data as described in the publication and configure local input/output paths in the relevant scripts.

Generate embeddings from protein FASTA files with ESM-2 or Bacformer, or from nucleotide FASTA files with DNABERT-S:

```bash
python 2.generating_FM_representations/generate_protein_embeddings.py /path/to/proteins esm2 /path/to/output
python 2.generating_FM_representations/generate_protein_embeddings.py /path/to/proteins bacformer /path/to/output
python 2.generating_FM_representations/generate_dna_embeddings.py --input_folder /path/to/genomes --output_folder /path/to/output
```

Train the classifiers after selecting the representation and setting its input paths and parameters:

```bash
python 3.model_training/main_MLP_fm.py       # Foundation model representations
python 3.model_training/main_MLP_keggog.py   # KO or OG features
```

Evaluation, explainability, and figure-generation scripts are in directories `4`–`7`. The model configurations used in the study are provided in `suplementary_hyperparameters/`.

## Environment

The analyses used Python 3.9/3.10 and PyTorch 2.6, alongside packages for foundation models, machine learning, and data analysis. See **Supplementary Table 6** in [`Suplementary_information.pdf`](Suplementary_information.pdf) for software versions.

## Citation

Matyskova, P., Selten, G., Pieterse, C. M. J., Abeln, S., & de Jonge, R. *Beyond the annotated: protein foundation models enable robust prediction of microbial root competence*.
