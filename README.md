# Beyond the annotated: protein foundation models enable robust prediction of microbial root competence

Code accompanying *Beyond the annotated: protein foundation models enable robust prediction of microbial root competence* (Matyskova et al.). The study compares KEGG Orthology (KO), OrthoFinder orthogroups (OG), ESM-2, Bacformer, and DNABERT-S representations for predicting bacterial root competence in *Arabidopsis thaliana*. It evaluates prediction within the training taxa, generalisation to unseen bacterial phyla and independent datasets, and the biological signals underlying model predictions.

## Repository structure

| Directory | Contents |
| --- | --- |
| `1.data/` | Phenotype labels, taxonomic train/test split, and reference files |
| `2.generating_FM_representations/` | ESM-2, Bacformer, and DNABERT-S embedding generation |
| `3.model_training/` | MLP classifiers, data loading, and nested cross-validation with Optuna |
| `4.model_performance/` | Model performance and generalisation evaluations |
| `5.explainability_analysis/` | Integrated Gradients and functional feature-importance analyses |
| `6.data_set_analysis/` | Taxonomic similarity and embedding-space analyses |
| `7.generating_figures/` | Scripts for plotting the analysis results |
| `suplementary_hyperparameters/` | Saved hyperparameter configurations for each model and outer fold |
| `suplementary_results/` | Functional analysis results for KOs, OGs, and pathways |

## Running the analyses

The numbered directories follow the analysis workflow. The input datasets and methodological details are described in the accompanying publication.

**Generate foundation-model representations** from protein FASTA (`.faa`/`.fasta`) or nucleotide FASTA (`.fna`) files:

```bash
python 2.generating_FM_representations/generate_protein_embeddings.py /path/to/proteins esm2 /path/to/output
python 2.generating_FM_representations/generate_protein_embeddings.py /path/to/proteins bacformer /path/to/output
python 2.generating_FM_representations/generate_dna_embeddings.py --input_folder /path/to/genomes --output_folder /path/to/output
```

**Train classifiers** using the generated embeddings or KO/OG feature matrices. Set the input paths, output paths, and representation-specific parameters in the relevant training script, then run:

```bash
python 3.model_training/main_MLP_fm.py
python 3.model_training/main_MLP_keggog.py
```

These scripts train models from scratch using three-fold nested cross-validation and Optuna hyperparameter optimisation. The phenotype labels and taxonomic split used in the study are included in `1.data/`.

**Evaluate and interpret results** with the scripts in `4.model_performance/` through `7.generating_figures/`, covering external and out-of-distribution evaluation, Integrated Gradients, representation analyses, and figure generation.

## Environment

Software versions used in the analyses are listed in **Supplementary Table 6** of [`Suplementary_information.pdf`](Suplementary_information.pdf).

## Citation

Matyskova, P., Selten, G., Pieterse, C. M. J., Abeln, S., and de Jonge, R. *Beyond the annotated: protein foundation models enable robust prediction of microbial root competence*.
