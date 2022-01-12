# celltype-ML

## Overview


## Analysis Walkthrough

### Installing the Conda environment

1. Create the environment from the `keras2.yml` file: `conda env create -f keras2.yml`

2. Activate the environment: `conda activate keras2`


### Partitioning sequences into training, validation, and test sets

The test set is composed of sequences from chromosomes 1 or 2. The validation set is composed of sequences from chromosomes 7 or 8. The remaining sequences in the dataset are partitioned into the training set. 

The following example bash command will partition sequences from `celltype1.bed` into the training set:

```bash
awk -F "\t" '{ if ($1 != "chr8" && $1 != "chr9" && $1 != "chr1" && $1 != "chr2" ) {print}}' celltype1.bed > celltype1_TRAINING.bed
```
The following example bash command will partition sequences from `celltype1.bed` into the validation set:

```bash
awk -F "\t" '{ if ($1 == "chr8" || $1 == "chr9" ) {print}}' celltype1.bed  > celltype1_VALIDATION.bed
```

The following example bash command will partition sequences from `celltype1.bed` into the test set:

```bash
awk -F "\t" '{ if ($1 == "chr1" || $1 == "chr2" ) {print}}' celltype1.bed
> celltype1_TEST.bed
```

The BED format files can then be converted into FASTA files using `bedtools`. We will need the assembly sequence in the reference genome of the species from which we have cell type measurements. These can generally be downloaded on the command line using a FTP link from [UCSC](https://hgdownload.soe.ucsc.edu/downloads.html). For example, the following command will download the mouse reference assembly sequence: `wget https://hgdownload.soe.ucsc.edu/downloads.html`

```bash
bedtools getfasta -fi mm10.fa -bed celltype1_TRAINING.bed > celltype1_TRAINING.fa
bedtools getfasta -fi mm10.fa -bed celltype1_VALIDATION.bed > celltype1_VALIDATION.fa
bedtools getfasta -fi mm10.fa -bed celltype1_TEST.bed > celltype1_TEST.fa
```

### Training the Convolutional Neural Network
