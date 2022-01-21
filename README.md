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
awk -F "\t" '{ if ($1 == "chr1" || $1 == "chr2" ) {print}}' celltype1.bed > celltype1_TEST.bed
```

The BED format files can then be converted into FASTA files using `bedtools`. We will need the assembly sequence in the reference genome of the species from which we have cell type measurements. These can generally be downloaded on the command line using a FTP link from [UCSC](https://hgdownload.soe.ucsc.edu/downloads.html). For example, the following command will download the mouse reference assembly sequence: `wget https://hgdownload.soe.ucsc.edu/downloads.html`

```bash
bedtools getfasta -fi mm10.fa -bed celltype1_TRAINING.bed > celltype1_TRAINING.fa
bedtools getfasta -fi mm10.fa -bed celltype1_VALIDATION.bed > celltype1_VALIDATION.fa
bedtools getfasta -fi mm10.fa -bed celltype1_TEST.bed > celltype1_TEST.fa
```

### Training the Convolutional Neural Network (CNN)

The Python script `keras_cnn.py` can be used to train cell type models. The model architecture is hard-coded to follow one of the architecture described in [Kaplow et al (bioRxiv, 2020)](https://www.biorxiv.org/content/10.1101/2020.12.04.410795v2.full) for a  starting point. The required arguments are the input positive class and negative class FASTA files for the training and validation set.

```
usage: keras_cnn.py [-h] [-b BATCH] [-e EPOCH] [-n NAME] -pt POSTRAIN -nt
                    NEGTRAIN -pv POSVAL -nv NEGVAL [-bl BASELR] [-ml MAXLR]
                    [-bm BASEMOMENTUM] [-mm MAXMOMENTUM]

optional arguments:
  -h, --help            show this help message and exit
  -b BATCH, --batch BATCH
                        Batch size
  -e EPOCH, --epoch EPOCH
                        Number of epochs
  -n NAME, --name NAME  Output model name
  -pt POSTRAIN, --postrain POSTRAIN
                        Path to positive training set input FASTA file
  -nt NEGTRAIN, --negtrain NEGTRAIN
                        Path to negative training set input FASTA file
  -pv POSVAL, --posval POSVAL
                        Path to positive training set input FASTA file
  -nv NEGVAL, --negval NEGVAL
                        Path to negative training set input FASTA file
  -bl BASELR, --baselr BASELR
                        Base learning rate
  -ml MAXLR, --maxlr MAXLR
                        Maximum learning rate
  -bm BASEMOMENTUM, --basemomentum BASEMOMENTUM
                        Base momentum
  -mm MAXMOMENTUM, --maxmomentum MAXMOMENTUM
                        Maximum momentum
```

When training many cell type models, it can be helpful to use shell and sbatch scripts to train the models in parallel. By iterating over the cell types in a shell script, jobs can be submitted to a GPU compute node in parallel. In the example below, assume FASTAs for cell types 1, 2, 3, 4, etc. are in the current working directory:

```bash
#!/bin/bash

ls *.fa | awk '{split($0, a, "_"); print a[1]}' | sort | uniq | while read line; do
  echo $line
  sbatch run_cnn.sb $line
done
```

In `run_cnn.sb`, the compute specifications and input variables to `keras_cnn.py` can be defined:

```bash
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=16G                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --job-name cnn
#SBATCH --error=cnn-%J.err.txt
#SBATCH --output=cnn-%J.out.txt

NAME="${1}_b64e30.h5"
PT="${1}_train_positive.fa"
NT="${1}_train_negative.fa"
PV="${1}_valid_positive.fa"
NV="${1}_valid_negative.fa"

python keras_cnn.py -b 64 -e 30 -n ${NAME} -pt ${PT} -nt ${NT} -pv ${PV} -nv ${NV}
```

### CNN Hyperparameter Tuning

The model performance and generalizability are especially sensitive to, but not limited to, the following hyperparameters:
- Ratio of the size of the positive set to size of the negative set
- Number of convolutional layers
- Number of convolutional filters
- Learning rate
- Number of epochs
- Batch size
- Dropout 
