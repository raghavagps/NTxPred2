# NTxPred2
A method for predicting neurotoxic activity of the peptides and proteins.
# Introduction
NTxPred2 is developed to empower researchers engaged in therapeutic peptide and protein development by providing innovative methods for quantifying and classifying neurotoxic peptides and proteins that target the central nervous system. It uses large language model word embeddings as features for predicting neurotoxic activity of peptides and proteins. The final model deploys prediction, protein-scanning, and design-based module which has been implemented using machine learning and protein language models. More information on NTxPred2.0 is available from its web server http://webs.iiitd.edu.in/raghava/ntxpred2. Please read/cite the content about NTxPred2 for complete information including algorithm behind the approach.
## Reference
Rathore et al. A large language model for predicting neurotoxic peptides and neurotoxins. #Journal Name#

## PIP Installation
PIP version is also available for easy installation and usage of this tool. The following command is required to install the package 
```
pip install ntxpred2
```
To know about the available option for the pip package, type the following command:
```
ntxpred2 -h
```

# Standalone

Standalone version of NTxPred2 is written in python3 and the following libraries are necessary for a successful run:

- Dependencies:
  - python=3.10.7
  - pytorch
  - esm=2.0.1

- scikit-learn: scikit-learn (sklearn) is a an open-source machine learning library. You can install it using pip (Python’s package installer). Open your terminal and type:
```
  pip install scikit-learn==1.5.2
```
- Pandas
```
  pip install pandas==1.5.3
```
- Numpy
The libraries pandas and numpy are automatically installed with the scikit-learn library.
- PyTorch: PyTorch is an open-source machine learning library. You can install it using pip (Python’s package installer). Open your terminal and type:
```
!pip install torch==2.1.0
```
- Transformers: The Transformers library provides state-of-the-art machine learning models like ESM. Install it with:
```
!pip install transformers==4.34.0
```
- joblib
```
!pip install joblib==1.4.2
```
- onnxruntime
```
!pip install onnxruntime==1.15.1
```

## Install using environment.yml

1. Create a new Conda environment from the environment.yml file:

```
conda env create -f environment.yml
```
2. Activate the newly created environment:
```
conda activate NTxPred2
```

# Important Note

- Due to large size of the model file, we have compressed model directory and uploaded on our webserver. https://webs.iiitd.edu.in/raghava/ntxpred2/download.html
- Download this zip file 
- It is crucial to unzip the file before attempting to use the code or model. The compressed file must be extracted to its original form for the code to function properly.

# Classification
Determines whether peptides and proteins are neurotoxic or non-neurotoxic based on their primary sequence. We have employed machine learning models and protein language models. The provided options include ESM2-t30 (peptide), ET (protein) and ET (combined, i.e., peptide+protein) models. You can select your preferred model for prediction. IF the sequences length are >7 and <= 50, use peptide model; if length of the sequences are >= 51, use protein model; and if lengths are of mixed type, use combined model. By default, this use the ESM2-t30 peptide model, which has demonstrated best performance on our evaluation on independent dataset as well as runtime efficient.

**Minimum USAGE**
To know about the available option for the standalone, type the following command: 
```
ntxpred2.py -h
```
To run the example, type the following command:
```
ntxpred2.py -i example.fasta

```
**Full Usage**: 
```
Following is complete list of all options, you may get these options
usage: ntxpred2.py [-h] 
                     [-i INPUT]
                     [-o OUTPUT]
                     [-t THRESHOLD]
                     [-j {1,2,3,4}]
                     [-m {1,2,3}] 
                     [-d {1,2}]
                     [-wd Working Directory]
```
```
Please provide following arguments

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Input: protein or peptide sequence in FASTA format or single sequence per line in single
                        letter code
  -o OUTPUT, --output OUTPUT
                        Output: File for saving results by default outfile.csv
  -t THRESHOLD, --threshold THRESHOLD
                        Threshold: Value between 0 to 1 by default 0.5.
  -j {1,2,3,4}, --job {1,2,3,4}
                        Job Type: 1: Prediction, 2: Protein Scanning, 3: Design, 4: Design all possible mutants
  -m {1,2,3}, --model {1,2,3}
                        Model selection: For peptides (Length <=50): 1: ESM2-t30, For proteins (Length >= 51): 2: ET,
                        For combined (Any Length): 3: ET
  -p POSITION, --Position POSITION
                        Position of mutation (1-indexed)
  -r RESIDUES, --Residues RESIDUES
                        Mutated residues (one or two of the 20 essential amino acids in upper case)
  -w {8,9,10,11,12,13,14,15,16,17,18,19,20}, --winleng {8,9,10,11,12,13,14,15,16,17,18,19,20}
                        Window Length: 8 to 20 (scan mode only), by default 12
  -wd WORKING, --working WORKING
                        Working Directory: Location for writing results
  -d {1,2}, --display {1,2}
                        Display: 1:Neurotoxic, 2: All peptides, by default 2
```

**Input File**: It allow users to provide input in two format; i) FASTA format (standard) (e.g. example.fasta) and ii) Simple Format. In case of simple format, file should have one peptide/protein sequence in a single line in single letter code (eg. example.seq). 

**Output File**: Program will save result in CSV format, in case user do not provide output file name, it will be stored in outfile.csv.

**Threshold**: User should provide threshold between 0 and 1, please note score is proportional to neurotoxic potential of peptide/protein.

**Jobs**:  In this program, four jobs have been incorporated;  
1) Prediction: Prediction for predicting given input peptide/protein sequence as neurotoxic and non-neurotoxic peptide/protein.
2) Protein Scanning: For the prediction of neurotoxic regions in a protein sequence.
3) Design: Generates mutant peptides/proteins with a single amino acid or dipeptide at particulal position provided by user and predict their neurotoxic activity. Provide residue (-r) and position (-p) while using this job.
4) Design all possible mutants: Design all possible mutants and predict their neurotoxic activity.

**Models**:  In this program, three models have been incorporated;  
i) Model1 for predicting given input peptide sequence as neurotoxic and non-neurotoxic peptide using protein language model ESM2-t30 based on word embeddings features; 

ii) Model2 for predicting given input protein sequence as neurotoxic and non-neurotoxic protein using machine learning model ET.

iii) Model3 for predicting given input combined sequence (both peptides and proteins) as neurotoxic and non-neurotoxic using machine learning model ET.

**Position**: User can provide position at which he/she wants insert any single amino acid or dipeptide for creating mutation. This option is available for only Design module.

**Residue**: Mutated residues (one or two of the 20 essential amino acids in upper case) (e.g., A for Alanine)

**Window length**: User can choose any pattern length between 8 and 20 in long sequences. This option is available for only protein scan module.

**Working Directory**: Location for writing results



NTxPred2.0 Package Files
=======================
It contain following files, brief description of these files given below

INSTALLATION  	: Installation instructions

LICENSE       	: License information

README.md     	: This file provide information about this package

ntxpred2.py 	:  Python program for classification

example.fasta	: Example file containing peptide sequences in FASTA format

example.seq	: Example file containing peptide sequences in simple format

## Installation via PIP
User can install Hemopi2.0 via PIP also
```
pip install ntxpred2
```
To know about the available option for the pip package, type the following command:

```
ntxpred2 -h
```
