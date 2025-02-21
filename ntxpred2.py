#########################################################################
# NTXpred2 is developed for predicting neurotoxic and non-neurotoxic    #
# peptide and proteins from their primary sequence. It is developed by  #
# Prof G. P. S. Raghava's group. Please cite : NTXpred2                 #
# Available at: https://webs.iiitd.edu.in/raghava/ntxpred2/             #
#########################################################################
import argparse  
import warnings
import os
import re
import numpy as np
import pandas as pd
import sklearn
from transformers import AutoTokenizer, EsmForSequenceClassification,EsmModel
from torch.utils.data import DataLoader, Dataset
import joblib
import torch
import pandas as pd
import torch
import esm
import pathlib
import shutil
import zipfile
import urllib.request

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(description='Please provide following arguments. Please make the suitable changes in the envfile provided in the folder.') 

## Read Arguments from command
parser.add_argument("-i", "--input", type=str, required=True, help="Input: protein or peptide sequence in FASTA format or single sequence per line in single letter code")
parser.add_argument("-o", "--output",type=str, default="outfile.csv", help="Output: File for saving results by default outfile.csv")
parser.add_argument("-t","--threshold", type=float,  help="Threshold: Value between 0 to 1 by default 0.5. ")
parser.add_argument("-j", "--job", type=int, choices=[1, 2, 3, 4], default=1, help="Job Type: 1: Prediction, 2: Protein Scanning, 3: Design, 4: Design all possible mutants")
parser.add_argument("-m", "--model", type=int, default=1, choices=[1, 2, 3], help="Model selection: For peptides (Length <=50): 1: ESM2-t30, For proteins (Length >= 51): 2: ET, For combined (Any Length): 3: ET")
parser.add_argument("-p", "--Position", type=int, help="Position of mutation (1-indexed)")
parser.add_argument("-r", "--Residues", type=str, help="Mutated residues (one or two of the 20 essential amino acids in upper case)")
parser.add_argument("-w","--winleng", type=int, choices =range(8, 21), help="Window Length: 8 to 20 (scan mode only), by default 12")
parser.add_argument("-wd", "--working", type=str, default=os.getcwd(),required=True, help="Working Directory: Location for writing results")
parser.add_argument("-d","--display", type=int, choices = [1,2], default=2, help="Display: 1:Neurotoxic, 2: All peptides, by default 2")


args = parser.parse_args()

nf_path = os.path.dirname(__file__)

###################################Model Calling##########################################

MODEL_URL = "https://webs.iiitd.edu.in/raghava/ntxpred2/download/Model.zip"
ZIP_PATH = os.path.join(os.path.dirname(__file__), "Model.zip")
EXTRACT_PATH = os.path.dirname(__file__)  # Extract in the same directory

# Check if the Model folder exists
if not os.path.exists(os.path.join(EXTRACT_PATH, "Model")):
    print("Downloading the model files...")

    try:
        # Download the ZIP file
        urllib.request.urlretrieve(MODEL_URL, ZIP_PATH)
        print("Download complete. Extracting files...")

        # Extract the ZIP file in the same directory
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(EXTRACT_PATH)

        print("Extraction complete. Removing ZIP file...")

        # Remove the ZIP file after extraction
        os.remove(ZIP_PATH)
        print("Model setup completed successfully.")

    except Exception as e:
        print(f"Error downloading or extracting model: {e}")
        raise
else:
    print("Model folder already exists. Skipping download.")


# Function to check the sequence residue
def readseq(file):
    with open(file) as f:
        records = f.read()
    records = records.split('>')[1:]
    seqid = []
    seq = []
    non_standard_detected = False  # Flag to track non-standard amino acids

    for fasta in records:
        array = fasta.split('\n')
        name, sequence = array[0].split()[0], ''.join(array[1:]).upper()
        
        # Check for non-standard amino acids
        filtered_sequence = re.sub('[^ACDEFGHIKLMNPQRSTVWY-]', '', sequence)
        if filtered_sequence != sequence:
            non_standard_detected = True
        
        seqid.append('>' + name)
        seq.append(filtered_sequence)
    
    if len(seqid) == 0:
        f = open(file, "r")
        data1 = f.readlines()
        for each in data1:
            seq.append(each.replace('\n', ''))
        for i in range(1, len(seq) + 1):
            seqid.append(">Seq_" + str(i))
    
    if non_standard_detected:
        print("Non-standard amino acids were detected. Processed sequences have been saved and used for further prediction.")
    else:
        print("No non-standard amino acids were detected.")
    
    df1 = pd.DataFrame(seqid)
    df2 = pd.DataFrame(seq)
    return df1, df2


# Function to check the length of sequences and suggest a model
def lenchk(file1):
    cc = []
    df1 = file1
    df1.columns = ['seq']
    
    # Analyze sequence lengths
    for seq in df1['seq']:
        cc.append(len(seq))
    
    # Check if any sequences are shorter than or equal to 7
    if any(length < 7 for length in cc):
        raise ValueError("Sequences with length < 7 detected. Please ensure all sequences have length at least 7. Prediction process stopped.")
    
    if all(length <= 50 for length in cc):
        print("All sequences are <= 50. Please use Model 1 (Peptide model).")
    elif all(length >= 51 for length in cc):
        print("All sequences are >= 51. Please use Model 2 (Protein model).")
    else:
        print("Mixed sequence lengths detected. Please use Model 3 (Combined model).")
    
    return df1


def prediction(inputfile1, inputfile2, model,out):
    df = pd.DataFrame()
    a=[]
    file_name = inputfile1
    file_name1 = out
    file_name2 = model
    file_name3 = inputfile2
    clf = joblib.load(file_name2)
    
    data_test1 = np.loadtxt(file_name, delimiter=',')
    data_test2 = np.loadtxt(file_name3, delimiter=',')
    data_test3 = np.concatenate([data_test1,data_test2], axis=1)
    X_test = data_test3
    y_p_score1=clf.predict_proba(X_test)
    y_p_s1=y_p_score1.tolist()
    df = pd.DataFrame(y_p_s1)
    df_1 = df.iloc[:,-1]
    df_1.to_csv(file_name1, index=None, header=False)


def prediction_mod(inputfile1, inputfile2, inputfile3, model,out):
    df = pd.DataFrame()
    a=[]
    file_name = inputfile1
    file_name1 = out
    file_name2 = model
    file_name3 = inputfile2
    file_name4 = inputfile3
    clf = joblib.load(file_name2)
    
    data_test1 = np.loadtxt(file_name, delimiter=',')
    data_test2 = np.loadtxt(file_name3, delimiter=',')
    data_test3 = np.loadtxt(file_name4, delimiter=',')
    data_test4 = np.concatenate([data_test1,data_test2,data_test3], axis=1)
    X_test = data_test4
    y_p_score1=clf.predict_proba(X_test)
    y_p_s1=y_p_score1.tolist()
    df = pd.DataFrame(y_p_s1)
    df_1 = df.iloc[:,-1]
    df_1.to_csv(file_name1, index=None, header=False)


# ESM2
# Define a function to process sequences
def process_sequences(df, df_2):
    df = pd.DataFrame(df, columns=['seq'])  # Assuming 'seq' is the column name
    df_2 = pd.DataFrame(df_2, columns=['SeqID'])
    # Process the sequences
    outputs = [(df_2.loc[index, 'SeqID'], row['seq']) for index, row in df.iterrows()]
    return outputs


# Function to prepare dataset for prediction
def prepare_dataset(sequences, tokenizer):
    seqs = [seq for _, seq in sequences]
    inputs = tokenizer(seqs, padding=True, truncation=True, return_tensors="pt")
    return inputs


# Function to write output to a file
def write_output(output_file, sequences, predictions, Threshold):
    with open(output_file, 'w') as f:
        f.write("SeqID,Seq,ESM Score,Prediction\n")
        for (seq_id, seq), pred in zip(sequences, predictions):
            final_pred = "Neurotoxic" if pred >= Threshold else "Non-Neurotoxic"
            f.write(f"{seq_id},{seq},{pred:.4f},{final_pred}\n")


# Function to make predictions
def make_predictions(model, inputs, device):
    # Move the model to the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    return probs


# Main function for ESM model integration
def run_esm_model(dfseq , df_2, output_file, Threshold):
    # Process sequences from the DataFrame
    sequences = process_sequences(dfseq, df_2)

    # Move the model to the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Prepare inputs for the model
    inputs = prepare_dataset(sequences, tokenizer)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Make predictions
    predictions = make_predictions(model, inputs, device)

    # Write the output to a file
    write_output(output_file, sequences, predictions, Threshold)


# Function to generate esm embeddings for protein dataset
class ProteinDataset(Dataset):
    """
    A custom Dataset class for protein sequences.
    """
    def __init__(self, sequences):
        # Ensure sequences is a list of strings
        if isinstance(sequences, pd.DataFrame):
            self.sequences = sequences.iloc[:, 0].tolist()  # Convert first column to list if it's a DataFrame
        else:
            self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def gen_emb(seq, output, batch_size=4096, seq_length=1500, repr_layers=[30]):
    # Load the tokenizer and model
    save_path = "Model/saved_esm2_t30_model"
    tokenizer = AutoTokenizer.from_pretrained(save_path)
    model = EsmModel.from_pretrained(save_path)

    model.eval()  # Set the model to evaluation mode

    # Set the maximum sequence length for padding/truncating sequences
    tokenizer.model_max_length = seq_length

    # Create a DataLoader for batching
    dataset = ProteinDataset(seq)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=lambda x: tokenizer(x, return_tensors="pt", padding=True, truncation=True, max_length=seq_length))

    # Generate embeddings
    all_embeddings = []
    with torch.no_grad():
        for inputs in data_loader:
            # Get model outputs, ensuring hidden states are included
            outputs = model(**inputs, output_hidden_states=True)  # Ensure hidden states are returned

            # Extract embeddings from specified layers (repr_layers)
            embeddings = outputs.hidden_states  # This will contain hidden states for each layer
            
            # Check if hidden_states is None
            if embeddings is None:
                print("Error: The model did not return hidden states.")
                return None  # Early exit if there's an issue

            # Extract embeddings from the specified layer (repr_layers[0] = 30 for example)
            selected_embeddings = embeddings[repr_layers[0]]  # Extract the specified layer (e.g., layer 30)
            
            # Take the mean over tokens for each sequence to obtain a fixed-size representation
            batch_embeddings = selected_embeddings.mean(dim=1).cpu().numpy()

            # Append batch embeddings
            all_embeddings.append(batch_embeddings)

    # Concatenate all batches into a single array
    all_embeddings = np.vstack(all_embeddings)
    # Save as a DataFrame
    embedding_df = pd.DataFrame(all_embeddings)
    print(f"Embedding shape: {embedding_df.shape}")
    print(f"Saving embeddings to: {output}")

    # Save the DataFrame to the specified output file
    embedding_df.to_csv(output, index=False)

    return embedding_df



def pred_prot_emb(input_csv, model_path, output_csv):
    """Predict protein embeddings using a trained ET model and selected features."""
    
    # Load input CSV
    df = pd.read_csv(input_csv)

    # Load trained model
    clf = joblib.load(model_path)

    # Column names to select
    sel_feat = ["3", "11", "17", "18", "46", "78", "89", "98", "108", "111", "112", "130", "171", "196", 
                "217", "223", "236", "240", "241", "247", "261", "280", "313", "334", "352", "368", 
                "385", "433", "445", "465", "471", "474", "478", "499", "512", "517", "552", "555", 
                "558", "586", "595", "601", "613", "619", "620", "625", "634", "639"]

    # Select required columns
    X_test = df[sel_feat]

    # Predict probabilities
    y_p_score = clf.predict_proba(X_test)

    # Extract probability of positive class
    df_pred = pd.DataFrame(y_p_score[:, -1])

    # Save predictions
    df_pred.to_csv(output_csv, index=False, header=False)


def pred_comb_emb(input_df, model, output):

    df = pd.DataFrame()
    a=[]
    file_name = pd.read_csv(input_df)
    # print(file_name.columns)
    file_name1 = output
    file_name2 = model
    clf = joblib.load(file_name2)

    # Column names to select    
    sel_feat = ["3", "11", "13", "14", "15", "29", "38", "78", "95", "97", "98", "102", "105", "112", "115", "119", "123", "143", "144", "169", "171", "187", "196", "199", "217", "224", "236", "240", "241", "247", "250", "261", "269", "277", "279", "280", "288", "291", "324", "331", "341", "345", "352", "375", "382", "385", "388", "393", "400", "409", "412", "418", "426", "431", "445", "446", "450", "455", "456", "463", "465", "471", "473", "475", "495", "499", "508", "509", "512", "517", "524", "526", "553", "586", "595", "607", "613", "618", "619", "620", "630", "632", "634", "639"]
    
    X_test = file_name[sel_feat]
    y_p_score1=clf.predict_proba(X_test)
    y_p_s1=y_p_score1.tolist()
    df = pd.DataFrame(y_p_s1)
    df_1 = df.iloc[:,-1]
    df_1.to_csv(file_name1, index=None, header=False)


# Function for generating pattern of a given length (protein scanning)
def seq_pattern(file1, file2, num):
    df1 = file1
    df1.columns = ['Seq']
    df2 = file2
    df2.columns = ['Name']
    cc = []
    dd = []
    ee = []
    ff = []
    gg = []
    for i in range(len(df1)):
        for j in range(len(df1['Seq'][i])):
            xx = df1['Seq'][i][j:j+num]
            if len(xx) == num:
                cc.append(df2['Name'][i])
                dd.append('Pattern_' + str(j + 1))
                ee.append(xx)
                ff.append(j + 1)  # Start position (1-based index)
                gg.append(j + num)  # End position (1-based index)
    df3 = pd.concat([pd.DataFrame(cc), pd.DataFrame(dd), pd.DataFrame(ff), pd.DataFrame(gg), pd.DataFrame(ee)], axis=1)
    df3.columns = ['SeqID', 'Pattern ID', 'Start', 'End', 'Seq']
    return df3


def class_assignment(file1,thr,out,wd):
    df1 = pd.read_csv(file1, header=None)
    df1.columns = ['ML Score']
    cc = []
    for i in range(0,len(df1)):
        if df1['ML Score'][i]>=float(thr):
            cc.append('Neurotoxic')
        else:
            cc.append('Non-Neurotoxic')
    df1['Prediction'] = cc
    df1 =  df1.round(3)
    output_file = os.path.join(wd, out)
    df1.to_csv(output_file, index=None)


def generate_mutant(original_seq, residues, position):
    std = "ACDEFGHIKLMNPQRSTVWY"
    if all(residue.upper() in std for residue in residues):
        if len(residues) == 1:
            mutated_seq = original_seq[:position-1] + residues.upper() + original_seq[position:]
        elif len(residues) == 2:
            mutated_seq = original_seq[:position-1] + residues[0].upper() + residues[1].upper() + original_seq[position+1:]
        else:
            print("Invalid residues. Please enter one or two of the 20 essential amino acids.")
            return None
    else:
        print("Invalid residues. Please enter one or two of the 20 essential amino acids.")
        return None
    return mutated_seq


def generate_mutants_from_dataframe(df, residues, position):
    mutants = []
    for index, row in df.iterrows():
        original_seq = row['seq']
        mutant_seq = generate_mutant(original_seq, residues, position)
        if mutant_seq:
            mutants.append((original_seq, mutant_seq, position))
    return mutants


# Function for generating all possible mutants
def all_mutants(file1,file2):
    std = list("ACDEFGHIKLMNPQRSTVWY")
    cc = []
    dd = []
    ee = []
    df2 = pd.DataFrame(file2)
    df2.columns = ['Name']
    df1 = pd.DataFrame(file1)
    df1.columns = ['Seq']
    for k in range(len(df1)):
        cc.append(df1['Seq'][k])
        dd.append('Original_'+'Seq'+str(k+1))
        ee.append(df2['Name'][k])
        for i in range(0,len(df1['Seq'][k])):
            for j in std:
                if df1['Seq'][k][i]!=j:
                    #dd.append('Mutant_'+df1['Seq'][k][i]+str(i+1)+j+'_Seq'+str(k+1))
                    dd.append('Mutant_'+df1['Seq'][k][i]+str(i+1)+j)
                    cc.append(df1['Seq'][k][:i] + j + df1['Seq'][k][i + 1:])
                    ee.append(df2['Name'][k])
    xx = pd.concat([pd.DataFrame(ee),pd.DataFrame(dd),pd.DataFrame(cc)],axis=1)
    xx.columns = ['SeqID','Mutant_ID','Seq']
    return xx


def extract_embeddings(model_path, fasta_file, output_dir, tokens_per_batch=4096, seq_length=1500, repr_layers=[30]):
    """Extract embeddings using a pre-trained ESM2 model."""
    
    # Convert string paths to pathlib.Path objects
    model_path = pathlib.Path(model_path)
    fasta_file = pathlib.Path(fasta_file)
    output_dir = pathlib.Path(output_dir)

    # Load model architecture
    model, alphabet = esm.pretrained.load_model_and_alphabet("esm2_t30_150M_UR50D")
    
    # Load saved weights
    state_dict = torch.load(model_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # Ensure model is on the correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    # Load FASTA dataset
    dataset = esm.data.FastaBatchedDataset.from_file(fasta_file)
    batches = dataset.get_batch_indices(tokens_per_batch, extra_toks_per_seq=1)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=alphabet.get_batch_converter(seq_length),
        batch_sampler=batches
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract embeddings
    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(f'Processing batch {batch_idx + 1} of {len(batches)}')

            toks = toks.to(device=device, non_blocking=True)

            out = model(toks, repr_layers=repr_layers, return_contacts=False)
            representations = {layer: t.to(device="cpu") for layer, t in out["representations"].items()}

            for i, label in enumerate(labels):
                entry_id = label.split()[0]
                filename = output_dir / f"{entry_id}.pt"
                truncate_len = min(seq_length, len(strs[i]))

                result = {
                    "entry_id": entry_id,
                    "mean_representations": {
                        layer: t[i, 1 : truncate_len + 1].mean(0).clone()
                        for layer, t in representations.items()
                    }
                }

                torch.save(result, filename)


def embeddings_to_dataframe(embedding_dir, output_csv=None):
    """Load .pt embeddings and convert them into a DataFrame."""
    
    files = list(pathlib.Path(embedding_dir).glob("*.pt"))
    data = [[(emb := torch.load(f))["entry_id"]] + emb["mean_representations"][30].tolist() for f in files]

    df = pd.DataFrame(data, columns=["Entry_ID"] + [f"{i}" for i in range(len(data[0]) - 1)])
    
    if output_csv:
        df.to_csv(output_csv, index=False)
    
    return df

import os

def convert_fasta(df, wd, filename="output.fasta"):
    filepath = os.path.join(wd, filename)
    
    with open(filepath, "w") as fasta_file:
        for seq in df["Seq"]:
            fasta_file.write(f">{seq}\n{seq}\n")




print('\n')
print('#####################################################################################')
print('# The program NTxPred2 is developed for predicting Neurotoxic and Non-Neurotoxic    #')
print('# peptides and proteins from their primary sequence, developed by Prof G. P. S.     #')
print("# Raghava's group. Available at: https://webs.iiitd.edu.in/raghava/ntxpred2/        #")
print('#####################################################################################')

# Provide guidance based on sequence lengths
print("\nUser Instructions:")
print("- If the length of the sequences is <= 50, use peptide model (Model 1).")
print("- If the length of the sequences is >=51, use protein model (Model 2).")
print("- If the sequence lengths are mixed, use combined model (Model 3).\n")

# Parameter initialization or assigning variable for command level arguments

Sequence= args.input        # Input variable 
 
# Output file 
if args.output is None:
    result_filename = "output.csv"
else:
    result_filename = args.output
         
# Threshold
if args.threshold is None:
    Threshold = {1: 0.45, 2: 0.55, 3: 0.5}.get(args.model, 0.5)
else:
    Threshold = float(args.threshold)


# Model
if args.model is None:
    Model = 1
else:
    Model = int(args.model)

# Display
if args.display is None:
    dplay = 2
else:
    dplay = int(args.display)

# Job Type
if args.job is None:
    Job = 1
else:
    Job = args.job

# Window Length 
if args.winleng == None:
    Win_len = int(12)
else:
    Win_len = int(args.winleng)


if args.Position is None:
    position = 1
else:
    position = args.Position

if args.Residues is None:
    residues = "AA"
else:
    residues = args.Residues

# Working Directory
wd = args.working

print('\nSummary of Parameters:')
print(f"Input File: {args.input} ; Model: {args.model} ; Job: {args.job} ; Threshold: {Threshold}")
print(f"Output File: {args.output} ; Display: {args.display}")

#------------------ Read input file ---------------------
f=open(Sequence,"r")
len1 = f.read().count('>')
f.close()

# Use the `readseq` function to process the input file
seqid, seq = readseq(Sequence)

# Check sequence lengths and suggest the appropriate model
seq = lenchk(seq)

# Combine sequence IDs and sequences into a single DataFrame
seqid_1 = list(map("{}".format, seqid[0].tolist()))
CM = pd.concat([pd.DataFrame(seqid_1), pd.DataFrame(seq)], axis=1)

# Save to csv file
CM.to_csv(f"{wd}/Sequence_1", header=False, index=None, sep="\n")


            
            #======================= Prediction Module starts from here =====================
if Job == 1:

            #=================================== Peptides ==================================    
    if Model == 1:
            print('\n======= You are using the Prediction module of NTxPred2. Your results will be stored in file: 'f"{wd}/{result_filename}"' =====\n')
            print('==== Predicting Neurotoxic Activity using ESM2-t30 model (peptide dataset): Processing sequences please wait ...')
            # Load the tokenizer and model
            model_save_path = "Model/Peptide/saved_model_t30"
            tokenizer = AutoTokenizer.from_pretrained(model_save_path)
            model = EsmForSequenceClassification.from_pretrained(model_save_path)
            model.eval()
            run_esm_model(seq, seqid_1, f"{wd}/{result_filename}", Threshold)
            df13 = pd.read_csv(f"{wd}/{result_filename}")
            #df13.rename(columns={"ML Score": "ESM Score"}, inplace=True)
            df13.columns = ['SeqID', 'Sequence', 'ESM Score', "Prediction"]
            df13['SeqID'] = df13['SeqID'].str.replace('>','')
            df13 = round(df13, 3)
            df13.to_csv(f"{wd}/{result_filename}", index=None)

            if dplay == 1:
                df13 = df13.loc[df13.Prediction == "Neurotoxic"]
                print(df13)
            elif dplay == 2:
                df13=df13
                print(df13)
            
            # Clean up temporary files 
            os.remove(f'{wd}/Sequence_1')  

            #=================================== Proteins ==================================
    if Model == 2:
            print('\n======= You are using the Prediction module of NTxPred2. Your results will be stored in file: 'f"{wd}/{result_filename}"' =====\n')
            print('==== Predicting Neurotoxic Activity using ET model with ESM2-t30 embeddings as features (protein dataset): Processing sequences please wait ...')
            #seq = seq.iloc[:, 0].tolist()  # Converts the first column to a list
            extract_embeddings("Model/saved_esm2_t30_model.pth", f'{wd}/Sequence_1', f'{wd}/esm2-t30_embeddings')
            df = embeddings_to_dataframe(f'{wd}/esm2-t30_embeddings', f'{wd}/esm2_embeddings.csv')
            df1 = pd.DataFrame(seqid)
            df1.rename(columns={0: "Entry_ID"}, inplace=True)
            df1["Entry_ID"] = df1["Entry_ID"].str.replace(r"^>", "", regex=True)
            df11 = df1.merge(df, on="Entry_ID", how="inner")
            df11.to_csv(f'{wd}/embeddings_in_order.csv')
            pred_prot_emb(f'{wd}/embeddings_in_order.csv', "Model/Protein/protein_dataset_et_model_sk_1_6_0.pkl",f'{wd}/seq.pred')
            class_assignment(f'{wd}/seq.pred', Threshold, 'seq.out', wd)
            df2 = pd.DataFrame(seq)
            df3 = pd.read_csv(f'{wd}/seq.out')
            df3 = round(df3,3)
            df4 = pd.concat([df1,df2,df3],axis=1)
            df4.columns = ['SeqID','Sequence','ML Score','Prediction']
            df4 = round(df4,3)
            df4.to_csv(f"{wd}/{result_filename}", index=None)
            if dplay == 1:
                df4 = df4.loc[df4.Prediction=="Neurotoxic"]
                print(df4)
            elif dplay == 2:
                df4=df4
                print(df4)
           
            # Clean up temporary files 
            os.remove(f'{wd}/Sequence_1') 
            os.remove(f'{wd}/seq.out')
            os.remove(f'{wd}/seq.pred')
            os.remove(f'{wd}/embeddings_in_order.csv')  
            os.remove(f'{wd}/esm2_embeddings.csv')
            shutil.rmtree(f'{wd}/esm2-t30_embeddings', ignore_errors=True)   

            #=================================== Combined ==================================        
    if Model == 3:
            print('\n======= You are using the Prediction module of NTxPred2. Your results will be stored in file: 'f"{wd}/{result_filename}"' =====\n')
            print('==== Predicting Neurotoxic Activity using ET model with ESM2-t30 embeddings as features (combined dataset): Processing sequences please wait ...')
            #seq = seq.iloc[:, 0].tolist()  # Converts the first column to a list
            extract_embeddings("Model/saved_esm2_t30_model.pth", f'{wd}/Sequence_1', f'{wd}/esm2-t30_embeddings')
            df = embeddings_to_dataframe(f'{wd}/esm2-t30_embeddings', f'{wd}/esm2_embeddings.csv')
            df1 = pd.DataFrame(seqid)
            df1.rename(columns={0: "Entry_ID"}, inplace=True)
            df1["Entry_ID"] = df1["Entry_ID"].str.replace(r"^>", "", regex=True)
            df11 = df1.merge(df, on="Entry_ID", how="inner")
            df11.to_csv(f'{wd}/embeddings_in_order.csv')
            pred_comb_emb(f'{wd}/embeddings_in_order.csv', "Model/Combined/combined_dataset_et_model_sk_1_6_0.pkl",f'{wd}/seq.pred')
            class_assignment(f'{wd}/seq.pred', Threshold, 'seq.out', wd)
            df2 = pd.DataFrame(seq)
            df3 = pd.read_csv(f'{wd}/seq.out')
            df3 = round(df3,3)
            df4 = pd.concat([df1,df2,df3],axis=1)
            df4.columns = ['SeqID','Sequence','ML Score','Prediction']
            df4.to_csv(f"{wd}/{result_filename}", index=None)
            if dplay == 1:
                df4 = df4.loc[df4.Prediction=="Neurotoxic"]
                print(df4)
            elif dplay == 2:
                df4 = df4
                print(df4) 
           
            # Clean up temporary files 
            os.remove(f'{wd}/Sequence_1')
            os.remove(f'{wd}/seq.out')
            os.remove(f'{wd}/seq.pred')
            os.remove(f'{wd}/embeddings_in_order.csv')  
            os.remove(f'{wd}/esm2_embeddings.csv')
            shutil.rmtree(f'{wd}/esm2-t30_embeddings', ignore_errors=True)  

    print('\n\n======= Thanks for using NTxPred2 =====')
    print('Please cite: NTxPred2\n\n')
           
            #======================= Protein Scanning Module starts from here =====================                            
elif Job == 2:
            #=================================== Peptides ==================================        
    if Model == 1:
        print('\n======= You are using the Protein Scanning module of NTxPred2. Your results will be stored in file: 'f"{wd}/{result_filename}"' =====\n')
        print('==== Scanning through ESM2-t30 model (peptide dataset): Processing sequences please wait ...')
        # Load the tokenizer and model
        model_save_path = "Model/Peptide/saved_model_t30"
        tokenizer = AutoTokenizer.from_pretrained(model_save_path)
        model = EsmForSequenceClassification.from_pretrained(model_save_path)
        model.eval()
        #seq = seq.iloc[:, 0].tolist()
        df_1 = seq_pattern(seq,seqid,Win_len)
        seq = df_1["Seq"].tolist()
        seqid_1=df_1["SeqID"].tolist()
        run_esm_model(seq, seqid_1, f"{wd}/{result_filename}", Threshold)
        df13 = pd.read_csv(f"{wd}/{result_filename}")
        df13 = df13.drop(columns=["SeqID", "Seq"])
        df21 = pd.concat([df_1, df13], axis=1)
        df21["SeqID"] = df21["SeqID"].str.lstrip(">")
        df21.columns = ['SeqID','Pattern ID', 'Start', 'End', 'Sequence', 'ESM Score', "Prediction"]
        df21 = round(df21, 3)
        df21.to_csv(f"{wd}/{result_filename}", index=None)

        if dplay == 1:
            df21 = df21.loc[df21.Prediction == "Neurotoxic"]
            print(df21)
        elif dplay == 2:
            df21=df21
            print(df21)

        # Clean up temporary files 
        os.remove(f'{wd}/Sequence_1')

            #=================================== Proteins ==================================
    if Model == 2:
            print('\n======= You are using the Protein Scanning module of NTxPred2. Your results will be stored in file: 'f"{wd}/{result_filename}"' =====\n')
            print('==== Predicting Neurotoxic Activity using ET model with ESM2-t30 embeddings as features (protein dataset): Processing sequences please wait ...')                        
            #seq = seq.iloc[:, 0].tolist()
            df_1 = seq_pattern(seq,seqid,Win_len)
            seq = df_1["Seq"].tolist()
            convert_fasta(df_1, wd, 'seq_pattern.fasta')
            extract_embeddings("Model/saved_esm2_t30_model.pth", f'{wd}/seq_pattern.fasta', f'{wd}/esm2-t30_embeddings')
            df = embeddings_to_dataframe(f'{wd}/esm2-t30_embeddings', f'{wd}/esm2_embeddings.csv')
            df = df.rename(columns={"Entry_ID": "Seq"})
            df11 = df_1.merge(df, on="Seq", how="inner")
            df11.to_csv(f'{wd}/pattern_embeddings_in_order.csv')
            pred_prot_emb(f'{wd}/pattern_embeddings_in_order.csv', "Model/Protein/protein_dataset_et_model_sk_1_6_0.pkl",f'{wd}/seq.pred')
            class_assignment(f'{wd}/seq.pred', Threshold, 'seq.out', wd)
            df3 = pd.read_csv(f'{wd}/seq.out')
            df4 = pd.concat([df_1,df3],axis=1)
            df4["SeqID"] = df4["SeqID"].str.lstrip(">")
            df4.columns = ['SeqID','Pattern ID', 'Start', 'End', 'Sequence', 'ML Score', "Prediction"]
            df4 = round(df4,3)
            df4.to_csv(f"{wd}/{result_filename}", index=None)
            if dplay == 1:
                df4 = df4.loc[df4.Prediction=="Neurotoxic"]
                print(df4)
            elif dplay == 2:
                df4=df4
                print(df4)
            
            # Clean up temporary files 
            os.remove(f'{wd}/Sequence_1')
            os.remove(f'{wd}/seq.out')
            os.remove(f'{wd}/seq.pred')  
            os.remove(f'{wd}/seq_pattern.fasta') 
            os.remove(f'{wd}/pattern_embeddings_in_order.csv')  
            os.remove(f'{wd}/esm2_embeddings.csv')
            shutil.rmtree(f'{wd}/esm2-t30_embeddings', ignore_errors=True)
    
                #=================================== Combined ==================================                            
    if Model == 3:
                
        print('\n======= You are using the Protein Scanning module of NTxPred2. Your results will be stored in file: 'f"{wd}/{result_filename}"' =====\n')
        print('==== Predicting Neurotoxic Activity using ET model with ESM2-t30 embeddings as features (combined dataset): Processing sequences please wait ...')
        #seq = seq.iloc[:, 0].tolist() 
        df_1 = seq_pattern(seq,seqid,Win_len)
        seq = df_1["Seq"].tolist()
        convert_fasta(df_1, wd, 'seq_pattern.fasta')
        extract_embeddings("Model/saved_esm2_t30_model.pth", f'{wd}/seq_pattern.fasta', f'{wd}/esm2-t30_embeddings')
        df = embeddings_to_dataframe(f'{wd}/esm2-t30_embeddings', f'{wd}/esm2_embeddings.csv')
        df = df.rename(columns={"Entry_ID": "Seq"})
        df11 = df_1.merge(df, on="Seq", how="inner")
        df11.to_csv(f'{wd}/pattern_embeddings_in_order.csv')
        pred_comb_emb(f'{wd}/pattern_embeddings_in_order.csv', "Model/Combined/combined_dataset_et_model_sk_1_6_0.pkl",f'{wd}/seq.pred')
        class_assignment(f'{wd}/seq.pred', Threshold, 'seq.out', wd)
        df3 = pd.read_csv(f'{wd}/seq.out')
        df4 = pd.concat([df_1,df3],axis=1)
        df4["SeqID"] = df4["SeqID"].str.lstrip(">")
        df4.columns = ['SeqID','Pattern ID', 'Start', 'End', 'Sequence', 'ML Score', "Prediction"]
        df4 = round(df4,3)
        df4.to_csv(f"{wd}/{result_filename}", index=None)
        if dplay == 1:
            df4 = df4.loc[df4.Prediction=="Neurotoxic"]
            print(df4)
        elif dplay == 2:
            df4=df4
            print(df4)

        # Clean up temporary files 
        os.remove(f'{wd}/Sequence_1')
        os.remove(f'{wd}/seq.out')
        os.remove(f'{wd}/seq.pred')  
        os.remove(f'{wd}/seq_pattern.fasta') 
        os.remove(f'{wd}/pattern_embeddings_in_order.csv')  
        os.remove(f'{wd}/esm2_embeddings.csv')
        shutil.rmtree(f'{wd}/esm2-t30_embeddings', ignore_errors=True) 
        
    print('\n\n======= Thanks for using NTxPred2 =====')
    print('Please cite: NTxPred2\n\n')

           
        #======================= Design Module starts from here =====================
elif Job == 3:
            #=================================== Peptides ==================================        
    if Model == 1:
        print('\n======= You are using the Design Module of NTxPred2. Your results will be stored in file: 'f"{wd}/{result_filename}"' =====\n')
        print('==== Predicting Neurotoxic Activity using ESM2-t30 model (peptide dataset): Processing sequences please wait ...')

        mutants = generate_mutants_from_dataframe(seq, residues, position)
        result_df = pd.DataFrame(mutants, columns=['Original Sequence', 'seq', 'Position'])
        out_len_mut = pd.DataFrame(result_df['seq'])
        model_save_path = "Model/Peptide/saved_model_t30"
        tokenizer = AutoTokenizer.from_pretrained(model_save_path)
        model = EsmForSequenceClassification.from_pretrained(model_save_path)
        model.eval()
        run_esm_model(seq, seqid_1, f"{wd}/{result_filename}", Threshold)
        run_esm_model(out_len_mut, seqid_1, f"{wd}/out_m", Threshold)
        df13 = pd.read_csv(f"{wd}/{result_filename}")
        df14 = pd.read_csv(f"{wd}/out_m")
        df14 = df14.drop(columns=['SeqID'])
        df15 = pd.concat([df13, df14], axis=1)
        seqid_1 = pd.Series(seqid_1, name="SeqID")
        df15 = pd.concat([seqid_1, result_df['Original Sequence'], df13['ESM Score'], df13['Prediction'], 
                            result_df['seq'], result_df['Position'], df14['ESM Score'], df14['Prediction']], axis=1)
        df15.columns = ['SeqID', 'Original Sequence', 'ESM Score', 'Prediction', 'Mutant Sequence', 'Position', 'ESM Score', 'Mutant Prediction']
        df15['SeqID'] = df15['SeqID'].str.replace('>', '')
        df15 = round(df15, 3)
        df15.to_csv(f"{wd}/{result_filename}", index=None)
        if dplay == 1:
            df15 = df15.loc[df15['Mutant Prediction'] == "Neurotoxic"]
            print(df15)
        elif dplay == 2:
            df15 = df15
            print(df15)
      
        # Clean up temporary files
        os.remove(f'{wd}/out_m')
        os.remove(f'{wd}/Sequence_1')
        

        #=================================== Proteins ==================================
    if Model == 2: 
        print('\n======= You are using the Design Module of NTxPred2. Your results will be stored in file: 'f"{wd}/{result_filename}"' =====\n')
        print('==== Predicting Neurotoxic Activity using ET model with ESM2-t30 embeddings as features (protein dataset): Processing sequences please wait ...')
        mutants = generate_mutants_from_dataframe(seq, residues, position)
        result_df = pd.DataFrame(mutants, columns=['Original Sequence', 'Mutant Sequence', 'Position'])
        result_df['Mutant Sequence'].to_csv(f'{wd}/out_len_mut', index=None, header=None)
        #seq=seq['seq'].tolist()
        out_len_mut = pd.DataFrame(result_df['Mutant Sequence'])
        ori_Sequence = pd.DataFrame(result_df["Original Sequence"])

        #prediction for original sequence
        ori_Sequence = ori_Sequence.rename(columns={"Original Sequence": "Seq"})
        convert_fasta(ori_Sequence, wd, "ori_Sequence.fasta")
        extract_embeddings("Model/saved_esm2_t30_model.pth", f'{wd}/ori_Sequence.fasta', f'{wd}/esm2-t30_embeddings')
        ori_Sequence_embeddings = embeddings_to_dataframe(f'{wd}/esm2-t30_embeddings', f'{wd}/esm2_embeddings.csv')
        ori_Sequence_embeddings = ori_Sequence_embeddings.rename(columns={"Entry_ID": "Seq"})
        df11 = ori_Sequence.merge(ori_Sequence_embeddings, on="Seq", how="inner")
        df11.to_csv(f'{wd}/ori_Sequence_embeddings_in_order.csv',  index=False)
        pred_prot_emb(f'{wd}/ori_Sequence_embeddings_in_order.csv', "Model/Protein/protein_dataset_et_model_sk_1_6_0.pkl",f'{wd}/seq.pred')
        class_assignment(f'{wd}/seq.pred', Threshold, 'seq.out', wd)
        df = pd.read_csv(f'{wd}/seq.out')
        df3 = round(df,3)
        df4 = pd.concat([ori_Sequence,df3],axis=1)
        df4.columns = ['Original Sequence','ML Score','Prediction']

        ##prediction for mutant sequence
        Mut_Sequence = pd.DataFrame(result_df["Mutant Sequence"])
        Mut_Sequence = Mut_Sequence.rename(columns={"Mutant Sequence": "Seq"})
        convert_fasta(Mut_Sequence, wd, "Mut_Sequence.fasta")
        extract_embeddings("Model/saved_esm2_t30_model.pth", f'{wd}/Mut_Sequence.fasta', f'{wd}/esm2-t30_embeddings')
        Mut_Sequence_embeddings = embeddings_to_dataframe(f'{wd}/esm2-t30_embeddings', f'{wd}/esm2_embeddings.csv')
        Mut_Sequence_embeddings = Mut_Sequence_embeddings.rename(columns={"Entry_ID": "Seq"})
        df12 = Mut_Sequence.merge(Mut_Sequence_embeddings, on="Seq", how="inner")
        df12.to_csv(f'{wd}/Mut_Sequence_embeddings_in_order.csv',  index=False)
        pred_prot_emb(f'{wd}/Mut_Sequence_embeddings_in_order.csv', "Model/Protein/protein_dataset_et_model_sk_1_6_0.pkl",f'{wd}/seq.pred')
        class_assignment(f'{wd}/seq.pred',Threshold, 'seq.out', wd)
        df = pd.read_csv(f'{wd}/seq.out')
        df33 = round(df,3)
        df44 = pd.concat([Mut_Sequence,df33],axis=1)
        df44.columns = ['Mutant Sequence','ML Score','Prediction']

        ##prediction of original sequence + mutant sequence
        df55 = pd.concat([seqid,df4,df44],axis=1)
        df55.columns = ['SeqID','Original Sequence','ML Score','Prediction','Mutant Sequence','ML Score','Prediction']
        df55["SeqID"] = df55["SeqID"].str.lstrip(">")
        df55.to_csv(f"{wd}/{result_filename}", index=None)

        if dplay == 1:
            df55 = df55.loc[df55['Mutant Prediction'] == "Neurotoxic"]
            print(df55)
        elif dplay == 2:
            df55 = df55
            print(df55)

     
        # Clean up temporary files 
        os.remove(f'{wd}/Sequence_1')
        os.remove(f'{wd}/seq.out')
        os.remove(f'{wd}/seq.pred')  
        os.remove(f'{wd}/Mut_Sequence.fasta') 
        os.remove(f'{wd}/ori_Sequence.fasta')
        os.remove(f'{wd}/out_len_mut')
        os.remove(f'{wd}/Mut_Sequence_embeddings_in_order.csv') 
        os.remove(f'{wd}/ori_Sequence_embeddings_in_order.csv')  
        os.remove(f'{wd}/esm2_embeddings.csv')
        shutil.rmtree(f'{wd}/esm2-t30_embeddings', ignore_errors=True) 

        #=================================== Combined ==================================
    if Model == 3: 
        print('\n======= You are using the Design Module of NTxPred2. Your results will be stored in file: 'f"{wd}/{result_filename}"' =====\n')
        print('==== Predicting Neurotoxic Activity using ET model with ESM2-t30 embeddings as features (combined dataset): Processing sequences please wait ...')

        mutants = generate_mutants_from_dataframe(seq, residues, position)
        result_df = pd.DataFrame(mutants, columns=['Original Sequence', 'Mutant Sequence', 'Position'])
        result_df['Mutant Sequence'].to_csv(f'{wd}/out_len_mut', index=None, header=None)
        #seq=seq['seq'].tolist()
        out_len_mut = pd.DataFrame(result_df['Mutant Sequence'])
        ori_Sequence = pd.DataFrame(result_df["Original Sequence"])

        #prediction for original sequence
        ori_Sequence = ori_Sequence.rename(columns={"Original Sequence": "Seq"})
        convert_fasta(ori_Sequence, wd, "ori_Sequence.fasta")
        extract_embeddings("Model/saved_esm2_t30_model.pth", f'{wd}/ori_Sequence.fasta', f'{wd}/esm2-t30_embeddings')
        ori_Sequence_embeddings = embeddings_to_dataframe(f'{wd}/esm2-t30_embeddings', f'{wd}/esm2_embeddings.csv')
        ori_Sequence_embeddings = ori_Sequence_embeddings.rename(columns={"Entry_ID": "Seq"})
        df11 = ori_Sequence.merge(ori_Sequence_embeddings, on="Seq", how="inner")
        df11.to_csv(f'{wd}/ori_Sequence_embeddings_in_order.csv',  index=False)
        pred_comb_emb(f'{wd}/ori_Sequence_embeddings_in_order.csv', "Model/Combined/combined_dataset_et_model_sk_1_6_0.pkl",f'{wd}/seq.pred')
        class_assignment(f'{wd}/seq.pred', Threshold, 'seq.out', wd)
        df = pd.read_csv(f'{wd}/seq.out')
        df3 = round(df,3)
        df4 = pd.concat([ori_Sequence,df3],axis=1)
        df4.columns = ['Original Sequence','ML Score','Prediction']

        ##prediction for mutant sequence
        Mut_Sequence = pd.DataFrame(result_df["Mutant Sequence"])
        Mut_Sequence = Mut_Sequence.rename(columns={"Mutant Sequence": "Seq"})
        convert_fasta(Mut_Sequence, wd, "Mut_Sequence.fasta")
        extract_embeddings("Model/saved_esm2_t30_model.pth", f'{wd}/Mut_Sequence.fasta', f'{wd}/esm2-t30_embeddings')
        Mut_Sequence_embeddings = embeddings_to_dataframe(f'{wd}/esm2-t30_embeddings', f'{wd}/esm2_embeddings.csv')
        Mut_Sequence_embeddings = Mut_Sequence_embeddings.rename(columns={"Entry_ID": "Seq"})
        df12 = Mut_Sequence.merge(Mut_Sequence_embeddings, on="Seq", how="inner")
        df12.to_csv(f'{wd}/Mut_Sequence_embeddings_in_order.csv',  index=False)
        pred_comb_emb(f'{wd}/Mut_Sequence_embeddings_in_order.csv', "Model/Combined/combined_dataset_et_model_sk_1_6_0.pkl",f'{wd}/seq.pred')
        class_assignment(f'{wd}/seq.pred',Threshold, 'seq.out', wd)
        df = pd.read_csv(f'{wd}/seq.out')
        df33 = round(df,3)
        df44 = pd.concat([Mut_Sequence,df33],axis=1)
        df44.columns = ['Mutant Sequence','ML Score','Prediction']

        ##prediction of original sequence + mutant sequence
        df55 = pd.concat([seqid,df4,df44],axis=1)
        df55.columns = ['SeqID','Original Sequence','ML Score','Prediction','Mutant Sequence','ML Score','Prediction']
        df55["SeqID"] = df55["SeqID"].str.lstrip(">")
        df55.to_csv(f"{wd}/{result_filename}", index=None)

        # Display results based on 'dplay'
        if dplay == 1:
            df55 = df55.loc[df55['Mutant Prediction'] == "Neurotoxic"]
            print(df55)
        elif dplay == 2:
            df55 = df55
            print(df55)

        # Clean up temporary files 
        os.remove(f'{wd}/Sequence_1')
        os.remove(f'{wd}/seq.out')
        os.remove(f'{wd}/seq.pred')  
        os.remove(f'{wd}/Mut_Sequence.fasta') 
        os.remove(f'{wd}/ori_Sequence.fasta')
        os.remove(f'{wd}/out_len_mut')
        os.remove(f'{wd}/Mut_Sequence_embeddings_in_order.csv') 
        os.remove(f'{wd}/ori_Sequence_embeddings_in_order.csv')  
        os.remove(f'{wd}/esm2_embeddings.csv')
        shutil.rmtree(f'{wd}/esm2-t30_embeddings', ignore_errors=True)

    print('\n\n======= Thanks for using NTxPred2 =====')
    print('Please cite: NTxPred2\n\n')


#======================= Design Module for all possible mutants starts from here =====================      
elif Job == 4:
            #=================================== Peptides ==================================        
    if Model == 1:
            print('\n======= You are using the Design Module for all possible mutants of NTxPred2. Your results will be stored in file: 'f"{wd}/{result_filename}"' =====\n')
            print('==== Predicting Neurotoxic Activity using ESM2-t30 model (peptide dataset): Processing sequences please wait ...')
            # Load the tokenizer and model
            model_save_path = "Model/Peptide/saved_model_t30"
            tokenizer = AutoTokenizer.from_pretrained(model_save_path)
            model = EsmForSequenceClassification.from_pretrained(model_save_path)
            model.eval()
            muts = all_mutants(seq, seqid_1)
            muts.to_csv(f'{wd}/muts.csv', index=None, header=None)  
            seq=muts['Seq'].tolist()
            seqid_1=muts['Mutant_ID'].tolist()
            run_esm_model(seq, seqid_1, f"{wd}/{result_filename}", Threshold)
            df13 = pd.read_csv(f"{wd}/{result_filename}")
            df13.columns = ['MutantID', 'Sequence', 'ESM Score', "Prediction"]
            seqid = pd.DataFrame(muts["SeqID"])
            df14 = pd.concat([seqid,df13],axis=1)
            df14['SeqID'] = df14['SeqID'].str.replace('>','')

            df14.to_csv(f"{wd}/{result_filename}", index=None)
            if dplay == 1:
                df14 = df14.loc[df14.Prediction == "Neurotoxic"]
                print(df14)
            elif dplay == 2:
                df14=df14
                print(df14)   

            # Clean up temporary files 
            os.remove(f'{wd}/Sequence_1') 
            os.remove(f'{wd}/muts.csv')    

           
    
            #=================================== Proteins ==================================          
    if Model == 2:
            print('\n======= You are using the Design module for all possible mutants of NTXpred2. Your results will be stored in file: 'f"{wd}/{result_filename}"' =====\n')
            print('==== Predicting Neurotoxic Activity using ET model with ESM2-t30 embeddings as features (protein dataset): Processing sequences please wait ...')
                        
            muts = all_mutants(seq, seqid_1)
            muts.to_csv(f'{wd}/muts.csv', index=None, header=None)  
            convert_fasta(muts,wd, "muts.fasta")
            extract_embeddings("Model/saved_esm2_t30_model.pth", f'{wd}/muts.fasta', f'{wd}/esm2-t30_embeddings')
            muts_embeddings = embeddings_to_dataframe(f'{wd}/esm2-t30_embeddings', f'{wd}/esm2_embeddings.csv')
            muts_embeddings = muts_embeddings.rename(columns={"Entry_ID": "Seq"})
            df12 = muts.merge(muts_embeddings, on="Seq", how="inner")
            df12.to_csv(f'{wd}/muts_embeddings_in_order.csv',  index=False)
            pred_prot_emb(f'{wd}/muts_embeddings_in_order.csv', "Model/Protein/protein_dataset_et_model_sk_1_6_0.pkl",f'{wd}/seq.pred')
            class_assignment(f'{wd}/seq.pred', Threshold, 'seq.out', wd)
            df = pd.read_csv(f'{wd}/seq.out')
            df33 = round(df,3)
            df44 = pd.concat([muts,df33],axis=1)
            df44.columns = ['SeqID','MutantID','Sequence','ML Score','Prediction']
            df44['SeqID'] = df44['SeqID'].str.replace('>','')

            df44.to_csv(f"{wd}/{result_filename}", index=None)
            if dplay == 1:
                df44 = df44.loc[df44.Prediction=="Neurotoxic"]
                print(df44)
            elif dplay == 2:
                df44 = df44
                print(df44) 

            # Clean up temporary files 
            os.remove(f'{wd}/Sequence_1')
            os.remove(f'{wd}/seq.out')
            os.remove(f'{wd}/seq.pred')  
            os.remove(f'{wd}/muts.fasta') 
            os.remove(f'{wd}/muts.csv')
            os.remove(f'{wd}/muts_embeddings_in_order.csv')  
            os.remove(f'{wd}/esm2_embeddings.csv')
            shutil.rmtree(f'{wd}/esm2-t30_embeddings', ignore_errors=True)  

            #=================================== Combined ==================================          
    if Model == 3:
            print('\n======= You are using the Design module for all possible mutants of NTxPred2. Your results will be stored in file: 'f"{wd}/{result_filename}"' =====\n')
            print('==== Predicting Neurotoxic Activity using ET model with ESM2-t30 embeddings as features (combined dataset): Processing sequences please wait ...')
                        
            muts = all_mutants(seq, seqid_1)
            muts.to_csv(f'{wd}/muts.csv', index=None, header=None)  
            convert_fasta(muts,wd, "muts.fasta")
            extract_embeddings("Model/saved_esm2_t30_model.pth", f'{wd}/muts.fasta', f'{wd}/esm2-t30_embeddings')
            muts_embeddings = embeddings_to_dataframe(f'{wd}/esm2-t30_embeddings', f'{wd}/esm2_embeddings.csv')
            muts_embeddings = muts_embeddings.rename(columns={"Entry_ID": "Seq"})
            df12 = muts.merge(muts_embeddings, on="Seq", how="inner")
            df12.to_csv(f'{wd}/muts_embeddings_in_order.csv',  index=False)
            pred_comb_emb(f'{wd}/muts_embeddings_in_order.csv', "Model/Combined/combined_dataset_et_model_sk_1_6_0.pkl",f'{wd}/seq.pred')
            class_assignment(f'{wd}/seq.pred', Threshold, 'seq.out', wd)
            df = pd.read_csv(f'{wd}/seq.out')
            df33 = round(df,3)
            df44 = pd.concat([muts,df33],axis=1)
            df44.columns = ['SeqID','MutantID','Sequence','ML Score','Prediction']
            df44['SeqID'] = df44['SeqID'].str.replace('>','')

            df44.to_csv(f"{wd}/{result_filename}", index=None)
            if dplay == 1:
                df44 = df44.loc[df44.Prediction=="Neurotoxic"]
                print(df44)
            elif dplay == 2:
                df44 = df44
                print(df44)   
            # Clean up temporary files
            os.remove(f'{wd}/Sequence_1') 
            os.remove(f'{wd}/seq.out')
            os.remove(f'{wd}/seq.pred')  
            os.remove(f'{wd}/muts.fasta') 
            os.remove(f'{wd}/muts.csv')
            os.remove(f'{wd}/muts_embeddings_in_order.csv')  
            os.remove(f'{wd}/esm2_embeddings.csv')
            shutil.rmtree(f'{wd}/esm2-t30_embeddings', ignore_errors=True)            
                                
    print('\n\n======= Thanks for using NTxPred2 =====')
    print('Please cite: NTxPred2\n\n')                   

