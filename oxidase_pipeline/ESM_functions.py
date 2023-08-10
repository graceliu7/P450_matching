import pandas as pd
import numpy as np

from copy import deepcopy
from Bio import SeqIO
from tqdm import tqdm 
import sys
sys.path.append('../')
from NCPR_functions import load_NCPR
import torch 
import esm 

def load_data(filename="../data/uniprot-Cytochrome+P450.fasta", tab_file="../data/uniprot-Cytochrome+P450.tab"):
    # Initialize lists to write fasta file information into
    id_list = []  # Uniprot ID
    seq_list = []  # Protein Sequence
    name_list = []  # Uniprot Description used for filtering
    species_list = []

    print(1)
    df = pd.read_csv(tab_file, delimiter = '\t')
    print(2)
    with open(filename) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            id = record.id.split('|')[-1]
            id_list.append(id)
            seq_list.append(record.seq)
            name = record.description.split(' ')[1:]
            name_list.append(record.description)
            organism_name =  df.loc[df['Entry name']==id, 'Organism'].values[0].split(' ')
            if len(organism_name)>1:
                species = organism_name[1]
                species_list.append(species)
            else:
                species = organism_name[0]
                species_list.append(species)
              
        print(3)

    return id_list, seq_list, name_list, species_list


def data_filter(id_list, seq_list, name_list, species_list, length_threshold=1000):
    # Place data into format described in the ESM github readme
    data = []

    """To generate padding I want to know what the longest sequence is, 
        and out of interest, how many sequences are over the threshold and 
        how many proteins are explicitly called P450"""

    # This for loop goes through the id and seq lists to write the data frame for ESM
    for i in range(len(id_list)):

        # Filter out all bifunctional and hybrid P450s
        if "Bifunctional" in name_list[i] or "hybrid" in name_list[i]:
            pass

        # Filter out all fragments
        elif "Fragment" in name_list[i]:
            pass

        # Filter out all sequences over the threshold
        elif len(str(seq_list[i])) > length_threshold:
            pass

        # Check the longest sequence of those written into the data frame
        else:
            id_seq = (id_list[i], str(seq_list[i]), species_list[i])
            data.append(id_seq)
    return data

def filter_with_NCPR(data, NCPR_df):
    filtered = []
    unique_species=NCPR_df['species'].unique()
    for item in data:
        if item[2] in unique_species:
            filtered.append(item)
    return filtered

def batch_data(data, batch_size=16):
    # Create batches from the data frame
    batched_data = []
    i = 0
    
    while i < len(data):
        # Conditions just if batch size does not divide data length
        if (i + batch_size) > len(data):
            batched_data.append(data[i : len(data) - 1])
        else:
            batched_data.append(data[i : i + batch_size])
        i = i + batch_size
    batched_data.pop()

    return batched_data

def get_batched_data(p450_fasta="../data/uniprot-Cytochrome+P450.fasta", p450_tab="../data/uniprot-Cytochrome+P450.tab", NCPR_npz='../data/NCPR_bert.npz',
NCPR_tab='../data/uniprot-NCPR.tab',NCPR_fasta='../data/uniprot-NCPR.fasta', batch_size=16):
    data = load_data(filename=p450_fasta, tab_file=p450_tab)
    data= data_filter(data[0], data[1], data[2], data[3])
    dict_data, NCPR_df=load_NCPR(NCPR_npz, NCPR_tab, NCPR_fasta)
    data = filter_with_NCPR(data, NCPR_df)
    return batch_data(data, batch_size=batch_size)


def pad_token_reps(token_rep, size=1000, value=0, location="pre"):
    dummy_array = np.zeros((1, size, len(token_rep[0][0])))
    for i in range(len(token_rep)):
        pad_len = size - len(token_rep[i])
        padded = token_rep[i]
        if location == "pre":
            for j in range(pad_len):
                padded = np.insert(padded, 0, 0, axis=0)
            dummy_array = np.append(dummy_array, [padded], axis=0)
        
        elif location == "post":
            for j in range(pad_len):
                padded = np.append(padded, 0, axis=0)
            dummy_array = np.append(dummy_array, padded, axis=0)
        else:
            raise Exception(
                "Location variable {text} cannot be recognized.".format(text=location)
            )

    return dummy_array

def load_model(modelname="esm1b_t33_650M_UR50S"):
    # Download model
    model, alphabet = torch.hub.load("facebookresearch/esm", modelname)
    # Load ESM-1b model and batch converter
    if modelname == "esm1b_t33_650M_UR50S":
        model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        embedding_size = 1280
        layer_size = 33
    elif modelname == "esm_msa1_t12_100M_UR50S":
        model, alphabet = esm.pretrained.esm_msa1_t12_100M_UR50S()
        embedding_size = 768
        layer_size = 12
    elif modelname == "esm1_t34_670M_UR50S":
        model, alphabet = esm.pretrained.esm1_t34_670M_UR50S()
        embedding_size = 1280
        layer_size = 34
    elif modelname == "esm1_t34_670M_UR50D":
        model, alphabet = esm.pretrained.esm1_t34_670M_UR50D()
        embedding_size = 1280
        layer_size = 34
    elif modelname == "esm1_t34_670M_UR100":
        model, alphabet = esm.pretrained.esm1_t34_670M_UR100()
        embedding_size = 1280
        layer_size = 34
    elif modelname == "esm1_t12_85M_UR50S":
        model, alphabet = esm.pretrained.esm1_t12_85M_UR50S()
        embedding_size = 768
        layer_size = 12
    elif modelname == "esm1_t6_43M_UR50S":
        model, alphabet = esm.pretrained.esm1_t6_43M_UR50S()
        embedding_size = 768
        layer_size = 6
    else:
        raise Exception("Model {name} not found.".format(name=modelname))
    batch_converter = alphabet.get_batch_converter()
    return model, alphabet, batch_converter, embedding_size, layer_size
