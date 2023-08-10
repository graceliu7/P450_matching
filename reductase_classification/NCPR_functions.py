import random
import numpy as np
import pandas as pd
import pickle as pkl

from copy import deepcopy
from tqdm import tqdm
from joblib import Parallel, delayed
from Bio import SeqIO

from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid


def load_NCPR(embed_file='data/NCPR_bert.npz', protein_file='data/uniprot-NCPR.tab', fasta_file='data/uniprot-NCPR.fasta'):
    """
    Arguments:
    embed_file: an npz folder of zipped files, with each file being named according 
    to the protein entry name, and containing the embeddings for that protein
    protein_file: a tab file containing information on each protein, including name,
    entry name, review status, protein names, gene names, organism and length
    fasta_file: a fasta file containing the names of each protein as well as the sequence

    Returns:
    dict_data: a dictionary with keys as the embedding index, and the values are the 
    average values of the embeddings
    df_p: a pandas dataframe containing information on each index, and the corresponding sequence,  
    entry name, protein name, gene name and species.
    """
    # initialize dictionary
    p_dict = {}
    p_dict['species'] = []
    p_dict['ID'] = []
    p_dict['embedding'] = []

    dict_data = {}

    # loading in data
    data = np.load(embed_file, allow_pickle=True)
    lst = list(data.keys())

    df = pd.read_csv(protein_file, sep='\t')

    for i in range(len(lst)):
        # npz files have keys, stored in 'lst', and the correponding files are opened one by one
        item = lst[i]
        p_dict['embedding'].append(data[item].item()['avg'])

        # the last part of each file name is the id, which will be used to match up the embedding to protien info
        id = item.split('|')[-1]
        
        p_dict['ID'].append(id)
        # matching up the embeddings to the proteins in the df, and adding in relevant information from the df
        inds = df['Entry name'] == id
        name = df.loc[inds, 'Organism'].values[0].split()
        genus = name[0]
        species = name[1]
        p_dict['species'].append(species)
    # creating the data frame with all the information about each embedding
    old_df_p = pd.DataFrame(p_dict)

    # parsing the fasta file to match ID with sequence
    fasta_sequences = SeqIO.parse(open(fasta_file),'fasta')
    fasta_dict = {}
    for fasta in fasta_sequences:
        name, sequence = fasta.id.split('|')[-1], str(fasta.seq)
        fasta_dict[name]=sequence

    # adding sequences to the DataFrame
    sequences = []
    for index, row in old_df_p.iterrows():
        sequences.append(fasta_dict[row['ID']])
    old_df_p['Sequence'] = sequences

    # dropping duplicated sequences
    df_p = old_df_p.drop_duplicates(subset='Sequence').reset_index()
    dict_data = df_p['embedding'].to_dict()
    del df_p['embedding']
    df_p['index'] = df_p.index


    # setting up lists to be used for adding the species and genus IDs into the dataframe
    species_id = []
    genus_id = []

    # identifying the unique species and genus
    unique_species = df_p['species'].unique()

    # looping through each row of the data frame to assign a species index and a genus index
    for i in range(len(df_p)):
        species_id.append(np.where(unique_species==df_p['species'][i])[0][0])

    # completing the Dataframe and returning
    df_p['species index'] = species_id
    return dict_data, df_p

def load_clusters(df_p, cluster_file='data/clustered_NCPR.csv'):
    """Loads the cluster number for each protein into the dataframe provided

    Args:
        df_p (pd.DataFrame): A dataframe containing protein data, including ID and species
        cluster_file (str, optional): path to the file containing information abotu the cluster number of each protein. 
        Defaults to 'data/clustered_NCPR.csv'.

    Returns:
        pd.DataFrame: A copy of the inputed DataFrame, with a new column dictating the cluster number
    """
    cluster_df = pd.read_csv(cluster_file)
    id = []
    for index, row in cluster_df.iterrows():
        id.append(row['protein_name'].split('|')[-1])
        temp_df = pd.DataFrame(list(zip(list(cluster_df['labels']), id)), columns=['label', 'ID'])
    temp_df
    merged_df = pd.merge(df_p, temp_df, on='ID')
    return merged_df


def train_test_split(dict_data, df_p, split=0.9, label_level='species', seed = 2147):
    """splits the data into train and test sets while making sure there is at least one example 
    of each unique label in the train set

    Args:
        dict_data (dictionary): has keys as the index of the embedding, and values as the average embeddings
        df_p (pd.DataFrame): matched to dict_data by the embedding index, contains information about species, 
        genus, species number and genus number
        split (float, optional): the percentage of the data to be used in the train set. Defaults to 0.9.
        label_level (str, optional): Either 'genus' or 'species', indicates the level of classification to be 
        used as the target. Defaults to 'species'.
        seed (int, optional): If an integer seed is given, it will be used for np.random. If a non-integer value 
        is passed, a random seed will be used. Defaults to 2147.

    Raises:
        Exception: If the seed is neither an integer nor 'random', then it will be raised as an invalid seed

    Returns:
        numpy array: xtrain, the train set data
        numpy array: ytrain, the train set target
        numpy array: xtest, the test set data
        numpy array: ytest, the test set target
    """
    # setting the seed
    if type(seed) is int:
        np.random.seed(seed)
    elif seed != 'random':
        raise Exception('invalid seed')
    
    # creating and shuffling a list of all the embedding indices, which will eventually become the test set, and a list of all unique species/genus
    test_ind = list(df_p['index'])
    np.random.shuffle(test_ind)
    unique_label = list(df_p[label_level].unique())
    np.random.shuffle(unique_label)

    # creating a list of all the indices to be used in the train set
    train_ind = []

    # looping through each unique label to make one of each is randomly selected and part of the training set
    # at the same time that it is added to train_ind, it is removed from test_ind. Therefore, at the end, the 
    # only indices left in test_ind will be those used in the test set
    for label in unique_label:
        lst = df_p[df_p[label_level] == label]['index'].values
        index = random.choice(lst)
        train_ind.append(index)
        test_ind.remove(index)

    # based on the split, calculate how many more samples are needed after one of each label was chosen
    # randomly add that number of samples into the train set, while removing it from the test set
    num_missing = int(split*len(df_p) - len(train_ind))
    for i in range(num_missing):
        train_ind.append(test_ind.pop())

    # using the two lists of train_ind and test_ind as well as either species or genus index, construct the train and test sets
    xtrain = np.array([dict_data[key] for key in train_ind])
    xtest = np.array([dict_data[key] for key in test_ind])
    if label_level == 'species':    
        ytrain = np.array([df_p.loc[df_p['index']==key, 'species index'].values[0] for key in train_ind])
        ytest = np.array([df_p.loc[df_p['index']==key, 'species index'].values[0] for key in test_ind])
    elif label_level == 'label':    
        ytrain = np.array([df_p.loc[df_p['index']==key, 'label'].values[0] for key in train_ind])
        ytest = np.array([df_p.loc[df_p['index']==key, 'label'].values[0] for key in test_ind])
    return xtrain, ytrain, xtest, ytest

def gridsearch(dict_data, df_p, model_name, params):
    """Implements the scikit-learn Grid Search algorithm on a given model for hyperparameterization

    Args:
        dict_data (dictionary): has keys as the index of the embedding, and values as the average embeddings
        df_p (pd.DataFrame): matched to dict_data by the embedding index, contains information about species, 
        genus, species number and genus number
        model_name (string): the string name of the model to be searched
        params (dictionary or list of dictionaies): the parameters to be searched

    Raises:
        Exception: If the model_name does not match one of the working models, an exception will occur

    Returns:
        cv_results: a table of the results of the grid search, including train time and accuracy score
        best_estimator: the trained model with the best performing set of parameters
    """
    
    # creating the x and y sets
    xtrain, ytrain, xtest, ytest = train_test_split(dict_data, df_p, seed='random')
    X = np.append(xtrain, xtest, axis=0)
    y = np.append(ytrain, ytest)

    # creation of the model
    if model_name == 'XGBoost':
        model = XGBClassifier(disable_default_eval_metric=True, use_label_encoder=False)
    elif model_name == 'SVC rbf':
        model = SVC()
    elif model_name == 'SVC poly':
        model = SVC(kernel='poly')
    elif model_name == 'SVC sigmoid':
        model = SVC(kernel='sigmoid')
    elif model_name == 'Decision Tree':
        model = DecisionTreeClassifier()
    elif model_name == 'Linear SVC':
        model = LinearSVC(max_iter=10000)
    elif model_name == 'Random Forest':
        model = RandomForestClassifier()
    else:
        raise Exception('Model class not found') 

    # implementing grid search and returning
    clf = GridSearchCV(model, params, verbose=1, cv=3, n_jobs=5)
    clf.fit(X, y)
    return clf.cv_results_, clf.best_estimator_

def try_params(params, data, labels, model_name, seed):
    """passes one set of parameters into a given model to evaluate the accuracy of that model 
    Args:
        params (dictionary): dictionary with keys as the names of the hyperparameters and values as the 
        values to be tested - only one set of parameters should be passed in at a time
        data (dictionary): dictionary with integer keys representing the index of the embedding and values 
        as the average values of the embeddings
        labels (pd.DataFrame): pandas DataFrame containing information on embedding indices, species and genus
        model_name (string): name of the model to be used. Currently supported ones are 'XGBoost', 'SVC rbf',
        'SVC poly', 'SVC sigmoid', 'Decision Tree', 'Linear SVC' and 'Random Forest'
        seed (int or str): seed for the random splitting of train and test set. If set to 'random', a random 
        seed will be used

    Raises:
        Exception: If model_name is not one of the currently supported names, an Exception will be thrown

    Returns:
        list: list of three elements. The first element is a dictionary of the param values that were tested, 
        the second element is the accuracy score and the third element is the trained model
    """
    # creates the train and test sets
    xtrain, ytrain, xtest, ytest = train_test_split(data, labels, seed=seed)


    if model_name == 'XGBoost':
        model = XGBClassifier(disable_default_eval_metric=True, use_label_encoder=False, nthread=1)
    elif model_name == 'SVC rbf':
        model = SVC()
    elif model_name == 'SVC poly':
        model = SVC(kernel='poly')
    elif model_name == 'SVC sigmoid':
        model = SVC(kernel='sigmoid')
    elif model_name == 'Decision Tree':
        model = DecisionTreeClassifier()
    elif model_name == 'Linear SVC':
        model = LinearSVC(max_iter=10000)
    elif model_name == 'Random Forest':
        model = RandomForestClassifier()
    else:
        raise Exception('Model class not found') 

    # setting the params for the model and fitting, making predictions, and calculating accuracy score
    model.set_params(**params)
    model.fit(xtrain, ytrain)
    y_pred = model.predict(xtest)
    score = accuracy_score(ytest, y_pred)
    # printing a message after each set of completed params
    print('Completed: params = {}'.format(params))
    
    return [params, score, model]

def apply_hyperparam(data, labels, model_name, param_grid, csv_fname, pkl_fname, n_jobs=3, seed='random', return_best_param=False):
    """Takes in a grid of parameters and calls hyperparam to find the accuracy scores for each set of params.
    Can be parallelized using the n_jobs argument. Saves results directly to a .csv file and the best model to 
    a .pkl file, has the option to return the best set of parameters 

    Args:
        data (dictionary): has keys as the index of the embedding, and values as the average embeddings
        labels (pd.DataFrame): matched to dict_data by the embedding index, contains information about species, 
        genus, species number and genus number
        model_name (string): the string name of the model to be searched
        param_grid (dictionary): the parameters to be searched. The keys should be the names of the parameters. 
        The values should be an iterable with all values to be tried for each parameter
        csv_fname (str): name of the csv file that hyperparameterization results should be saved to
        pkl_fname (str): name of the pkl file that the best performing model wil be saved to
        n_jobs (int, optional): Number of jobs to run in parallel. Defaults to 3.
        seed (str, optional): The seed that should be used to initialize the train test split in the hyperparam
        function. Defaults to 'random'.
        return_best_param (boolean, optional): Whether or not the best set of parameters should be returned
        by the function
    """
    # initializing the parameter grid
    param_grid = ParameterGrid(param_grid)

    # calling hyperparam while running n_jobs in parallel, and storing them in a list
    results = Parallel(n_jobs=n_jobs)(delayed(try_params)(params, data, labels, model_name, seed) for params in tqdm(param_grid))

    # getting the column names and formatting the results to create a dataframe. Each row should contain the 
    # paramters used, as well as the accuracy score. The df is saved to a csv
    column_names = list(results[0][0].keys())
    column_names.append('Score')
    rows = []
    scores = []
    for i in range(len(results)):
        row = list(results[i][0].values())
        row.append(results[i][1])
        rows.append(row)
        scores.append(results[i][1])
    df = pd.DataFrame(rows, columns=column_names)
    df.to_csv(csv_fname, header=True)

    # identifying the best performing parameters and saving the associated model
    best_index = scores.index(max(scores))
    best_model = results[best_index][2]
    pkl.dump(best_model, open(pkl_fname, 'wb'))

    # returning the best params as a dictionary if return_best_param is set to True
    if return_best_param:
        best_params = results[best_index][0]
        return best_params
    else:
        pass 

def get_initial_accuracy(xtrain, ytrain, xtest, ytest, model_name, iterations=10):
    """finds the accuracy score for a particular model in a number of iterations

    Args:
        dict_data (dictionary): has keys as the index of the embedding, and values as the average embeddings
        df_p (pd.DataFrame): matched to dict_data by the embedding index, contains information about species, 
        genus, species number and genus number
        model_name (string): the name of the model to be used
        iterations (int, optional): the number of iterations to get the accuracy score of the model. 
        Defaults to 10.
        params (dictionary, optional): the params to be used for the model. If none is given, the default params
        will be used
        seed (int or string, optional): if an integer given, that will be the seed for the train_test_splitting.
        if set to 'random', a random seed will be used for each run.

    Raises:
        Exception: string does not match one of the existing models

    Returns:
        np.Array: an array containing all the accuracy scores from each iteration
        Object: the model that had the highest score. 
    """
    
    scores = []
    best_score = 0
    # iterates a given number of times, based on the keyword argument
    for i in tqdm(range(iterations)):
        # initiating the model based on the model_name argument
        if model_name == 'XGBoost':
            model = XGBClassifier(disable_default_eval_metric=True, use_label_encoder=False)
        elif model_name == 'SVC rbf':
            model = SVC()
        elif model_name == 'SVC poly':
            model = SVC(kernel='poly')
        elif model_name == 'SVC sigmoid':
            model = SVC(kernel='sigmoid')
        elif model_name == 'Decision Tree':
            model = DecisionTreeClassifier()
        elif model_name == 'Linear SVC':
            model = LinearSVC(max_iter=10000, dual=True)
        elif model_name == 'Random Forest':
            model = RandomForestClassifier()
        else:
            raise Exception('Model class not found') 

        # fitting the model and getting the accuracy score
        model.fit(xtrain, ytrain)
        y_pred = model.predict(xtest)
        score = accuracy_score(ytest, y_pred)
        scores.append(score)
        
        # if the current model outperforms the best model, set it as the new best model
        if score > best_score:
            best_model = deepcopy(model)
            best_score = deepcopy(score)
    return np.array(scores), best_model