import os
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import re

def windowing(sequence):
    sliced_sequence = []
    for idx, residue in enumerate(sequence):
        if residue == 'R':
            sliced_sequence.append(sequence[idx-9:idx+10])
    return sliced_sequence    

def get_path(parent_dir):
    paths = []
    for dirname, _, filenames in os.walk(parent_dir):
        paths = [os.path.join(dirname, filename) for filename in filenames]
        paths.sort()
    return paths

def preprocess_data(data):
    names = data[data.isnull().any(axis=1)].reset_index()
    features = data.dropna().reset_index() # position and sequence
    combined = pd.concat([names, features], axis=1, sort=False, ignore_index=True)
    combined.drop(columns=[0,2,3], inplace=True)
    combined.rename(columns={1:"name", 4:"position", 5:"sequence"}, inplace=True)
    return combined

def label_data(data_neg, data_pos):
    seq_neg = np.array(data_neg.sequence.values)
    seq_pos = np.array(data_pos.sequence.values)
    label_neg = np.zeros(len(data_neg), dtype=int)
    label_pos = np.ones(len(data_pos), dtype=int)
    # concate negative and positive dataset
    data_x = np.concatenate((seq_neg, seq_pos), axis=0, out=None)
    data_y = np.concatenate((label_neg, label_pos), axis=0, out=None)
    # shuffle dataset
    data_x, data_y = shuffle(data_x, data_y, random_state=30)
    return data_x, data_y

# tokenize and encode sequences in the training set
def tokenize(data_x, tokenizer):
    processed_data = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in data_x]
    tokens = tokenizer(processed_data, add_special_tokens=False, padding=False)
    return tokens['input_ids'], tokens['attention_mask']

from keras.preprocessing.text import Tokenizer
def neo_tokenize(data_x):
    asam = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V','X']
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(asam)
    
    window = 19 #3 to 19 odd number
    start_w = 9-int(window/2)
    end_w = 9+int(window/2)+1

    dataset_X_token = []
    for i in range(len(data_x)):
        temp = tokenizer.texts_to_sequences(data_x[i])
        dataset_X_token = np.append(dataset_X_token, temp)
    dataset_X_token = dataset_X_token-1
    dataset_X_token = dataset_X_token.reshape(len(data_x),19)
    dataset_X_token = dataset_X_token[:, range(start_w, end_w)]
    return dataset_X_token