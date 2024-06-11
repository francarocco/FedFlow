# importing libraries

import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
import datetime as dtm
import pickle
import numpy as np

import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
from tensorflow.keras import backend as K

from utilities import *
from networks import *

from fedflow import FedFlow


n_steps_in = 9
n_steps_out = 3
n_features = 6 # 6 considered features
to_predict = 0 # the feature to predict is located at column 0
num_epochs = 3
num_clients = 5
num_rounds = 25
num_epochs_federated = 3
splits = [0, 1, 2]
seeds = [26, 6, 76]

print(tf.config.list_physical_devices('GPU'),flush=True)
print('Starting custom8 federated training with lstm Network at CITY level.',flush=True)




path_files = f'../data'


print('Generating clients datasets', flush=True)

for split in splits:

    print('Reading clients datasets', flush=True)
    X_train = []
    y_train = []
    X_val = []
    y_val = []
    X_test = []
    y_test = []
    for i in range(num_clients):
        path_X_test_i = f'../data/xtest_{i}_split{split}.npy'
        path_y_test_i = f'../data/ytest_{i}_split{split}.npy'
        X_test_i = np.load(path_X_test_i)
        y_test_i = np.load(path_y_test_i)
        path_X_val_i = f'../data/xval_{i}_split{split}.npy'
        path_y_val_i = f'../data/yval_{i}_split{split}.npy'
        X_val_i = np.load(path_X_val_i)
        y_val_i = np.load(path_y_val_i)
        path_X_train_i = f'../data/xtrain_{i}_split{split}.npy'
        path_y_train_i = f'../data/ytrain_{i}_split{split}.npy'
        X_train_i = np.load(path_X_train_i)
        y_train_i = np.load(path_y_train_i)
        
        X_train.append(X_train_i)
        y_train.append(y_train_i)
        X_val.append(X_val_i)
        y_val.append(y_val_i)
        X_test.append(X_test_i)
        y_test.append(y_test_i)


    # dataset have been already generated for LSTM
    print('Reading global training, validation, test set',flush=True)
    path_X_train_global = f'../data/xtrain_city_split{split}.npy'
    path_y_train_global = f'../data/ytrain_city_split{split}.npy'
    X_train_global = np.load(path_X_train_global)
    y_train_global = np.load(path_y_train_global)
    path_X_val_global = f'../data/xval_city_split{split}.npy'
    path_y_val_global = f'../data/yval_city_split{split}.npy'
    X_val_global = np.load(path_X_val_global)
    y_val_global = np.load(path_y_val_global)
    path_X_test_global = f'../data/xtest_city_split{split}.npy'
    path_y_test_global = f'../data/ytest_city_split{split}.npy'
    X_test_global = np.load(path_X_test_global)
    y_test_global = np.load(path_y_test_global)


    #opening average stops values for clients: 
    path_factors = f'../data/custom_vector_euclidean_city.pkl'
    with open(path_factors, 'rb') as f:
        similarity_lists = pickle.load(f)
        
    path_files = f'../data'





    similarity_lists = calculate_similarities(context_vectors) # calculate similarities among clients
    fedflow_models_generation(path_files,input_shape,output_shape,similarity_lists,X_train,y_train,X_val,y_val,split,epochs) # perform local models generation with FedFlow
    fedflow_finetuning(path_files,X_train,y_train,X_val,y_val,split,epochs_finetuning) # final fine-tuning for further personalization
    