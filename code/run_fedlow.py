# importing libraries
import pickle
import numpy as np

import tensorflow as tf


from tensorflow.keras import backend as K

from utilities.utilities import *
from utilities.networks import *

from fedflow import FedFlow


n_steps_in = 9
n_steps_out = 3
n_features = 6 # 6 considered features
epochs = 3
num_clients = 5
num_rounds = 25
epochs_finetuning = 3
splits = range(5)

print(tf.config.list_physical_devices('GPU'),flush=True)



path_files = f'../data'


for split in splits:

    print(f'Reading clients datasets for split {split}', flush=True)
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


    path_files = f'../data'

    #opening average stops values for clients: 
    path_factors = f'{path_files}/context_vectors.pkl'
    with open(path_factors, 'rb') as f:
        context_vectors = pickle.load(f)

    #opening average stops values for clients: 
    path_scaler = f'{path_files}/scalers/scaler_split{split}.pkl'
    with open(path_scaler, 'rb') as f:
        scaler = pickle.load(f)

    ff = FedFlow()

    input_shape = (n_steps_in,n_features)
    output_shape = n_steps_out

    similarity_lists = ff.calculate_similarities(context_vectors) # calculate similarities among clients
    ff.fedflow_models_generation(path_files,input_shape,output_shape,similarity_lists,X_train,y_train,X_val,y_val,split,epochs) # perform local models generation with FedFlow
    ff.fedflow_finetuning(path_files,X_train,y_train,X_val,y_val,X_test,y_test,split,epochs_finetuning,scaler,n_steps_in,n_steps_out) # final fine-tuning for further personalization
    