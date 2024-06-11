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

class FedFlow:

    def __init__(self):  
        self = self

      
    def calculate_scaling_factor(similarity_lists,i): # calculate scaling factors for client i
        return similarity_lists[i]

    def weight_scaling_factor(X_train,i):
        # Get the total number of training data points across all clients
        global_count = sum(X.shape[0] for X in X_train)

        # Calculate the weight scaling factor for each client
        scaling_factor = X_train[i].shape[0] / global_count

        return scaling_factor

    def scale_model_weights(weight, scalar):
        '''function for scaling a models weights'''
        weight_final = []
        steps = len(weight)
        for i in range(steps):
            weight_final.append(scalar * weight[i])
        return weight_final



    def sum_scaled_weights(scaled_weight_list):
        '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
        avg_grad = list()
        #get the average grad accross all client gradients
        for grad_list_tuple in zip(*scaled_weight_list):
            layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
            avg_grad.append(layer_mean)
            
        return avg_grad
        
    def custom_scaled_weights_client(local_weight_list, client_scaling_factor): # calculate new weights for client i
        num_clients = len(local_weight_list)
        print(f'len local weight list: {len(local_weight_list)}', flush=True)
        print(f'client scaling factor: {len(client_scaling_factor)}', flush=True)
        temp_weight_i = []
        for j in range(num_clients):
            weight_j = self.scale_model_weights(local_weight_list[j],client_scaling_factor[j])
            temp_weight_i.append(weight_j)
        updated_local_weigths = self.sum_scaled_weights(temp_weight_i)
        return updated_local_weigths

    def calculate_similarities(context_vectors):
        # normalize clients context vectors
        context_vectors[:-1,:] = min_max_normalize(context_vectors[:-1,:])
        context_vectors[-1:,:] = min_max_normalize(context_vectors[-1:,:])

        similarity_lists = [] # calculate similarities based on Euclidean Distances
        euclidean_distances = np.zeros((context_vectors.shape[1], context_vectors.shape[1]))
        for i in range(context_vectors.shape[1]):
            for j in range(context_vectors.shape[1]):
                distance = np.linalg.norm(context_vectors[:,i] - context_vectors[:,j])
                euclidean_distances[i,j] = distance
        for i in range(euclidean_distances.shape[0]):
            euclidean_distances_area = euclidean_distances[i,:]
            euclidean_distances_area = 1/(1+euclidean_distances_area)
            euclidean_distances_area = euclidean_distances_area/sum(euclidean_distances_area)
            similarity_lists.append(euclidean_distances_area)
        return similarity_lists

    def fedflow_models_generation(path_files,input_shape,output_shape,similarity_lists,X_train,y_train,X_val,y_val,split,epochs):
        # network definition
        input_shape = (n_steps_in,n_features)
        output_shape = n_steps_out
        global_model = lstm_definition(input_shape,output_shape)
    
        accuracy_df = pd.DataFrame(columns=['test_accuracy', 'validation_accuracy', 'trainin_accuracy'])
        custom_global_weights = []
        for i in range(len(X_train)):
            custom_global_weights.append(global_model.get_weights()) #randomly initialize clients' models

        for r in range(num_rounds):
            global_weights = global_model.get_weights() #getting global weights
            scaled_local_weight_list = list()
            local_weight_list = list()
            
            for i in range(len(X_train)):
                #local training for round r
                local_model = lstm_definition(input_shape, output_shape)
                local_model.set_weights(custom_global_weights[i]) #getting weights of client i
                local_model.fit(X_train[i],y_train[i], epochs=epochs, verbose=2, validation_data=(X_val[i], y_val[i]),batch_size = 32)
                
                #scale the model weights and add to list
                scaling_factor = self.weight_scaling_factor(X_train, i)
                scaled_weights = self.scale_model_weights(local_model.get_weights(), scaling_factor)
                scaled_local_weight_list.append(scaled_weights)
                local_weight_list.append(local_model.get_weights())
                
                #clear session to free memory after each communication round
                K.clear_session()
            
            updated_custom_global_weights = []
            
            for i in range(len(X_train)):
                client_scaling_factor = self.calculate_scaling_factor(similarity_lists,i) #taking the weights for client i, based on the similarities
                updated_local_weigths = self.custom_scaled_weights_client(local_weight_list, client_scaling_factor)
                updated_custom_global_weights.append(updated_local_weigths)

            

            #calculate local model based on custom strategy
            custom_global_weights = updated_custom_global_weights.copy()
            
    

        for i in range(len(X_train)):
            # Saving local models
            local_model = lstm_definition(input_shape, output_shape)
            local_model.set_weights(custom_global_weights[i])
            path_local_model = f'{path_files}/models/fedflow_city_client{i}_split{split}.h5'
            local_model.save(path_local_model)


    def fedflow_finetuning(path_files,X_train,y_train,X_val,y_val,split,epochs_finetuning,scaler):   
        for i in range(len(X_train)):
                    
            # working with local model obtained through FedFlow
            path_model = f'{path_files}/models/fedflow_city_client{i}_split{split}.h5'
            local_federated_model = tf.keras.models.load_model(path_model)
            
            #Starting fine tuning of client i
            print(f'***Starting fine-tuning for client{i}***',flush=True)
            local_federated_model.fit(X_train[i],y_train[i], epochs=epochs_finetuning, verbose=2, validation_data=(X_val[i], y_val[i]),batch_size = 32)
            print('***Starting testing***',flush=True)
            scores = local_federated_model.evaluate(X_test[i],y_test[i],verbose=2)
            
            # evaluate complete metrics 
            forecasts = local_federated_model.predict(X_test[i])
            # results inversion
            forecasts_inverted = inverse_transform(scaler,forecasts)
            y_test_inverted = inverse_transform(scaler,y_test[i])
            # valutazione dei risultati
            AE, SE = evaluate_forecasts(y_test_inverted, forecasts_inverted, n_steps_in, n_steps_out)
            APE = evaluate_MAPE_forecasts(y_test_inverted, forecasts_inverted, n_steps_in, n_steps_out)

            print('Saving model and results',flush=True)
            path_model = f'{path_files}/models/local_fedflow_city_client{i}_split{split}.h5'
            local_federated_model.save(path_model)

            path_forecasts = f'{path_files}/results/forecasts_local_fedflow_city_client{i}_split{split}.npy'
            np.save(path_forecasts,forecasts)

            path_AE = f'{path_files}/results/AE_local_fedflow_city_client{i}_split{split}.pkl'
            AE.to_pickle(path_AE)

            path_SE = f'{path_files}/results/SE_fedflow_city_client{i}_split{split}.pkl'
            SE.to_pickle(path_SE)
            
            path_APE = f'{path_files}/results/APE_fedflow_city_client{i}_split{split}.pkl'
            APE.to_pickle(path_APE)

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
    