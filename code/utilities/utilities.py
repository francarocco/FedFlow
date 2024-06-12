
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
import datetime as dtm
import pickle
import numpy as np

import tensorflow as tf
from tensorflow.keras import backend as K


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA



# defintion of utility function

# Function to extract the part before '-'
def extract_route_id(route):
    return route.split('-')[0]


# split a multivariate sequence into samples
def split_multivariate_sequences(sequences, n_steps_in, n_steps_out,to_predict):
 X, y = list(), list()
 for i in range(len(sequences)):
  # find the end of this pattern
  end_ix = i + n_steps_in
  out_end_ix = end_ix + n_steps_out
  # check if we are beyond the dataset
  if out_end_ix > len(sequences):
   break
  # gather input and output parts of the pattern
  seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, to_predict]
  X.append(seq_x)
  y.append(seq_y)
 return np.array(X), np.array(y)

# function for generating supervised dataset, route by route, block_id by block_id
def df_to_supervised_routes(df_train,scaler,n_steps_in,n_features,n_steps_out):
    X_train = np.empty((0, n_steps_in, n_features))
    y_train = np.empty((0, n_steps_out)) 
    routes = df_train.route_id.unique()
    for r in routes: # working route by route
        df_route = df_train[df_train['route_id']==r]
        dates = df_route.date.unique()
        for d in dates: # working day by day
            df_date = df_route[df_route['date']==d]
            blocks = df_date.block_id.unique()
            for b in blocks: # working block by block
                df_block = df_date[df_date['block_id']==b]
                my_df = pd.DataFrame()
                my_df['occupancy'] = df_block['occupancy']
                my_df['stop_id'] = df_block['stop_id']
                my_df['month'] = df_block['month']
                my_df['weekday'] = df_block['weekday']
                my_df['timeslot'] = df_block['timeslot']
                my_df['route_id'] = df_block['route_id']
                values = my_df.values
                scaled = scaler.transform(values)
                Xi, yi = split_multivariate_sequences(scaled, n_steps_in, n_steps_out,to_predict)
                if Xi.shape[0]>0:
                    X_train = np.concatenate((X_train,Xi),axis=0)
                    y_train = np.concatenate((y_train,yi),axis=0)
    return X_train, y_train


# make a persistence forecast
def persistence(last_ob, n_seq):
	return [last_ob for i in range(n_seq)]

# evaluate the persistence model
def make_forecasts_naive_forecasting(test, n_lag, n_seq):
    test = np.array(test)
    forecasts = list()
    for i in range(len(test)):
        X, y = test[i, 0:n_lag], test[i, n_lag:]
        # make forecast
        forecast = persistence(X[-1], n_seq)
        # store the forecast
        forecasts.append(forecast)
    return forecasts

# evaluate the RMSE for each forecast time step
def evaluate_forecasts_naive_forecasting(test, forecasts, n_lag, n_seq):
  AE=pd.DataFrame()
  SE=pd.DataFrame()
  test = np.array(test)
  forecast = np.array(forecasts)
  for i in range(n_seq):
    actual = test[:,(n_lag+i)]
    predicted = [forecast[i] for forecast in forecasts]
    rmse = sqrt(mean_squared_error(actual, predicted))
    print('t+%d RMSE: %f' % ((i+1), rmse))
    mae = mean_absolute_error(actual,predicted)
    print('t+%d MAE: %f' % ((i+1), mae))
    AEi=pd.DataFrame()
    absolute_error=np.absolute(np.asarray(actual)-np.asarray(predicted))
    AEi['Absolute Error']=absolute_error
    AEi['Horizon'] = pd.Series(['horizon = %d ' % (i+1) for x in range(len(AEi.index))])
    AE= pd.concat([AE,AEi], ignore_index=True)
    SEi=pd.DataFrame()
    square_error=np.square(np.asarray(actual)-np.asarray(predicted))
    SEi['Square Error']=square_error
    SEi['Horizon'] = pd.Series(['horizon = %d ' % (i+1) for x in range(len(SEi.index))])
    SE= pd.concat([SE,SEi], ignore_index=True)

  return AE, SE

# evaluate the RMSE for each forecast time step
def evaluate_forecasts(y_test, forecasts, n_lag, n_seq):
	AE=pd.DataFrame()
	SE=pd.DataFrame()
	for i in range(n_seq):
		actual = [row[i] for row in y_test]
		predicted = [forecast[i] for forecast in forecasts]
		rmse = sqrt(mean_squared_error(actual, predicted))
		mae = mean_absolute_error(actual,predicted)
		print('t+%d RMSE: %f' % ((i+1), rmse),flush=True)
		print('t+%d MAE: %f' % ((i+1), mae),flush=True)
		AEi=pd.DataFrame()
		absolute_error=np.absolute(np.asarray(actual)-np.asarray(predicted))
		AEi['Absolute Error']=absolute_error
		AEi['Horizon'] = pd.Series(['horizon = %d ' % (i+1) for x in range(len(AEi.index))])
		AE=pd.concat([AE,AEi],ignore_index=True)
		SEi=pd.DataFrame()
		squared_error=np.square(np.asarray(actual)-np.asarray(predicted))
		SEi['Square Error']=squared_error
		SEi['Horizon'] = pd.Series(['horizon = %d ' % (i+1) for x in range(len(SEi.index))])
		SE=pd.concat([SE,SEi],ignore_index=True)
	return AE, SE
    
def custom_mean_absolute_percentage_error(y_true, y_pred, epsilon=1):
    """
    Calculate the mean absolute percentage error with epsilon.

    Parameters:
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,)
        Estimated target values.

    epsilon : float, optional (default=1)
        Small value to add to the denominator to avoid division by zero.

    Returns:
    float
        Mean absolute percentage error.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    absolute_percentage_error = np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon))
    return np.mean(absolute_percentage_error) * 100    
    
# evaluate the MAPE
def evaluate_MAPE_forecasts(y_test, forecasts, n_lag, n_seq):
	APE=pd.DataFrame()
	for i in range(n_seq):
		actual = [row[i] for row in y_test]
		predicted = [forecast[i] for forecast in forecasts]
		#rmse = sqrt(mean_squared_error(actual, predicted))
		#mape = mean_absolute_percentage_error(actual,predicted)
		#print('t+%d RMSE: %f' % ((i+1), rmse),flush=True)
		#print('t+%d MAPE: %f' % ((i+1), mape),flush=True)
		APEi=pd.DataFrame()
		absolute_percentage_error=[custom_mean_absolute_percentage_error([actual[j]],[predicted[j]]) for j in range(len(actual))]
		APEi['Absolute Percentage Error']=absolute_percentage_error
		APEi['Horizon'] = pd.Series(['horizon = %d ' % (i+1) for x in range(len(APEi.index))])
		APE=pd.concat([APE,APEi],ignore_index=True)
	return APE

# evaluate the MAPE
def evaluate_MAPE_forecasts_naive_forecasting(y_test, forecasts, n_lag, n_seq):
    APE=pd.DataFrame()
    y_test = np.array(y_test)
    forecast = np.array(forecasts)
    for i in range(n_seq):
        actual = y_test[:,i]
        predicted = [forecast[i] for forecast in forecasts]
        #rmse = sqrt(mean_squared_error(actual, predicted))
        #mape = mean_absolute_percentage_error(actual,predicted)
        #print('t+%d RMSE: %f' % ((i+1), rmse),flush=True)
        #print('t+%d MAPE: %f' % ((i+1), mape),flush=True)
        APEi=pd.DataFrame()
        absolute_percentage_error=[custom_mean_absolute_percentage_error([actual[j]],[predicted[j]]) for j in range(len(actual))]
        APEi['Absolute Percentage Error']=absolute_percentage_error
        APEi['Horizon'] = pd.Series(['horizon = %d ' % (i+1) for x in range(len(APEi.index))])
        APE=pd.concat([APE,APEi],ignore_index=True)
    return APE

def inverse_transform(scaler, preds):
    # Initialize an empty matrix
    inverted = np.empty((preds.shape[0], 0))
    for i in range(preds.shape[1]):
    #for i in [0]:
        # my scaler is defined for this kind of structure:
        for_scaler = pd.DataFrame()
        for_scaler['occupancy'] = preds[:,i]
        for_scaler['stop_id'] = preds[:,i]
        for_scaler['month'] = preds[:,i]
        for_scaler['weekday'] = preds[:,i]
        for_scaler['timeslot'] = preds[:,i]
        for_scaler['route_id'] = preds[:,i]
        values_for_scaler = for_scaler.values
        inverted_values = scaler.inverse_transform(values_for_scaler)
        inverted_pred = inverted_values[:,0]
        inverted = np.column_stack((inverted,inverted_pred))
    return(inverted)
# finding the best hyperparameters

def adf_test(series):
    result = adfuller(series)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    return result[1]  # Return p-value

# evaluate the persistence model
def make_arima_forecasts(train,test, n_lag, n_seq, order):
    forecasts = []
    actuals = []
    history  = [x for x in train]
    for t in range(len(test)):
        if (t+n_seq)<len(test):
            last_obs = test[t]
            history.append(last_obs)
            actual = test[t:(t+n_seq)]
            actuals.append(actual)
            try:
                model = ARIMA(history[-n_lag:], order=order)
                model_fit = model.fit()
                output = model_fit.forecast(steps=n_seq)
                forecasts.append(output)
            except Exception as e:
                print(f"Error during forecasting at step {t} with window: {e}")
                forecasts.append([0] * n_seq)
                
    return forecasts, actuals

# evaluate the RMSE for each forecast time step
def evaluate_arima_forecasts(test, forecasts, n_lag, n_seq):
    AE=pd.DataFrame()
    SE=pd.DataFrame()
    test = np.array(test)
    forecasts = np.array(forecasts)
    for i in range(n_seq):
        actual = test[:,i]
        predicted = [forecast[i] for forecast in forecasts]
        #rmse = sqrt(mean_squared_error(actual, predicted))
        #print('t+%d RMSE: %f' % ((i+1), rmse))
        #mae = mean_absolute_error(actual,predicted)
        #print('t+%d MAE: %f' % ((i+1), mae))
        AEi=pd.DataFrame()
        absolute_error=np.absolute(np.asarray(actual)-np.asarray(predicted))
        AEi['Absolute Error']=absolute_error
        AEi['Horizon'] = pd.Series(['horizon = %d ' % (i+1) for x in range(len(AEi.index))])
        AE= pd.concat([AE,AEi], ignore_index=True)
        SEi=pd.DataFrame()
        square_error=np.square(np.asarray(actual)-np.asarray(predicted))
        SEi['Square Error']=square_error
        SEi['Horizon'] = pd.Series(['horizon = %d ' % (i+1) for x in range(len(SEi.index))])
        SE= pd.concat([SE,SEi], ignore_index=True)
    return AE, SE

# evaluate the MAPE
def evaluate_arima_MAPE_forecasts(y_test, forecasts, n_lag, n_seq):
    APE=pd.DataFrame()
    y_test = np.array(y_test)
    forecasts = np.array(forecasts)
    for i in range(n_seq):
        actual = y_test[:,i]
        predicted = [forecast[i] for forecast in forecasts]
        #rmse = sqrt(mean_squared_error(actual, predicted))
        #mape = mean_absolute_percentage_error(actual,predicted)
        #print('t+%d RMSE: %f' % ((i+1), rmse),flush=True)
        #print('t+%d MAPE: %f' % ((i+1), mape),flush=True)
        APEi=pd.DataFrame()
        absolute_percentage_error=[custom_mean_absolute_percentage_error([actual[j]],[predicted[j]]) for j in range(len(actual))]
        APEi['Absolute Percentage Error']=absolute_percentage_error
        APEi['Horizon'] = pd.Series(['horizon = %d ' % (i+1) for x in range(len(APEi.index))])
        APE=pd.concat([APE,APEi],ignore_index=True)
    return APE

'''
def weight_scalling_factor(clients_trn_data, client_name):
    client_names = list(clients_trn_data.keys())
    #get the bs
    bs = list(clients_trn_data[client_name])[0][0].shape[0]
    #first calculate the total training data points across clinets
    global_count = sum([tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy() for client_name in client_names])*bs
    # get the total number of data points held by a client
    local_count = tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy()*bs
    return local_count/global_count
'''

# Function to calculate Euclidean distance between two matrices 
def euclidean_distance(val1, val2):
    return abs(val1 - val2)
    
# Function to normalize the distances to obtain similarity scores
def normalize_distances(distances):
    max_dist = np.max(distances)
    min_dist = np.min(distances)
    return (max_dist - distances) / (max_dist - min_dist)
   
# Min-Max Normalization
def min_max_normalize(matrix):
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    normalized_matrix = (matrix - min_val) / (max_val - min_val)
    return normalized_matrix