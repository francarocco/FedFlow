import pandas as pd
import matplotlib.pyplot as plt
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





def extract_route_id(route):
    """
    Extract the base route ID from a route string in the format 'route-block'.

    Parameters:
    - route (str): The full route identifier (e.g., 'R12-B3').

    Returns:
    - str: The extracted route ID (e.g., 'R12').
    """
    return route.split('-')[0]



def divide_vector(vector, num_subvectors):
    """
    Evenly divide a list/vector into `num_subvectors` parts, distributing remainder as evenly as possible.

    Parameters:
    - vector (list): List to divide.
    - num_subvectors (int): Number of parts to divide into.

    Returns:
    - List[List]: A list of subvectors.
    """
    total_elements = len(vector)
    subvector_length = total_elements // num_subvectors
    remainder = total_elements % num_subvectors

    subvectors = []
    start_index = 0
    for i in range(num_subvectors):
        if i < remainder:
            subvector_end = start_index + subvector_length + 1
        else:
            subvector_end = start_index + subvector_length

        subvectors.append(vector[start_index:subvector_end])
        start_index = subvector_end

    return subvectors



def split_multivariate_sequences(sequences, n_steps_in, n_steps_out,to_predict):
    """
    Convert multivariate time series data into supervised learning samples.

    Parameters:
    - sequences (np.ndarray): Input time series data (n_samples, n_features).
    - n_steps_in (int): Input sequence length.
    - n_steps_out (int): Output prediction length.
    - to_predict (int): Index of the target feature to predict.

    Returns:
    - Tuple[np.ndarray, np.ndarray]: Input (X) and output (y) samples.
    """
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the dataset
        if out_end_ix <= len(sequences):
            # gather input and output parts of the pattern
            seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, to_predict]
            X.append(seq_x)
            y.append(seq_y)
    return np.array(X), np.array(y)


def df_to_supervised_routes(df_train,scaler,n_steps_in,n_features,n_steps_out):
    """
    Convert structured time series data by route, date, and block into supervised format.

    Parameters:
    - df_train (pd.DataFrame): Dataframe containing training data.
    - scaler (object): Fitted scaler (e.g., MinMaxScaler).
    - n_steps_in (int): Length of input sequence.
    - n_features (int): Number of input features.
    - n_steps_out (int): Length of output sequence.

    Returns:
    - Tuple[np.ndarray, np.ndarray]: Supervised inputs and outputs.
    """
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
    

def preprocessing_s1(areas,split,n_train_start,n_val,n_test,df,path_data,scaler,n_steps_in,n_features,n_steps_out):
    """
    Preprocess data for scenario S1 where each area is assigned to a separate client.

    Saves processed .npy files to disk for training, validation, and testing.

    Parameters:
    - areas (List): List of area identifiers.
    - split (int): Current time split.
    - n_train_start (int): Initial number of training weeks.
    - n_val (int), n_test (int): Validation and test window sizes.
    - df (pd.DataFrame): Full dataset.
    - path_data (str): Path to save preprocessed data.
    - scaler (object): Fitted scaler.
    - n_steps_in (int): Length of input sequence.
    - n_features (int): Number of input features.
    - n_steps_out (int): Length of output sequence.
    """
    for i in range(len(areas)): #working area by area
        a = areas[i]
        df_area =  df[df['area']==a]
        weeks = np.array(df_area.week.unique())

        # time-based cross validation
        train_end = n_train_start + split # training set grows split by split
        val_end = train_end + n_val
        test_end = val_end + n_test

        train_weeks = weeks[:train_end]  # First 'train_end' weeks for training
        val_weeks = weeks[train_end:val_end]  # The following 'n_val' week for validation
        test_weeks = weeks[val_end:test_end]  # The following 'n_test' week for validation

        # filtering the DataFrame for each split
        df_train = df_area[df_area['week'].isin(train_weeks)]
        df_train.sort_values(by='timestamp',inplace=True)
        df_val = df_area[df_area['week'].isin(val_weeks)]
        df_val.sort_values(by='timestamp',inplace=True)
        df_test = df_area[df_area['week'].isin(test_weeks)]
        df_test.sort_values(by='timestamp',inplace=True)
        
        
        # generating train, validation and test supervised series:
        X_train, y_train = df_to_supervised_routes(df_train,scaler,n_steps_in,n_features,n_steps_out)
        X_train = X_train.astype(np.float32)
        y_train = y_train.astype(np.float32)
        path_X_train = f'{path_data}xtrain_{i}_split{split}.npy'
        path_y_train = f'{path_data}ytrain_{i}_split{split}.npy'
        np.save(path_X_train,X_train)
        np.save(path_y_train,y_train)

        X_val, y_val = df_to_supervised_routes(df_val,scaler,n_steps_in,n_features,n_steps_out)
        X_val = X_val.astype(np.float32)
        y_val = y_val.astype(np.float32)
        path_X_val = f'{path_data}xval_{i}_split{split}.npy'
        path_y_val = f'{path_data}yval_{i}_split{split}.npy'
        np.save(path_X_val,X_val)
        np.save(path_y_val,y_val)

        X_test, y_test = df_to_supervised_routes(df_test,scaler,n_steps_in,n_features,n_steps_out)   
        X_test = X_test.astype(np.float32)
        y_test = y_test.astype(np.float32)
        path_X_test = f'{path_data}xtest_{i}_split{split}.npy'
        path_y_test = f'{path_data}ytest_{i}_split{split}.npy'
        np.save(path_X_test,X_test)
        np.save(path_y_test,y_test)


def preprocessing_s2(num_clients,routes_per_area,seeds,split,n_train_start,n_val,n_test,df,path_data,scaler,n_steps_in,n_features,n_steps_out):
    """
    Preprocess data for scenario S2 where each area is divided among multiple clients.

    Parameters:
    - num_clients (int): Number of clients per area.
    - routes_per_area (List[List[str]]): Routes grouped by area.
    - seeds (List[int]): Random seeds for reproducibility across splits.
    - n_train_start (int): Initial number of training weeks.
    - n_val (int), n_test (int): Validation and test window sizes.
    - df (pd.DataFrame): Full dataset.
    - path_data (str): Path to save preprocessed data.
    - scaler (object): Fitted scaler.
    - n_steps_in (int): Length of input sequence.
    - n_features (int): Number of input features.
    - n_steps_out (int): Length of output sequence.
    """
    weeks = np.array(df.week.unique())
    all_routes = np.array(df.route_id.unique())

    # time-based cross validation
    train_end = n_train_start + split # training set grows split by split
    val_end = train_end + n_val
    test_end = val_end + n_test

    train_weeks = weeks[:train_end]  # First 'train_end' weeks for training
    val_weeks = weeks[train_end:val_end]  # The following 'n_val' week for validation
    test_weeks = weeks[val_end:test_end]  # The following 'n_test' week for validation
    
    # Filter the DataFrame for each split
    df_train = df[df['week'].isin(train_weeks)]
    df_train.sort_values(by='timestamp',inplace=True)
    df_val = df[df['week'].isin(val_weeks)]
    df_val.sort_values(by='timestamp',inplace=True)
    df_test = df[df['week'].isin(test_weeks)]
    df_test.sort_values(by='timestamp',inplace=True)

    # partitioning all the routes for the number of clients:
    all_routes_divided = []
    rng = np.random.default_rng(seed = seeds[split])
    
    for i in range(len(all_routes)):
        routes_area = routes_per_area[i].copy() #taking the routes of the area
        rng.shuffle(routes_area) 
        routes_clients_area_i = divide_vector(routes_area, num_clients) # divide the routes in num_clients set
        all_routes_divided.append(routes_clients_area_i)

    temp_routes_per_clients = []
    routes_clients = []
    
    for i in range(num_clients): # working client by client
        routes_clients_i = []
        for j in range(len(all_routes)): # a sublist for each area
            if j < len(all_routes)/2:
                routes_clients_i.append(all_routes_divided[j][i])
            else: 
                routes_clients_i.append(all_routes_divided[j][-i])
        temp_routes_per_clients.append(routes_clients_i)
        flat_list_i = [item for sublist in temp_routes_per_clients[i] for item in sublist] #creating a single list per client
        routes_clients.append(flat_list_i) # append the list of the single client into a list of all the clients

        df_train_i = df_train[df_train['route_id'].isin(routes_clients[i])]
        df_val_i = df_val[df_val['route_id'].isin(routes_clients[i])]
        df_test_i = df_test[df_test['route_id'].isin(routes_clients[i])]
        
        # generating train, validation and test supervised series:
        X_train, y_train = df_to_supervised_routes(df_train_i,scaler,n_steps_in,n_features,n_steps_out)
        X_train = X_train.astype(np.float32)
        y_train = y_train.astype(np.float32)
        path_X_train = f'{path_data}xtrain_{i}_split{split}_s2.npy'
        path_y_train = f'{path_data}ytrain_{i}_split{split}_s2.npy'
        np.save(path_X_train,X_train)
        np.save(path_y_train,y_train)

        X_val, y_val = df_to_supervised_routes(df_val_i,scaler,n_steps_in,n_features,n_steps_out)
        X_val = X_val.astype(np.float32)
        y_val = y_val.astype(np.float32)
        path_X_val = f'{path_data}xval_{i}_split{split}_s2.npy'
        path_y_val = f'{path_data}yval_{i}_split{split}_s2.npy'
        np.save(path_X_val,X_val)
        np.save(path_y_val,y_val)

        X_test, y_test = df_to_supervised_routes(df_test_i,scaler,n_steps_in,n_features,n_steps_out)   
        X_test = X_test.astype(np.float32)
        y_test = y_test.astype(np.float32)
        path_X_test = f'{path_data}xtest_{i}_split{split}_s2.npy'
        path_y_test = f'{path_data}ytest_{i}_split{split}_s2.npy'
        np.save(path_X_test,X_test)
        np.save(path_y_test,y_test)


def persistence(last_ob, n_seq):
    """
    Generate a naive (persistence) forecast by repeating the last observed value.

    Parameters:
    - last_ob (float): The last observed value in the time series.
    - n_seq (int): Number of steps to forecast.

    Returns:
    - List[float]: Forecasted values repeated over the horizon.
    """
    return [last_ob for i in range(n_seq)]


def make_forecasts_naive_forecasting(test, n_lag, n_seq):
    """
    Apply naive forecasting to a test dataset using the persistence model.

    Parameters:
    - test (np.ndarray): Array of test sequences (each with lag + future steps).
    - n_lag (int): Number of lag steps used for input.
    - n_seq (int): Number of future steps to predict.

    Returns:
    - List[List[float]]: Forecasts for each sequence in the test set.
    """
    test = np.array(test)
    forecasts = list()
    for i in range(len(test)):
        X, y = test[i, 0:n_lag], test[i, n_lag:]
        # make forecast
        forecast = persistence(X[-1], n_seq)
        # store the forecast
        forecasts.append(forecast)
    return forecasts


def evaluate_forecasts_naive_forecasting(test, forecasts, n_lag, n_seq):
  """
    Evaluate naive forecasts using RMSE and MAE per prediction horizon.

    Parameters:
    - test (np.ndarray): Ground truth sequences, including both lag and forecast steps.
    - forecasts (List[List[float]]): Forecasted sequences from the naive model.
    - n_lag (int): Number of input steps (used to align forecasts with actuals).
    - n_seq (int): Number of prediction steps (forecast horizon).

    Returns:
    - Tuple[pd.DataFrame, pd.DataFrame]: 
        - AE: Absolute Errors per horizon
        - SE: Squared Errors per horizon
    """
  AE=pd.DataFrame()
  SE=pd.DataFrame()
  test = np.array(test)
  forecasts = np.array(forecasts)
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


def evaluate_forecasts(y_test, forecasts, n_lag, n_seq):
    """
    Evaluate forecast accuracy using RMSE and MAE for each forecast horizon.

    Parameters:
    - y_test (List[List[float]]): True values for each forecast step.
    - forecasts (List[List[float]]): Predicted values.
    - n_lag (int): Number of lag steps (not used directly here but retained for consistency).
    - n_seq (int): Number of forecast steps (horizons).

    Returns:
    - Tuple[pd.DataFrame, pd.DataFrame]: 
        - AE: Absolute errors per forecast step.
        - SE: Squared errors per forecast step.
    """
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
    Compute mean absolute percentage error (MAPE) with a safeguard for small values.

    Parameters:
    - y_true (array-like): True values.
    - y_pred (array-like): Predicted values.
    - epsilon (float): Small constant to avoid division by zero.

    Returns:
    - float: MAPE value in percentage.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    absolute_percentage_error = np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon))
    return np.mean(absolute_percentage_error) * 100    
    

def evaluate_MAPE_forecasts(y_test, forecasts, n_lag, n_seq):
    """
    Evaluate MAPE for each forecast horizon using custom error function.

    Parameters:
    - y_test (List[List[float]]): True output values.
    - forecasts (List[List[float]]): Predicted output values.
    - n_lag (int): Lag length (not used here).
    - n_seq (int): Forecast horizon length.

    Returns:
    - pd.DataFrame: Absolute percentage error per forecast step.
    """
    APE=pd.DataFrame()
    for i in range(n_seq):
        actual = [row[i] for row in y_test]
        predicted = [forecast[i] for forecast in forecasts]
        APEi=pd.DataFrame()
        absolute_percentage_error=[custom_mean_absolute_percentage_error([actual[j]],[predicted[j]]) for j in range(len(actual))]
        APEi['Absolute Percentage Error']=absolute_percentage_error
        APEi['Horizon'] = pd.Series(['horizon = %d ' % (i+1) for x in range(len(APEi.index))])
        APE=pd.concat([APE,APEi],ignore_index=True)
    return APE

def evaluate_MAPE_forecasts_naive_forecasting(y_test, forecasts, n_seq):
    """
    Evaluate MAPE for naive (persistence-based) forecasts.

    Parameters:
    - y_test (np.ndarray): True values, shape = (samples, forecast_horizon).
    - forecasts (List[List[float]]): Predicted values, shape = (samples, forecast_horizon).
    - n_seq (int): Forecast horizon.

    Returns:
    - pd.DataFrame: Absolute percentage error per forecast horizon.
    """

    APE=pd.DataFrame()
    y_test = np.array(y_test)
    forecasts = np.array(forecasts)
    for i in range(n_seq):
        actual = y_test[:,i]
        predicted = [forecast[i] for forecast in forecasts]
        APEi=pd.DataFrame()
        absolute_percentage_error=[custom_mean_absolute_percentage_error([actual[j]],[predicted[j]]) for j in range(len(actual))]
        APEi['Absolute Percentage Error']=absolute_percentage_error
        APEi['Horizon'] = pd.Series(['horizon = %d ' % (i+1) for x in range(len(APEi.index))])
        APE=pd.concat([APE,APEi],ignore_index=True)
    return APE

def inverse_transform(scaler, preds):
    """
    Invert scaling applied to prediction values using the original scaler structure.

    Assumes scaler expects a 6-feature input:
    ['occupancy', 'stop_id', 'month', 'weekday', 'timeslot', 'route_id']

    Parameters:
    - scaler: A fitted scaler (e.g., MinMaxScaler) used during training.
    - preds (np.ndarray): Scaled predictions, shape = (samples, forecast_horizon)

    Returns:
    - np.ndarray: Inverse-transformed values of shape (samples, forecast_horizon)
    """
    # Initialize an empty matrix
    inverted = np.empty((preds.shape[0], 0))
    for i in range(preds.shape[1]):
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
    """
    Perform Augmented Dickey-Fuller test for stationarity.

    Parameters:
    - series (array-like): Time series data.

    Returns:
    - float: p-value of the test (used to determine stationarity).
    """
    result = adfuller(series)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    return result[1]  # Return p-value

# evaluate the persistence model
def make_arima_forecasts(train,test, n_lag, n_seq, order):
    """
    Generate multi-step forecasts using ARIMA.

    Parameters:
    - train (List[float]): Training time series.
    - test (List[float]): Test data series.
    - n_lag (int): Size of historical window to use.
    - n_seq (int): Forecast horizon (number of future steps).
    - order (Tuple[int]): ARIMA order (p, d, q).

    Returns:
    - forecasts (List[List[float]]): Predicted values per step.
    - actuals (List[List[float]]): Ground truth values aligned with forecasts.
    """
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
    """
    Evaluate ARIMA forecasts using absolute and squared error per time step.

    Parameters:
    - test (np.ndarray): True values, shape = (samples, forecast_horizon).
    - forecasts (List[List[float]]): Forecasted values.
    - n_lag (int): Historical window size (unused but retained for compatibility).
    - n_seq (int): Forecast horizon.

    Returns:
    - Tuple[pd.DataFrame, pd.DataFrame]: 
        - AE: Absolute Error
        - SE: Squared Error
    """
    AE=pd.DataFrame()
    SE=pd.DataFrame()
    test = np.array(test)
    forecasts = np.array(forecasts)
    for i in range(n_seq):
        actual = test[:,i]
        predicted = [forecast[i] for forecast in forecasts]
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
    """
    Evaluate ARIMA forecasts using MAPE per forecast horizon.

    Parameters:
    - y_test (np.ndarray): Ground truth values.
    - forecasts (List[List[float]]): Predicted values.
    - n_lag (int): Unused.
    - n_seq (int): Forecast horizon.

    Returns:
    - pd.DataFrame: Absolute Percentage Error per forecast step.
    """
    APE=pd.DataFrame()
    y_test = np.array(y_test)
    forecasts = np.array(forecasts)
    for i in range(n_seq):
        actual = y_test[:,i]
        predicted = [forecast[i] for forecast in forecasts]
        APEi=pd.DataFrame()
        absolute_percentage_error=[custom_mean_absolute_percentage_error([actual[j]],[predicted[j]]) for j in range(len(actual))]
        APEi['Absolute Percentage Error']=absolute_percentage_error
        APEi['Horizon'] = pd.Series(['horizon = %d ' % (i+1) for x in range(len(APEi.index))])
        APE=pd.concat([APE,APEi],ignore_index=True)
    return APE


def euclidean_distance(val1, val2):
    """
    Compute absolute (Euclidean for scalars) distance between two values.

    Parameters:
    - val1, val2 (float): Numeric values.

    Returns:
    - float: Absolute distance.
    """
    return abs(val1 - val2)
    

def normalize_distances(distances):
    """
    Normalize an array of distances to [0, 1] range (as similarity scores).

    Parameters:
    - distances (np.ndarray): Raw distances.

    Returns:
    - np.ndarray: Similarity scores, where higher is more similar.
    """
    max_dist = np.max(distances)
    min_dist = np.min(distances)
    return (max_dist - distances) / (max_dist - min_dist)
   

def min_max_normalize(matrix):
    """
    Normalize a matrix to the [0, 1] range using Min-Max scaling.

    Parameters:
    - matrix (np.ndarray): Matrix of values.

    Returns:
    - np.ndarray: Normalized matrix.
    """
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    normalized_matrix = (matrix - min_val) / (max_val - min_val)
    return normalized_matrix