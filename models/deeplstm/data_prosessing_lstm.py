import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib #from sklearn.externals import joblib  # save scaler
import matplotlib.pyplot as plt

# Process full sequence to stacked sequence
def Generate_data(X_data0, y_data0, window_size=50):
    """
    Process a full sequence into stacked subsequences of fixed window size.

    Args:
        X_data0 (list or ndarray): List or array of input sequences.
        y_data0 (list or ndarray): List or array of corresponding target sequences.
        window_size (int): Number of timesteps per subsequence window. Default is 50.

    Returns:
        tuple:
            X_data_new0 (ndarray): Array of stacked input windows with shape (batch, num_windows, window_size).
            y_data_new0 (ndarray): Array of target values at the end of each window with shape (batch, num_windows, features).
    """
    X_new_temp = []
    y_new_temp = []
    for ii in range(len(X_data0)):
        X_temp = X_data0[ii]
        y_temp = y_data0[ii]
        X_new = []
        y_new = []
        for jj in range(int(np.floor(len(X_temp) / window_size))):
            X_new.append(X_temp[jj * window_size:(jj + 1) * window_size])
            y_new.append(y_temp[(jj + 1) * window_size - 1, :])
            # y_new.append(y_temp[(jj + 1) * window_size - 1])

        X_new_temp.append(np.array(X_new))
        y_new_temp.append(np.array(y_new))

    X_data_new0 = np.array(X_new_temp)
    y_data_new0 = np.array(y_new_temp)

    return X_data_new0, y_data_new0

def preprocess_data_lstm(X_data, y_data, train_indices, X_pred, y_pred_ref, model_type='LSTM-s', windowsize=10, scaler_path=None):
    """
    Preprocess data for LSTM models, scaling and optionally stacking windows.

    Args:
        X_data (ndarray): Training input data of shape (samples, timesteps, features).
        y_data (ndarray): Training target data of shape (samples, timesteps, features).
        train_indices (array-like): Indices defining the training subset.
        X_pred (ndarray): Input data for prediction.
        y_pred_ref (ndarray): True target data for prediction reference.
        model_type (str): 'LSTM-s' for stacked windows or 'LSTM-f' for flat sequence. Default 'LSTM-s'.
        windowsize (int): Window size for stacking when using 'LSTM-s'. Default is 10.
        scaler_path (str): Directory path to save fitted scalers. Default is None.

    Returns:
        tuple:
            input_dim (int): Number of input features.
            timesteps (int): Number of timesteps per input sequence.
            output_dim (int): Number of output features.
            X_data_new (ndarray): Preprocessed input data for training/validation.
            y_data_new (ndarray): Preprocessed target data for training/validation.
            X_train (ndarray): Training input data subset.
            y_train (ndarray): Training target data subset.
            X_test (ndarray): Validation input data subset.
            y_test (ndarray): Validation target data subset.
            X_pred (ndarray): Preprocessed input data for prediction.
            y_pred_ref (ndarray): Preprocessed target reference for prediction.
            scaler_X (MinMaxScaler): Fitted scaler for input data.
            scaler_y (MinMaxScaler): Fitted scaler for target data.
    """
    # Scale data
    X_data_flatten = np.reshape(X_data, [X_data.shape[0]*X_data.shape[1], 1])
    scaler_X = MinMaxScaler(feature_range=(-1, 1))
    scaler_X.fit(X_data_flatten)
    X_data_flatten_map = scaler_X.transform(X_data_flatten)
    X_data_map = np.reshape(X_data_flatten_map, [X_data.shape[0], X_data.shape[1], 1])

    y_data_flatten = np.reshape(y_data, [y_data.shape[0]*y_data.shape[1], y_data.shape[2]])
    scaler_y = MinMaxScaler(feature_range=(-1, 1))
    scaler_y.fit(y_data_flatten)
    y_data_flatten_map = scaler_y.transform(y_data_flatten)
    y_data_map = np.reshape(y_data_flatten_map, [y_data.shape[0], y_data.shape[1], y_data.shape[2]])

    # Unknown data
    #X_pred = mat['input_pred_tf']
    #y_pred_ref = mat['target_pred_tf']
    # X_pred = np.reshape(X_pred, [X_pred.shape[0], X_pred.shape[1], 1])
    # y_pred_ref = np.reshape(y_pred_ref, [y_pred_ref.shape[0], y_pred_ref.shape[1], 1])


    # Scale data
    X_pred_flatten = np.reshape(X_pred, [X_pred.shape[0]*X_pred.shape[1], 1])
    X_pred_flatten_map = scaler_X.transform(X_pred_flatten)
    X_pred_map = np.reshape(X_pred_flatten_map, [X_pred.shape[0], X_pred.shape[1], 1])

    y_pred_ref_flatten = np.reshape(y_pred_ref, [y_pred_ref.shape[0]*y_pred_ref.shape[1], y_pred_ref.shape[2]])
    y_pred_ref_flatten_map = scaler_y.transform(y_pred_ref_flatten)
    y_pred_ref_map = np.reshape(y_pred_ref_flatten_map, [y_pred_ref.shape[0], y_pred_ref.shape[1], y_pred_ref.shape[2]])

    if model_type == 'LSTM-s':
        X_data_new, y_data_new = Generate_data(X_data_map, y_data_map, windowsize)
        X_data_new = np.reshape(X_data_new, [X_data_new.shape[0], X_data_new.shape[1], X_data_new.shape[2]])
        # y_data = np.reshape(y_data, [y_data.shape[0], y_data.shape[1], 1])

    elif model_type == 'LSTM-f':
        X_data_new = X_data_map
        y_data_new = y_data_map

    X_train = X_data_new[0:len(train_indices[0])]  # Training set
    y_train = y_data_new[0:len(train_indices[0])]  # Training set
    X_test = X_data_new[len(train_indices[0]):]    # Validation set
    y_test = y_data_new[len(train_indices[0]):]    # Validation set

    # Testing set
    if model_type == 'LSTM-s':
        X_pred, y_pred_ref = Generate_data(X_pred_map, y_pred_ref_map, windowsize)
        X_pred = np.reshape(X_pred, [X_pred.shape[0], X_pred.shape[1], X_pred.shape[2]])
    elif model_type == 'LSTM-f':
        X_pred = X_pred_map         
        y_pred_ref = y_pred_ref_map

    input_dim = X_train.shape[2]  # number of input features
    timesteps = X_train.shape[1]
    output_dim = y_train.shape[2]  # number of output features
    
    
    # Save scaler
    joblib.dump(scaler_X, scaler_path+'/scaler_X.save')
    joblib.dump(scaler_y, scaler_path+'/scaler_y.save')
    # And now to load...
    # scaler_X = joblib.load(scaler_path+'/scaler_X.save')
    # scaler_y = joblib.load(scaler_path+'/scaler_y.save')
    
    return input_dim, timesteps, output_dim, X_data_new, y_data_new, X_train, y_train, X_test, y_test, X_pred, y_pred_ref, scaler_X, scaler_y






def postprocess_data_lstm(scaler_y, y_train, y_train_pred, y_test, y_test_pred, y_pred_ref, y_pure_preds):
    """
    Inverse transform predictions using the fitted target scaler.

    Args:
        scaler_y (MinMaxScaler): Fitted scaler for target data.
        y_train (ndarray): True training target data.
        y_train_pred (ndarray): Predicted training data to inverse transform.
        y_test (ndarray): True validation target data.
        y_test_pred (ndarray): Predicted validation data to inverse transform.
        y_pred_ref (ndarray): True prediction reference target data.
        y_pure_preds (ndarray): Predicted values for prediction data.

    Returns:
        tuple:
            y_train_pred (ndarray): Inverse transformed training predictions.
            y_test_pred (ndarray): Inverse transformed validation predictions.
            y_pure_preds (ndarray): Inverse transformed prediction values.
    """
    # X_train = X_data_new[0:len(train_indices[0])]
    # y_train = y_data_new[0:len(train_indices[0])]
    # X_test = X_data_new[len(train_indices[0]):]
    # y_test = y_data_new[len(train_indices[0]):]

    # y_train_pred = model_best.predict(X_train)
    # y_test_pred = model_best.predict(X_test)
    # y_pure_preds = model_best.predict(X_pred)

    # Reverse map to original magnitude
    # X_train_orig = X_data[0:len(train_indices[0])]
    # y_train_orig = y_data[0:len(train_indices[0])]
    # X_test_orig = X_data[len(train_indices[0]):]
    # y_test_orig = y_data[len(train_indices[0]):]
    # X_pred_orig = mat['input_pred_tf']
    # y_pred_ref_orig = mat['target_pred_tf']    
    
    y_train_pred_flatten = np.reshape(y_train_pred, [y_train_pred.shape[0]*y_train_pred.shape[1], y_train_pred.shape[2]])
    y_train_pred = scaler_y.inverse_transform(y_train_pred_flatten)
    y_train_pred = np.reshape(y_train_pred, [y_train.shape[0], y_train.shape[1], y_train.shape[2]])

    y_test_pred_flatten = np.reshape(y_test_pred, [y_test_pred.shape[0]*y_test_pred.shape[1], y_test_pred.shape[2]])
    y_test_pred = scaler_y.inverse_transform(y_test_pred_flatten)
    y_test_pred = np.reshape(y_test_pred, [y_test.shape[0], y_test.shape[1], y_test.shape[2]])

    y_pure_preds_flatten = np.reshape(y_pure_preds, [y_pure_preds.shape[0]*y_pure_preds.shape[1], y_pure_preds.shape[2]])
    y_pure_preds = scaler_y.inverse_transform(y_pure_preds_flatten)
    y_pure_preds = np.reshape(y_pure_preds, [y_pred_ref.shape[0], y_pred_ref.shape[1], y_pred_ref.shape[2]])

    return  y_train_pred, y_test_pred, y_pure_preds




