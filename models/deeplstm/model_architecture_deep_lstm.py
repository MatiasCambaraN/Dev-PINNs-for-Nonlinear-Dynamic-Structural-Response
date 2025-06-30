from keras.models import Sequential
from keras.layers import CuDNNLSTM, Dense, Activation, Dropout
from keras.optimizers import Adam

def build_lstm_model(input_dim,
                     output_dim,
                     model_type='LSTM-s',
                     lstm_units=100,
                     num_lstm_layers=2,
                     dense_units=100,
                     use_dropout=False,
                     dropout_rate=0.2,
                     activation='relu',
                     use_final_relu=True):
    """
    Builds a customizable LSTM model with default settings matching the provided architecture.
    
    Args:
        input_dim (int): Number of input features.
        output_dim (int): Number of output features.
        model_type (str): 'LSTM-s' applies ReLU after dense layer, otherwise skipped.
        lstm_units (int): Units per LSTM layer.
        num_lstm_layers (int): Number of CuDNNLSTM layers.
        dense_units (int): Units in dense layer before output.
        use_dropout (bool): Whether to add dropout after LSTM layers.
        dropout_rate (float): Dropout rate if enabled.
        activation (str): Activation function after LSTM layers.
        use_final_relu (bool): Applies ReLU after Dense if True or if model_type == 'LSTM-s'.
        learning_rate (float): Initial learning rate.
        decay (float): Learning rate decay.

    Returns:
        model (Sequential): Compiled Keras model.
    """
    model = Sequential()

    # First LSTM layer with input shape
    model.add(CuDNNLSTM(lstm_units, return_sequences=True, stateful=False, input_shape=(None, input_dim)))
    model.add(Activation(activation))

    # Additional LSTM layers
    for _ in range(num_lstm_layers - 1):
        model.add(CuDNNLSTM(lstm_units, return_sequences=True, stateful=False))
        model.add(Activation(activation))
        if use_dropout:
            model.add(Dropout(dropout_rate))

    # Dense + optional ReLU
    model.add(Dense(dense_units))
    if use_final_relu or model_type == 'LSTM-s':
        model.add(Activation('relu'))

    model.add(Dense(output_dim))

    # Optimizer and compilation
    optimizer = Adam(learning_rate=0.001, decay=0.0001) #RMSprop(lr=0.001, decay=0.0001)
    model.compile(loss='mean_squared_error', 
                  optimizer=optimizer, 
                  metrics=['mse'])

    model.summary()
    return model



