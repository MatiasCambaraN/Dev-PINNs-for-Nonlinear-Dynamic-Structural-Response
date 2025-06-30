import os
import time
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import tensorflow as tf
from random import shuffle

def train_model( X_data_new, y_data_new, model, gpu_id=0, batch_size=10, epochs=50000, model_path='my_best_model.h5'):
    """
    Train a Keras model with the provided data and save the best model based on validation loss.
    Parameters:
    - X_data_new: Input data for training (numpy array).
    - y_data_new: Target data for training (numpy array).
    - model: Keras model to be trained.
    - gpu_id: ID of the GPU to use for training.
    - batch_size: Number of samples per gradient update.
    - epochs: Number of epochs to train the model.
    - model_path: Path to save the best model.
    Returns:
    - A dictionary containing the path to the best model, training loss, validation loss, and best loss.
    """
    best_loss = 100
    train_loss = []
    test_loss = []
    #history = []

    with tf.device(f'/device:GPU:{gpu_id}'):

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.gpu_options.per_process_gpu_memory_fraction = 0.4
        session = tf.Session(config=config)
        # tf.Session(config=tf.ConfigProto(log_device_placement=True))

        start = time.time()

        
        for e in range(epochs):
            print('epoch = ', e + 1)

            Ind = list(range(len(X_data_new)))
            shuffle(Ind)
            ratio_split = 0.7
            Ind_train = Ind[0:round(ratio_split * len(X_data_new))]
            Ind_test = Ind[round(ratio_split * len(X_data_new)):]

            X_train = X_data_new[Ind_train]
            y_train = y_data_new[Ind_train]
            X_test = X_data_new[Ind_test]
            y_test = y_data_new[Ind_test]

            model.fit(X_train, y_train,
                    batch_size=batch_size,
                    # validation_split=0.2,
                    validation_data=(X_test, y_test),
                    shuffle=True,
                    epochs=1)
            score0 = model.evaluate(X_train, y_train, batch_size=batch_size, verbose=0)
            score = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
            train_loss.append(score0[0])
            test_loss.append(score[0])

            # Save the best trained model with minimum testing loss
            if test_loss[e] < best_loss:
                best_loss = test_loss[e]
                model.save(model_path)
                
                # Guardar perdidas de entrenamiento y validación
                np.savez(os.path.splitext(model_path)[0] + '_loss.npz', train_loss=np.array(train_loss), test_loss=np.array(test_loss))
                print(f"Best model saved with loss: {best_loss:.4f}")

        end = time.time()
        running_time = (end - start)/3600
        print('Running Time: ', running_time, ' hour')
        
    # Plot training and testing loss
    #plt.figure()
    #plt.plot(np.array(train_loss), 'b-')
    #plt.plot(np.array(test_loss), 'm-')

    # plt.figure()
    # plt.plot(np.log(np.array(train_loss)), 'b-')
    # plt.plot(np.log(np.array(test_loss)), 'm-')

    return {
        'model': model,
        'train_loss':np.array(train_loss),
        'val_loss':np.array(test_loss),
        'best_loss': best_loss
    }


def plot_losses(train_loss, val_loss, title='Train vs Validation Loss'):
    plt.figure()
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    


def load_best_model(model_path):
    """
    Load the best model from the specified path.
    Parameters:
    - model_path: Path to the saved model.
    Returns:
    - The loaded Keras model.
    """
    if os.path.exists(model_path):
        return load_model(model_path)
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")


def load_loss_data(model_path):
    """
    Load training and validation loss data from a .npz file.
    Parameters:
    - model_path: Path to the saved model (without extension).
    Returns:
    - A dictionary containing train_loss and val_loss arrays.
    """
    loss_file = os.path.splitext(model_path)[0] + '_loss.npz'
    if os.path.exists(loss_file):
        data = np.load(loss_file)
        return {
            'train_loss': data['train_loss'],
            'val_loss': data['test_loss']
        }
    else:
        raise FileNotFoundError(f"Loss file not found at {loss_file}")