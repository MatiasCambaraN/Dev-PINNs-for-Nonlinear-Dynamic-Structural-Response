import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv1D, Activation, Dense
import numpy as np
import time
import matplotlib.pyplot as plt

class DeepPhyCNNutt:
    # Initialize the class
    def __init__(self, eta_tt, ag, Phi_t,
                 num_filters=64,
                 kernel_size=50,
                 num_conv_layers=5,
                 num_dense_layers=2,
                 dense_units=50,
                 activation='relu'):

        # Save data 
        self.eta_tt = eta_tt
        self.ag = ag
        # self.lift = lift
        # self.ag_c = ag_c
        self.Phi_t = Phi_t
        # self.Phi_tt = Phi_tt
        
        # Save hyperparameters
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.num_conv_layers = num_conv_layers
        self.num_dense_layers = num_dense_layers
        self.dense_units = dense_units
        self.activation = activation

        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        # placeholders for data
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        # self.eta_tf = tf.placeholder(tf.float32, shape=[None, None, self.eta.shape[2]])
        # self.eta_t_tf = tf.placeholder(tf.float32, shape=[None, None, self.eta.shape[2]])
        self.eta_tt_tf = tf.placeholder(tf.float32, shape=[None, None, self.eta_tt.shape[2]])
        self.ag_tf = tf.placeholder(tf.float32, shape=[None, None, 1])

        # physics informed neural networks
        self.eta_pred, self.eta_t_pred, self.eta_tt_pred, = self.net_structure(self.ag_tf)

        # loss
        # for measurements
        self.loss = tf.reduce_mean(tf.square(self.eta_tt_tf - self.eta_tt_pred)) + tf.reduce_mean(tf.square(self.eta_pred[:,:,0:10]))

        # optimizers
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 20000,
                                                                         'maxfun': 50000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1 * np.finfo(float).eps})

        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        self.train_op = self.optimizer_Adam.minimize(self.loss)

        # Init session
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def CNN_model(self, X):
        model = Sequential()

        # Add configurable Conv1D layers
        for _ in range(self.num_conv_layers):
            model.add(Conv1D(self.num_filters, self.kernel_size, strides=1,
                             padding='same', use_bias=True, input_shape=(None, 1)))
            model.add(Activation(self.activation))

        # Add configurable Dense layers
        for _ in range(self.num_dense_layers):
            model.add(Dense(self.dense_units))
            model.add(Activation(self.activation))

        # Output layer
        model.add(Dense(self.eta_tt.shape[2]))

        model.summary()
        return model(X)

    def net_structure(self, ag):
        eta = self.CNN_model(ag)

        Phi_ut = np.reshape(self.Phi_t, [1, self.eta_tt.shape[1], self.eta_tt.shape[1]])
        Phi_ut = np.repeat(Phi_ut, self.eta_tt.shape[0], axis=0)
        eta_t = tf.matmul(tf.cast(Phi_ut, dtype=tf.float32), eta)
        eta_tt = tf.matmul(tf.cast(Phi_ut, dtype=tf.float32), eta_t)

        return eta, eta_t, eta_tt
    
    def train(self, num_epochs, batch_size, learning_rate, bfgs):

        Loss = []

        for epoch in range(num_epochs):
            
            N = self.eta_tt.shape[0]

            start_time = time.time()
            for it in range(0, N, batch_size):
                tf_dict = {self.eta_tt_tf: self.eta_tt, self.ag_tf: self.ag, self.learning_rate: learning_rate}
                self.sess.run(self.train_op, tf_dict)
                
                # Print
                if it % (10*batch_size) == 0:
                    elapsed = time.time() - start_time
                    loss_value, learning_rate_value = self.sess.run([self.loss, self.learning_rate], tf_dict)
                    print('Epoch: %d, It: %d, Loss: %.3e, Time: %.2f, Learning Rate: %.3e'
                          %(epoch, it/batch_size, loss_value, elapsed, learning_rate_value))
                    start_time = time.time()

            Loss.append(self.sess.run(self.loss, tf_dict))

        if bfgs == 1:
            tf_dict_all = {self.eta_tt_tf: self.eta_tt, self.ag_tf: self.ag, self.learning_rate: learning_rate}

            self.optimizer.minimize(self.sess,
                                    feed_dict=tf_dict_all,
                                    fetches=[self.loss],
                                    loss_callback=self.callback)

            Loss.append(self.sess.run(self.loss, tf_dict))

        return Loss

    def callback(self, loss):
        print('Loss:', loss)

    def predict(self, ag_star):
        
        tf_dict = {self.ag_tf: ag_star}

        eta_star = self.sess.run(self.eta_pred, tf_dict)
        eta_t_star = self.sess.run(self.eta_t_pred, tf_dict)
        eta_tt_star = self.sess.run(self.eta_tt_pred, tf_dict)

        return eta_star, eta_t_star, eta_tt_star
    
    

class DeepPhyCNNu:
    # Initialize the class
    def __init__(self, eta, eta_t, g, ag, lift, Phi_t,
                 num_filters=64,
                 kernel_size=40,
                 num_conv_layers=5,
                 num_dense_layers=2,
                 dense_units=50,
                 activation='relu'):

        # Save data
        self.eta = eta
        self.eta_t = eta_t
        # self.eta_tt = eta_tt
        self.g = g
        self.ag = ag
        self.lift = lift
        # self.ag_c = ag_c
        self.Phi_t = Phi_t
        # self.Phi_tt = Phi_tt
        
        # Save hyperparameters
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.num_conv_layers = num_conv_layers
        self.num_dense_layers = num_dense_layers
        self.dense_units = dense_units
        self.activation = activation

        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        # placeholders for data
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.eta_tf = tf.placeholder(tf.float32, shape=[None, None, self.eta.shape[2]])
        self.eta_t_tf = tf.placeholder(tf.float32, shape=[None, None, self.eta.shape[2]])
        self.g_tf = tf.placeholder(tf.float32, shape=[None, None, self.g.shape[2]])
        self.ag_tf = tf.placeholder(tf.float32, shape=[None, None, 1])
        self.lift_tf = tf.placeholder(tf.float32, shape=[None, None, 1])

        # physics informed neural networks
        self.eta_pred, self.eta_t_pred, self.eta_dot_pred, self.g_pred, self.lift_pred = self.net_structure(self.ag_tf)

        # loss
        # for measurements
        self.loss_u = tf.reduce_mean(tf.square(self.eta_tf - self.eta_pred))
        self.loss_udot = tf.reduce_mean(tf.square(self.eta_t_tf - self.eta_dot_pred))
        self.loss_ut = tf.reduce_mean(tf.square(self.eta_dot_pred - self.eta_t_pred))
        self.loss_g = tf.reduce_mean(tf.square(self.g_tf - self.g_pred))
        self.loss_f = tf.reduce_mean(tf.square(self.lift_tf - self.lift_pred))
        self.loss = self.loss_u + self.loss_ut + self.loss_udot + self.loss_g + self.loss_f

        # optimizers
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 20000,
                                                                         'maxfun': 50000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1 * np.finfo(float).eps})

        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        self.train_op = self.optimizer_Adam.minimize(self.loss)

        # Init session
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def CNN_model(self, X):
        model = Sequential()

        # Add configurable Conv1D layers
        for i in range(self.num_conv_layers):
            if i == 0:
                model.add(Conv1D(self.num_filters, self.kernel_size, strides=1,
                                 padding='same', use_bias=True, input_shape=(None, 1)))
            else:
                model.add(Conv1D(self.num_filters, self.kernel_size, strides=1,
                                 padding='same', use_bias=True))
            model.add(Activation(self.activation))

        # Add configurable Dense layers
        for _ in range(self.num_dense_layers):
            model.add(Dense(self.dense_units))
            model.add(Activation(self.activation))

        # Output layer
        model.add(Dense(3 * self.eta.shape[2]))

        model.summary()
        return model(X)

    def net_structure(self, ag):
        output = self.CNN_model(ag)
        eta = output[:, :, 0:1]
        eta_dot = output[:, :, 1:2]
        g = output[:, :, 2:]

        Phi_ut = np.reshape(self.Phi_t, [1, self.eta.shape[1], self.eta.shape[1]])
        Phi_ut = np.repeat(Phi_ut, self.eta.shape[0], axis=0)
        eta_t = tf.matmul(tf.cast(Phi_ut, dtype=tf.float32), eta)
        eta_tt = tf.matmul(tf.cast(Phi_ut, dtype=tf.float32), eta_dot)

        lift = eta_tt + g
        return eta, eta_t, eta_dot, g, lift
    
    def train(self, num_epochs, batch_size, learning_rate, bfgs):

        Loss_u = []
        Loss_udot = []
        Loss_ut = []
        Loss_g = []
        Loss_f = []
        Loss = []

        for epoch in range(num_epochs):
            
            N = self.eta.shape[0]

            start_time = time.time()
            for it in range(0, N, batch_size):
                tf_dict = {self.eta_tf: self.eta, self.eta_t_tf: self.eta_t, self.g_tf: self.g,
                           self.ag_tf: self.ag, self.lift_tf: self.lift, self.learning_rate: learning_rate}
                self.sess.run(self.train_op, tf_dict)
                
                # Print
                if it % (10*batch_size) == 0:
                    elapsed = time.time() - start_time
                    loss_value, learning_rate_value = self.sess.run([self.loss, self.learning_rate], tf_dict)
                    print('Epoch: %d, It: %d, Loss: %.3e, Time: %.2f, Learning Rate: %.3e'
                          %(epoch, it/batch_size, loss_value, elapsed, learning_rate_value))
                    start_time = time.time()

            Loss_u.append(self.sess.run(self.loss_u, tf_dict))
            Loss_ut.append(self.sess.run(self.loss_ut, tf_dict))
            Loss_udot.append(self.sess.run(self.loss_udot, tf_dict))
            Loss_g.append(self.sess.run(self.loss_g, tf_dict))
            Loss_f.append(self.sess.run(self.loss_f, tf_dict))
            Loss.append(self.sess.run(self.loss, tf_dict))

        if bfgs == 1:
            tf_dict_all = {self.eta_tf: self.eta, self.eta_t_tf: self.eta_t, self.g_tf: self.g,
                       self.ag_tf: self.ag, self.lift_tf: self.lift, self.learning_rate: learning_rate}

            self.optimizer.minimize(self.sess,
                                    feed_dict=tf_dict_all,
                                    fetches=[self.loss],
                                    loss_callback=self.callback)

            Loss_u.append(self.sess.run(self.loss_u, tf_dict))
            Loss_ut.append(self.sess.run(self.loss_ut, tf_dict))
            Loss_udot.append(self.sess.run(self.loss_udot, tf_dict))
            Loss_g.append(self.sess.run(self.loss_g, tf_dict))
            Loss_f.append(self.sess.run(self.loss_f, tf_dict))
            Loss.append(self.sess.run(self.loss, tf_dict))

        return Loss_u, Loss_udot, Loss_ut, Loss_g, Loss_f, Loss

    def callback(self, loss):
        print('Loss:', loss)

    def predict(self, ag_star):
        
        tf_dict = {self.ag_tf: ag_star}

        eta_star = self.sess.run(self.eta_pred, tf_dict)
        eta_t_star = self.sess.run(self.eta_t_pred, tf_dict)
        eta_dot_star = self.sess.run(self.eta_dot_pred, tf_dict)
        g_star = self.sess.run(self.g_pred, tf_dict)

        return eta_star, eta_t_star, eta_dot_star, g_star




def plot_losses(train_loss=None, val_loss=None, other_losses=None, title='Train vs Validation Loss'):
    """
    Plots the training and validation losses.
    Args:
        train_loss (list): List of training losses.
        val_loss (list, optional): List of validation losses. Defaults to None.
        other_losses (dict, optional): Dictionary of other losses to plot. Defaults to None.
        title (str): Title of the plot.
    """
    plt.figure()
    if train_loss is not None:
        plt.plot(train_loss, label='Train Loss', color='blue')
    if val_loss is not None:
        plt.plot(val_loss, label='Validation Loss', color='orange')
    if other_losses is not None:
        for key, loss_color in other_losses.items():
            loss, color = loss_color
            plt.plot(loss, label=f'{key}', color=color)
    if train_loss is None and val_loss is None and other_losses is None:
        raise ValueError("At least one loss must be provided to plot.")
    plt.title(title)
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()