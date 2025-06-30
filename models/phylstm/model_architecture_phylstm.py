import tensorflow as tf
from keras.models import Sequential
from keras.layers import CuDNNLSTM, Activation, Dense
from random import shuffle
import numpy as np
import time
import matplotlib.pyplot as plt

class DeepPhyLSTM2:
    # Initialize the class
    def __init__(self, eta, eta_t, g, ag, ag_c, lift, Phi_t, save_path,
                 lstm_units=100,
                 lstm_layers=3,
                 dense_units=100,
                 dense_layers=1,
                 activation='relu',
                 stateful=False):

        # Save data
        self.eta = eta
        self.eta_t = eta_t
        self.g = g
        self.ag = ag
        self.lift = lift
        self.ag_c = ag_c
        self.Phi_t = Phi_t
        self.save_path = save_path
        
        # Save Hyperparameters
        self.lstm_units = lstm_units
        self.lstm_layers = lstm_layers
        self.dense_units = dense_units
        self.dense_layers = dense_layers
        self.activation = activation
        self.stateful = stateful

        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        # placeholders for data
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.best_loss = tf.placeholder(tf.float32, shape=[])
        self.eta_tf = tf.placeholder(tf.float32, shape=[None, None, self.eta.shape[2]])
        self.eta_t_tf = tf.placeholder(tf.float32, shape=[None, None, self.eta.shape[2]])
        self.g_tf = tf.placeholder(tf.float32, shape=[None, None, self.eta.shape[2]])
        self.ag_tf = tf.placeholder(tf.float32, shape=[None, None, 1])
        self.lift_tf = tf.placeholder(tf.float32, shape=[None, None, 1])
        self.ag_c_tf = tf.placeholder(tf.float32, shape=[None, None, 1])
        self.Phi_tf = tf.placeholder(tf.float32, shape=[None, self.eta.shape[1], self.eta.shape[1]])

        # physics informed neural networks
        self.eta_pred, self.eta_t_pred, self.eta_tt_pred, self.eta_dot_pred, self.g_pred = self.net_structure(self.ag_tf)
        self.eta_t_pred_c, self.eta_dot_pred_c, self.lift_c_pred = self.net_f(self.ag_c_tf)

        # loss
        # for measurements
        self.loss_u = tf.reduce_mean(tf.square(self.eta_tf - self.eta_pred))
        self.loss_udot = tf.reduce_mean(tf.square(self.eta_t_tf - self.eta_dot_pred))
        self.loss_g = tf.reduce_mean(tf.square(self.g_tf - self.g_pred))
        # for collocations
        self.loss_ut_c = tf.reduce_mean(tf.square(self.eta_t_pred_c - self.eta_dot_pred_c))
        self.loss_e = tf.reduce_mean(tf.square(tf.matmul(self.lift_tf, tf.ones([self.lift.shape[0], 1, self.eta.shape[2]], dtype=tf.float32)) - self.lift_c_pred))

        # total loss
        self.loss = self.loss_u + self.loss_udot + self.loss_ut_c + self.loss_e

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

        init = tf.global_variables_initializer()
        self.sess.run(init)

        self.saver = tf.train.Saver()

    def LSTM_model(self, X):
        model = Sequential()

        # Add LSTM layers
        for i in range(self.lstm_layers):
            return_seq = True  # all layers return sequences
            if i == 0:
                model.add(CuDNNLSTM(self.lstm_units, return_sequences=return_seq,
                                    stateful=self.stateful, input_shape=(None, 1)))
            else:
                model.add(CuDNNLSTM(self.lstm_units, return_sequences=return_seq, stateful=self.stateful))
            model.add(Activation(self.activation))

        # Dense layers
        for _ in range(self.dense_layers):
            model.add(Dense(self.dense_units))
            model.add(Activation(self.activation))

        # Output
        model.add(Dense(3 * self.eta.shape[2]))
        return model(X)

    def LSTM_model_f(self, X):
        model = Sequential()

        # LSTM layers
        for i in range(self.lstm_layers):
            return_seq = True
            if i == 0:
                model.add(CuDNNLSTM(self.lstm_units, return_sequences=return_seq,
                                    stateful=self.stateful, input_shape=(None, 3 * self.eta.shape[2])))
            else:
                model.add(CuDNNLSTM(self.lstm_units, return_sequences=return_seq, stateful=self.stateful))
            model.add(Activation(self.activation))

        # Dense layers
        for _ in range(self.dense_layers):
            model.add(Dense(self.dense_units))
            model.add(Activation(self.activation))

        # Output
        model.add(Dense(self.eta.shape[2]))
        return model(X)

    def net_structure(self, ag):
        output = self.LSTM_model(ag)
        eta = output[:, :, 0:self.eta.shape[2]]
        eta_dot = output[:, :, self.eta.shape[2]:2*self.eta.shape[2]]
        g = output[:, :, 2*self.eta.shape[2]:]

        eta_t = tf.matmul(self.Phi_tf, eta)
        eta_tt = tf.matmul(self.Phi_tf, eta_dot)
        
        return eta, eta_t, eta_tt, eta_dot, g

    def net_f(self, ag):
        eta, eta_t, eta_tt, eta_dot, g = self.net_structure(ag)

        eta_dot1 = eta_dot[:, :, 0:1]
        # eta_dot2 = eta_dot[:, :, 1:] - eta_dot[:, :, 0:self.eta.shape[2] - 1]
        f = self.LSTM_model_f(tf.concat([eta, eta_dot1, g], 2))
        lift = eta_tt + f
        return eta_t, eta_dot, lift
    
    def train(self, num_epochs, learning_rate, bfgs):

        Loss_u = []
        Loss_udot = []
        Loss_ut_c = []
        Loss_g = []
        Loss_e = []
        Loss = []
        Loss_val = []
        best_loss = 100

        for epoch in range(num_epochs):

            Ind = list(range(self.ag.shape[0]))
            shuffle(Ind)
            ratio_split = 0.8
            Ind_tr = Ind[0:round(ratio_split * self.ag.shape[0])]
            Ind_val = Ind[round(ratio_split * self.ag.shape[0]):]
            # self.batch_size = len(Ind_val)

            self.ag_tr = self.ag[Ind_tr]
            self.eta_tr = self.eta[Ind_tr]
            self.eta_t_tr = self.eta_t[Ind_tr]
            self.g_tr = self.g[Ind_tr]
            self.ag_val = self.ag[Ind_val]
            self.eta_val = self.eta[Ind_val]
            self.eta_t_val = self.eta_t[Ind_val]
            self.g_val = self.g[Ind_val]

            start_time = time.time()

            tf_dict = {self.eta_tf: self.eta_tr, self.eta_t_tf: self.eta_t_tr, self.g_tf: self.g_tr,
                       self.ag_tf: self.ag_tr, self.lift_tf: self.lift, self.ag_c_tf: self.ag_c,
                       self.Phi_tf: self.Phi_t, self.learning_rate: learning_rate}

            tf_dict_val = {self.eta_tf: self.eta_val, self.eta_t_tf: self.eta_t_val, self.g_tf: self.g_val,
                           self.ag_tf: self.ag_val, self.lift_tf: self.lift, self.ag_c_tf: self.ag_c,
                           self.Phi_tf: self.Phi_t, self.learning_rate: learning_rate}
            
            self.sess.run(self.train_op, tf_dict)

            loss_value, learning_rate_value = self.sess.run([self.loss, self.learning_rate], tf_dict)
            loss_val_value = self.sess.run(self.loss, tf_dict_val)

            Loss_u.append(self.sess.run(self.loss_u, tf_dict))
            Loss_udot.append(self.sess.run(self.loss_udot, tf_dict))
            Loss_g.append(self.sess.run(self.loss_g, tf_dict))
            Loss_ut_c.append(self.sess.run(self.loss_ut_c, tf_dict))
            Loss_e.append(self.sess.run(self.loss_e, tf_dict))
            Loss.append(self.sess.run(self.loss, tf_dict))
            Loss_val.append(self.sess.run(self.loss, tf_dict_val))

            elapsed = time.time() - start_time
            print('Epoch: %d, Loss: %.3e, Loss_val: %.3e, Best_loss: %.3e, Time: %.2f, Learning Rate: %.3e'
                  % (epoch, loss_value, loss_val_value, best_loss, elapsed, learning_rate_value))

        if bfgs == 1:

            tf_dict_all = {self.eta_tf: self.eta_tr, self.eta_t_tf: self.eta_t_tr, self.g_tf: self.g_tr,
                       self.ag_tf: self.ag_tr, self.lift_tf: self.lift, self.ag_c_tf: self.ag_c,
                       self.Phi_tf: self.Phi_t, self.learning_rate: learning_rate, self.best_loss: best_loss}

            self.optimizer.minimize(self.sess,
                                    feed_dict=tf_dict_all,
                                    fetches=[self.loss, self.best_loss],
                                    loss_callback=self.callback)
                                    # step_callback=self.step_callback)

            Loss_u.append(self.sess.run(self.loss_u, tf_dict_all))
            Loss_udot.append(self.sess.run(self.loss_udot, tf_dict_all))
            Loss_g.append(self.sess.run(self.loss_g, tf_dict))
            Loss_ut_c.append(self.sess.run(self.loss_ut_c, tf_dict_all))
            Loss_e.append(self.sess.run(self.loss_e, tf_dict_all))
            Loss.append(self.sess.run(self.loss, tf_dict_all))
            Loss_val.append(self.sess.run(self.loss, tf_dict_val))

        return Loss_u, Loss_udot, Loss_g, Loss_ut_c, Loss_e, Loss, Loss_val, best_loss

    def callback(self, loss, best_loss):

        global Loss_BFGS
        global Loss_val_BFGS
        Loss_BFGS = np.append(Loss_BFGS, loss)

        loss_val = self.sess.run(self.loss, feed_dict={self.eta_tf: self.eta_val, self.eta_t_tf: self.eta_t_val, self.g_tf: self.g_val,
                               self.ag_tf: self.ag_val, self.lift_tf: self.lift, self.ag_c_tf: self.ag_c, self.Phi_tf: self.Phi_t})

        Loss_val_BFGS = np.append(Loss_val_BFGS, loss_val)

        print('Loss: %.3e, Loss_val: %.3e, Best_loss: %.3e'
              % (loss, loss_val, best_loss))

    def step_callback(self, loss):

        loss_val = self.sess.run(self.loss, feed_dict={self.eta_tf: self.eta_val, self.eta_t_tf: self.eta_t_val, self.g_tf: self.g_val,
                                 self.ag_tf: self.ag_val, self.lift_tf: self.lift, self.ag_c_tf: self.ag_c, self.Phi_tf: self.Phi_t})

        eta_star, eta_t_star, eta_tt_star, eta_dot_star, g_star = self.predict(self.ag_val, self.Phi_t[0:self.ag_val.shape[0]])
        loss_val = np.mean(np.square(eta_star, self.eta_val))

        print('Loss_val: %.3e', loss_val)

    def callback1(self, loss):
        print('Loss:', loss)

    def predict(self, ag_star, Phi_star):

        tf_dict = {self.ag_tf: ag_star, self.Phi_tf: Phi_star}

        eta_star = self.sess.run(self.eta_pred, tf_dict)
        eta_t_star = self.sess.run(self.eta_t_pred, tf_dict)
        eta_tt_star = self.sess.run(self.eta_tt_pred, tf_dict)
        eta_dot_star = self.sess.run(self.eta_dot_pred, tf_dict)
        g_star = self.sess.run(self.g_pred, tf_dict)

        return eta_star, eta_t_star, eta_tt_star, eta_dot_star, g_star

    def predict_c(self, ag_star, Phi_star):

        tf_dict = {self.ag_c_tf: ag_star, self.Phi_tf: Phi_star}
        lift_star = self.sess.run(self.lift_c_pred, tf_dict)

        return lift_star

    def predict_best_model(self, path, ag_star, Phi_star):
        # best_model = tf.train.import_meta_graph(path)
        self.saver.restore(sess=self.sess, save_path=path)

        tf_dict = {self.ag_tf: ag_star, self.Phi_tf: Phi_star}

        eta_star = self.sess.run(self.eta_pred, tf_dict)
        eta_t_star = self.sess.run(self.eta_t_pred, tf_dict)
        eta_tt_star = self.sess.run(self.eta_tt_pred, tf_dict)
        eta_dot_star = self.sess.run(self.eta_dot_pred, tf_dict)
        g_star = self.sess.run(self.g_pred, tf_dict)

        return eta_star, eta_t_star, eta_tt_star, eta_dot_star, g_star




class DeepPhyLSTM3:
    # Initialize the class
    def __init__(self, eta, eta_t, g, ag, ag_c, lift, Phi_t, save_path=None,
                 lstm_units=100,
                 lstm_layers=3,
                 dense_units=None,
                 activation='relu',
                 stateful=False):

        # Save data and config
        self.eta = eta
        self.eta_t = eta_t
        self.g = g
        self.ag = ag
        self.lift = lift
        self.ag_c = ag_c
        self.Phi_t = Phi_t
        self.save_path = save_path
        
        # Save Hyperparameters
        self.lstm_units = lstm_units
        self.lstm_layers = lstm_layers
        self.dense_units = dense_units
        self.activation = activation
        self.stateful = stateful

        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        # placeholders for data
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.best_loss = tf.placeholder(tf.float32, shape=[])
        self.eta_tf = tf.placeholder(tf.float32, shape=[None, None, self.eta.shape[2]])
        self.eta_t_tf = tf.placeholder(tf.float32, shape=[None, None, self.eta.shape[2]])
        self.g_tf = tf.placeholder(tf.float32, shape=[None, None, self.eta.shape[2]])
        self.ag_tf = tf.placeholder(tf.float32, shape=[None, None, 1])
        self.lift_tf = tf.placeholder(tf.float32, shape=[None, None, 1])
        self.ag_c_tf = tf.placeholder(tf.float32, shape=[None, None, 1])
        self.Phi_tf = tf.placeholder(tf.float32, shape=[None, self.eta.shape[1], self.eta.shape[1]])

        # physics informed neural networks
        self.eta_pred, self.eta_t_pred, self.eta_tt_pred, self.eta_dot_pred, self.g_pred, self.g_t_pred = self.net_structure(self.ag_tf)
        self.eta_t_pred_c, self.eta_dot_pred_c, self.g_t_pred_c, self.g_dot_pred_c, self.lift_c_pred = self.net_f(self.ag_c_tf)

        # loss
        # for measurements
        self.loss_u = tf.reduce_mean(tf.square(self.eta_tf - self.eta_pred))
        self.loss_udot = tf.reduce_mean(tf.square(self.eta_t_tf - self.eta_dot_pred))
        self.loss_g = tf.reduce_mean(tf.square(self.g_tf - self.g_pred))
        # for collocations
        self.loss_ut_c = tf.reduce_mean(tf.square(self.eta_t_pred_c - self.eta_dot_pred_c))
        self.loss_gt_c = tf.reduce_mean(tf.square(self.g_t_pred_c - self.g_dot_pred_c))
        self.loss_e = tf.reduce_mean(tf.square(tf.matmul(self.lift_tf, tf.ones([self.lift.shape[0], 1, self.eta.shape[2]], dtype=tf.float32)) - self.lift_c_pred))

        self.loss = self.loss_u + self.loss_udot + self.loss_ut_c + self.loss_gt_c + self.loss_e

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

        init = tf.global_variables_initializer()
        self.sess.run(init)

        self.saver = tf.train.Saver()

    def _build_lstm(self, input_shape, out_dim):
        model = Sequential()
        for i in range(self.lstm_layers):
            return_seq = True
            if i == 0:
                model.add(CuDNNLSTM(self.lstm_units, return_sequences=return_seq,
                                    stateful=self.stateful, input_shape=input_shape))
            else:
                model.add(CuDNNLSTM(self.lstm_units, return_sequences=return_seq,
                                    stateful=self.stateful))
            model.add(Activation(self.activation))
        
        if self.dense_units:
            model.add(Dense(self.dense_units))
            model.add(Activation(self.activation))

        model.add(Dense(out_dim))
        return model

    def LSTM_model(self, X):
        model = self._build_lstm(input_shape=(None, 1), out_dim=3 * self.eta.shape[2])
        return model(X)

    def LSTM_model_f(self, X):
        model = self._build_lstm(input_shape=(None, 3 * self.eta.shape[2]), out_dim=self.eta.shape[2])
        return model(X)

    def LSTM_model_g(self, X):
        model = self._build_lstm(input_shape=(None, 2 * self.eta.shape[2]), out_dim=self.eta.shape[2])
        return model(X)

    def net_structure(self, ag):
        output = self.LSTM_model(ag)
        eta = output[:, :, 0:self.eta.shape[2]]
        eta_dot = output[:, :, self.eta.shape[2]:2*self.eta.shape[2]]
        g = output[:, :, 2*self.eta.shape[2]:]

        eta_t = tf.matmul(self.Phi_tf, eta)
        eta_tt = tf.matmul(self.Phi_tf, eta_dot)
        g_t = tf.matmul(self.Phi_tf, g)

        return eta, eta_t, eta_tt, eta_dot, g, g_t

    def net_f(self, ag):
        eta, eta_t, eta_tt, eta_dot, g, g_t = self.net_structure(ag)
        f = self.LSTM_model_f(tf.concat([eta, eta_dot, g], 2))
        lift = eta_tt + f

        eta_dot1 = eta_dot[:, :, 0:1]
        g_dot = self.LSTM_model_g(tf.concat([eta_dot1, g], 2))
        return eta_t, eta_dot, g_t, g_dot, lift
    
    def train(self, num_epochs, learning_rate, bfgs):

        Loss_u = []
        Loss_udot = []
        Loss_ut_c = []
        Loss_gt_c = []
        Loss_g = []
        Loss_e = []
        Loss = []
        Loss_val = []
        best_loss = 100

        for epoch in range(num_epochs):

            Ind = list(range(self.ag.shape[0]))
            shuffle(Ind)
            ratio_split = 0.8
            Ind_tr = Ind[0:round(ratio_split * self.ag.shape[0])]
            Ind_val = Ind[round(ratio_split * self.ag.shape[0]):]

            self.ag_tr = self.ag[Ind_tr]
            self.eta_tr = self.eta[Ind_tr]
            self.eta_t_tr = self.eta_t[Ind_tr]
            self.g_tr = self.g[Ind_tr]
            self.ag_val = self.ag[Ind_val]
            self.eta_val = self.eta[Ind_val]
            self.eta_t_val = self.eta_t[Ind_val]
            self.g_val = self.g[Ind_val]

            start_time = time.time()

            tf_dict = {self.eta_tf: self.eta_tr, self.eta_t_tf: self.eta_t_tr, self.g_tf: self.g_tr,
                       self.ag_tf: self.ag_tr, self.lift_tf: self.lift, self.ag_c_tf: self.ag_c,
                       self.Phi_tf: self.Phi_t, self.learning_rate: learning_rate}

            tf_dict_val = {self.eta_tf: self.eta_val, self.eta_t_tf: self.eta_t_val, self.g_tf: self.g_val,
                           self.ag_tf: self.ag_val, self.lift_tf: self.lift, self.ag_c_tf: self.ag_c,
                           self.Phi_tf: self.Phi_t, self.learning_rate: learning_rate}

            self.sess.run(self.train_op, tf_dict)

            loss_value, learning_rate_value = self.sess.run([self.loss, self.learning_rate], tf_dict)
            loss_val_value = self.sess.run(self.loss, tf_dict_val)

            Loss_u.append(self.sess.run(self.loss_u, tf_dict))
            Loss_udot.append(self.sess.run(self.loss_udot, tf_dict))
            Loss_g.append(self.sess.run(self.loss_g, tf_dict))
            Loss_ut_c.append(self.sess.run(self.loss_ut_c, tf_dict))
            Loss_gt_c.append(self.sess.run(self.loss_gt_c, tf_dict))
            Loss_e.append(self.sess.run(self.loss_e, tf_dict))
            Loss.append(self.sess.run(self.loss, tf_dict))
            Loss_val.append(self.sess.run(self.loss, tf_dict_val))

            # Save the best val model
            if loss_val_value < best_loss and loss_val_value < 1e-2:
                best_loss = loss_val_value

                # self.saver.save(sess=self.sess, save_path=self.save_path)

            elapsed = time.time() - start_time
            print('Epoch: %d, Loss: %.3e, Loss_val: %.3e, Best_loss: %.3e, Time: %.2f, Learning Rate: %.3e'
                  % (epoch, loss_value, loss_val_value, best_loss, elapsed, learning_rate_value))

        if bfgs == 1:

            start_time = time.time()

            tf_dict_all = {self.eta_tf: self.eta_tr, self.eta_t_tf: self.eta_t_tr, self.g_tf: self.g_tr,
                       self.ag_tf: self.ag_tr, self.lift_tf: self.lift, self.ag_c_tf: self.ag_c,
                       self.Phi_tf: self.Phi_t, self.learning_rate: learning_rate, self.best_loss: best_loss}

            self.optimizer.minimize(self.sess,
                                    feed_dict=tf_dict_all,
                                    fetches=[self.loss, self.best_loss],
                                    loss_callback=self.callback)
                                    # step_callback=self.step_callback)

            Loss_u.append(self.sess.run(self.loss_u, tf_dict_all))
            Loss_udot.append(self.sess.run(self.loss_udot, tf_dict_all))
            Loss_g.append(self.sess.run(self.loss_g, tf_dict))
            Loss_ut_c.append(self.sess.run(self.loss_ut_c, tf_dict_all))
            Loss_gt_c.append(self.sess.run(self.loss_gt_c, tf_dict_all))
            Loss_e.append(self.sess.run(self.loss_e, tf_dict_all))
            Loss.append(self.sess.run(self.loss, tf_dict_all))
            Loss_val.append(self.sess.run(self.loss, tf_dict_val))

        return Loss_u, Loss_udot, Loss_g, Loss_ut_c, Loss_gt_c, Loss_e, Loss, Loss_val, best_loss

    def callback(self, loss, best_loss):

        global Loss_BFGS
        global Loss_val_BFGS
        Loss_BFGS = np.append(Loss_BFGS, loss)

        loss_val = self.sess.run(self.loss, feed_dict={self.eta_tf: self.eta_val, self.eta_t_tf: self.eta_t_val, self.g_tf: self.g_val,
                               self.ag_tf: self.ag_val, self.lift_tf: self.lift, self.ag_c_tf: self.ag_c, self.Phi_tf: self.Phi_t})

        Loss_val_BFGS = np.append(Loss_val_BFGS, loss_val)

        print('Loss: %.3e, Loss_val: %.3e, Best_loss: %.3e'
              % (loss, loss_val, best_loss))

    def step_callback(self, loss):

        loss_val = self.sess.run(self.loss, feed_dict={self.eta_tf: self.eta_val, self.eta_t_tf: self.eta_t_val, self.g_tf: self.g_val,
                                 self.ag_tf: self.ag_val, self.lift_tf: self.lift, self.ag_c_tf: self.ag_c, self.Phi_tf: self.Phi_t})

        eta_star, eta_t_star, eta_tt_star, eta_dot_star, g_star = self.predict(self.ag_val, self.Phi_t[0:self.ag_val.shape[0]])
        loss_val = np.mean(np.square(eta_star, self.eta_val))

        print('Loss_val: %.3e', loss_val)

    def callback1(self, loss):
        print('Loss:', loss)

    def predict(self, ag_star, Phi_star):

        tf_dict = {self.ag_tf: ag_star, self.Phi_tf: Phi_star}

        eta_star = self.sess.run(self.eta_pred, tf_dict)
        eta_t_star = self.sess.run(self.eta_t_pred, tf_dict)
        eta_tt_star = self.sess.run(self.eta_tt_pred, tf_dict)
        eta_dot_star = self.sess.run(self.eta_dot_pred, tf_dict)
        g_star = self.sess.run(self.g_pred, tf_dict)

        return eta_star, eta_t_star, eta_tt_star, eta_dot_star, g_star

    def predict_c(self, ag_star, Phi_star):

        tf_dict = {self.ag_c_tf: ag_star, self.Phi_tf: Phi_star}
        lift_star = self.sess.run(self.lift_c_pred, tf_dict)

        return lift_star

    def predict_best_model(self, path, ag_star, Phi_star):
        self.saver.restore(sess=self.sess, save_path=path)

        tf_dict = {self.ag_tf: ag_star, self.Phi_tf: Phi_star}

        eta_star = self.sess.run(self.eta_pred, tf_dict)
        eta_t_star = self.sess.run(self.eta_t_pred, tf_dict)
        eta_tt_star = self.sess.run(self.eta_tt_pred, tf_dict)
        eta_dot_star = self.sess.run(self.eta_dot_pred, tf_dict)
        g_star = self.sess.run(self.g_pred, tf_dict)

        return eta_star, eta_t_star, eta_tt_star, eta_dot_star, g_star





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