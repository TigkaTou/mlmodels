import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle


class BaseModel(object):
    def __init__(self, n_classes, n_features, learning_rate=0.1):
        self.n_classes = n_classes
        self.n_features = n_features
        self.sess = tf.InteractiveSession()
        self.weights_init()
        self.forward_pass()
        self.learning_rate = learning_rate
        self.cost_op()
        self.optimize_op()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    @staticmethod
    def xavier_init(fan_in, fan_out, constant=1):
        """ Xavier initialization of network weights"""
        low = -constant*np.sqrt(6.0/(fan_in + fan_out))
        high = constant*np.sqrt(6.0/(fan_in + fan_out))
        return tf.random_uniform((fan_in, fan_out), minval=low,
                                 maxval=high, dtype=tf.float32)

    def forward_pass(self):
        with tf.name_scope("DNN"):
            hidden = tf.nn.relu(tf.add(tf.matmul(self.x,
                                self.variables["w_h"]),
                                self.variables["b_h"]))
            self.logits = tf.add(tf.matmul(hidden, self.variables["w_o"]),
                                 self.variables["b_o"])
            self.y_hat = tf.nn.softmax(self.logits)

    def weights_init(self):
        self.x = tf.placeholder(tf.float32, [None, self.n_features])
        self.y = tf.placeholder(tf.float32, [None, self.n_classes])
        self.variables = {}
        self.variables["w_h"] = tf.get_variable("w_h", initializer=
                                    self.xavier_init(self.n_features, 1000))
        self.variables["b_h"] = tf.get_variable("b_h",
                                                initializer=tf.zeros(1000))
        self.variables["w_o"] = tf.get_variable("w_o", initializer=
                                    self.xavier_init(1000, self.n_classes))
        self.variables["b_o"] = tf.get_variable("b_o",
                                        initializer=tf.zeros(self.n_classes))

    def cost_op(self):
        with tf.name_scope("cost"):
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                                        logits=self.logits, labels=self.y))
        return self.cost

    def optimize_op(self):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = tf.train.AdamOptimizer(
                        self.learning_rate).minimize(self.cost)
        return self.optimizer

    def train(self, X, Y, learning_rate=0.1, epochs=10, batch_size=100):
        N = X.shape[0]
        epoch_cost = []
        for e in range(epochs):
            total_cost = 0
            X, Y = shuffle(X, Y, random_state=0)
            for i in range(0, N, batch_size):
                train_X = X[i:i + batch_size]
                train_Y = Y[i:i + batch_size]
                c, _ = self.sess.run([self.cost, self.optimizer],
                                     feed_dict={self.x: train_X,
                                     self.y: train_Y})
                total_cost += c
            epoch_cost.append(total_cost)

    def predict(self, X):
        predictions = self.sess.run(self.y_hat, feed_dict={self.x: X})
        predictions = predictions.argmax(axis=1)
        return predictions
