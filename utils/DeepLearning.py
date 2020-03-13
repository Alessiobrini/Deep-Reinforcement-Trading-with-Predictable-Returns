# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 16:53:31 2020

@author: aless
"""

# inspired by https://towardsdatascience.com/deep-reinforcement-learning-build-a-deep-q-network-dqn-to-play-cartpole-with-tensorflow-2-and-gym-8e105744b998

import tensorflow as tf
import numpy as np
import pdb

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import Huber

# tf.debugging.set_log_device_placement(True)


# Class to create a Deep Neural Network model
class DeepQNetworkModel(tf.keras.Model):
    
    def __init__(self, num_states, hidden_units, num_actions, batch_norm, activation, kernel_initializer):
        # call the parent constructor
        super(DeepQNetworkModel, self).__init__() # name='Deep Q Network'
        # set random seed 
        tf.random.set_seed(1234)
        # set flag for batch norm as attribute
        self.bnflag = batch_norm
        # In setting input_shape, the batch dimension is not included.
        # input layer
        self.inp = InputLayer(input_shape=(num_states,))
        # batch norm layer for inputs
        if self.bnflag:
            self.bn1 = BatchNormalization()
        
        # set of hidden layers
        self.hids = []
        
        for i in hidden_units:
            self.hids.append(Dense(i, kernel_initializer=kernel_initializer))
            self.hids.append(Activation(activation))
        #leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.2)
        # output layer
        self.out = Dense(num_actions)
        self.act_out = Activation('linear')

    # When you annotate a function with tf.function, you can still call it like any 
    # other function. But it will be compiled into a graph, which means you get the 
    # benefits of faster execution, running on GPU or TPU, or exporting to SavedModel
    #@tf.function
    def call(self, inputs):
        #pdb.set_trace()
        # build the input layer
        if self.bnflag:
            z = self.inp(inputs)
            z = self.bn1(z)
        else:
            z = self.inp(inputs)
        # build the hidden layer
        for layer in self.hids:
            z = layer(z)
        # build the output layer
        z = self.out(z)
        z = self.act_out(z)
        
        return z


# network

class DQN:
    
    def __init__(self, num_states, hidden_units, gamma, max_experiences, min_experiences, 
                 batch_size, lr, action_space, batch_norm, summary_writer, activation, 
                 kernel_initializer, plot_steps):
        self.batch_size = batch_size
        self.optimizer = tf.optimizers.Adam(lr)
        self.gamma = gamma
        self.experience = {'s': [], 'a': [], 'r': [], 's2': []}
        self.test_experience = None
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences
        self.action_space = action_space
        self.num_actions = len(self.action_space.values)
        self.batch_norm = batch_norm
        self.model = DeepQNetworkModel(num_states, hidden_units, self.num_actions, batch_norm, activation, kernel_initializer)
        self.summary_writer = summary_writer
        self.plot_steps = plot_steps


    def predict(self, inputs):
        return self.model(np.atleast_2d(inputs.astype('float32')))
    
    s
    #@tf.function
    def train(self, TargetNet, BaselineNet, iteration):

        if len(self.experience['s']) < self.min_experiences:
            return 0
        
        ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)
        states = np.asarray([self.experience['s'][i] for i in ids])
        actions = np.asarray([self.experience['a'][i] for i in ids])
        rewards = np.asarray([self.experience['r'][i] for i in ids])
        states_next = np.asarray([self.experience['s2'][i] for i in ids])
        value_next = np.max(TargetNet.predict(states_next), axis=1)
        actual_values = rewards+self.gamma*value_next

        with tf.GradientTape(persistent=True) as tape:
            
            encoded_actions = [self.action_space.values.tolist().index(act) 
                               for act in actions]     
            selected_action_values = tf.math.reduce_sum(
                self.predict(states) * tf.one_hot(encoded_actions, self.num_actions), axis=1)
            loss = tf.math.reduce_mean(tf.square(actual_values - selected_action_values))
    
            baseline_action_values = tf.math.reduce_sum(
                BaselineNet.predict(states) * tf.one_hot(encoded_actions, self.num_actions), axis=1)           
            loss_baseline = tf.math.reduce_mean(tf.square(actual_values - baseline_action_values))


        variables = self.model.trainable_variables
        # compute gradient of the loss with respect to the variables (weights)
        gradients = tape.gradient(loss, variables)

        # provide a list of (gradient, variable) pairs.
        self.optimizer.apply_gradients(zip(gradients, variables))

        with self.summary_writer.as_default():
            
            tf.summary.scalar('L2 loss/TrainNet', loss, step=iteration)
            tf.summary.scalar('L2 loss/BaselineNet', loss_baseline, step=iteration)
            
            for g,v in zip(gradients, variables):
                grad_norm = tf.sqrt(tf.reduce_mean(g**2))
                tf.summary.scalar(v.name.split('/',1)[1] + '/grad_norm', grad_norm, step=iteration)
                
            #pdb.set_trace()
            if ((iteration % self.plot_steps) == 0) or (iteration == self.min_experiences):
                for i,layer in enumerate(self.model.layers[1:]):
                    with tf.name_scope('layer{0}'.format(i)):
                        if 'dense' in layer.name:
                            tf.summary.histogram(layer.name + '/weights', 
                                                 layer.get_weights()[0], step=iteration)
                            tf.summary.histogram(layer.name + '/biases', 
                                                 layer.get_weights()[1], step=iteration)
                            tf.summary.histogram(layer.name + '/Wx+b_pre_activation', 
                                                 layer.activation(states), step=iteration)
                        elif 'activation' in layer.name:
                            tf.summary.histogram(layer.name + '/activation', 
                                                 layer.activation(states), step=iteration)
                        elif 'batch' in layer.name:
                            tf.summary.histogram(layer.name + '/bnorm_inputs', 
                                                 layer(states), step=iteration)
                            tf.summary.histogram(layer.name + '/inputs', 
                                                 states, step=iteration)           
            self.summary_writer.flush()

    
    def test(self, BaselineNet, iteration):
        if self.test_experience:
            
            test_states = np.vstack(self.experience['s'])
            Q_avg = tf.reduce_mean(tf.reduce_max(self.model(test_states), axis=1))
            Q_avg_baseline = tf.reduce_mean(tf.reduce_max(BaselineNet.model(test_states), axis=1))
            #pdb.set_trace()
            with self.summary_writer.as_default():
            
                tf.summary.scalar('Q_avg_test/TrainNet', Q_avg, step=iteration)
                tf.summary.scalar('Q_avg_test/BaselineNet', Q_avg_baseline, step=iteration)
    
        
    def eps_greedy_action(self, states, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice(self.action_space.values)
        else:
            return self.action_space.values[np.argmax(self.predict(np.atleast_2d(states))[0])]
        
    
    def add_experience(self, exp):
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)
            
    def add_test_experience(self, exp, iteration):
        self.test_experience = self.experience['s']


    def copy_weights(self, TrainNet):
        variables1 = self.model.trainable_variables
        variables2 = TrainNet.model.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())
            
    def store_graph(self, prof_outdir):
        tf.summary.trace_on(graph=True, profiler=True)
        with self.summary_writer.as_default():
            tf.summary.trace_export("graph", 0, prof_outdir)