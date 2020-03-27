# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 16:53:31 2020

@author: aless
"""

# inspired by https://towardsdatascience.com/deep-reinforcement-learning-build-a-deep-q-network-dqn-to-play-cartpole-with-tensorflow-2-and-gym-8e105744b998

import tensorflow as tf
import numpy as np
import pdb
import copy
from sys import getsizeof

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import Huber

# tf.debugging.set_log_device_placement(True)
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# Class to create a Deep Neural Network model
class DeepQNetworkModel(tf.keras.Model):
    
    def __init__(self, tfseed, num_states, hidden_units, num_actions, batch_norm, activation, 
                 kernel_initializer, mom_batch_norm, trainable_batch_norm, modelname='Deep Q Network'):
        # call the parent constructor
        super(DeepQNetworkModel, self).__init__(name=modelname) # name='Deep Q Network'
        #self.training = training
        # set random seed 
        tf.random.set_seed(tfseed)
        # set flag for batch norm as attribute
        self.bnflag = batch_norm
        # In setting input_shape, the batch dimension is not included.
        # input layer
        self.input_layer = InputLayer(input_shape=(num_states,))
        # batch norm layer for inputs
        if self.bnflag:
            self.bnorm_layer = BatchNormalization(momentum=mom_batch_norm,
                                                  epsilon=0.0,
                                                  trainable=trainable_batch_norm,
                                                  center=False, 
                                                  scale=False)
        
        # set of hidden layers
        self.hids = []
        
        for i in hidden_units:
            
            self.hids.append(Dense(i, kernel_initializer=kernel_initializer))
            # check what type of activation is set
            if activation == 'leaky_relu':
                leaky_relu = tf.nn.leaky_relu
                self.hids.append(Activation(leaky_relu))
            elif activation == 'relu6':
                relu6 = tf.nn.relu6
                self.hids.append(Activation(relu6))
            elif activation == 'elu':
                elu = tf.nn.elu
                self.hids.append(Activation(elu))
            else:
                self.hids.append(Activation(activation))
        # output layer with linear activation by default
        self.output_layer = Dense(num_actions)


    def call(self, inputs, training=True, store_intermediate_outputs=False):

        if not store_intermediate_outputs:
            # build the input layer
            if self.bnflag:
                z = self.input_layer(inputs)
                self.inputs = z
                z = self.bnorm_layer(z, training)
                self.bninputs = z
            else:
                z = self.input_layer(inputs)
                self.inputs = z
            # build the hidden layer
            for layer in self.hids:
                z = layer(z)
                layer.out = z
            # build the output layer
            z = self.output_layer(z)
            self.output_layer.out = z 

        else:
            # build the input layer
            if self.bnflag:
                z = self.input_layer(inputs)
                z = self.bnorm_layer(z, training)
            else:
                z = self.input_layer(inputs)
            # build the hidden layer
            for layer in self.hids:
                z = layer(z)
            # build the output layer
            z = self.output_layer(z)

        return z

# Class to create the DQN algorithm
class DQN:
    
    def __init__(self, tfseed, num_states, hidden_units, gamma, max_experiences, min_experiences, 
                 batch_size, lr, action_space, batch_norm, summary_writer, activation, 
                 kernel_initializer, plot_steps, selected_loss, mom_batch_norm, 
                 trainable_batch_norm, DQN_type, clipgrad, clipnorm, clipmin, clipmax, optimizer_name, modelname='Deep Q Network'):
        self.batch_size = batch_size
        if optimizer_name == 'sgd':
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        elif optimizer_name == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        elif optimizer_name == 'rmsprop':
            self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
        self.gamma = gamma
        self.experience = {'s': [], 'a': [], 'r': [], 's2': []}
        self.test_experience = None
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences
        self.action_space = action_space
        self.num_actions = len(self.action_space.values)
        self.batch_norm = batch_norm
        self.model = DeepQNetworkModel(tfseed, num_states, hidden_units, self.num_actions, 
                                       batch_norm, activation, kernel_initializer, mom_batch_norm, 
                                       trainable_batch_norm, modelname)
        self.summary_writer = summary_writer
        self.plot_steps = plot_steps
        self.selected_loss = selected_loss
        self.DQN_type = DQN_type
        self.clipgrad = clipgrad
        self.clipnorm = clipnorm
        self.clipmin = clipmin
        self.clipmax = clipmax
        
        if self.selected_loss == 'mse':
            self.loss = tf.keras.losses.MeanSquaredError()
        elif self.selected_loss == 'huber':
            self.loss = tf.keras.losses.Huber()


    def predict(self, inputs, training=True, store_intermediate_outputs=False):
        return self.model(np.atleast_2d(inputs.astype('float32')), training, store_intermediate_outputs)
    
    def train(self, TargetNet, iteration):

        if len(self.experience['s']) < self.min_experiences:
            return 0
        
        # find the index of streams included in the experience buffer that will composed the training batch
        ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)
        states = np.asarray([self.experience['s'][i] for i in ids])
        actions = np.asarray([self.experience['a'][i] for i in ids])
        rewards = np.asarray([self.experience['r'][i] for i in ids])
        states_next = np.asarray([self.experience['s2'][i] for i in ids])

        with tf.GradientTape() as tape: #persistent=True
            
            # compute current action values
            # find index of actions included in the batch
            encoded_actions = [self.action_space.values.tolist().index(act) 
                               for act in actions]
            selected_action_values = tf.math.reduce_sum(
                self.predict(states,store_intermediate_outputs=True) * tf.one_hot(encoded_actions, self.num_actions), axis=1)
                       
            # compute target action values
            if self.DQN_type == 'DQN':
                value_next = np.max(TargetNet.predict(states_next), axis=1)
            elif self.DQN_type == 'DDQN':
                greedy_target_action = tf.math.argmax(self.predict(states_next), 1)
                value_next = tf.math.reduce_sum(
                    TargetNet.predict(states_next) * tf.one_hot(greedy_target_action, self.num_actions), axis=1)
            
            actual_values = rewards+self.gamma*value_next
            
            # compute loss function for the train model
            loss = self.loss(actual_values,selected_action_values)
            
            
        variables = self.model.trainable_variables

        # compute gradient of the loss with respect to the variables (weights)
        gradients = tape.gradient(loss, variables)

        if self.clipgrad == 'norm':
            gradients = [(tf.clip_by_norm(gv, self.clipnorm)) for gv in gradients]
        elif self.clipgrad == 'value':
            gradients = [(tf.clip_by_value(gv,self.clipmin, self.clipmax)) for gv in gradients]
        
        # provide a list of (gradient, variable) pairs.
        self.optimizer.apply_gradients(zip(gradients, variables))
        
        with self.summary_writer.as_default():
            
            tf.summary.scalar('Mean Squared Loss/Train', loss, step=iteration)
            # tf.summary.scalar('Train Mean Squared Loss/BaselineNet', loss_baseline, step=iteration)
            
            
            if ((iteration % self.plot_steps) == 0) or (iteration == self.min_experiences):
                for i,layer in enumerate(self.model.layers[1:]):
                    with tf.name_scope('layer{0}'.format(i)):
                        if 'dense' in layer.name:                            
                            tf.summary.histogram(layer.name + '/weights', 
                                                  layer.get_weights()[0], step=iteration)
                            tf.summary.histogram(layer.name + '/biases', 
                                                  layer.get_weights()[1], step=iteration)
                            tf.summary.histogram(layer.name + '/Wx+b_pre_activation', 
                                                  layer.out, step=iteration)
                        elif 'activation' in layer.name:        
                            tf.summary.histogram(layer.name + '/activation', 
                                                  layer.out, step=iteration)             
                        elif 'batch' in layer.name:               
                            tf.summary.histogram(layer.name + '/bnorm_inputs_1', 
                                                  self.model.bninputs[:,0], step=iteration)
                            tf.summary.histogram(layer.name + '/bnorm_inputs_2', 
                                                  self.model.bninputs[:,1], step=iteration)
                            tf.summary.histogram(layer.name + '/inputs_1', 
                                                  self.model.inputs[:,0], step=iteration)
                            tf.summary.histogram(layer.name + '/inputs_2', 
                                                  self.model.inputs[:,1], step=iteration)
                        
            for g,v in zip(gradients, variables):
                grad_mean = tf.reduce_mean(g)
                grad_norm = tf.sqrt(tf.reduce_mean(g**2))
                tf.summary.scalar(v.name.split('/',1)[1] + '/grad_mean', grad_mean, step=iteration)
                tf.summary.scalar(v.name.split('/',1)[1] + '/grad_norm', grad_norm, step=iteration)
                                                                
            self.summary_writer.flush()
        
    
    def test(self, TargetNet,iteration):
        if self.test_experience:

            encoded_test_actions = [self.action_space.values.tolist().index(act) 
                                for act in np.asarray(self.test_experience['a'])]     
            
            test_pred = self.predict(np.asarray(self.test_experience['s']))
            
            selected_test_action_values = tf.math.reduce_sum( 
                test_pred * tf.one_hot(encoded_test_actions, self.num_actions), axis=1)
            
            
            test_loss = self.loss(self.actual_test_values,selected_test_action_values)
            # compute average maximum Q values for the fixed test states
            Q_avg = tf.reduce_mean(tf.reduce_max(test_pred, axis=1))
            with self.summary_writer.as_default(): 
                tf.summary.scalar('Q_avg_test/TrainNet', Q_avg, step=iteration)
                tf.summary.scalar('Mean Squared Loss/Train', test_loss.numpy(), step=iteration)
                self.summary_writer.flush()

    def compute_test_target(self, TargetNet):
        if self.test_experience:
            if self.DQN_type == 'DQN':
                test_value_next = np.max(TargetNet.predict(np.asarray(self.test_experience['s2'])), axis=1)
            elif self.DQN_type == 'DDQN':
                greedy_target_action = tf.math.argmax(self.predict(np.asarray(self.test_experience['s2'])), 1)
                test_value_next = tf.math.reduce_sum(
                    TargetNet.predict(np.asarray(self.test_experience['s2'])) * tf.one_hot(greedy_target_action, self.num_actions), axis=1)
            
            actual_test_values = np.asarray(self.test_experience['r']) + self.gamma*test_value_next
            
            self.actual_test_values = actual_test_values
        
    def eps_greedy_action(self, states, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice(self.action_space.values)
        else:
            return self.action_space.values[np.argmax(self.predict(np.atleast_2d(states), training = False)[0])]
        
    
    def add_experience(self, exp):
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)
            
    def add_test_experience(self):
        self.test_experience = copy.deepcopy(self.experience)


    def copy_weights(self, TrainNet):
        variables1 = self.model.trainable_variables
        variables2 = TrainNet.model.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())
