# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 16:53:31 2020

@author: aless
"""

# inspired by https://towardsdatascience.com/deep-reinforcement-learning-build-a-deep-q-network-dqn-to-play-cartpole-with-tensorflow-2-and-gym-8e105744b998

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import pdb

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense




# tf.debugging.set_log_device_placement(True)
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# Class to create a Deep Neural Network model
class DeepQNetworkModel(tf.keras.Model):
    
    def __init__(self, tfseed, num_states, hidden_units, num_actions, batch_norm, activation, 
                  kernel_initializer, mom_batch_norm, trainable_batch_norm, modelname='Deep Q Network'):
        # call the parent constructor
        super(DeepQNetworkModel, self).__init__(name=modelname) # name='Deep Q Network'
        # set random seed 
        tf.random.set_seed(tfseed)
        # set flag for batch norm as attribute
        self.bnflag = batch_norm
        # In setting input_shape, the batch dimension is not included.
        # input layer
        self.input_layer = InputLayer(input_shape=(num_states,))
        # batch norm layer for inputs
        if self.bnflag:
            self.bnorm_layer = BatchNormalization(momentum=mom_batch_norm,epsilon=0.0,
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

       
    # When you annotate a function with tf.function, you can still call it like any 
    # other function. But it will be compiled into a graph, which means you get the 
    # benefits of faster execution, running on GPU or TPU, or exporting to SavedModel
    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=tf.constant(True)):
        #print('Tracing:\n    {a}\n    {b}\n'.format(a=inputs, b=training))
          
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


# def DeepQNetworkModel(tfseed, num_states, hidden_units, num_actions, batch_norm, activation, 
#                       kernel_initializer, mom_batch_norm, trainable_batch_norm, modelname='Deep Q Network'):

#         # set random seed 
#         tf.random.set_seed(tfseed)

#         DQNmodel = tf.keras.Sequential()
#         DQNmodel.add(InputLayer(input_shape=(num_states,)))
#         if batch_norm: 
#             DQNmodel.add(BatchNormalization(momentum=mom_batch_norm,
#                                             epsilon=0.0,
#                                             trainable=trainable_batch_norm,
#                                             center=False, 
#                                             scale=False))
        
#         for i in hidden_units:
#             DQNmodel.add(Dense(i, kernel_initializer=kernel_initializer))
#             if activation == 'leaky_relu':
#                 leaky_relu = tf.nn.leaky_relu
#                 DQNmodel.add(Activation(leaky_relu))
#             elif activation == 'relu6':
#                 relu6 = tf.nn.relu6
#                 DQNmodel.add(Activation(relu6))
#             elif activation == 'elu':
#                 elu = tf.nn.elu
#                 DQNmodel.add(Activation(elu))
#             else:
#                 DQNmodel.add(Activation(activation))
        
#         DQNmodel.add(Dense(num_actions))
        
#         return DQNmodel
        
    

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
        self.plot_steps = plot_steps
        self.selected_loss = selected_loss
        self.DQN_type = DQN_type
        self.clipgrad = clipgrad
        self.clipnorm = clipnorm
        self.clipmin = clipmin
        self.clipmax = clipmax
        self.summary_writer = summary_writer
        
        if self.selected_loss == 'mse':
            self.loss = tf.keras.losses.MeanSquaredError()
        elif self.selected_loss == 'huber':
            self.loss = tf.keras.losses.Huber()
            
            
    def predict(self, inputs, training=tf.constant(True)):
        # return self.model(tf.numpy_function(np.atleast_2d,[tf.Variable(inputs)], tf.float32), training=training)
        return self.model(inputs, training=training)
    
    @tf.function
    def compute_gradients(self,states,encoded_actions,rewards,states_next,TargetNet):
        with tf.GradientTape() as tape: 

            selected_action_values = tf.math.reduce_sum(tf.math.multiply(self.predict(states),
                                      tf.one_hot(encoded_actions, self.num_actions)), axis=1)

            if self.DQN_type == 'DQN':
                value_next = tf.reduce_max(TargetNet.predict(states_next), axis=1)
            elif self.DQN_type == 'DDQN':
                greedy_target_action = tf.math.argmax(self.predict(states_next), 1)
                value_next = tf.math.reduce_sum(
                    TargetNet.predict(states_next) * tf.one_hot(greedy_target_action, self.num_actions), axis=1)
                
            actual_values = tf.math.add(rewards,tf.math.multiply(self.gamma,value_next))
            # compute loss function for the train model
            loss = self.loss(actual_values,selected_action_values)
            
        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        if self.clipgrad == 'norm':
            gradients = [(tf.clip_by_norm(gv, self.clipnorm)) for gv in gradients]
        elif self.clipgrad == 'value':
            gradients = [(tf.clip_by_value(gv,self.clipmin, self.clipmax)) for gv in gradients]

        self.optimizer.apply_gradients(zip(gradients, variables))
        
        return loss, gradients, variables
    
    
    def train(self,TargetNet,iteration):
        if len(self.experience['s']) < self.min_experiences:
            return 0
        
        # find the index of streams included in the experience buffer that will composed the training batch
        ids = tf.random.uniform([self.batch_size],0,len(self.experience['s']),dtype=tf.int64)
        states = tf.constant([self.experience['s'][i] for i in ids], dtype=tf.float32)
        actions = tf.constant([self.experience['a'][i] for i in ids], dtype=tf.int64)
        rewards = tf.constant([self.experience['r'][i] for i in ids], dtype=tf.float32)
        states_next = tf.constant([self.experience['s2'][i] for i in ids], dtype=tf.float32)
        encoded_actions = tf.constant([self.action_space.values.tolist().index(act.numpy()) for act in actions], 
                                      dtype=tf.int64)
        
        loss, gradients, variables = self.compute_gradients(states,encoded_actions,rewards,states_next,TargetNet)
        #pdb.set_trace()
        with self.summary_writer.as_default():

            tf.summary.scalar('Train Mean Squared Loss/TrainNet', loss, step=iteration)
            #tf.summary.scalar('Train Mean Squared Loss/BaselineNet', loss_baseline, step=iteration)
            
            if self.batch_norm:
                # plot batch norm moving averages of train network
                W_bn = self.model.layers[1].get_weights()
                if len(W_bn) == 4:
                    tf.summary.scalar('Batch Norm Parameters TrainNet/Input_1/Gamma', W_bn[0][0], step=iteration)
                    tf.summary.scalar('Batch Norm Parameters TrainNet/Input_1/Beta', W_bn[1][0], step=iteration)
                    tf.summary.scalar('Batch Norm Parameters TrainNet/Input_1/Mu', W_bn[2][0], step=iteration)
                    tf.summary.scalar('Batch Norm Parameters TrainNet/Input_1/SigmaSquare', W_bn[3][0], step=iteration)
                    
                    tf.summary.scalar('Batch Norm Parameters TrainNet/Input_2/Gamma', W_bn[0][1], step=iteration)
                    tf.summary.scalar('Batch Norm Parameters TrainNet/Input_2/Beta', W_bn[1][1], step=iteration)
                    tf.summary.scalar('Batch Norm Parameters TrainNet/Input_2/Mu', W_bn[2][1], step=iteration)
                    tf.summary.scalar('Batch Norm Parameters TrainNet/Input_2/SigmaSquare', W_bn[3][1], step=iteration)
                elif len(W_bn) == 2:
                    tf.summary.scalar('Batch Norm Parameters TrainNet/Input_1/Mu', W_bn[0][0], step=iteration)
                    tf.summary.scalar('Batch Norm Parameters TrainNet/Input_1/SigmaSquare', W_bn[1][0], step=iteration)
                    
                    tf.summary.scalar('Batch Norm Parameters TrainNet/Input_2/Mu', W_bn[0][1], step=iteration)
                    tf.summary.scalar('Batch Norm Parameters TrainNet/Input_2/SigmaSquare', W_bn[1][1], step=iteration)
                    
                # # plot batch norm moving averages of train network
                # W_bn_target = TargetNet.model.layers[1].get_weights()
                # if len(W_bn_target) == 4:
                #     tf.summary.scalar('Batch Norm Parameters TargetNet/Input_1/Gamma', W_bn_target[0][0], step=iteration)
                #     tf.summary.scalar('Batch Norm Parameters TargetNet/Input_1/Beta', W_bn_target[1][0], step=iteration)
                #     tf.summary.scalar('Batch Norm Parameters TargetNet/Input_1/Mu', W_bn_target[2][0], step=iteration)
                #     tf.summary.scalar('Batch Norm Parameters TargetNet/Input_1/SigmaSquare', W_bn_target[3][0], step=iteration)
                    
                #     tf.summary.scalar('Batch Norm Parameters TargetNet/Input_2/Gamma', W_bn_target[0][1], step=iteration)
                #     tf.summary.scalar('Batch Norm Parameters TargetNet/Input_2/Beta', W_bn_target[1][1], step=iteration)
                #     tf.summary.scalar('Batch Norm Parameters TargetNet/Input_2/Mu', W_bn_target[2][1], step=iteration)
                #     tf.summary.scalar('Batch Norm Parameters TargetNet/Input_2/SigmaSquare', W_bn_target[3][1], step=iteration)
                # elif len(W_bn_target) == 2:
                #     tf.summary.scalar('Batch Norm Parameters TargetNet/Input_1/Mu', W_bn_target[0][0], step=iteration)
                #     tf.summary.scalar('Batch Norm Parameters TargetNet/Input_1/SigmaSquare', W_bn_target[1][0], step=iteration)
                    
                #     tf.summary.scalar('Batch Norm Parameters TargetNet/Input_2/Mu', W_bn_target[0][1], step=iteration)
                #     tf.summary.scalar('Batch Norm Parameters TargetNet/Input_2/SigmaSquare', W_bn_target[1][1], step=iteration)

            #pdb.set_trace()
            for g,v in zip(gradients, variables):
                grad_mean = tf.reduce_mean(g)
                grad_norm = tf.math.sqrt(tf.reduce_mean(g**2))
                tf.summary.scalar(str(g.shape) + '/grad_mean', grad_mean.numpy(), step=iteration)
                tf.summary.scalar(str(g.shape) + '/grad_norm', grad_norm.numpy(), step=iteration)
        
    
    
    # @tf.function  #(experimental_relax_shapes=True)
    # def test(self,TargetNet,BaselineNet,iteration,summary_writer):
    #     if self.test_experience:
            
    #         # extract fixed streams of test experience
    #         test_states = np.asarray(self.test_experience['s'])
    #         test_actions = np.asarray(self.test_experience['a'])
    #         test_rewards = np.asarray(self.test_experience['r'])
    #         test_states_next = np.asarray(self.test_experience['s2'])
            
    #         # compute average maximum Q values for the fixed test states
    #         Q_avg = tf.reduce_mean(tf.reduce_max(self.predict(test_states), axis=1))
    #         Q_avg_baseline = tf.reduce_mean(tf.reduce_max(BaselineNet.predict(test_states), axis=1))
            
    #         # compute loss function over fixed states
    #         # Firstly compute current action values
    #         # find index of actions included in the batch
    #         encoded_actions = [self.action_space.values.tolist().index(act) 
    #                             for act in test_actions]     
    #         selected_action_values = tf.math.reduce_sum(
    #             self.predict(test_states) * tf.one_hot(encoded_actions, self.num_actions), axis=1)
                
    #         # compute target action values
    #         if self.DQN_type == 'DQN':
    #             value_next = np.max(TargetNet.predict(test_states_next), axis=1)
    #         elif self.DQN_type == 'DDQN':
    #             greedy_target_action = tf.math.argmax(self.predict(test_states_next), 1)
    #             value_next = tf.math.reduce_sum(
    #                 TargetNet.predict(test_states_next) * tf.one_hot(greedy_target_action, self.num_actions), axis=1)
            
    #         actual_values = test_rewards+self.gamma*value_next
            
    #         # compute loss function for the train model
    #         if self.selected_loss == 'mse':
    #             loss = tf.keras.losses.MeanSquaredError()(actual_values,selected_action_values)
    #         elif self.selected_loss == 'huber':
    #             loss = tf.keras.losses.Huber()(actual_values,selected_action_values)
            
    #         # compute loss function for the baseline model
    #         baseline_action_values = tf.math.reduce_sum(
    #             BaselineNet.predict(test_states) * tf.one_hot(encoded_actions, self.num_actions), axis=1)           
    #         if self.selected_loss == 'mse':
    #             loss_baseline = tf.keras.losses.MeanSquaredError()(actual_values,baseline_action_values)
    #         elif self.selected_loss == 'huber':
    #             loss_baseline = tf.keras.losses.Huber()(actual_values,baseline_action_values)
            
    
    def eps_greedy_action(self, states, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice(self.action_space.values)
        else:
            #pdb.set_trace()
            return self.action_space.values[tf.math.argmax(self.predict(states, 
                                                                   training = tf.constant(False))[0])]

    def add_experience(self, exp):
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)
            
    def add_test_experience(self, exp):
        self.test_experience = self.experience


    def copy_weights(self, TrainNet):
        variables1 = self.model.trainable_variables
        variables2 = TrainNet.model.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())
            
    # def store_graph(self, prof_outdir):
    #     tf.summary.trace_on(graph=True, profiler=True)
    #     with self.summary_writer.as_default():
    #         tf.summary.trace_export("graph", 0, prof_outdir)