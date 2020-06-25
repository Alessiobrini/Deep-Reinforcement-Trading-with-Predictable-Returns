# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 12:01:33 2020

@author: aless
"""

import tensorflow as tf
# from tensorflow_addons.optimizers import RectifiedAdam
import numpy as np
import pdb
import copy
from sys import getsizeof

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.keras.experimental import LinearCosineDecay

from utils.SumTreePER import PER_buffer
from utils.ExplorativeNoises import OrnsteinUhlenbeckActionNoise


################################ Class to create a Deep Q Network model ################################ 
class CriticNetwork(tf.keras.Model):
    
    def __init__(self, seed, num_states, hidden_units, num_actions, batch_norm_input, batch_norm_hidden,
                 activation, kernel_initializer, mom_batch_norm, trainable_batch_norm, modelname='Critic Network'):
        # call the parent constructor
        super(CriticNetwork, self).__init__(name=modelname)

        # set dimensionality of input/output depending on the model
        inp_shape = (num_states + num_actions,)
        out_shape = 1
        
        # set random seed 
        tf.random.set_seed(seed)
        # set flag for batch norm as attribute
        self.bnflag_input = batch_norm_input
        self.batch_norm_hidden = batch_norm_hidden
        # In setting input_shape, the batch dimension is not included.
        # input layer
        self.input_layer = InputLayer(input_shape=inp_shape)
        # batch norm layer for inputs
        if self.bnflag_input:
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
                
            if self.batch_norm_hidden:
                self.hids.append(BatchNormalization(momentum=mom_batch_norm,
                                                    trainable=trainable_batch_norm))
        # output layer with linear activation by default
        self.output_layer = Dense(out_shape, kernel_initializer=tf.keras.initializers.RandomUniform(-3e-3, 3e-3))


    def call(self, states, actions, training=True, store_intermediate_outputs=False):
        
        inputs = tf.concat([states,actions], axis=1)

        if store_intermediate_outputs:
            # build the input layer
            if self.bnflag_input:
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
            if self.bnflag_input:
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
    
class ActorNetwork(tf.keras.Model):
    
    def __init__(self, seed, num_states, hidden_units, num_actions, action_limit, batch_norm_input, batch_norm_hidden,
                 activation, kernel_initializer, mom_batch_norm, trainable_batch_norm, modelname='Actor Network'):
        # call the parent constructor
        super(ActorNetwork, self).__init__(name=modelname)

        # set dimensionality of input/output depending on the model
        inp_shape = num_states
        out_shape = num_actions
        # set boundaries for action
        self.action_limit = action_limit

        # set random seed 
        tf.random.set_seed(seed)
        # set flag for batch norm as attribute
        self.bnflag_input = batch_norm_input
        self.batch_norm_hidden = batch_norm_hidden
        # In setting input_shape, the batch dimension is not included.
        # input layer
        self.input_layer = InputLayer(input_shape=inp_shape)
        # batch norm layer for inputs
        if self.bnflag_input:
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
                
            if self.batch_norm_hidden:
                self.hids.append(BatchNormalization(momentum=mom_batch_norm,
                                                    trainable=trainable_batch_norm))
        # output layer with linear activation by default
        self.output_layer = Dense(out_shape, activation='tanh', 
                                  kernel_initializer=tf.keras.initializers.RandomUniform(-3e-3, 3e-3))


    def call(self, inputs, training=True, store_intermediate_outputs=False):

        if store_intermediate_outputs:
            # build the input layer
            if self.bnflag_input:
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
            z = z * self.action_limit
            self.output_layer.out = z 
 
        else:
            # build the input layer
            if self.bnflag_input:
                z = self.input_layer(inputs)
                z = self.bnorm_layer(z, training)
            else:
                z = self.input_layer(inputs)
            # build the hidden layer
            for layer in self.hids:
                z = layer(z)
            # build the output layer
            z = self.output_layer(z)
            z = z * self.action_limit
        return z
    
############################### DDPG ALGORITHM ################################ 
# https://spinningup.openai.com/en/latest/algorithms/ddpg.html#exploration-vs-exploitation
class DDPG:
    
    def __init__(self, seed, num_states, num_actions, hidden_units, gamma, start_train, max_experiences, min_experiences, 
                 batch_size, lr_Q, lr_p, lr_schedule, exp_decay_steps, exp_decay_rate_Q, exp_decay_rate_p, batch_norm_input, 
                 batch_norm_hidden, summary_writer, activation, kernel_initializer, plot_hist, plot_steps_hist, plot_steps, 
                 selected_loss, mom_batch_norm, trainable_batch_norm, DQN_type, use_PER, PER_e,PER_a,PER_b,final_PER_b,PER_b_steps,
                 PER_b_growth, final_PER_a,PER_a_steps,PER_a_growth, clipgrad, clipnorm, clipvalue, clipglob_steps, 
                 optimizer_name,optimizer_decay, beta_1, beta_2, eps_opt, update_target,tau, stddev_noise, theta, action_limit,
                 modelname='Train',pretraining_mode = False):
        self.batch_size = batch_size
        
        if lr_schedule == 'exponential':
            lr_Q = ExponentialDecay(initial_learning_rate=lr_Q, 
                                  decay_steps=exp_decay_steps, 
                                  decay_rate=exp_decay_rate_Q)
            lr_p = ExponentialDecay(initial_learning_rate=lr_p, 
                                  decay_steps=exp_decay_steps, 
                                  decay_rate=exp_decay_rate_p)
        elif lr_schedule == 'piecewise':
            lr_Q = PiecewiseConstantDecay(boundaries=[500000], 
                                        values=[0.001, 0.0001])
            lr_p = PiecewiseConstantDecay(boundaries=[500000], 
                                        values=[0.001, 0.0001])
            
        if optimizer_name == 'sgd':
            self.optimizer_Q = tf.keras.optimizers.SGD(learning_rate=lr_Q, momentum=0.0, 
                                                     nesterov=False, decay=optimizer_decay)
            self.optimizer_p = tf.keras.optimizers.SGD(learning_rate=lr_p, momentum=0.0, 
                                                     nesterov=False, decay=optimizer_decay)
        elif optimizer_name == 'adam':
            self.optimizer_Q = tf.keras.optimizers.Adam(learning_rate=lr_Q, beta_1=beta_1, beta_2=beta_2, 
                                                      epsilon=eps_opt, amsgrad=False, decay=optimizer_decay)
            self.optimizer_p = tf.keras.optimizers.Adam(learning_rate=lr_p, beta_1=beta_1, beta_2=beta_2, 
                                                  epsilon=eps_opt, amsgrad=False, decay=optimizer_decay)
        elif optimizer_name == 'rmsprop':
            self.optimizer_Q = tf.keras.optimizers.RMSprop(learning_rate=lr_Q, rho=beta_1, momentum=0.0, epsilon=eps_opt,
                                                         centered=False, decay=optimizer_decay)
            self.optimizer_p = tf.keras.optimizers.RMSprop(learning_rate=lr_p, rho=beta_1, momentum=0.0, epsilon=eps_opt,
                                                         centered=False, decay=optimizer_decay)

        self.beta_1 = beta_1
        self.eps_opt = eps_opt
        self.gamma = gamma
        self.use_PER = use_PER
        if self.use_PER:
            self.PERmemory = PER_buffer(PER_e,PER_a,PER_b,final_PER_b,PER_b_steps,PER_b_growth,
                                        final_PER_a,PER_a_steps,PER_a_growth,max_experiences) # experience is stored as object of this class
        else:
            self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'f': []}
        self.test_experience = None
        self.start_train = start_train
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences
        self.action_limit = action_limit
        self.num_actions = num_actions 
        self.batch_norm_input = batch_norm_input
        self.batch_norm_hidden = batch_norm_hidden
        self.Q_model = CriticNetwork(seed, num_states, hidden_units, num_actions, 
                                       batch_norm_input, batch_norm_hidden, activation, kernel_initializer, 
                                       mom_batch_norm, trainable_batch_norm, modelname='Q'+ modelname)   
        self.p_model = ActorNetwork(seed, num_states, hidden_units, num_actions, action_limit,
                                       batch_norm_input, batch_norm_hidden, activation, kernel_initializer, 
                                       mom_batch_norm, trainable_batch_norm, modelname='p'+ modelname) 
        self.summary_writer = summary_writer
        self.plot_hist = plot_hist
        self.plot_steps = plot_steps
        self.plot_steps_hist = plot_steps_hist
        self.selected_loss = selected_loss
        self.DQN_type = DQN_type
        self.clipgrad = clipgrad
        self.clipnorm = clipnorm
        self.clipvalue = clipvalue
        self.update_target = update_target
        self.tau = tau
        self.global_norms = []
        self.clipglob_steps = clipglob_steps
        self.optimizer_name = optimizer_name
        self.pretraining_mode = pretraining_mode

        
        nb_actions = int(num_states/2)
        self.action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev_noise) * np.ones(nb_actions),
                                                         theta=theta)
        self.nb_actions = nb_actions
      
        if self.selected_loss == 'mse':
            self.loss = tf.keras.losses.MeanSquaredError()
        elif self.selected_loss == 'huber':
            self.loss = tf.keras.losses.Huber()
    
    def train(self, TargetNet, iteration, env=None):

        if iteration < self.start_train:
            return 0
        
        if self.use_PER:
            b_idx, minibatch = self.PERmemory.sample_batch(self.batch_size)
            states = np.asarray(minibatch['s'])
            actions = np.asarray(minibatch['a'])
            rewards = np.asarray(minibatch['r'])
            states_next = np.asarray(minibatch['s2'])
            if self.pretraining_mode:
                factors = np.asarray(minibatch['f'])
        else:
            # find the index of streams included in the experience buffer that will 
            #composed the training batch
            ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)
            states = np.asarray([self.experience['s'][i] for i in ids])
            actions = np.asarray([self.experience['a'][i] for i in ids])
            rewards = np.asarray([self.experience['r'][i] for i in ids])
            states_next = np.asarray([self.experience['s2'][i] for i in ids])
            if self.pretraining_mode:
                factors = np.asarray([self.experience['f'][i] for i in ids])
        
        with tf.GradientTape() as tape:
            
            # compute current action values
            current_q = self.Q_model(states.astype('float32'), actions.astype('float32'))
            
            # compute target action values
            action_next = TargetNet.p_model(states_next.astype('float32'))
            # stat_act_next = np.concatenate((states_next,action_next),axis=1)
            target_q = TargetNet.Q_model(states_next.astype('float32'), action_next)
            
            target_values = rewards+self.gamma*target_q
            
            if self.use_PER:
                # compute weights
                if iteration < self.max_experiences:
                    N = iteration + 1
                else:
                    N = self.max_experiences
                #N = len([x for x in self.PERmemory.experience['r'] if x!=0])
                prob = self.PERmemory.tree[b_idx]/self.PERmemory.total_priority
                self.PERmemory.PER_b = min(self.PERmemory.final_PER_b, self.PERmemory.PER_b + self.PERmemory.PER_b_growth)
                w_IS = (N * prob)**(-self.PERmemory.PER_b)
                scaled_w_IS = w_IS/np.max(w_IS)

                # update priorities
                self.PERmemory.batch_update(b_idx, np.abs(target_values-current_q))
  
                # compute loss function for the train model
                loss_q = self.loss(y_true=target_values,y_pred=current_q,
                                 sample_weight=scaled_w_IS)
            else:
                loss_q = self.loss(y_true=target_values,y_pred=current_q)
                
              
        variables_q = self.Q_model.trainable_variables

        # compute gradient of the loss with respect to the variables (weights)
        gradients_q = tape.gradient(loss_q, variables_q)

        if self.clipgrad == 'norm':
            gradients_q = [(tf.clip_by_norm(gv, self.clipnorm)) for gv in gradients_q]
        elif self.clipgrad == 'value':
            gradients_q = [(tf.clip_by_value(gv,-self.clipvalue, self.clipvalue)) for gv in gradients_q]
        elif self.clipgrad == 'globnorm':
            if iteration <= self.clipglob_steps:
                gbnorm = tf.linalg.global_norm(gradients_q)
                self.global_norms.append(gbnorm.numpy())
                if iteration == self.clipglob_steps:
                    self.clipglob = np.mean(self.global_norms)
            else:              
                gradients_q, gbnorm  = tf.clip_by_global_norm(gradients_q,self.clipglob)
        
        # provide a list of (gradient, variable) pairs.
        self.optimizer_Q.apply_gradients(zip(gradients_q, variables_q))
        
        with tf.GradientTape() as tape:            
            current_q_pg = self.Q_model(states.astype('float32'), self.p_model(states.astype('float32')))
            loss_p = - tf.math.reduce_mean(current_q_pg)
            
        # compute gradient of the loss with respect to the variables (weights)
        variables_p = self.p_model.trainable_variables
        gradients_p = tape.gradient(loss_p, variables_p)
        self.optimizer_p.apply_gradients(zip(gradients_p, variables_p))
        
                   
        # if (((iteration % self.plot_steps) == 0) or (iteration == self.start_train)) and (not self.pretraining_mode):
        #     with self.summary_writer.as_default():
                
        #         tf.summary.scalar('Mean Squared Loss/Train', loss, step=iteration)
        #         # tf.summary.scalar('Train Mean Squared Loss/BaselineNet', loss_baseline, step=iteration)
        #         # tf.summary.scalar('Learning Rate/Initial LR', self.optimizer.learning_rate, step=iteration)
        #         tf.summary.scalar('Learning Rate/LR', self.optimizer._decayed_lr(tf.float32), step=iteration)
        #         if self.clipgrad == 'globnorm':
        #             tf.summary.scalar('Norm/Global grad norm', gbnorm, step=iteration)
        #             if iteration > self.clipglob_steps:
        #                 tf.summary.scalar('Norm/Clip Glob', self.clipglob, step=iteration)
                    
        #         else:
        #             gbnorm = tf.linalg.global_norm(gradients)
        #             tf.summary.scalar('Norm/Global grad norm', gbnorm, step=iteration)
                
                
                
        #         if self.plot_hist and ((iteration % self.plot_steps_hist) == 0):
        #             for i,layer in enumerate(self.model.layers[1:]):
        #                 with tf.name_scope('layer{0}'.format(i)):
        #                     if 'dense' in layer.name:                            
        #                         tf.summary.histogram(layer.name + '/weights', 
        #                                               layer.get_weights()[0], step=iteration)
        #                         tf.summary.histogram(layer.name + '/biases', 
        #                                               layer.get_weights()[1], step=iteration)
        #                         tf.summary.histogram(layer.name + '/Wx+b_pre_activation', 
        #                                               layer.out, step=iteration)
        #                     elif 'activation' in layer.name:        
        #                         tf.summary.histogram(layer.name + '/activation', 
        #                                               layer.out, step=iteration)             
        #                     elif 'batch' in layer.name:               
        #                         tf.summary.histogram(layer.name + '/bnorm_inputs_1', 
        #                                               self.model.bninputs[:,0], step=iteration)
        #                         tf.summary.histogram(layer.name + '/bnorm_inputs_2', 
        #                                               self.model.bninputs[:,1], step=iteration)
        #                         tf.summary.histogram(layer.name + '/inputs_1', 
        #                                               self.model.inputs[:,0], step=iteration)
        #                         tf.summary.histogram(layer.name + '/inputs_2', 
        #                                               self.model.inputs[:,1], step=iteration)
                                            
        #         for g,v in zip(gradients, variables):

        #             grad_mean = tf.reduce_mean(g)
        #             grad_square_sum = tf.reduce_sum(tf.math.square(g))
        #             grad_norm = tf.sqrt(grad_square_sum)
        #             sq_norm = tf.square(grad_norm)
        #             tf.summary.scalar(v.name.split('/',1)[1] + 'Gradients/grad_mean', grad_mean, step=iteration)
        #             tf.summary.scalar(v.name.split('/',1)[1] + 'Gradients/grad_norm', grad_norm, step=iteration)
        #             tf.summary.scalar(v.name.split('/',1)[1] + 'Gradients/grad_sq_norm', sq_norm, step=iteration)

                    
        #             tf.summary.histogram(v.name.split('/',1)[1] + 'hist/grads', g, step=iteration)
        #             tf.summary.histogram(v.name.split('/',1)[1] + 'hist/grads_squared', tf.square(g), step=iteration)

        #             slots = self.optimizer.get_slot_names()

        #             if slots:
        #                 for slot in slots:
        #                     tf.summary.scalar(v.name.split('/',1)[1] + 'Gradients/' + slot,
        #                                       tf.reduce_mean(self.optimizer.get_slot(v,slot)), step=iteration)
                        
        #                 if self.optimizer_name == 'adam':
        #                     mean_slt_ratio = tf.reduce_mean(self.optimizer.get_slot(v,'m')/
        #                                                self.optimizer.get_slot(v,'v')) 
        #                     mean_slt_sqrtratio = tf.reduce_mean(self.optimizer.get_slot(v,'m')/
        #                                                    (tf.math.sqrt(self.optimizer.get_slot(v,'v')) + self.eps_opt))
        #                     tf.summary.scalar(v.name.split('/',1)[1] + 'Gradients/mv_ratio', 
        #                                       mean_slt_ratio,step=iteration)                        
        #                     tf.summary.scalar(v.name.split('/',1)[1] + 'Gradients/mv_sqrt_ratio',
        #                                       mean_slt_sqrtratio, step=iteration)
                            
        #                     slt_ratio = (self.optimizer.get_slot(v,'m')/
        #                                                self.optimizer.get_slot(v,'v')) 
        #                     slt_sqrtratio = (self.optimizer.get_slot(v,'m')/
        #                                                    (tf.math.sqrt(self.optimizer.get_slot(v,'v')) + self.eps_opt))
        #                     tf.summary.histogram(v.name.split('/',1)[1] + 'hist/mv_ratio', 
        #                                       slt_ratio,step=iteration)                        
        #                     tf.summary.histogram(v.name.split('/',1)[1] + 'hist/mv_sqrt_ratio',
        #                                       slt_sqrtratio, step=iteration)

        #         self.summary_writer.flush()
        # elif (((iteration % self.plot_steps) == 0) or (iteration == self.start_train)) and self.pretraining_mode:
        #     with self.summary_writer.as_default():
        #         tf.summary.scalar('Mean Squared Loss/PreTrain', loss, step=iteration)
        #         tf.summary.scalar('Learning Rate/PReTrain_LR', self.optimizer._decayed_lr(tf.float32), step=iteration)
        #         self.summary_writer.flush()
          
    def noisy_action(self, states):
        
        a = self.p_model(np.atleast_2d(states.astype('float32')), training=False)
        a += self.action_noise()
        act_min, act_max = -self.action_limit, self.action_limit
        return np.clip(a, act_min, act_max)
    
    def uniform_action(self):
        act_min, act_max = -self.action_limit, self.action_limit
        a = np.random.uniform(act_min,act_max,self.nb_actions)
        return a
        
    def add_experience(self, exp):
        
        if self.use_PER:
            self.PERmemory.add(exp)
        else:
            if len(self.experience['s']) >= self.max_experiences:
                for key in self.experience.keys():
                    self.experience[key].pop(0)
            for key, value in exp.items():
                self.experience[key].append(value)
            
    def add_test_experience(self):

        if self.use_PER:
            ids = np.random.randint(low=0, high=self.min_experiences, size=self.batch_size)
            self.test_experience = {'s': np.asarray([self.PERmemory.experience['s'][i] for i in ids]),
                                    'a': np.asarray([self.PERmemory.experience['a'][i] for i in ids]),
                                    'r': np.asarray([self.PERmemory.experience['r'][i] for i in ids]),
                                    's2':np.asarray([self.PERmemory.experience['s2'][i] for i in ids]),
                                    'f':np.asarray([self.PERmemory.experience['f'][i] for i in ids])}

        else:
            ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)
            self.test_experience = {'s': np.asarray([self.experience['s'][i] for i in ids]),
                                    'a': np.asarray([self.experience['a'][i] for i in ids]),
                                    'r': np.asarray([self.experience['r'][i] for i in ids]),
                                    's2':np.asarray([self.experience['s2'][i] for i in ids]),
                                    'f':np.asarray([self.experience['f'][i] for i in ids])}



    def copy_weights(self, TrainNet):
        variables1 = self.Q_model.trainable_variables
        variables2 = TrainNet.Q_model.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            vsoft = (1 - self.tau) * v1 + self.tau * v2
            v1.assign(vsoft.numpy())
            
        variables1 = self.p_model.trainable_variables
        variables2 = TrainNet.p_model.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            vsoft = (1 - self.tau) * v1 + self.tau * v2
            v1.assign(vsoft.numpy())
                
                