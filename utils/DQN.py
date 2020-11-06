# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 16:53:31 2020

@author: aless
"""

# inspired by https://towardsdatascience.com/deep-reinforcement-learning-build-a-deep-q-network-dqn-to-play-cartpole-with-tensorflow-2-and-gym-8e105744b998
import numpy as np
import pdb
from typing import Union, Optional

import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers.schedules import InverseTimeDecay
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.keras.optimizers.schedules import PolynomialDecay
# from tensorflow_addons.optimizers import RectifiedAdam
from utils.SumTreePER import PER_buffer

#tf.debugging.set_log_device_placement(True)
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# tf_config = tf.ConfigProto()
# tf_config.gpu_options.allow_growth = True
# tf_config.gpu_options.per_process_gpu_memory_fraction = 0.9
# tf_config.allow_soft_placement = True


################################ Class to create a Deep Q Network model ################################ 
class DeepNetworkModel(tf.keras.Model):
    
    def __init__(self, 
                 seed: int, 
                 input_shape: int, 
                 hidden_units: list, 
                 num_actions: int, 
                 batch_norm_input: bool, 
                 batch_norm_hidden: bool,
                 activation: str, 
                 kernel_initializer: str, 
                 modelname: str = 'Deep Q Network'):
        # call the parent constructor
        super(DeepNetworkModel, self).__init__(name=modelname)

        # set dimensionality of input/output depending on the model
        inp_shape = input_shape
        out_shape = num_actions
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
            self.bnorm_layer = BatchNormalization(center=False,scale=False)
        
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
                self.hids.append(BatchNormalization())
        # output layer with linear activation by default
        self.output_layer = Dense(out_shape)


    def call(self, 
             inputs: Union[np.ndarray or tf.Tensor], 
             training: bool = True, 
             store_intermediate_outputs: bool = False):
        
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
                if 'batch' in layer.name:
                    z = layer(z, training)
                else:
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
                if 'batch' in layer.name:
                    z = layer(z, training)
                else:
                    z = layer(z)
            # build the output layer
            z = self.output_layer(z)
        return z

################################ Class to create a Deep Recurrent Q Network model ################################  
class DeepRecurrentNetworkModel(tf.keras.Model):
    
    def __init__(self, 
                 seed: int, 
                 input_shape: int, 
                 hidden_memory_units: list,
                 hidden_units: list, 
                 num_actions: int, 
                 batch_norm_input: bool, 
                 batch_norm_hidden: bool,
                 activation: str, 
                 kernel_initializer: str, 
                 modelname: str = 'Deep Recurrent Q Network'):
        # call the parent constructor
        super(DeepRecurrentNetworkModel, self).__init__(name=modelname)
        # set dimensionality of input/output depending on the model
        inp_shape = input_shape
        out_shape = num_actions
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
            self.bnorm_layer = BatchNormalization(center=False,scale=False)
        
        # set of hidden layers
        self.hids = []
        
        for i in range(len(hidden_memory_units)):
            if i == len(hidden_memory_units)-1:
                self.hids.append(LSTM(hidden_memory_units[i]))
            else:
                self.hids.append(LSTM(hidden_memory_units[i], return_sequences=True))           
            if self.batch_norm_hidden:
                self.hids.append(BatchNormalization())
        if hidden_units:
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
                    self.hids.append(BatchNormalization())
        # output layer with linear activation by default
        self.output_layer = Dense(out_shape)


    def call(self, 
             inputs: Union[np.ndarray or tf.Tensor], 
             training: bool = True, 
             store_intermediate_outputs: bool = False):

        if len(inputs.shape) != 3:
            inputs = tf.reshape(inputs, tf.TensorShape([1]).concatenate(inputs.shape))
        
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
                if 'batch' in layer.name:
                    z = layer(z, training)
                else:
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
                if 'batch' in layer.name:
                    z = layer(z, training)
                else:
                    z = layer(z)
            # build the output layer
            z = self.output_layer(z)
        return z
    
 
############################### DQN ALGORITHM ################################
class DQN:
    
    def __init__(self, 
                 seed: int,
                 DQN_type: str,
                 recurrent_env: bool,
                 gamma: float,
                 max_experiences: int, 
                 update_target: str,
                 tau: float,
                 input_shape: int, 
                 hidden_units: list, 
                 hidden_memory_units: list,
                 batch_size: int, 
                 selected_loss: str,
                 lr: float, 
                 start_train: int, 
                 optimizer_name: str,
                 batch_norm_input: str, 
                 batch_norm_hidden: str, 
                 activation: str, 
                 kernel_initializer: str, 
                 plot_hist: bool, 
                 plot_steps_hist: int, 
                 plot_steps: int,  
                 summary_writer, #TODO need to add proper type hint
                 action_space, 
                 use_PER: bool = False, 
                 PER_e: Optional[float] = None,
                 PER_a: Optional[float] = None,
                 PER_b: Optional[float] = None,
                 final_PER_b: Optional[float] = None,
                 PER_b_steps: Optional[int] = None,
                 PER_b_growth: Optional[float] = None, 
                 final_PER_a: Optional[float] = None,
                 PER_a_steps: Optional[int] = None,
                 PER_a_growth: Optional[float] = None, 
                 clipgrad: bool = False, 
                 clipnorm: Optional[Union[str or float]] = None, 
                 clipvalue: Optional[Union[str or float]] = None, 
                 clipglob_steps: Optional[int] = None, 
                 beta_1: float = 0.9, 
                 beta_2: float = 0.999, 
                 eps_opt: float = 1e-07, 
                 std_rwds: bool = False,
                 lr_schedule: Optional[str] = None, 
                 exp_decay_steps: Optional[int] = None, 
                 exp_decay_rate: Optional[float] = None,
                 rng = None,
                 modelname: str = 'Deep Network',
                 pretraining_mode: bool = False,
                 stop_train: int = 1e+10):
        
        if rng is not None: self.rng = rng
        
        self.batch_size = batch_size
                    
        if lr_schedule == 'exponential':
            lr = ExponentialDecay(initial_learning_rate=lr, 
                                  decay_steps=exp_decay_steps, 
                                  decay_rate=exp_decay_rate)
        elif lr_schedule == 'piecewise':
            lr = PiecewiseConstantDecay(boundaries=[100000], 
                                        values=[0.001, 0.0001])
        elif lr_schedule == 'inverse_time':
            lr = InverseTimeDecay(initial_learning_rate=lr, 
                                  decay_steps=exp_decay_steps, 
                                  decay_rate=exp_decay_rate)
        elif lr_schedule == 'polynomial':
            lr = PolynomialDecay(initial_learning_rate=lr, 
                                 decay_steps=exp_decay_steps, 
                                 end_learning_rate=1e-6, 
                                 power=1.0,)
            
        
        if optimizer_name == 'sgd':
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.0, 
                                                     nesterov=False)
        elif optimizer_name == 'sgdmom':
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.95, 
                                                     nesterov=False)
        elif optimizer_name == 'sgdnest':
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.0, 
                                                     nesterov=True)
        elif optimizer_name == 'adadelta':
            self.optimizer = tf.keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95, 
                                                          epsilon=eps_opt)
        elif optimizer_name == 'adagrad':
            self.optimizer = tf.keras.optimizers.Adagrad(learning_rate=lr, initial_accumulator_value=0.1,
                                                         epsilon=eps_opt)
        elif optimizer_name == 'adamax':
            self.optimizer = tf.keras.optimizers.Adamax(learning_rate=lr, beta_1=beta_1, beta_2=beta_2,
                                                        epsilon=eps_opt)
        elif optimizer_name == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1, beta_2=beta_2, 
                                                      epsilon=eps_opt, amsgrad=False)
        elif optimizer_name == 'amsgrad':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1, beta_2=beta_2, 
                                                      epsilon=eps_opt, amsgrad=True)
        elif optimizer_name == 'nadam':
            self.optimizer = tf.keras.optimizers.Nadam(learning_rate=lr, beta_1=beta_1, beta_2=beta_2, 
                                                      epsilon=eps_opt)
        elif optimizer_name == 'rmsprop':
            self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr, rho=beta_1, momentum=0.0, epsilon=eps_opt,
                                                         centered=False)
        # elif optimizer_name == 'radam':
        #     self.optimizer = RectifiedAdam(lr=lr,total_steps=1500000,warmup_proportion=0.025,min_lr=1e-5,
        #                                     epsilon=eps_opt)
        
        self.recurrent_env = recurrent_env
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
        self.stop_train = stop_train
        self.max_experiences = max_experiences
        self.action_space = action_space
        self.num_actions = len(self.action_space.values)
        self.batch_norm_input = batch_norm_input
        self.batch_norm_hidden = batch_norm_hidden
        if recurrent_env:
            self.model = DeepRecurrentNetworkModel(seed, input_shape, hidden_memory_units, hidden_units, self.num_actions, 
                                           batch_norm_input, batch_norm_hidden, activation, kernel_initializer, 
                                           modelname)
        else:
            self.model = DeepNetworkModel(seed, input_shape, hidden_units, self.num_actions, 
                                           batch_norm_input, batch_norm_hidden, activation, kernel_initializer, 
                                           modelname)
        
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
        self.std_rwds = std_rwds
                
        if self.std_rwds:
            self.rwds_run_mean = 0
            self.rwds_run_std = 0
        
        if self.selected_loss == 'mse':
            self.loss = tf.keras.losses.MeanSquaredError()
        elif self.selected_loss == 'huber':
            self.loss = tf.keras.losses.Huber()
  
    def train(self, TargetNet, iteration, env=None):
        if iteration < self.start_train or iteration > self.stop_train:
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
            ids = self.rng.randint(low=0, high=len(self.experience['s']), size=self.batch_size)
            states = np.asarray([self.experience['s'][i] for i in ids])
            actions = np.asarray([self.experience['a'][i] for i in ids])
            rewards = np.asarray([self.experience['r'][i] for i in ids])
            states_next = np.asarray([self.experience['s2'][i] for i in ids])
            if self.pretraining_mode:
                factors = np.asarray([self.experience['f'][i] for i in ids])

        with tf.GradientTape() as tape: #persistent=True
            
            # compute current action values
            # find index of actions included in the batch
            encoded_actions = [self.action_space.values.tolist().index(act) 
                                for act in actions]
            selected_action_values = tf.math.reduce_sum(
                self.model(np.atleast_2d(states.astype('float32')),
                           store_intermediate_outputs=True) * tf.one_hot(encoded_actions, self.num_actions), axis=1)
            
            if self.pretraining_mode:
                assert env, "Market Env not passed as argument"
                #compute fixed target for pretraining
                OptRate, DiscFactorLoads = env.opt_trading_rate_disc_loads()
                # Optimal traded quantity between period

                if len(DiscFactorLoads) == 1:
                    retprod = factors * DiscFactorLoads
                else:
                    retprod = factors @ DiscFactorLoads
                
                OptNextHolding = (1 - OptRate) * states_next[:,1] + OptRate * \
                              (1/(env.kappa * (env.sigma)**2)) * retprod
                
                OptAction = OptNextHolding - states[:,1]
                DiscOptActionIdx = [(np.abs(self.action_space.values - a)).argmin() for a in OptAction]

                #DiscOptAction = [self.action_space.values[i] for i in DiscOptActionIdx]
                value_next = tf.math.reduce_sum(
                        TargetNet.model(np.atleast_2d(states_next.astype('float32'))) * tf.one_hot(DiscOptActionIdx, self.num_actions), axis=1)
            else:
                # compute target action values
                if self.DQN_type == 'DQN':
                    value_next = np.max(TargetNet.model(states_next.astype('float32')), axis=1)
                elif self.DQN_type == 'DDQN':
                    greedy_target_action = tf.math.argmax(self.model(states_next.astype('float32')), 1)
                    value_next = tf.math.reduce_sum(
                        TargetNet.model(states_next.astype('float32')) * tf.one_hot(greedy_target_action, self.num_actions), axis=1)


            if self.std_rwds:
                if iteration == self.start_train:
                    actual_values = np.array(rewards+self.gamma*value_next)
                    sample_mean = actual_values.mean()
                    sample_std = actual_values.std()
                    self.rwds_run_mean = sample_mean
                    self.rwds_run_std = sample_std
                    
                    std_rewards = (actual_values - self.rwds_run_mean)/self.rwds_run_std
                    actual_values = std_rewards#+self.gamma*value_next
                else:
                    actual_values = np.array(rewards+self.gamma*value_next)
                    sample_mean = actual_values.mean()
                    sample_std = actual_values.std()
                    self.rwds_run_mean = 0.99 * self.rwds_run_mean + (1 - 0.99) * sample_mean
                    self.rwds_run_std = 0.99 * self.rwds_run_std + (1 - 0.99) * sample_std
                    
                    std_rewards = (actual_values - self.rwds_run_mean)/self.rwds_run_std
                    actual_values = std_rewards#+self.gamma*value_next
                
            else:
                actual_values = rewards+self.gamma*value_next

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
                self.PERmemory.batch_update(b_idx, np.abs(actual_values-selected_action_values))
                # compute loss function for the train model
                loss = self.loss(y_true=actual_values,y_pred=selected_action_values,
                                 sample_weight=scaled_w_IS.reshape(-1,1))
            else:
                loss = self.loss(y_true=actual_values,y_pred=selected_action_values)
            
        variables = self.model.trainable_variables

        # compute gradient of the loss with respect to the variables (weights)
        gradients = tape.gradient(loss, variables)

        if self.clipgrad == 'norm':
            gradients = [(tf.clip_by_norm(gv, self.clipnorm)) for gv in gradients]
        elif self.clipgrad == 'value':
            gradients = [(tf.clip_by_value(gv,-self.clipvalue, self.clipvalue)) for gv in gradients]
        elif self.clipgrad == 'globnorm':
            #gbnorm = tf.linalg.global_norm(gradients)
            if iteration <= self.clipglob_steps:
                gbnorm = tf.linalg.global_norm(gradients)
                self.global_norms.append(gbnorm.numpy())
                if iteration == self.clipglob_steps:
                    self.clipglob = np.mean(self.global_norms)
            else:
                # gbnorm = tf.linalg.global_norm(gradients)
                # self.global_norms.append(gbnorm.numpy())
                # clipglob = self.beta_1 * np.mean(self.global_norms) + (1 - self.beta_1) * gbnorm                
                gradients, gbnorm  = tf.clip_by_global_norm(gradients,self.clipglob)#, use_norm = gbnorm)
        
        # provide a list of (gradient, variable) pairs.
        self.optimizer.apply_gradients(zip(gradients, variables))
        
        if (((iteration % self.plot_steps) == 0) or (iteration == self.start_train)) and (not self.pretraining_mode):
            with self.summary_writer.as_default():
                
                tf.summary.scalar('Mean Squared Loss/Train', loss, step=iteration)
                # tf.summary.scalar('Train Mean Squared Loss/BaselineNet', loss_baseline, step=iteration)
                # tf.summary.scalar('Learning Rate/Initial LR', self.optimizer.learning_rate, step=iteration)
                tf.summary.scalar('Learning Rate/LR', self.optimizer._decayed_lr(tf.float32), step=iteration)
                if self.clipgrad == 'globnorm':
                    tf.summary.scalar('Norm/Global grad norm', gbnorm, step=iteration)
                    if iteration > self.clipglob_steps:
                        tf.summary.scalar('Norm/Clip Glob', self.clipglob, step=iteration)
                    
                else:
                    gbnorm = tf.linalg.global_norm(gradients)
                    tf.summary.scalar('Norm/Global grad norm', gbnorm, step=iteration)
                
                
                
                if self.plot_hist and ((iteration % self.plot_steps_hist) == 0):
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
                
                # if self.std_rwds:
                #     tf.summary.histogram('Rewards/Original', 
                #                           rewards, step=iteration)
                #     tf.summary.histogram('Rewards/Normalized', 
                #                           std_rewards, step=iteration)
                # else:
                #     tf.summary.histogram('Rewards/Original', 
                #                          rewards, step=iteration)
                            
                for g,v in zip(gradients, variables):

                    grad_mean = tf.reduce_mean(g)
                    grad_square_sum = tf.reduce_sum(tf.math.square(g))
                    grad_norm = tf.sqrt(grad_square_sum)
                    sq_norm = tf.square(grad_norm)
                    tf.summary.scalar(v.name.split('/',1)[1] + 'Gradients/grad_mean', grad_mean, step=iteration)
                    tf.summary.scalar(v.name.split('/',1)[1] + 'Gradients/grad_norm', grad_norm, step=iteration)
                    tf.summary.scalar(v.name.split('/',1)[1] + 'Gradients/grad_sq_norm', sq_norm, step=iteration)

                    
                    tf.summary.histogram(v.name.split('/',1)[1] + 'hist/grads', g, step=iteration)
                    tf.summary.histogram(v.name.split('/',1)[1] + 'hist/grads_squared', tf.square(g), step=iteration)

                    slots = self.optimizer.get_slot_names()

                    if slots:
                        for slot in slots:
                            tf.summary.scalar(v.name.split('/',1)[1] + 'Gradients/' + slot,
                                              tf.reduce_mean(self.optimizer.get_slot(v,slot)), step=iteration)
                        
                        if self.optimizer_name == 'adam':
                            mean_slt_ratio = tf.reduce_mean(self.optimizer.get_slot(v,'m')/
                                                       self.optimizer.get_slot(v,'v')) 
                            mean_slt_sqrtratio = tf.reduce_mean(self.optimizer.get_slot(v,'m')/
                                                           (tf.math.sqrt(self.optimizer.get_slot(v,'v')) + self.eps_opt))
                            tf.summary.scalar(v.name.split('/',1)[1] + 'Gradients/mv_ratio', 
                                              mean_slt_ratio,step=iteration)                        
                            tf.summary.scalar(v.name.split('/',1)[1] + 'Gradients/mv_sqrt_ratio',
                                              mean_slt_sqrtratio, step=iteration)
                            
                            slt_ratio = (self.optimizer.get_slot(v,'m')/
                                                       self.optimizer.get_slot(v,'v')) 
                            slt_sqrtratio = (self.optimizer.get_slot(v,'m')/
                                                           (tf.math.sqrt(self.optimizer.get_slot(v,'v')) + self.eps_opt))
                            tf.summary.histogram(v.name.split('/',1)[1] + 'hist/mv_ratio', 
                                              slt_ratio,step=iteration)                        
                            tf.summary.histogram(v.name.split('/',1)[1] + 'hist/mv_sqrt_ratio',
                                              slt_sqrtratio, step=iteration)

                self.summary_writer.flush()
        elif (((iteration % self.plot_steps) == 0) or (iteration == self.start_train)) and self.pretraining_mode:
            with self.summary_writer.as_default():
                tf.summary.scalar('Mean Squared Loss/PreTrain', loss, step=iteration)
                tf.summary.scalar('Learning Rate/PReTrain_LR', self.optimizer._decayed_lr(tf.float32), step=iteration)
                self.summary_writer.flush()
    
    # def test(self, TargetNet,iteration):
    #     if ((iteration % self.plot_steps) == 0) or (iteration == self.start_train):
    #         if self.test_experience:
    
    #             encoded_test_actions = [self.action_space.values.tolist().index(act) 
    #                                 for act in self.test_experience['a']]     
                
    #             test_pred = self.predict(self.test_experience['s'])
                
    #             selected_test_action_values = tf.math.reduce_sum( 
    #                 test_pred * tf.one_hot(encoded_test_actions, self.num_actions), axis=1)
                
                
    #             test_loss = self.loss(self.actual_test_values,selected_test_action_values)
    #             # compute average maximum Q values for the fixed test states
    #             Q_avg = tf.reduce_mean(tf.reduce_max(test_pred, axis=1))
                
    #             with self.summary_writer.as_default(): 
    #                 tf.summary.scalar('Q_avg_test/TrainNet', Q_avg, step=iteration)
    #                 tf.summary.scalar('Mean Squared Loss/Test', test_loss.numpy(), step=iteration)
    #                 self.summary_writer.flush()

    # def compute_test_target(self, TargetNet):
    #     if self.test_experience:
    #         if self.DQN_type == 'DQN':
    #             test_value_next = np.max(TargetNet.predict(self.test_experience['s2']), axis=1)
    #         elif self.DQN_type == 'DDQN':
    #             greedy_target_action = tf.math.argmax(self.predict(self.test_experience['s2']), 1)
    #             test_value_next = tf.math.reduce_sum(
    #                 TargetNet.predict(self.test_experience['s2']) * tf.one_hot(greedy_target_action, self.num_actions), axis=1)
            
    #         actual_test_values = self.test_experience['r'] + self.gamma*test_value_next
    #         self.actual_test_values = actual_test_values

    
    def compute_portfolio_distance(self, env, OptRate, DiscFactorLoads, iteration):
        
        
        states = self.test_experience['s2']
        factors = self.test_experience['f']
        
        max_action = self.action_space.values[tf.math.argmax(self.model(np.atleast_2d(states.astype('float32'))),
                                                     axis=1)]
                
        # Optimal traded quantity between period
        OptNextHolding = (1 - OptRate) * states[:,1] + OptRate * \
                      (1/(env.kappa * (env.sigma)**2)) * \
                        np.sum(DiscFactorLoads * factors, axis=1)
        
        OptAction = OptNextHolding - states[:,1]
        DiscOptAction = [self.action_space.values[(np.abs(self.action_space.values - a)).argmin()] for a in OptAction]
 
        pdist = tf.reduce_mean(tf.math.squared_difference(max_action,DiscOptAction))

        with self.summary_writer.as_default(): 
            tf.summary.scalar('Portfolio Distance/Test Squared Loss', pdist, step=iteration)
            self.summary_writer.flush()
        
    
    def eps_greedy_action(self, states, epsilon):
        if self.rng.random() < epsilon:
            return self.rng.choice(self.action_space.values)
        else:
            return self.action_space.values[np.argmax(self.model(np.atleast_2d(states.astype('float32')), training = False)[0])]
    
    def alpha_beta_greedy_action(self, states, factors, epsilon, OptRate, DiscFactorLoads, alpha, env):
        if self.rng.random() < epsilon:
            if self.rng.random() < alpha:
                return self.rng.choice(self.action_space.values)
            else:
                if not self.recurrent_env:
                    if len(DiscFactorLoads) == 1:
                        retprod = factors * DiscFactorLoads
                    else:
                        retprod = factors @ DiscFactorLoads
                        
                        OptNextHolding = (1 - OptRate) * states[1] + OptRate * \
                                      (1/(env.kappa * (env.sigma)**2)) * retprod
                        
                        OptAction = OptNextHolding - states[1]
                else:
                    if len(DiscFactorLoads) == 1:
                        retprod = factors[-1,:] * DiscFactorLoads
                    else:
                        retprod = factors[-1,:] @ DiscFactorLoads
                    
                    OptNextHolding = (1 - OptRate) * states[-1,1] + OptRate * \
                                  (1/(env.kappa * (env.sigma)**2)) * retprod
                    
                    OptAction = OptNextHolding - states[-1,1]
                DiscOptActionIdx = (np.abs(self.action_space.values - OptAction)).argmin()
                DiscOptAction = self.action_space.values[DiscOptActionIdx]
                return DiscOptAction
        else:
            return self.action_space.values[np.argmax(self.model(np.atleast_2d(states.astype('float32')), training = False)[0])]
        
    def greedy_action(self, states):
        return self.action_space.values[np.argmax(self.model(np.atleast_2d(states.astype('float32')), training = False)[0])]
        
    def add_experience(self, exp):
        
        if self.use_PER:
            self.PERmemory.add(exp)
        else:
            if len(self.experience['s']) >= self.max_experiences:
                for key in self.experience.keys():
                    self.experience[key].pop(0)
            for key, value in exp.items():
                self.experience[key].append(value)
            
    # def add_test_experience(self):

    #     if self.use_PER:
    #         ids = np.random.randint(low=0, high=self.min_experiences, size=self.batch_size)
    #         self.test_experience = {'s': np.asarray([self.PERmemory.experience['s'][i] for i in ids]),
    #                                 'a': np.asarray([self.PERmemory.experience['a'][i] for i in ids]),
    #                                 'r': np.asarray([self.PERmemory.experience['r'][i] for i in ids]),
    #                                 's2':np.asarray([self.PERmemory.experience['s2'][i] for i in ids]),
    #                                 'f':np.asarray([self.PERmemory.experience['f'][i] for i in ids])}

    #     else:
    #         ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)
    #         self.test_experience = {'s': np.asarray([self.experience['s'][i] for i in ids]),
    #                                 'a': np.asarray([self.experience['a'][i] for i in ids]),
    #                                 'r': np.asarray([self.experience['r'][i] for i in ids]),
    #                                 's2':np.asarray([self.experience['s2'][i] for i in ids]),
    #                                 'f':np.asarray([self.experience['f'][i] for i in ids])}



    def copy_weights(self, TrainNet):
        if self.update_target == 'soft':
            variables1 = self.model.trainable_variables
            variables2 = TrainNet.model.trainable_variables
            for v1, v2 in zip(variables1, variables2):
                vsoft = (1 - self.tau) * v1 + self.tau * v2
                v1.assign(vsoft.numpy())
        else:
            variables1 = self.model.trainable_variables
            variables2 = TrainNet.model.trainable_variables
            for v1, v2 in zip(variables1, variables2):
                v1.assign(v2.numpy())
                
