# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 12:01:33 2020

@author: aless
"""

import tensorflow as tf
from tensorflow_addons.optimizers import RectifiedAdam, AdamW
import numpy as np
import pdb
import sys
from typing import Union, Optional
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.initializers import RandomUniform, VarianceScaling
from utils.exploration import (
    OrnsteinUhlenbeckActionNoise,
    GaussianActionNoise,
    PER_buffer,
)
from utils.math_tools import scale_action


################################ Class to create a Deep Q Network model ################################
# FEEDFORWARD NETWORKS
class CriticNetwork(tf.keras.Model):
    def __init__(
        self,
        seed: int,
        num_states: int,
        hidden_units: list,
        num_actions: int,
        batch_norm_input: bool,
        batch_norm_hidden: bool,
        activation: str,
        kernel_initializer: str,
        output_init: float,
        delayed_actions: bool,
        modelname="Critic Network",
    ):
        # call the parent constructor
        super(CriticNetwork, self).__init__(name=modelname)

        # set dimensionality of input/output depending on the model
        self.delayed_actions = delayed_actions
        if self.delayed_actions:
            inp_shape = (num_states,)
        else:
            inp_shape = (num_states + num_actions,)
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
            self.bnorm_layer = BatchNormalization(center=False, scale=False)

        # set of hidden layers
        self.hids = []

        for i in hidden_units:
            if kernel_initializer == "ddpg_type":
                self.hids.append(
                    Dense(
                        i,
                        kernel_initializer=VarianceScaling(
                            scale=(1.0 / 3.0), mode="fan_in", distribution="uniform"
                        ),
                    )
                )
            else:
                self.hids.append(Dense(i, kernel_initializer=kernel_initializer))
            # check what type of activation is set
            if activation == "leaky_relu":
                leaky_relu = tf.nn.leaky_relu
                self.hids.append(Activation(leaky_relu))
            elif activation == "relu6":
                relu6 = tf.nn.relu6
                self.hids.append(Activation(relu6))
            elif activation == "elu":
                elu = tf.nn.elu
                self.hids.append(Activation(elu))
            else:
                self.hids.append(Activation(activation))

            if self.batch_norm_hidden:
                self.hids.append(BatchNormalization())
        # output layer with linear activation by default
        self.output_layer = Dense(
            out_shape, kernel_initializer=RandomUniform(-output_init, output_init)
        )

    def call(self, states, actions, training=True, store_intermediate_outputs=False):

        if self.delayed_actions:
            inputs = states
        else:
            inputs = tf.concat([states, actions], axis=1)

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

            if self.batch_norm_hidden:
                for i, layer in enumerate(self.hids):
                    if "batch" in layer.name:
                        z = layer(z, training)
                    elif i == len(self.hids) - 3 and self.delayed_actions:
                        z = layer(tf.concat([z, actions], axis=1))
                    else:
                        z = layer(z)
                    layer.out = z
            else:
                for i, layer in enumerate(self.hids):
                    if i == len(self.hids) - 2 and self.delayed_actions:
                        actions = (actions - actions.mean()) / actions.std()
                        z = layer(tf.concat([z, actions], axis=1))
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
            if self.batch_norm_hidden:
                for i, layer in enumerate(self.hids):
                    if "batch" in layer.name:
                        z = layer(z, training)
                    elif i == len(self.hids) - 3 and self.delayed_actions:
                        z = layer(tf.concat([z, actions], axis=1))
                    else:
                        z = layer(z)
            else:
                for i, layer in enumerate(self.hids):
                    if i == len(self.hids) - 2 and self.delayed_actions:
                        z = layer(tf.concat([z, actions], axis=1))
                    else:
                        z = layer(z)

            # build the output layer
            z = self.output_layer(z)
        return z


class ActorNetwork(tf.keras.Model):
    def __init__(
        self,
        seed: int,
        num_states: int,
        hidden_units: list,
        num_actions: int,
        batch_norm_input: bool,
        batch_norm_hidden: bool,
        activation: str,
        kernel_initializer: str,
        output_init: float,
        modelname="Actor Network",
    ):

        # call the parent constructor
        super(ActorNetwork, self).__init__(name=modelname)

        # set dimensionality of input/output depending on the model
        inp_shape = num_states
        out_shape = num_actions
        # set boundaries for action

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
            self.bnorm_layer = BatchNormalization(center=False, scale=False)

        # set of hidden layers
        self.hids = []

        for i in hidden_units:
            if kernel_initializer == "ddpg_type":
                self.hids.append(
                    Dense(
                        i,
                        kernel_initializer=VarianceScaling(
                            scale=(1.0 / 3.0), mode="fan_in", distribution="uniform"
                        ),
                    )
                )
            else:
                self.hids.append(Dense(i, kernel_initializer=kernel_initializer))
            # check what type of activation is set
            if activation == "leaky_relu":
                leaky_relu = tf.nn.leaky_relu
                self.hids.append(Activation(leaky_relu))
            elif activation == "relu6":
                relu6 = tf.nn.relu6
                self.hids.append(Activation(relu6))
            elif activation == "elu":
                elu = tf.nn.elu
                self.hids.append(Activation(elu))
            else:
                self.hids.append(Activation(activation))

            if self.batch_norm_hidden:
                self.hids.append(BatchNormalization())
        # output layer with linear activation by default
        self.output_layer = Dense(
            out_shape,
            activation="tanh",
            kernel_initializer=RandomUniform(-output_init, output_init),
        )

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
                if "batch" in layer.name:
                    z = layer(z, training)
                else:
                    z = layer(z)
                layer.out = z

            # build the output layer
            z = self.output_layer(z)

            # z = z * self.action_limit
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
                if "batch" in layer.name:
                    z = layer(z, training)
                else:
                    z = layer(z)
            # build the output layer
            z = self.output_layer(z)
            # z = z * self.action_limit
        return z


####################################################################################
# RECURRENT NETWORKS
class CriticRecurrentNetwork(tf.keras.Model):
    def __init__(
        self,
        seed: int,
        num_states: int,
        hidden_memory_units: list,
        hidden_units: list,
        num_actions: int,
        batch_norm_input: bool,
        batch_norm_hidden: bool,
        activation: str,
        kernel_initializer: str,
        output_init: float,
        delayed_actions: bool,
        modelname="Critic Network",
    ):
        # call the parent constructor
        super(CriticRecurrentNetwork, self).__init__(name=modelname)

        # set dimensionality of input/output depending on the model
        self.delayed_actions = delayed_actions
        if self.delayed_actions:
            inp_shape = (num_states,)
        else:
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
            self.bnorm_layer = BatchNormalization(center=False, scale=False)

        # set of hidden layers
        self.hids = []

        for i in range(len(hidden_memory_units)):
            if i == len(hidden_memory_units) - 1:
                self.hids.append(LSTM(hidden_memory_units[i]))
            else:
                self.hids.append(LSTM(hidden_memory_units[i], return_sequences=True))
            if self.batch_norm_hidden:
                self.hids.append(BatchNormalization())
        if hidden_units:
            for i in hidden_units:
                if kernel_initializer == "ddpg_type":
                    self.hids.append(
                        Dense(
                            i,
                            kernel_initializer=VarianceScaling(
                                scale=(1.0 / 3.0), mode="fan_in", distribution="uniform"
                            ),
                        )
                    )
                else:
                    self.hids.append(Dense(i, kernel_initializer=kernel_initializer))
                # check what type of activation is set
                if activation == "leaky_relu":
                    leaky_relu = tf.nn.leaky_relu
                    self.hids.append(Activation(leaky_relu))
                elif activation == "relu6":
                    relu6 = tf.nn.relu6
                    self.hids.append(Activation(relu6))
                elif activation == "elu":
                    elu = tf.nn.elu
                    self.hids.append(Activation(elu))
                else:
                    self.hids.append(Activation(activation))

                if self.batch_norm_hidden:
                    self.hids.append(BatchNormalization())
        # output layer with linear activation by default
        self.output_layer = Dense(
            out_shape, kernel_initializer=RandomUniform(-output_init, output_init)
        )

    def call(self, states, actions, training=True, store_intermediate_outputs=False):
        if self.delayed_actions:
            inputs = states
        else:
            inputs = tf.concat([states, actions], axis=1)

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
            if self.batch_norm_hidden:
                for i, layer in enumerate(self.hids):
                    if "batch" in layer.name:
                        z = layer(z, training)
                    elif i == len(self.hids) - 3 and self.delayed_actions:
                        z = layer(tf.concat([z, actions], axis=1))
                    else:
                        z = layer(z)
                    layer.out = z
            else:
                for i, layer in enumerate(self.hids):
                    if i == len(self.hids) - 2 and self.delayed_actions:
                        z = layer(tf.concat([z, actions], axis=1))
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
            if self.batch_norm_hidden:
                for i, layer in enumerate(self.hids):
                    if "batch" in layer.name:
                        z = layer(z, training)

                    elif i == len(self.hids) - 3 and self.delayed_actions:
                        z = layer(tf.concat([z, actions], axis=1))

                    else:
                        z = layer(z)

            else:
                for i, layer in enumerate(self.hids):
                    if i == len(self.hids) - 2 and self.delayed_actions:
                        z = layer(tf.concat([z, actions], axis=1))

                    else:
                        z = layer(z)

            # build the output layer
            z = self.output_layer(z)
        return z


class ActorRecurrentNetwork(tf.keras.Model):
    def __init__(
        self,
        seed: int,
        num_states: int,
        hidden_memory_units: list,
        hidden_units: list,
        num_actions: int,
        batch_norm_input: bool,
        batch_norm_hidden: bool,
        activation: str,
        kernel_initializer: str,
        output_init: float,
        modelname="Actor Network",
    ):

        # call the parent constructor
        super(ActorRecurrentNetwork, self).__init__(name=modelname)

        # set dimensionality of input/output depending on the model
        inp_shape = num_states
        out_shape = num_actions
        # set boundaries for action

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
            self.bnorm_layer = BatchNormalization(center=False, scale=False)

        # set of hidden layers
        self.hids = []

        for i in range(len(hidden_memory_units)):
            if i == len(hidden_memory_units) - 1:
                self.hids.append(LSTM(hidden_memory_units[i]))
            else:
                self.hids.append(LSTM(hidden_memory_units[i], return_sequences=True))
            if self.batch_norm_hidden:
                self.hids.append(BatchNormalization())
        if hidden_units:
            for i in hidden_units:
                if kernel_initializer == "ddpg_type":
                    self.hids.append(
                        Dense(
                            i,
                            kernel_initializer=VarianceScaling(
                                scale=(1.0 / 3.0), mode="fan_in", distribution="uniform"
                            ),
                        )
                    )
                else:
                    self.hids.append(Dense(i, kernel_initializer=kernel_initializer))
                # check what type of activation is set
                if activation == "leaky_relu":
                    leaky_relu = tf.nn.leaky_relu
                    self.hids.append(Activation(leaky_relu))
                elif activation == "relu6":
                    relu6 = tf.nn.relu6
                    self.hids.append(Activation(relu6))
                elif activation == "elu":
                    elu = tf.nn.elu
                    self.hids.append(Activation(elu))
                else:
                    self.hids.append(Activation(activation))

                if self.batch_norm_hidden:
                    self.hids.append(BatchNormalization())
        # output layer with linear activation by default
        self.output_layer = Dense(
            out_shape,
            activation="tanh",
            kernel_initializer=RandomUniform(-output_init, output_init),
        )

    def call(self, inputs, training=True, store_intermediate_outputs=False):

        if len(inputs.shape) != 3:
            inputs = tf.reshape(inputs, [1] + inputs.shape)

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
                if "batch" in layer.name:
                    z = layer(z, training)
                else:
                    z = layer(z)
                layer.out = z
            # build the output layer
            z = self.output_layer(z)
            # z = z * self.action_limit
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
                if "batch" in layer.name:
                    z = layer(z, training)
                else:
                    z = layer(z)
            # build the output layer
            z = self.output_layer(z)
            # z = z * self.action_limit
        return z


############################### DDPG ALGORITHM ################################
# https://spinningup.openai.com/en/latest/algorithms/ddpg.html#exploration-vs-exploitation
class DDPG:
    def __init__(
        self,
        seed: int,
        recurrent_env: bool,
        gamma: float,
        max_experiences: int,
        update_target: str,
        tau_Q: float,
        tau_p: float,
        num_states: int,  # TODO check dimensionality
        num_actions: int,
        hidden_units_Q: list,
        hidden_units_p: list,
        hidden_memory_units: list,
        batch_size: int,
        selected_loss: str,
        lr_Q: float,
        lr_p: float,
        start_train: int,
        optimizer_name: str,
        batch_norm_input: str,
        batch_norm_hidden: str,
        activation_Q: str,
        activation_p: str,
        kernel_initializer: str,
        plot_hist: bool,
        plot_steps_hist: int,
        plot_steps: int,
        summary_writer,  # TODO need to add proper type hint
        stddev_noise: float,
        theta: float,
        mu: Union[float or np.ndarray],
        action_limit: Union[float or int],
        output_init: float,
        weight_decay_Q: bool,
        weight_decay_p: bool,
        delayed_actions: bool,
        noise: str,
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
        lr_schedule: Optional[str] = None,
        exp_decay_steps: Optional[int] = None,
        exp_decay_rate_Q: Optional[float] = None,
        exp_decay_rate_p: Optional[float] = None,
        DDPG_type: str = "DDPG",
        noise_clip: float = None,
        stddev_pol_noise: float = None,
        rng=None,
        modelname: str = "Train",
        pretraining_mode: bool = False,
    ):

        if rng is not None:
            self.rng = rng

        self.batch_size = batch_size

        if lr_schedule == "exponential":
            lr_Q = ExponentialDecay(
                initial_learning_rate=lr_Q,
                decay_steps=exp_decay_steps,
                decay_rate=exp_decay_rate_Q,
            )
            lr_p = ExponentialDecay(
                initial_learning_rate=lr_p,
                decay_steps=exp_decay_steps,
                decay_rate=exp_decay_rate_p,
            )
        elif lr_schedule == "piecewise":
            lr_Q = PiecewiseConstantDecay(boundaries=[500000], values=[0.001, 0.0001])
            lr_p = PiecewiseConstantDecay(boundaries=[500000], values=[0.001, 0.0001])

        if DDPG_type == "DDPG":
            if optimizer_name == "sgd":
                self.optimizer_Q = tf.keras.optimizers.SGD(
                    learning_rate=lr_Q, momentum=0.0, nesterov=False
                )
                self.optimizer_p = tf.keras.optimizers.SGD(
                    learning_rate=lr_p, momentum=0.0, nesterov=False
                )
            elif optimizer_name == "adam":
                self.optimizer_Q = tf.keras.optimizers.Adam(
                    learning_rate=lr_Q,
                    beta_1=beta_1,
                    beta_2=beta_2,
                    epsilon=eps_opt,
                    amsgrad=False,
                )
                self.optimizer_p = tf.keras.optimizers.Adam(
                    learning_rate=lr_p,
                    beta_1=beta_1,
                    beta_2=beta_2,
                    epsilon=eps_opt,
                    amsgrad=False,
                )
            elif optimizer_name == "rmsprop":
                self.optimizer_Q = tf.keras.optimizers.RMSprop(
                    learning_rate=lr_Q,
                    rho=beta_1,
                    momentum=0.0,
                    epsilon=eps_opt,
                    centered=False,
                )
                self.optimizer_p = tf.keras.optimizers.RMSprop(
                    learning_rate=lr_p,
                    rho=beta_1,
                    momentum=0.0,
                    epsilon=eps_opt,
                    centered=False,
                )
            elif optimizer_name == "adamw":
                self.optimizer_Q = AdamW(
                    weight_decay=weight_decay_Q,
                    learning_rate=lr_Q,
                    beta_1=beta_1,
                    beta_2=beta_2,
                    epsilon=eps_opt,
                    amsgrad=False,
                )
                # self.optimizer_p = AdamW(weight_decay = weight_decay, learning_rate=lr_p, beta_1=beta_1, beta_2=beta_2,
                #                                       epsilon=eps_opt, amsgrad=False)
                self.optimizer_p = tf.keras.optimizers.Adam(
                    learning_rate=lr_p,
                    beta_1=beta_1,
                    beta_2=beta_2,
                    epsilon=eps_opt,
                    amsgrad=False,
                )
            else:
                print("Choose an available optimizer!")
                sys.exit()

        elif DDPG_type == "TD3":
            if optimizer_name == "sgd":
                self.optimizer_Q1 = tf.keras.optimizers.SGD(
                    learning_rate=lr_Q, momentum=0.0, nesterov=False
                )
                self.optimizer_Q2 = tf.keras.optimizers.SGD(
                    learning_rate=lr_Q, momentum=0.0, nesterov=False
                )
                self.optimizer_p = tf.keras.optimizers.SGD(
                    learning_rate=lr_p, momentum=0.0, nesterov=False
                )
            elif optimizer_name == "adam":
                self.optimizer_Q1 = tf.keras.optimizers.Adam(
                    learning_rate=lr_Q,
                    beta_1=beta_1,
                    beta_2=beta_2,
                    epsilon=eps_opt,
                    amsgrad=False,
                )
                self.optimizer_Q2 = tf.keras.optimizers.Adam(
                    learning_rate=lr_Q,
                    beta_1=beta_1,
                    beta_2=beta_2,
                    epsilon=eps_opt,
                    amsgrad=False,
                )
                self.optimizer_p = tf.keras.optimizers.Adam(
                    learning_rate=lr_p,
                    beta_1=beta_1,
                    beta_2=beta_2,
                    epsilon=eps_opt,
                    amsgrad=False,
                )
            elif optimizer_name == "rmsprop":
                self.optimizer_Q1 = tf.keras.optimizers.RMSprop(
                    learning_rate=lr_Q,
                    rho=beta_1,
                    momentum=0.0,
                    epsilon=eps_opt,
                    centered=False,
                )
                self.optimizer_Q2 = tf.keras.optimizers.RMSprop(
                    learning_rate=lr_Q,
                    rho=beta_1,
                    momentum=0.0,
                    epsilon=eps_opt,
                    centered=False,
                )
                self.optimizer_p = tf.keras.optimizers.RMSprop(
                    learning_rate=lr_p,
                    rho=beta_1,
                    momentum=0.0,
                    epsilon=eps_opt,
                    centered=False,
                )
            elif optimizer_name == "adamw":
                self.optimizer_Q1 = AdamW(
                    weight_decay=weight_decay_Q,
                    learning_rate=lr_Q,
                    beta_1=beta_1,
                    beta_2=beta_2,
                    epsilon=eps_opt,
                    amsgrad=False,
                )
                self.optimizer_Q2 = AdamW(
                    weight_decay=weight_decay_Q,
                    learning_rate=lr_Q,
                    beta_1=beta_1,
                    beta_2=beta_2,
                    epsilon=eps_opt,
                    amsgrad=False,
                )
                self.optimizer_p = AdamW(
                    weight_decay=weight_decay_p,
                    learning_rate=lr_p,
                    beta_1=beta_1,
                    beta_2=beta_2,
                    epsilon=eps_opt,
                    amsgrad=False,
                )
            else:
                print("Choose an available optimizer!")
                sys.exit()

        self.beta_1 = beta_1
        self.eps_opt = eps_opt
        self.gamma = gamma
        self.use_PER = use_PER
        if self.use_PER:
            self.PERmemory = PER_buffer(
                PER_e,
                PER_a,
                PER_b,
                final_PER_b,
                PER_b_steps,
                PER_b_growth,
                final_PER_a,
                PER_a_steps,
                PER_a_growth,
                max_experiences,
            )  # experience is stored as object of this class
        else:
            self.experience = {"s": [], "a": [], "r": [], "s2": [], "f": []}
        self.test_experience = None
        self.start_train = start_train
        self.max_experiences = max_experiences
        self.action_limit = action_limit
        self.num_actions = num_actions
        self.batch_norm_input = batch_norm_input
        self.batch_norm_hidden = batch_norm_hidden

        if DDPG_type == "DDPG":
            if recurrent_env:
                self.Q_model = CriticRecurrentNetwork(
                    seed,
                    num_states,
                    hidden_memory_units,
                    hidden_units_Q,
                    num_actions,
                    batch_norm_input,
                    batch_norm_hidden,
                    activation_Q,
                    kernel_initializer,
                    output_init,
                    delayed_actions,
                    modelname="Q" + modelname,
                )
                self.p_model = ActorRecurrentNetwork(
                    seed,
                    num_states,
                    hidden_memory_units,
                    hidden_units_p,
                    num_actions,
                    batch_norm_input,
                    batch_norm_hidden,
                    activation_p,
                    kernel_initializer,
                    output_init,
                    modelname="p" + modelname,
                )
            else:
                self.Q_model = CriticNetwork(
                    seed,
                    num_states,
                    hidden_units_Q,
                    num_actions,
                    batch_norm_input,
                    batch_norm_hidden,
                    activation_Q,
                    kernel_initializer,
                    output_init,
                    delayed_actions,
                    modelname="Q" + modelname,
                )
                self.p_model = ActorNetwork(
                    seed,
                    num_states,
                    hidden_units_p,
                    num_actions,
                    batch_norm_input,
                    batch_norm_hidden,
                    activation_p,
                    kernel_initializer,
                    output_init,
                    modelname="p" + modelname,
                )
        elif DDPG_type == "TD3":
            if recurrent_env:
                self.Q1_model = CriticRecurrentNetwork(
                    seed,
                    num_states,
                    hidden_memory_units,
                    hidden_units_Q,
                    num_actions,
                    batch_norm_input,
                    batch_norm_hidden,
                    activation_Q,
                    kernel_initializer,
                    output_init,
                    delayed_actions,
                    modelname="Q" + modelname,
                )
                self.Q2_model = CriticRecurrentNetwork(
                    seed,
                    num_states,
                    hidden_memory_units,
                    hidden_units_Q,
                    num_actions,
                    batch_norm_input,
                    batch_norm_hidden,
                    activation_Q,
                    kernel_initializer,
                    output_init,
                    delayed_actions,
                    modelname="Q" + modelname,
                )
                self.p_model = ActorRecurrentNetwork(
                    seed,
                    num_states,
                    hidden_memory_units,
                    hidden_units_p,
                    num_actions,
                    batch_norm_input,
                    batch_norm_hidden,
                    activation_p,
                    kernel_initializer,
                    output_init,
                    modelname="p" + modelname,
                )
            else:
                self.Q1_model = CriticNetwork(
                    seed,
                    num_states,
                    hidden_units_Q,
                    num_actions,
                    batch_norm_input,
                    batch_norm_hidden,
                    activation_Q,
                    kernel_initializer,
                    output_init,
                    delayed_actions,
                    modelname="Q" + modelname,
                )
                self.Q2_model = CriticNetwork(
                    seed,
                    num_states,
                    hidden_units_Q,
                    num_actions,
                    batch_norm_input,
                    batch_norm_hidden,
                    activation_Q,
                    kernel_initializer,
                    output_init,
                    delayed_actions,
                    modelname="Q" + modelname,
                )
                self.p_model = ActorNetwork(
                    seed,
                    num_states,
                    hidden_units_p,
                    num_actions,
                    batch_norm_input,
                    batch_norm_hidden,
                    activation_p,
                    kernel_initializer,
                    output_init,
                    modelname="p" + modelname,
                )
        else:
            print("Choose proper algorithm (DDPG or TD3)")
            sys.exit()

        self.summary_writer = summary_writer
        self.plot_hist = plot_hist
        self.plot_steps = plot_steps
        self.plot_steps_hist = plot_steps_hist
        self.selected_loss = selected_loss
        self.clipgrad = clipgrad
        self.clipnorm = clipnorm
        self.clipvalue = clipvalue
        self.update_target = update_target
        self.tau_Q = tau_Q
        self.tau_p = tau_p

        self.global_norms = []
        self.clipglob_steps = clipglob_steps
        self.optimizer_name = optimizer_name
        self.pretraining_mode = pretraining_mode
        self.DDPG_type = DDPG_type
        self.noise_clip = noise_clip

        nb_actions = int(num_states / 2)
        if noise == "G":
            self.action_noise = GaussianActionNoise(
                mu=np.ones(nb_actions) * mu,
                sigma=float(stddev_noise) * np.ones(nb_actions),
                rng=self.rng,
            )
        elif noise == "OU":
            self.action_noise = OrnsteinUhlenbeckActionNoise(
                mu=np.ones(nb_actions) * mu,
                sigma=float(stddev_noise) * np.ones(nb_actions),
                theta=theta,
                rng=self.rng,
            )

        if DDPG_type == "TD3":
            self.policy_noise = GaussianActionNoise(
                mu=np.ones(nb_actions) * mu,
                sigma=float(stddev_pol_noise) * np.ones(nb_actions),
                rng=self.rng,
            )
        self.nb_actions = nb_actions

        if self.selected_loss == "mse":
            self.loss = tf.keras.losses.MeanSquaredError()
        elif self.selected_loss == "huber":
            self.loss = tf.keras.losses.Huber()

    def train(self, TargetNet, iteration, env=None):

        if iteration < self.start_train:
            return 0

        if self.use_PER:
            b_idx, minibatch = self.PERmemory.sample_batch(self.batch_size)
            states = np.asarray(minibatch["s"])
            actions = np.asarray(minibatch["a"])
            rewards = np.asarray(minibatch["r"])
            states_next = np.asarray(minibatch["s2"])

        else:
            # find the index of streams included in the experience buffer that will
            # compose the training batch
            ids = self.rng.randint(
                low=0, high=len(self.experience["s"]), size=self.batch_size
            )
            states = np.asarray([self.experience["s"][i] for i in ids])
            actions = np.asarray([self.experience["a"][i] for i in ids])
            rewards = np.asarray([self.experience["r"][i] for i in ids])
            states_next = np.asarray([self.experience["s2"][i] for i in ids])

        with tf.GradientTape(persistent=True) as tape1:
            # compute current action values
            if self.DDPG_type == "TD3":
                current_q1 = self.Q1_model(
                    states.astype("float32"),
                    actions.astype("float32"),
                    store_intermediate_outputs=True,
                )
                current_q2 = self.Q2_model(
                    states.astype("float32"),
                    actions.astype("float32"),
                    store_intermediate_outputs=True,
                )

                noise = tf.cast(
                    tf.clip_by_value(
                        self.policy_noise(), -self.noise_clip, self.noise_clip
                    ),
                    tf.float32,
                )

                action_next = tf.clip_by_value(
                    TargetNet.p_model(states_next.astype("float32")) + noise,
                    -self.action_limit,
                    self.action_limit,
                )  # TODO nonsense here because we should clip between -1 and 1

                target_q1 = TargetNet.Q1_model(
                    states_next.astype("float32"), action_next
                )
                target_q2 = TargetNet.Q2_model(
                    states_next.astype("float32"), action_next
                )

                target_q = tf.math.minimum(target_q1, target_q2)
                target_values = rewards + self.gamma * target_q

                # not implemented PER with TD3 (TODO)
                loss_q1 = self.loss(y_true=target_values, y_pred=current_q1)
                loss_q2 = self.loss(y_true=target_values, y_pred=current_q2)

            else:
                current_q = self.Q_model(
                    states.astype("float32"),
                    actions.astype("float32"),
                    store_intermediate_outputs=True,
                )
                # compute target action values
                # here the action is in the range (-1,1)
                action_next = TargetNet.p_model(states_next.astype("float32"))
                target_q = TargetNet.Q_model(states_next.astype("float32"), action_next)
                target_values = rewards + self.gamma * target_q

                if self.use_PER:
                    # compute weights
                    if iteration < self.max_experiences:
                        N = iteration + 1
                    else:
                        N = self.max_experiences
                    # N = len([x for x in self.PERmemory.experience['r'] if x!=0])
                    prob = self.PERmemory.tree[b_idx] / self.PERmemory.total_priority
                    self.PERmemory.PER_b = min(
                        self.PERmemory.final_PER_b,
                        self.PERmemory.PER_b + self.PERmemory.PER_b_growth,
                    )
                    w_IS = (N * prob) ** (-self.PERmemory.PER_b)
                    scaled_w_IS = w_IS / np.max(w_IS)

                    # update priorities
                    self.PERmemory.batch_update(
                        b_idx, np.abs(target_values - current_q)
                    )

                    # compute loss function for the train model
                    loss_q = self.loss(
                        y_true=target_values,
                        y_pred=current_q,
                        sample_weight=scaled_w_IS,
                    )
                else:
                    loss_q = self.loss(y_true=target_values, y_pred=current_q)

        if self.DDPG_type == "TD3":
            variables_q1 = self.Q1_model.trainable_variables
            variables_q2 = self.Q2_model.trainable_variables

            # compute gradient of the loss with respect to the variables (weights)
            gradients_q1 = tape1.gradient(loss_q1, variables_q1)
            gradients_q2 = tape1.gradient(loss_q2, variables_q2)

            if self.clipgrad == "norm":
                gradients_q1 = [
                    (tf.clip_by_norm(gv, self.clipnorm)) for gv in gradients_q1
                ]
                gradients_q2 = [
                    (tf.clip_by_norm(gv, self.clipnorm)) for gv in gradients_q2
                ]
            elif self.clipgrad == "value":
                gradients_q1 = [
                    (tf.clip_by_value(gv, -self.clipvalue, self.clipvalue))
                    for gv in gradients_q1
                ]
                gradients_q2 = [
                    (tf.clip_by_value(gv, -self.clipvalue, self.clipvalue))
                    for gv in gradients_q2
                ]

            # provide a list of (gradient, variable) pairs.
            self.optimizer_Q1.apply_gradients(zip(gradients_q1, variables_q1))
            self.optimizer_Q2.apply_gradients(zip(gradients_q2, variables_q2))

        else:
            variables_q = self.Q_model.trainable_variables

            # compute gradient of the loss with respect to the variables (weights)
            gradients_q = tape1.gradient(loss_q, variables_q)

            if self.clipgrad == "norm":
                gradients_q = [
                    (tf.clip_by_norm(gv, self.clipnorm)) for gv in gradients_q
                ]
            elif self.clipgrad == "value":
                gradients_q = [
                    (tf.clip_by_value(gv, -self.clipvalue, self.clipvalue))
                    for gv in gradients_q
                ]
            elif self.clipgrad == "globnorm":
                if iteration <= self.clipglob_steps:
                    gbnorm = tf.linalg.global_norm(gradients_q)
                    self.global_norms.append(gbnorm.numpy())
                    if iteration == self.clipglob_steps:
                        self.clipglob = np.mean(self.global_norms)
                else:
                    gradients_q, gbnorm = tf.clip_by_global_norm(
                        gradients_q, self.clipglob
                    )

            # provide a list of (gradient, variable) pairs.
            self.optimizer_Q.apply_gradients(zip(gradients_q, variables_q))

        with tf.GradientTape() as tape2:
            if self.DDPG_type == "TD3":
                current_q_pg = self.Q1_model(
                    states.astype("float32"),
                    self.p_model(
                        states.astype("float32"), store_intermediate_outputs=True
                    ),
                )
            else:
                current_q_pg = self.Q_model(
                    states.astype("float32"),
                    self.p_model(
                        states.astype("float32"), store_intermediate_outputs=True
                    ),
                )
            loss_p = -tf.math.reduce_mean(current_q_pg)

        # compute gradient of the loss with respect to the variables (weights)
        variables_p = self.p_model.trainable_variables
        gradients_p = tape2.gradient(loss_p, variables_p)
        self.optimizer_p.apply_gradients(zip(gradients_p, variables_p))

        # if (((iteration % self.plot_steps) == 0) or (iteration == self.start_train)) and (not self.pretraining_mode):
        #     with self.summary_writer.as_default():

        #         tf.summary.scalar('Mean Squared Loss/Q_model_Train', loss_q, step=iteration)
        #         tf.summary.scalar('Mean Squared Loss/p_model_Train', loss_p, step=iteration)
        #         tf.summary.scalar('Learning Rate/Q_model_LR', self.optimizer_Q._decayed_lr(tf.float32), step=iteration)
        #         tf.summary.scalar('Learning Rate/p_model_LR', self.optimizer_p._decayed_lr(tf.float32), step=iteration)
        #         # if self.clipgrad == 'globnorm':
        #         #     tf.summary.scalar('Norm/Global grad norm', gbnorm, step=iteration)
        #         #     if iteration > self.clipglob_steps:
        #         #         tf.summary.scalar('Norm/Clip Glob', self.clipglob, step=iteration)

        #         # else:
        #         #     gbnorm = tf.linalg.global_norm(gradients)
        #         #     tf.summary.scalar('Norm/Global grad norm', gbnorm, step=iteration)

        #         if self.plot_hist and ((iteration % self.plot_steps_hist) == 0):
        #             models = [self.Q_model, self.p_model]
        #             for model in models:
        #                 for i,layer in enumerate(model.layers[1:]):
        #                     with tf.name_scope('{1}_layer{0}'.format(i,model.name)):
        #                         if 'dense' in layer.name:
        #                             tf.summary.histogram(layer.name + '/weights',
        #                                                   layer.get_weights()[0], step=iteration)
        #                             tf.summary.histogram(layer.name + '/biases',
        #                                                   layer.get_weights()[1], step=iteration)
        #                             tf.summary.histogram(layer.name + '/Wx+b_pre_activation',
        #                                                   layer.out, step=iteration)
        #                         elif 'activation' in layer.name:
        #                             tf.summary.histogram(layer.name + '/activation',
        #                                                   layer.out, step=iteration)
        #                         elif 'batch' in layer.name:
        #                             tf.summary.histogram(layer.name + '/bnorm_inputs_1',
        #                                                   model.bninputs[:,0], step=iteration)
        #                             tf.summary.histogram(layer.name + '/bnorm_inputs_2',
        #                                                   model.bninputs[:,1], step=iteration)
        #                             tf.summary.histogram(layer.name + '/inputs_1',
        #                                                   model.inputs[:,0], step=iteration)
        #                             tf.summary.histogram(layer.name + '/inputs_2',
        #                                                   model.inputs[:,1], step=iteration)
        #                             if model.name == 'QTrain' and not model.delayed_actions:
        #                                 tf.summary.histogram(layer.name + '/bnorm_inputs_3',
        #                                                   model.bninputs[:,2], step=iteration)
        #                                 tf.summary.histogram(layer.name + '/inputs_3',
        #                                                   model.inputs[:,2], step=iteration)

        #         for g_q,v_q in zip(gradients_q,variables_q):
        #         # store q gradients
        #             grad_mean_Q = tf.reduce_mean(g_q)
        #             grad_square_sum_Q = tf.reduce_sum(tf.math.square(g_q))
        #             grad_norm_Q = tf.sqrt(grad_square_sum_Q)
        #             sq_norm_Q = tf.square(grad_norm_Q)
        #             tf.summary.scalar(v_q.name.split('/',1)[1] + 'Gradients_{}/grad_mean'.format(self.Q_model.name),
        #                               grad_mean_Q, step=iteration)
        #             tf.summary.scalar(v_q.name.split('/',1)[1] + 'Gradients_{}/grad_norm'.format(self.Q_model.name),
        #                               grad_norm_Q, step=iteration)
        #             tf.summary.scalar(v_q.name.split('/',1)[1] + 'Gradients_{}/grad_sq_norm'.format(self.Q_model.name),
        #                               sq_norm_Q, step=iteration)
        #             tf.summary.histogram(v_q.name.split('/',1)[1] + '_hist_{}/grads'.format(self.Q_model.name),
        #                                  g_q, step=iteration)
        #             tf.summary.histogram(v_q.name.split('/',1)[1] + '_hist_{}/grads_squared'.format(self.Q_model.name),
        #                                  tf.square(g_q), step=iteration)

        #         for g_p,v_p in zip(gradients_p,variables_p):
        #         # store p gradients
        #             grad_mean_p = tf.reduce_mean(g_p)
        #             grad_square_sum_p = tf.reduce_sum(tf.math.square(g_p))
        #             grad_norm_p = tf.sqrt(grad_square_sum_p)
        #             sq_norm_p = tf.square(grad_norm_p)
        #             tf.summary.scalar(v_p.name.split('/',1)[1] + 'Gradients_{}/grad_mean'.format(self.p_model.name),
        #                               grad_mean_p, step=iteration)
        #             tf.summary.scalar(v_p.name.split('/',1)[1] + 'Gradients_{}/grad_norm'.format(self.p_model.name),
        #                               grad_norm_p, step=iteration)
        #             tf.summary.scalar(v_p.name.split('/',1)[1] + 'Gradients_{}/grad_sq_norm'.format(self.p_model.name),
        #                               sq_norm_p, step=iteration)
        #             tf.summary.histogram(v_p.name.split('/',1)[1] + 'hist_{}/grads'.format(self.p_model.name),
        #                                  g_p, step=iteration)
        #             tf.summary.histogram(v_p.name.split('/',1)[1] + 'hist_{}/grads_squared'.format(self.p_model.name),
        #                                  tf.square(g_p), step=iteration)

        #         if self.use_PER:
        #             action_buffer = np.array(self.PERmemory.experience['a'], dtype=object)
        #             tf.summary.histogram('BufferReplay/Actions', action_buffer, step=iteration)

        #         self.summary_writer.flush()

    def noisy_action(self, states):
        a = self.p_model(np.atleast_2d(states.astype("float32")), training=False)
        a = tf.reshape(
            a,
            [
                1,
            ],
        )
        a += self.action_noise()

        # return np.clip(a, -self.action_limit, self.action_limit)
        return np.clip(a, -1.0, 1.0)

    def uniform_action(self):
        # return np.random.uniform(-self.action_limit, self.action_limit,self.nb_actions)

        return scale_action(
            self.action_limit,
            self.rng.uniform(-self.action_limit, self.action_limit, self.nb_actions),
        )

    def add_experience(self, exp):

        if self.use_PER:
            self.PERmemory.add(exp)
        else:
            if len(self.experience["s"]) >= self.max_experiences:
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

    def copy_weights_Q(self, TrainNet):
        if self.DDPG_type == "TD3":
            variables1q = self.Q1_model.trainable_variables
            variables2q = TrainNet.Q1_model.trainable_variables
            for v1q, v2q in zip(variables1q, variables2q):
                vsoftq = (1 - self.tau_Q) * v1q + self.tau_Q * v2q
                v1q.assign(vsoftq.numpy())

            variables1q = self.Q2_model.trainable_variables
            variables2q = TrainNet.Q2_model.trainable_variables
            for v1q, v2q in zip(variables1q, variables2q):
                vsoftq = (1 - self.tau_Q) * v1q + self.tau_Q * v2q
                v1q.assign(vsoftq.numpy())
        else:
            variables1q = self.Q_model.trainable_variables
            variables2q = TrainNet.Q_model.trainable_variables
            for v1q, v2q in zip(variables1q, variables2q):
                vsoftq = (1 - self.tau_Q) * v1q + self.tau_Q * v2q
                v1q.assign(vsoftq.numpy())

    def copy_weights_p(self, TrainNet):
        variables1p = self.p_model.trainable_variables
        variables2p = TrainNet.p_model.trainable_variables
        for v1p, v2p in zip(variables1p, variables2p):
            vsoftp = (1 - self.tau_p) * v1p + self.tau_p * v2p
            v1p.assign(vsoftp.numpy())
