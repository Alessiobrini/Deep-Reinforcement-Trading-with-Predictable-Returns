# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 16:53:31 2020

@author: aless
"""
import gin
import numpy as np
from typing import Union, Optional, Tuple
import sys
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers.schedules import InverseTimeDecay
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from utils.exploration import PER_buffer
import pdb

################################ Class to create a Deep Q Network model ################################
class DeepNetworkModel(tf.keras.Model):
    def __init__(
        self,
        seed: int,
        input_shape: int,
        hidden_units: list,
        num_actions: int,
        batch_norm_input: bool,
        batch_norm_hidden: bool,
        activation: str,
        kernel_initializer: str,
        modelname: str = "Deep Q Network",
    ):
        """
        Instantiate Deep Q Network Class

        Parameters
        ----------
        seed: int
            Seed for experiment reproducibility

        input_shape: int
            Shape of input of the neural network

        hidden_units: list
            List of sizes of hidden layer. The length of the list determines
            the depth of the Q network

        num_actions: int
            Number of possible action which is the size of the output

        batch_norm_input: bool
            Boolean to regulate the presence of a Batch Norm layer after the input

        batch_norm_hidden: bool
            Boolean to regulate the presence of a Batch Norm layer after each hidden layer

        activation: str
            Choice of activation function. It can be 'leaky_relu',
            'relu6' or 'elu'

        kernel_initializer: str
            Choice of weight initialization as aliased in TF2.0 documentation

        modelname: str
            Name of the model

        """
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
            self.bnorm_layer = BatchNormalization(center=False, scale=False)

        # set of hidden layers
        self.hids = []

        for i in hidden_units:
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
        self.output_layer = Dense(out_shape)

    def call(
        self,
        inputs: Union[np.ndarray or tf.Tensor],
        training: bool = True,
    ) -> Union[np.ndarray or tf.Tensor]:
        """
        Instantiate Deep Q Network Class

        Parameters
        ----------

        inputs: Union[np.ndarray or tf.Tensor]
            Inputs to the neural network

        training: bool
            Boolean to regulate if inference or test time

        Returns
        ----------
        z: Union[np.ndarray or tf.Tensor]
            Outputs of the neural network after a forward pass

        """
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
        return z




############################### DQN ALGORITHM ################################
@gin.configurable()
class DQN:
    """DQN algorithm class"""

    def __init__(
        self,
        seed: int,
        DQN_type: str,
        gamma: float,
        epsilon: float,
        min_eps_pct: float,
        min_eps: float,
        max_exp_pct: float,
        update_target: str,
        copy_step: int,
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
        action_space,
        use_PER: bool = False,
        PER_e: Optional[float] = None,
        PER_a: Optional[float] = None,
        PER_b: Optional[float] = None,
        final_PER_b: Optional[float] = None,
        PER_b_growth: Optional[float] = None,
        final_PER_a: Optional[float] = None,
        PER_a_growth: Optional[float] = None,
        sample_type: str = "TDerror",
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        eps_opt: float = 1e-07,
        lr_schedule: Optional[str] = None,
        exp_decay_pct: Optional[float] = None,
        exp_decay_rate: Optional[float] = None,
        rng=None,
        N_train:int = 100000,
        modelname: str = "Deep Network",
    ):
        """
        Instantiate DQN Class

        Parameters
        ----------
        seed: int
            Seed for experiment reproducibility

        DQN_type: str
            DQN variant choice. It can be 'DQN' or 'DDQN'

        recurrent_env: bool
            Boolean to regulate if the environment is recurrent or not

        gamma: float
            Discount parameter for the target update

        max_exp_pct: int
            Max size of the experience replay buffer as a pct of the total iterations

        update_target: str
            Choice for target update. It can be 'hard' or 'soft'

        tau: float
            When the update is 'soft', tau regulates the amount of the update
            towards the current parameters

        input_shape: int
            Shape of input of the neural network

        hidden_units: list
            List of sizes of hidden layers. The length of the list determines
            the depth of the Q network

        hidden_memory_units: list,
            List of sizes of recurrent hidden layers. The length of the list determines
            the depth of the Q network

        batch_size: int
            Size of the batch to perform an update

        selected_loss: str
            Choice for the loss function. It can be 'mse' or 'huber'

        lr: float
            Initial learning rate

        start_train: int
            Number of iteration after which the training starts

        optimizer_name: str
            Choice for the optimizer. It can be 'sgd', 'sgdmom', 'sgdnest',
            'adagrad', 'adadelta', 'adamax', 'adam', 'amsgrad', 'nadam', or 'rmsprop'

        batch_norm_input: bool
            Boolean to regulate the presence of a Batch Norm layer after the input

        batch_norm_hidden: bool
            Boolean to regulate the presence of a Batch Norm layer after each hidden layer

        activation: str
            Choice of activation function. It can be 'leaky_relu',
            'relu6' or 'elu'

        kernel_initializer: str
            Choice of weight initialization as aliased in TF2.0 documentation

        plot_hist: bool
            Boolean to regulate if plot the histogram of intermediate outputs
            in tensorboard

        plot_steps_hist: int
            Number of steps at which the histogram of intermediate outputs are
            plotted in tensorboard

        plot_steps: int
            Number of steps at which all the other variables are
            stored in tensorboard

        summary_writer, #TODO need to add proper type hint
            Tensorabord writer
        action_space: class
            Space of possible action as class that inherits from gym

        use_PER: bool = False
            Boolean to regulate if use Prioritized Experience Replay (PER) or not

        PER_e: Optional[float]
            Correction for priorities

        PER_a: Optional[float]
            Amount of prioritization

        PER_b: Optional[float]
            Amount of correction for introduced bias when using PER

        final_PER_b: Optional[float] = None
            Final value for b after the anneal

        PER_b_growth: Optional[float]
            Rate of increase of the b

        final_PER_a: Optional[float] = None
            Final value for a after the anneal

        PER_a_growth: Optional[float]
            Rate of increase of the a

        sample_type : str
            Type of sampling in PER. It can be 'TDerror', 'diffTDerror' or 'reward'

        clipgrad: bool
            Choice of the gradients to clip. It can be 'norm', 'value' or 'globnorm'

        clipnorm: Optional[Union[str or float]]
            Boolean for clipping the norm of the gradients

        clipvalue: Optional[Union[str or float]]
            Boolean for clipping the value of the gradients

        clipglob_steps: Optional[int]
            Boolean for clipping the global norm of the gradients

        beta_1: float = 0.9
            Parameter for adaptive optimizer

        beta_2: float = 0.999
            Parameter for adaptive optimizer

        eps_opt: float = 1e-07
            Corrective parameter for adaptive optimizer

        std_rwds: bool = False
            Boolean to regulate if standardize rewards or not

        lr_schedule: Optional[str]
            Choice for the learning rate schedule. It can be 'exponential',
            'piecewise', 'inverse_time' or 'polynomial'

        exp_decay_pct: Optional[float]
             Amount of steps to reach the desired level of decayed learning rate as pct
             of the total iteration

        exp_decay_rate: Optional[float]
            Rate of decay to reach the desired level of decayed learning rate

        rng = None
            Random number generator for reproducibility

        modelname: str
            Name for the model

        """

        if rng is not None:
            self.rng = rng

        self.batch_size = batch_size

        exp_decay_steps = int(N_train * exp_decay_pct)
        if lr_schedule == "exponential":
            lr = ExponentialDecay(
                initial_learning_rate=lr,
                decay_steps=exp_decay_steps,
                decay_rate=exp_decay_rate,
            )

        if optimizer_name == "adam":
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=lr,
                beta_1=beta_1,
                beta_2=beta_2,
                epsilon=eps_opt,
                amsgrad=False,
            )
        elif optimizer_name == "rmsprop":
            self.optimizer = tf.keras.optimizers.RMSprop(
                learning_rate=lr,
                rho=beta_1,
                momentum=0.0,
                epsilon=eps_opt,
                centered=False,
            )

        self.beta_1 = beta_1
        self.eps_opt = eps_opt
        self.gamma = gamma

        self.max_experiences = int(N_train * max_exp_pct)

        self.use_PER = use_PER
        if self.use_PER:

            if PER_b_growth:
                PER_b_steps = N_train
                PER_b_growth = (final_PER_b - PER_b) / PER_b_steps
            else:
                PER_b_growth = 0.0
                PER_b_steps= None

            if PER_a_growth:
                PER_a_steps = PER_a_steps = N_train
                PER_a_growth  = (final_PER_a - PER_a) / PER_a_steps
            else:
                PER_a_growth = 0.0
                PER_a_steps = None

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
                self.max_experiences,
                rng,
                sample_type,
            )  # experience is stored as object of this class
        else:
            self.experience = {"s": [], "a": [], "r": [], "s2": [], "a_unsc": []}

        self.start_train = start_train
        self.action_space = action_space
        self.num_actions = len(self.action_space.values)
        self.batch_norm_input = batch_norm_input
        self.batch_norm_hidden = batch_norm_hidden

        self.model = DeepNetworkModel(
            seed,
            input_shape,
            hidden_units,
            self.num_actions,
            batch_norm_input,
            batch_norm_hidden,
            activation,
            kernel_initializer,
            modelname,
        )

        self.target_model = DeepNetworkModel(
            seed,
            input_shape,
            hidden_units,
            self.num_actions,
            batch_norm_input,
            batch_norm_hidden,
            activation,
            kernel_initializer,
            "Target " + modelname,
        )

        self.selected_loss = selected_loss
        self.DQN_type = DQN_type
        self.update_target = update_target
        self.copy_step = copy_step
        self.tau = tau
        self.optimizer_name = optimizer_name

        if self.selected_loss == "mse":
            self.loss = tf.keras.losses.MeanSquaredError()
        elif self.selected_loss == "huber":
            self.loss = tf.keras.losses.Huber()

        self.epsilon = epsilon
        self.min_eps = min_eps
        self.min_eps_pct = min_eps_pct

    def train(
        self,
        iteration: int,
        side_only: bool = False,
        bcm: bool = False,
        bcm_scale: float = 0.01,
    ):

        """Parameters
        ----------
        iteration: int
            Number of iteration update
        side_only: bool
            Regulate the decoupling between side and size of the bet

        bcm: bool
            Regulate the part of the loss relative to the behaviorl cloning of an expert
        """
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
            # composed the training batch
            ids = self.rng.randint(
                low=0, high=len(self.experience["s"]), size=self.batch_size
            )

            states = np.asarray([self.experience["s"][i] for i in ids])
            actions = np.asarray([self.experience["a"][i] for i in ids])
            rewards = np.asarray([self.experience["r"][i] for i in ids])
            states_next = np.asarray([self.experience["s2"][i] for i in ids])

        with tf.GradientTape() as tape:

            # compute current action values
            # find index of actions included in the batch
            encoded_actions = [
                self.action_space.values.tolist().index(act) for act in actions
            ]
            selected_action_values = tf.math.reduce_sum(
                self.model(
                    np.atleast_2d(states.astype("float32")),
                )
                * tf.one_hot(encoded_actions, self.num_actions),
                axis=1,
            )

            # compute target action values
            if self.DQN_type == "DQN":
                value_next = np.max(
                    self.target_model(states_next.astype("float32")), axis=1
                )
            elif self.DQN_type == "DDQN":
                greedy_target_action = tf.math.argmax(
                    self.model(states_next.astype("float32")), 1
                )
                value_next = tf.math.reduce_sum(
                    self.target_model(states_next.astype("float32"))
                    * tf.one_hot(greedy_target_action, self.num_actions),
                    axis=1,
                )

            actual_values = rewards + self.gamma * value_next

            if self.use_PER:
                # compute weights
                if iteration < self.max_experiences:
                    N = iteration + 1
                else:
                    N = self.max_experiences
                prob = self.PERmemory.tree[b_idx] / self.PERmemory.total_priority
                self.PERmemory.PER_b = min(
                    self.PERmemory.final_PER_b,
                    self.PERmemory.PER_b + self.PERmemory.PER_b_growth,
                )
                w_IS = (N * prob) ** (-self.PERmemory.PER_b)
                scaled_w_IS = w_IS / np.max(w_IS)

                # update priorities
                if self.PERmemory.sample_type == "rewards":
                    self.PERmemory.batch_update(b_idx, np.abs(rewards))
                elif (
                    self.PERmemory.sample_type == "TDerror"
                    or self.PERmemory.sample_type == "diffTDerror"
                ):
                    self.PERmemory.batch_update(
                        b_idx, np.abs(actual_values - selected_action_values)
                    )
                else:
                    print("Sample type for PER not available")
                    sys.exit()

                # compute loss function for the train model
                loss = self.loss(
                    y_true=actual_values,
                    y_pred=selected_action_values,
                    sample_weight=scaled_w_IS.reshape(-1, 1),
                )

            else:

                loss = self.loss(y_true=actual_values, y_pred=selected_action_values)

        variables = self.model.trainable_variables

        # compute gradient of the loss with respect to the variables (weights)
        gradients = tape.gradient(loss, variables)

        # provide a list of (gradient, variable) pairs.
        self.optimizer.apply_gradients(zip(gradients, variables))

    def eps_greedy_action(
        self, states: np.ndarray, epsilon: float, side_only: bool = False
    ) -> Tuple[Union[float or int], np.ndarray]:
        """Parameters
        ----------
        states: np.ndarray
            Current state representation

        epsilon: float
            Epsilon parameter for exploration

        side_only: bool
            Regulate the decoupling between side and size of the bet

        Returns
        ----------
        action: Union[float or int]
            Epsilon greedy selected action
        qvalues : np.ndarray
            Q values associated to the actions space
        """
        if not side_only:
            if self.rng.random() < epsilon:
                action = self.rng.choice(self.action_space.values)
                return action, None
            else:
                action = self.action_space.values[
                    np.argmax(
                        self.model(
                            np.atleast_2d(states.astype("float32")), training=False
                        )[0]
                    )
                ]
                return action, None
        else:
            if self.rng.random() < epsilon:
                action = self.rng.choice(self.action_space.values)
                return action, None
            else:
                qvalues = self.model(
                    np.atleast_2d(states.astype("float32")), training=False
                )
                action = self.action_space.values[np.argmax(qvalues[0])]
                return action, qvalues

    def greedy_action(
        self, states: np.ndarray, side_only: bool = False
    ) -> Tuple[Union[float or int], np.ndarray]:
        """Parameters
        ----------
        states: np.ndarray
            Current state representation

        side_only: bool
            Regulate the decoupling between side and size of the bet

        Returns
        ----------
        action: Union[float or int]
            Greedy selected action

        qvalues : np.ndarray
            Q values associated to the actions space
        """
        if not side_only:
            qvalues = self.model(
                np.atleast_2d(states.astype("float32")), training=False
            )
            action = self.action_space.values[np.argmax(qvalues[0])]
            return action, None
        else:

            qvalues = self.model(
                np.atleast_2d(states.astype("float32")), training=False
            )
            action = self.action_space.values[np.argmax(qvalues[0])]
            return action, qvalues

    def add_experience(self, exp):
        """Parameters
        ----------
        exp: dict
            Sequences of experience to store

        """
        if self.use_PER:
            self.PERmemory.add(exp)
        else:
            if len(self.experience["s"]) >= self.max_experiences:
                for key in self.experience.keys():
                    if self.experience[key]:  # check if the list is not empty
                        self.experience[key].pop(0)
            for key, value in exp.items():
                self.experience[key].append(value)

    def copy_weights(self):
        """Parameters
        ----------
        """
        if self.update_target == "soft":
            variables1 = self.target_model.trainable_variables
            variables2 = self.model.trainable_variables
            for v1, v2 in zip(variables1, variables2):
                vsoft = (1 - self.tau) * v1 + self.tau * v2
                v1.assign(vsoft.numpy())
        else:
            variables1 = self.target_model.trainable_variables
            variables2 = self.model.trainable_variables
            for v1, v2 in zip(variables1, variables2):
                v1.assign(v2.numpy())

    def update_epsilon(self):
        self.epsilon = max(self.min_eps, self.epsilon - self.eps_decay)

    def _get_exploration_length(self, N_train):
        steps_to_min_eps = int(N_train * self.min_eps_pct)
        self.eps_decay = (self.epsilon - self.min_eps) / steps_to_min_eps
