import argparse
import gin
import logging
import os
import time
import pdb
import sys
from typing import Union
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from utils.common import (
    GeneratePathFolder,
    save_gin,
)
from utils.spaces import (
    ActionSpace,
    ResActionSpace,
)
from agents.DQN import DQN
from utils.simulation import DataHandler
from utils.tools import get_action_boundaries, get_bet_size, CalculateLaggedSharpeRatio
from utils.test import Out_sample_vs_gp

from utils.mixin_core import MixinCore


@gin.configurable()
class DQN_runner(MixinCore):
    def __init__(
        self,
        env_cls: object,
        MV_res: bool,
        experiment_type: str,
        seed: int,
        episodes: int,
        N_train: int,
        len_series: Union[int or None],
        dt: int, 
        start_train: int,
        save_freq: int,
        use_GPU: bool,
        outputDir: str = "_outputs",
        outputClass: str = "DQN",
        outputModel: str = "test",
        varying_pars: Union[list or None] = None,
        varying_type: str = "chunk",
        num_cores: int = None,
    ):

        self.logging.info("Starting model setup")
        self._setattrs()

        self.rng = np.random.RandomState(self.seed)

        if self.use_GPU:
            gpu_devices = tf.config.experimental.list_physical_devices("GPU")
            for device in gpu_devices:
                tf.config.experimental.set_memory_growth(device, True)
        else:
            my_devices = tf.config.experimental.list_physical_devices(device_type="CPU")
            tf.config.experimental.set_visible_devices(
                devices=my_devices, device_type="CPU"
            )

        if self.dt != 1.0:
            # self.len_series = self.len_series * (1/self.dt)
            self.N_train = int(self.N_train * (1/self.dt))

        if self.episodes:
            self.N_train = self.episodes * self.len_series
            self.col_names_oos = [
                str(e) for e in np.arange(0, self.episodes + 1, save_freq)[1:]
            ]
        else:
            self.len_series = self.N_train
            self.save_freq_n = self.N_train // save_freq
            self.col_names_oos = [
                str(int(i)) for i in np.arange(0, self.N_train + 1, self.save_freq_n)
            ][1:]

        self.savedpath = GeneratePathFolder(
            outputDir,
            outputClass,
            outputModel,
            varying_pars,
            varying_type,
            self.N_train,
        )
        if save_freq and not os.path.exists(os.path.join(self.savedpath, "ckpt")):
            os.makedirs(os.path.join(self.savedpath, "ckpt"))
        elif save_freq and os.path.exists(os.path.join(self.savedpath, "ckpt")):
            pass

        logging.info("Successfully generated path to save outputs...")

    def run(self):
        """Wrapper for keyboard interrupt."""
        try:
            self.set_up_training()
            # if self.episodes:
            #     self.training_episodic_agent()
            # else:
            self.training_agent()
        except (KeyboardInterrupt, SystemExit):
            self.logging.debug("Exit on KeyboardInterrupt or SystemExit")
            sys.exit()

    def set_up_training(self):

        self.logging.debug("Simulating Data")

        self.data_handler = DataHandler(N_train=self.N_train, rng=self.rng)
        if self.experiment_type == "GP":
            self.data_handler.generate_returns()
        else:
            self.data_handler.generate_returns()
            # TODO check if these method really fit and change the parameters in the gin file
            self.data_handler.estimate_parameters()

        self.logging.debug("Instantiating action space")
        if self.MV_res:
            self.action_space = ResActionSpace()
        else:
            action_range, ret_quantile, holding_quantile = get_action_boundaries(
                N_train=self.N_train,
                f_speed=self.data_handler.f_speed,
                returns=self.data_handler.returns,
                factors=self.data_handler.factors,
            )

            gin.query_parameter("%ACTION_RANGE")[0] = action_range
            self.action_space = ActionSpace()

        self.logging.debug("Instantiating market environment")
        self.env = self.env_cls(
            N_train=self.N_train,
            f_speed=self.data_handler.f_speed,
            returns=self.data_handler.returns,
            factors=self.data_handler.factors,
        )

        self.logging.debug("Instantiating DQN model")
        input_shape = self.env.get_state_dim()

        self.train_agent = DQN(
            input_shape=input_shape,
            action_space=self.action_space,
            rng=self.rng,
            N_train=self.N_train,
        )

        self.logging.debug("Set up length of training and instantiate test env")
        self.train_agent._get_exploration_length(self.N_train)

        self.logging.debug("Instantiating Out of sample tester")
        self.oos_test = Out_sample_vs_gp(
            savedpath=self.savedpath,
            tag="DQN",
            experiment_type=self.experiment_type,
            env_cls=self.env_cls,
            MV_res=self.MV_res,
            N_test=2000
        )

        self.oos_test.init_series_to_fill(iterations=self.col_names_oos)

    def training_agent(self):
        """
        Main routine to train and test the DRL algorithm. The steps are:

        1. Load the dataset, metadata, any model output and any pre-loaded
        data (cached_data).
        2. Start the Backtrader engine and initialize the broker object.
        3. Instantiate the environment.
        4. Instantiate the model for the agent.
        5. Train the model according to a chosen technique.
        6. Test the model out-of-sample.
        7. Log the performance data, plot, save configuration file and
            the runner logger output.

        Once this is done, the backtest is over and all of the artifacts
        are saved in `_exp/experiment_name/_backtests/`.
        """

        self.logging.debug("Training...")
        # pdb.set_trace()
        CurrState = self.env.reset()

        # CurrOptState = env.opt_reset()
        # OptRate, DiscFactorLoads = env.opt_trading_rate_disc_loads()

        for i in tqdm(iterable=range(self.N_train + 1), desc="Training DQNetwork"):

            self.train_agent.update_epsilon()
            epsilon = self.train_agent.epsilon
            side_only = self.action_space.side_only
            copy_step = self.train_agent.copy_step

            action, qvalues = self.train_agent.eps_greedy_action(
                CurrState, epsilon, side_only=side_only
            )
            if not side_only:
                unscaled_action = action
            else:
                unscaled_action = get_bet_size(
                    qvalues,
                    action,
                    action_limit=self.action_space.action_range[0],
                    zero_action=self.action_space.zero_action,
                    rng=self.rng,
                )
            if self.MV_res:
                NextState, Result, _ = self.env.MV_res_step(
                    CurrState, unscaled_action, i
                )
            else:
                NextState, Result, _ = self.env.step(CurrState, unscaled_action, i)

            self.env.store_results(Result, i)

            exp = {
                "s": CurrState,
                "a": action,
                "a_unsc": unscaled_action,
                "r": Result["Reward_DQN"],
                "s2": NextState,
            }

            self.train_agent.add_experience(exp)

            self.train_agent.train(i, side_only)

            if (i % copy_step == 0) and (i > self.train_agent.start_train):
                self.train_agent.copy_weights()

            CurrState = NextState

            if (i % self.save_freq_n == 0) and (i > self.train_agent.start_train):
                self.train_agent.model.save_weights(
                    os.path.join(self.savedpath, "ckpt", "DQN_{}_ep_weights".format(i)),
                    save_format="tf",
                )

                self.logging.debug("Testing...")
                self.oos_test.run_test(it=i, test_agent=self.train_agent)

            # if executeGP:
            #     NextOptState, OptResult = env.opt_step(
            #         CurrOptState, OptRate, DiscFactorLoads, i
            #     )
            #     env.store_results(OptResult, i)
            #     CurrOptState = NextOptState

        self.oos_test.save_series()

        save_gin(os.path.join(self.savedpath, "config.gin"))
        logging.info("Config file saved")
