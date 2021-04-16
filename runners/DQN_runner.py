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
        action_space_fn: object,
        MV_res: bool,
        experiment_type: str,
        seed: int,
        episodes: int,
        N_train:int,
        len_series:Union[int or None],
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

        if self.episodes:
            self.N_train = self.episodes * self.len_series
            self.col_names_oos = [
                str(e) for e in np.arange(0, self.episodes + 1, save_freq)[1:]
            ]
        else:
            self.len_series = self.N_train
            self.save_freq_n = self.N_train // save_freq
            self.col_names_oos = [str(int(i)) for i in np.arange(0,self.N_train+1,self.save_freq_n)][1:]

        self.savedpath = GeneratePathFolder(
            outputDir,
            outputClass,
            outputModel,
            varying_pars,
            varying_type,
            self.N_train
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
            if self.episodes:
                self.training_episodic_agent()
            else:
                self.training_agent()
        except (KeyboardInterrupt, SystemExit):
            self.logging.debug("Exit on KeyboardInterrupt or SystemExit")
            sys.exit()

    def set_up_training(self):

        self.logging.debug("Simulating Data")
        
        self.data_handler = DataHandler(N_train=self.N_train, rng=self.rng)
        if self.experiment_type == 'GP':
            self.data_handler.generate_returns()
        else:
            self.data_handler.generate_returns()
            # TODO check if these method really fit and change the parameters in the gin file
            self.data_handler.estimate_parameters()
        
        self.logging.debug("Instantiating action space")
        if self.MV_res:
            self.action_space = self.action_space_fn()
        else:
            action_range, ret_quantile, holding_quantile = get_action_boundaries(
                N_train=self.N_train,
                f_speed=self.data_handler.f_speed,
                returns= self.data_handler.returns,
                factors = self.data_handler.factors,
            )

            gin.query_parameter('%ACTION_RANGE')[0] = action_range
            self.action_space = self.action_space_fn()

        self.logging.debug("Instantiating market environment")
        self.env = self.env_cls(N_train=self.N_train,
                                f_speed=self.data_handler.f_speed,
                                returns= self.data_handler.returns,
                                factors = self.data_handler.factors,
                                )

        self.logging.debug("Instantiating DQN model")
        input_shape = self.env.get_state_dim()

        self.train_agent = DQN(
            input_shape=input_shape, action_space=self.action_space, rng=self.rng, N_train=self.N_train
        )

        self.logging.debug("Set up length of training and instantiate test env")
        self.train_agent._get_exploration_length(self.N_train)

        self.logging.debug("Instantiating Out of sample tester")
        self.oos_test = Out_sample_vs_gp(
            savedpath=self.savedpath,
            tag="DQN",
            experiment_type = self.experiment_type,
            env_cls = self.env_cls,
            MV_res=self.MV_res
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
        CurrState, _ = self.env.reset()
 
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
                NextState, Result, _ = self.env.MV_res_step(CurrState, unscaled_action, i)
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
                    os.path.join(
                        self.savedpath, "ckpt", "DQN_{}_ep_weights".format(i)
                    ),
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

    # def training_episodic_agent(self):
    #     """
    #     Main routine to train and test the DRL algorithm. The steps are:

    #     1. Load the dataset, metadata, any model output and any pre-loaded
    #     data (cached_data).
    #     2. Start the Backtrader engine and initialize the broker object.
    #     3. Instantiate the environment.
    #     4. Instantiate the model for the agent.
    #     5. Train the model according to a chosen technique.
    #     6. Test the model out-of-sample.
    #     7. Log the performance data, plot, save configuration file and
    #         the runner logger output.

    #     Once this is done, the backtest is over and all of the artifacts
    #     are saved in `_exp/experiment_name/_backtests/`.
    #     """
        
    #     self.logging.debug("Start episodic training...")
    #     for e in tqdm(iterable=range(self.episodes), desc="Running episodes..."):

    #         self.logging.debug("Rpisodic training...")

    #         self.collect_rollouts()

    #         self.update()

    #         if self.save_freq and ((e + 1) % self.save_freq == 0):
    #             self.train_agent.model.save_weights(
    #                 os.path.join(
    #                     self.savedpath, "ckpt", "DQN_{}_ep_weights".format(e + 1)
    #                 ),
    #                 save_format="tf",
    #             )

    #             self.logging.debug("Testing...")
    #             if len(self.test_symbols)>1:
    #                 if e+1 == self.episodes: 
    #                     last_episode=True 
    #                 else: 
    #                     last_episode=False
    #                 self.oos_test.run_multiple_tests(episode=e + 1, last_episode=last_episode)
    #             else:
    #                 self.oos_test.run_test(episode=e + 1)

    #     if len(self.test_symbols)>1:
    #         self.oos_test.save_avg_series(self.savedpath)
    #     else:
    #         self.oos_test.save_series(self.savedpath)

    #     if self.universal:
    #         np.save(os.path.join(self.savedpath,'symbols.npy'),self.used_symbols)
    #     save_gin(os.path.join(self.savedpath, "config.gin"))
    #     logging.info("Config file saved")


    # def testing_agent(self):

    #     self.logging.debug("Loading Data")
    #     df_train, df_test, df = self.load_data()

    #     self.logging.debug("Instantiating action space")
    #     self.action_space = ActionSpace()
        
    #     self.test_env = self.env_cls(dataframe=df_test)

    #     self.logging.debug("Instantiating benchmark agent")
    #     self.benchmark_agent = self.bnch_agent_cls(
    #         symbol=self.symbol, 
    #         start_year=self.start_year,
    #         split_year=self.split_year,
    #         end_year=self.end_year, 
    #         df=df,
    #     )
    #     self.benchmark_agent._get_parameters()
    #     self.benchmark_test_env = self.bnch_env_cls(
    #         dataframe=self.benchmark_agent.df_test
    #     )

    #     self.logging.debug("Loading pretrained model")
    #     input_shape = self.test_env.get_state_dim()

    #     self.train_agent = self.agent_cls(
    #         input_shape=input_shape, action_space=self.action_space, rng=self.rng
    #     )

    #     self.train_agent.model.load_weights(
    #             os.path.join(self.savedpath, "ckpt", "DQN_{}_ep_weights".format(self.episodes))
    #         )
        

    #     self.logging.debug("Instantiating Out of sample tester")
    #     self.oos_test = Out_sample_vs_gp(
    #         test_agent=self.train_agent,
    #         test_env=self.test_env,
    #         benchmark_agent=self.benchmark_agent,
    #         benchmark_env=self.benchmark_test_env,
    #         savedpath=self.savedpath,
    #         test_symbols = [self.symbol],
    #         tag="DQN",
    #     )

    #     self.logging.debug("Testing...")
    #     res_df, res_bench_df = self.oos_test.run_test(self.oos_test, return_output=True)

    #     return res_df, res_bench_df


    # def collect_rollouts(self):

    #     CurrState = self.env.reset()

    #     for i in range(len(self.env.returns) - 1):

    #         self.train_agent.update_epsilon()
    #         epsilon = self.train_agent.epsilon
    #         side_only = self.action_space.side_only

    #         action, qvalues = self.train_agent.eps_greedy_action(
    #             CurrState, epsilon, side_only=side_only
    #         )
    #         if not side_only:
    #             unscaled_action = action
    #         else:
    #             unscaled_action = get_bet_size(
    #                 qvalues,
    #                 action,
    #                 action_limit=self.action_space.action_range[0],
    #                 zero_action=self.action_space.zero_action,
    #                 rng=self.rng,
    #             )

    #         NextState, Result = self.env.step(CurrState, unscaled_action, i)

    #         exp = {
    #             "s": CurrState,
    #             "a": action,
    #             "a_unsc": unscaled_action,
    #             "r": Result["Reward_DQN"],
    #             "s2": NextState,
    #         }

    #         self.train_agent.add_experience(exp)
    #         CurrState = NextState

    # def update(self):

    #     copy_step = self.train_agent.copy_step
    #     side_only = self.action_space.side_only

    #     for i in range(len(self.env.returns) - 1):

    #         self.train_agent.train(i, side_only)

    #         self.iters += 1
    #         if (self.iters % copy_step == 0) and (i > self.train_agent.start_train):
    #             self.train_agent.copy_weights()
