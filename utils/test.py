# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 12:04:21 2020

@author: aless
"""
from tqdm import tqdm
from utils.spaces import (
    ActionSpace,
    ResActionSpace,
)
from utils.env import MarketEnv
from utils.tools import CalculateLaggedSharpeRatio, RunModels
from utils.common import format_tousands
import gin
import os
import numpy as np
import pandas as pd
from typing import Union, Optional
from pathlib import Path
import pdb, sys
from utils.tools import get_bet_size
import torch
import torch.nn as nn
from utils.math_tools import unscale_action, unscale_asymmetric_action
from utils.simulation import DataHandler


@gin.configurable()
class Out_sample_vs_gp:
    def __init__(
        self,
        n_seeds: int,
        N_test: int,
        rnd_state: int,
        savedpath: str,
        tag: str,
        experiment_type: str,
        env_cls: object,
        MV_res: bool,
    ):

        variables = []
        variables.append("NetPNL_{}".format(tag))
        variables.append("Reward_{}".format(tag))
        variables.append("OptNetPNL")
        variables.append("OptReward")

        self.variables = variables
        self.rnd_state = rnd_state
        self.n_seeds = n_seeds
        self.N_test = N_test
        self.tag = tag
        self.savedpath = savedpath
        self.experiment_type = experiment_type
        self.env_cls = env_cls
        self.MV_res = MV_res

    def run_test(self, test_agent: object, it: int = 0, return_output: bool = False):

        rng = np.random.RandomState(self.rnd_state)
        seeds = rng.choice(1000, self.n_seeds, replace=False)
        self.rng_test = np.random.RandomState(self.rnd_state)

        avg_pnls = []
        avg_rews = []
        avg_srs = []
        abs_pnl_rl = []
        abs_pnl_gp = []
        abs_rew_rl = []
        abs_rew_gp = []
        abs_sr_rl = []
        abs_sr_gp = []
        abs_hold_rl = []
        abs_hold_gp = []
        avg_pnlstd = []
        avg_pdist = []

        for s in seeds:

            data_handler = DataHandler(N_train=self.N_test, rng=self.rng_test)

            if self.experiment_type == "GP":
                data_handler.generate_returns()
            else:
                data_handler.generate_returns()
                # TODO check if these method really fit and change the parameters in the gin file
                data_handler.estimate_parameters()

            self.test_env = self.env_cls(
                N_train=self.N_test,
                f_speed=data_handler.f_speed,
                returns=data_handler.returns,
                factors=data_handler.factors,
            )

            CurrState, _ = self.test_env.reset()

            CurrOptState = self.test_env.opt_reset()
            OptRate, DiscFactorLoads = self.test_env.opt_trading_rate_disc_loads()

            for i in tqdm(iterable=range(self.N_test + 1), desc="Testing DQNetwork"):

                if self.tag == "DQN":

                    side_only = test_agent.action_space.side_only

                    action, qvalues = test_agent.greedy_action(
                        CurrState, side_only=side_only
                    )
                    if side_only:
                        action = get_bet_size(
                            qvalues,
                            action,
                            action_limit=test_agent.action_space.action_range[0],
                            zero_action=test_agent.action_space.zero_action,
                            rng=self.rng,
                        )

                    if self.MV_res:
                        NextState, Result, _ = self.test_env.MV_res_step(
                            CurrState, action, i
                        )
                    else:
                        NextState, Result, NextFactors = self.test_env.step(
                            CurrState, action, i,
                        )
                    self.test_env.store_results(Result, i)

                elif self.tag == "PPO":
                    side_only = test_agent.action_space.side_only
                    test_agent.model.eval()
                    CurrState = torch.from_numpy(CurrState).float()
                    CurrState = CurrState.to(test_agent.device)

                    # PPO actions
                    with torch.no_grad():
                        dist, qvalues = test_agent.model(CurrState.unsqueeze(0))

                    if test_agent.policy_type == "continuous":
                        # action = dist.sample()
                        action = dist.mean
                        action = nn.Tanh()(action).cpu().numpy().ravel()[0]

                        if self.MV_res:
                            action = unscale_asymmetric_action(
                                test_agent.action_space.action_range[0],test_agent.action_space.action_range[1], action
                            )
                        else:
                            action = unscale_action(
                                test_agent.action_space.values[-1], action
                            )

                    elif test_agent.policy_type == "discrete":
                        action = test_agent.action_space.values[dist.sample()]

                    if side_only:
                        action = get_bet_size(
                            qvalues,
                            action,
                            action_limit=test_agent.action_space.action_range[0],
                            zero_action=test_agent.action_space.zero_action,
                            rng=self.rng,
                        )
                    if self.MV_res:
                        NextState, Result, _ = self.test_env.MV_res_step(
                            CurrState.cpu(), action, i, tag="PPO"
                        )
                    else:
                        NextState, Result, _ = self.test_env.step(
                            CurrState.cpu(), action, i, tag="PPO"
                        )

                    self.test_env.store_results(Result, i)

                CurrState = NextState

                # benchmark agent
                NextOptState, OptResult = self.test_env.opt_step(
                    CurrOptState, OptRate, DiscFactorLoads, i
                )

                self.test_env.store_results(OptResult, i)

                CurrOptState = NextOptState

            if return_output:
                return self.test_env.res_df

            else:

                # select interesting variables and express as a percentage of the GP results
                pnl_str = list(
                    filter(lambda x: "NetPNL_{}".format(self.tag) in x, self.variables)
                )
                opt_pnl_str = list(filter(lambda x: "OptNetPNL" in x, self.variables))
                rew_str = list(
                    filter(lambda x: "Reward_{}".format(self.tag) in x, self.variables)
                )
                opt_rew_str = list(filter(lambda x: "OptReward" in x, self.variables))

                # pnl
                pnl = self.test_env.res_df[pnl_str + opt_pnl_str].iloc[:-1]
                cum_pnl = pnl.cumsum()

                if (
                    data_handler.datatype == "garch"
                    or data_handler.datatype == "garch_mr"
                ):
                    ref_pnl = np.array(cum_pnl[pnl_str]) - np.array(
                        cum_pnl[opt_pnl_str]
                    )
                else:
                    ref_pnl = (
                        np.array(cum_pnl[pnl_str]) / np.array(cum_pnl[opt_pnl_str])
                    ) * 100

                # rewards
                rew = self.test_env.res_df[rew_str + opt_rew_str].iloc[:-1]
                cum_rew = rew.cumsum()
                if (
                    data_handler.datatype == "garch"
                    or data_handler.datatype == "garch_mr"
                ):
                    ref_rew = np.array(cum_rew[rew_str]) - np.array(
                        cum_rew[opt_rew_str]
                    )
                else:
                    ref_rew = (
                        np.array(cum_rew[rew_str]) / np.array(cum_rew[opt_rew_str])
                    ) * 100

                # SR
                mean = np.array(pnl[pnl_str]).mean()
                std = np.array(pnl[pnl_str]).std()
                sr = (mean / std) * (252 ** 0.5)

                # Holding
                hold = self.test_env.res_df["NextHolding_{}".format(self.tag)].iloc[
                    -2
                ]  # avoid last observation
                opthold = self.test_env.res_df["OptNextHolding"].iloc[-2]

                pdist_avg = (
                    (
                        self.test_env.res_df["NextHolding_{}".format(self.tag)].values
                        - self.test_env.res_df["OptNextHolding"].values
                    )
                    ** 2
                ).mean()

                opt_mean = np.array(pnl[opt_pnl_str]).mean()
                opt_std = np.array(pnl[opt_pnl_str]).std()
                optsr = (opt_mean / opt_std) * (252 ** 0.5)

                perc_SR = (sr / optsr) * 100
                pnl_std = (std / opt_std) * 100

                avg_pnls.append(ref_pnl[-1])
                avg_rews.append(ref_rew[-1])
                avg_srs.append(perc_SR)
                abs_pnl_rl.append(cum_pnl.iloc[-1].values[0])
                abs_pnl_gp.append(cum_pnl.iloc[-1].values[1])
                abs_rew_rl.append(cum_rew.iloc[-1].values[0])
                abs_rew_gp.append(cum_rew.iloc[-1].values[1])
                abs_sr_rl.append(sr)
                abs_sr_gp.append(optsr)
                abs_hold_rl.append(hold)
                abs_hold_gp.append(opthold)
                avg_pnlstd.append(pnl_std)
                avg_pdist.append(pdist_avg)

        self._collect_results(
            avg_pnls,
            avg_rews,
            avg_srs,
            abs_pnl_rl,
            abs_pnl_gp,
            abs_rew_rl,
            abs_rew_gp,
            abs_sr_rl,
            abs_sr_gp,
            abs_hold_rl,
            abs_hold_gp,
            avg_pnlstd,
            avg_pdist,
            it=it,
        )

    def init_series_to_fill(self, iterations):
        self.mean_series_pnl = pd.DataFrame(index=range(1), columns=iterations)
        self.mean_series_rew = pd.DataFrame(index=range(1), columns=iterations)
        self.mean_series_sr = pd.DataFrame(index=range(1), columns=iterations)
        self.mean_series_pnl_std = pd.DataFrame(index=range(1), columns=iterations)

        self.abs_series_pnl_rl = pd.DataFrame(index=range(1), columns=iterations)
        self.abs_series_pnl_gp = pd.DataFrame(index=range(1), columns=iterations)

        self.abs_series_rew_rl = pd.DataFrame(index=range(1), columns=iterations)
        self.abs_series_rew_gp = pd.DataFrame(index=range(1), columns=iterations)

        self.abs_series_sr_rl = pd.DataFrame(index=range(1), columns=iterations)
        self.abs_series_sr_gp = pd.DataFrame(index=range(1), columns=iterations)

        self.abs_series_hold_rl = pd.DataFrame(index=range(1), columns=iterations)
        self.abs_series_hold_gp = pd.DataFrame(index=range(1), columns=iterations)

        self.mean_series_pdist = pd.DataFrame(index=range(1), columns=iterations)

    def _collect_results(
        self,
        pnl,
        rew,
        sr,
        abs_prl,
        abs_pgp,
        abs_rewrl,
        abs_rewgp,
        abs_srrl,
        abs_srgp,
        abs_hold,
        abs_opthold,
        pnl_std,
        pdist_avg,
        it,
    ):

        self.mean_series_pnl.loc[0, str(it)] = np.mean(pnl)
        self.mean_series_rew.loc[0, str(it)] = np.mean(rew)
        self.mean_series_sr.loc[0, str(it)] = np.mean(sr)
        self.mean_series_pnl_std.loc[0, str(it)] = np.mean(pnl_std)

        self.abs_series_pnl_rl.loc[0, str(it)] = np.mean(abs_prl)
        self.abs_series_pnl_gp.loc[0, str(it)] = np.mean(abs_pgp)

        self.abs_series_rew_rl.loc[0, str(it)] = np.mean(abs_rewrl)
        self.abs_series_rew_gp.loc[0, str(it)] = np.mean(abs_rewgp)

        self.abs_series_sr_rl.loc[0, str(it)] = np.mean(abs_srrl)
        self.abs_series_sr_gp.loc[0, str(it)] = np.mean(abs_srgp)

        self.abs_series_hold_rl.loc[0, str(it)] = np.mean(abs_hold)
        self.abs_series_hold_gp.loc[0, str(it)] = np.mean(abs_opthold)

        self.mean_series_pdist.loc[0, str(it)] = np.mean(pdist_avg)

    def save_series(self):

        self.mean_series_pnl.to_parquet(
            os.path.join(
                self.savedpath,
                "NetPnl_OOS_{}_{}.parquet.gzip".format(
                    format_tousands(self.N_test), self.tag
                ),
            ),
            compression="gzip",
        )
        self.mean_series_rew.to_parquet(
            os.path.join(
                self.savedpath,
                "Reward_OOS_{}_{}.parquet.gzip".format(
                    format_tousands(self.N_test), self.tag
                ),
            ),
            compression="gzip",
        )
        self.mean_series_sr.to_parquet(
            os.path.join(
                self.savedpath,
                "SR_OOS_{}_{}.parquet.gzip".format(
                    format_tousands(self.N_test), self.tag
                ),
            ),
            compression="gzip",
        )

        self.mean_series_pnl_std.to_parquet(
            os.path.join(
                self.savedpath,
                "PnLstd_OOS_{}_{}.parquet.gzip".format(
                    format_tousands(self.N_test), self.tag
                ),
            ),
            compression="gzip",
        )

        self.abs_series_pnl_rl.to_parquet(
            os.path.join(
                self.savedpath,
                "AbsNetPnl_OOS_{}_{}.parquet.gzip".format(
                    format_tousands(self.N_test), self.tag
                ),
            ),
            compression="gzip",
        )
        self.abs_series_pnl_gp.to_parquet(
            os.path.join(
                self.savedpath,
                "AbsNetPnl_OOS_{}_GP.parquet.gzip".format(format_tousands(self.N_test)),
            ),
            compression="gzip",
        )
        self.abs_series_rew_rl.to_parquet(
            os.path.join(
                self.savedpath,
                "AbsRew_OOS_{}_{}.parquet.gzip".format(
                    format_tousands(self.N_test), self.tag
                ),
            ),
            compression="gzip",
        )
        self.abs_series_rew_gp.to_parquet(
            os.path.join(
                self.savedpath,
                "AbsRew_OOS_{}_GP.parquet.gzip".format(format_tousands(self.N_test)),
            ),
            compression="gzip",
        )
        self.abs_series_sr_rl.to_parquet(
            os.path.join(
                self.savedpath,
                "AbsSR_OOS_{}_{}.parquet.gzip".format(
                    format_tousands(self.N_test), self.tag
                ),
            ),
            compression="gzip",
        )
        self.abs_series_sr_gp.to_parquet(
            os.path.join(
                self.savedpath,
                "AbsSR_OOS_{}_GP.parquet.gzip".format(format_tousands(self.N_test)),
            ),
            compression="gzip",
        )
        self.abs_series_hold_rl.to_parquet(
            os.path.join(
                self.savedpath,
                "AbsHold_OOS_{}_{}.parquet.gzip".format(
                    format_tousands(self.N_test), self.tag
                ),
            ),
            compression="gzip",
        )
        self.abs_series_hold_gp.to_parquet(
            os.path.join(
                self.savedpath,
                "AbsHold_OOS_{}_GP.parquet.gzip".format(format_tousands(self.N_test)),
            ),
            compression="gzip",
        )

        self.mean_series_pdist.to_parquet(
            os.path.join(
                self.savedpath,
                "Pdist_OOS_{}_GP.parquet.gzip".format(format_tousands(self.N_test)),
            ),
            compression="gzip",
        )
