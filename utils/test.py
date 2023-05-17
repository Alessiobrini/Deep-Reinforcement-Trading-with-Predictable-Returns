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
from utils.tools import get_bet_size, get_action_boundaries
import torch
import torch.nn as nn
from utils.math_tools import unscale_action, unscale_asymmetric_action, scale_asymmetric_action
from utils.simulation import DataHandler
from utils.plot import optimal_vf
import matplotlib.pyplot as plt
import seaborn as sns


@gin.configurable()
class Out_sample_vs_gp:
    def __init__(
        self,
        N_test: int,
        rnd_state: int,
        savedpath: str,
        tag: str,
        experiment_type: str,
        env_cls: object,
        MV_res: bool,
        universal_train: bool = False,
        mv_solution: bool = False,
        stochastic_policy: bool = True,
    ):

        variables = []
        variables.append("NetPNL_{}".format(tag))
        variables.append("Reward_{}".format(tag))
        variables.append("OptNetPNL")
        variables.append("OptReward")

        self.variables = variables
        self.rnd_state = rnd_state
        self.N_test = N_test
        self.tag = tag
        self.savedpath = savedpath
        self.experiment_type = experiment_type
        self.env_cls = env_cls
        self.MV_res = MV_res
        self.mv_solution = mv_solution
        self.stochastic_policy = stochastic_policy


    def run_test(self, test_agent: object, it: int = 0, return_output: bool = False):

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
        abs_wealth_rl = []
        abs_wealth_gp = []
        
        

        if 'alpha' in gin.query_parameter('%INP_TYPE'):
            data_handler = DataHandler(N_train=self.N_test, rng=self.rng_test)
        else:
            data_handler = DataHandler(N_train=self.N_test, rng=self.rng_test)

        if self.experiment_type == "GP":
            data_handler.generate_returns()
        else:
            # TODO Check if the seeds variaes here
            data_handler.generate_returns()
            data_handler.estimate_parameters()
            # TODO ############################
        cond = not isinstance(gin.query_parameter("%ACTION_RANGE")[0],list)
        # cond = (np.abs(gin.query_parameter("%ACTION_RANGE")[0][0]) != np.abs(gin.query_parameter("%ACTION_RANGE")[0][1]) ) 
        if cond and not self.MV_res:
            action_range, _, _ = get_action_boundaries(
                N_train=self.N_test,
                f_speed=data_handler.f_speed,
                returns=data_handler.returns,
                factors=data_handler.factors,
            )
            gin.query_parameter("%ACTION_RANGE")[0] = action_range
        
        if self.MV_res:
            test_agent.action_space = ResActionSpace()
        else:
            test_agent.action_space = ActionSpace()


        self.test_env = self.env_cls(
            N_train=self.N_test,
            f_speed=data_handler.f_speed,
            returns=data_handler.returns,
            factors=data_handler.factors,
        )
        
        # Store also the value functions
        self.test_env.res_df['Vf_PPO'] = np.nan
        self.test_env.res_df['OptVf'] = np.nan

        CurrState = self.test_env.reset()

        CurrOptState = self.test_env.opt_reset()
        OptRate, DiscFactorLoads = self.test_env.opt_trading_rate_disc_loads()
        if self.mv_solution:
            CurrMVState = self.test_env.opt_reset()

        for i in tqdm(iterable=range(self.N_test + 1), desc="Testing DQNetwork"):

            if self.tag == "DQN":

                side_only = test_agent.action_space.side_only

                action, values = test_agent.greedy_action(
                    CurrState, side_only=side_only
                )
                if side_only:
                    action = get_bet_size(
                        values,
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
                
                test_agent.model.eval()
                CurrState = torch.from_numpy(CurrState).float()
                CurrState = CurrState.to(test_agent.device)
                # PPO actions
                with torch.no_grad():
                    dist, values = test_agent.model(CurrState.unsqueeze(0))
                if test_agent.policy_type == "continuous":
                    action = dist.mean
                    if test_agent.action_clipping_type == 'tanh':
                        action = nn.Tanh()(test_agent.tanh_stretching*action)
                    elif test_agent.action_clipping_type == 'clip':
                        action = torch.clip(action,-test_agent.gaussian_clipping, test_agent.gaussian_clipping)
            

                    if self.MV_res:
                        action = unscale_asymmetric_action(
                            test_agent.action_space.action_range[0],test_agent.action_space.action_range[1], action
                        )
                    else:
                        if test_agent.action_space.asymmetric:
                            action = unscale_asymmetric_action(
                                np.array(test_agent.action_space.action_range[0]),
                                np.array(test_agent.action_space.action_range[1]),
                                action.numpy().ravel(),
                                test_agent.gaussian_clipping
                            )
                        else:
                            action = unscale_action(
                                test_agent.action_space.action_range[0], action,test_agent.gaussian_clipping
                            )

                elif test_agent.policy_type == "discrete":
                    action = test_agent.action_space.values[torch.max(dist.logits, axis=1)[1]]


                if self.MV_res:
                    NextState, Result = self.test_env.MV_res_step(
                        CurrState.cpu(), action, i, tag="PPO"
                    )
                else:
                    NextState, Result, _ = self.test_env.step(
                        CurrState.cpu(), action, i, tag="PPO"
                    )

                self.test_env.store_results(Result, i)
            CurrState = NextState


            
            
            self.test_env.res_df.loc[i,'Vf_PPO'] = values.numpy()[0,0]
            opt_values = optimal_vf(np.asarray(CurrOptState,dtype=float)[1:], 
             self.test_env.discount_rate, 
             self.test_env.kappa, 
             self.test_env.CostMultiplier, 
             self.test_env.f_param, 
             self.test_env.HalfLife, 
             self.test_env.sigma)
            # TODO get just the first value because they are all equal. 
            # Function adapted to when I was creating multiple different states
            self.test_env.res_df.loc[i,'OptVf'] = opt_values[0]
            
            
            
            
            # benchmark agent
            NextOptState, OptResult = self.test_env.opt_step(
                CurrOptState, OptRate, DiscFactorLoads, i
            )
            self.test_env.store_results(OptResult, i)
            CurrOptState = NextOptState
            if self.mv_solution:
                NextMVState, MVResult = self.test_env.mv_step(
                    CurrMVState, i
                )
                self.test_env.store_results(MVResult, i)
                CurrMVState = NextMVState
        
        
        # value function plot

        # fig,ax = plt.subplots(figsize=(12,8))
        # sns.scatterplot(x='Vf_PPO', y='OptVf', data=self.test_env.res_df, ax=ax)
        
        # # Fit regression line
        # sns.regplot(x='Vf_PPO', y='OptVf', data=self.test_env.res_df, ax=ax)
        
        # # Add labels and title
        # ax.set_xlabel('PPO Vf')
        # ax.set_ylabel('GP Vf')
        # plt.show()
        # pdb.set_trace()
        # self.test_env.res_df['OptVf'] = (self.test_env.res_df['OptVf']-self.test_env.res_df['OptVf'].mean())/self.test_env.res_df['OptVf'].std() * 100
        fig,(ax1,ax2) = plt.subplots(2,1, figsize=(12,8))
        
        self.test_env.res_df['Vf_PPO'].plot(ax=ax1)
        self.test_env.res_df['OptVf'].plot(ax=ax2, color='tab:orange')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('PPO Vf')        
        ax2.set_xlabel('Time')
        ax2.set_ylabel('GP Vf')
        fig.suptitle('Corr Coeff fullpath {} \n Corr Coeff t=600 on {}'.format(np.round(self.test_env.res_df[['Vf_PPO','OptVf']].corr().values[0,1],2),
                                                                                np.round(self.test_env.res_df[['Vf_PPO','OptVf']].loc[600:].corr().values[0,1],2)))
        plt.show()
        # fig.savefig('outputs/vf_figs/vf_{}.png'.format(self.rnd_state))
        # pdb.set_trace()
        

        
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
            abs_hold_rl.append(0.0)
            abs_hold_gp.append(0.0)
            avg_pnlstd.append(pnl_std)
            avg_pdist.append(0.0)

            if self.test_env.cash:
                # Wealth
                wealth = self.test_env.res_df["Wealth_{}".format(self.tag)].iloc[:-1]  # avoid last observation
                abs_wealth_rl.append(wealth.iloc[-1])
                optwealth = self.test_env.res_df["OptWealth"].iloc[:-1] 
                abs_wealth_gp.append(optwealth.iloc[-1])


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
            abs_wealthrl=abs_wealth_rl,
            abs_wealthgp=abs_wealth_gp,
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
    
        self.abs_series_wealth_rl = pd.DataFrame(index=range(1), columns=iterations)
        self.abs_series_wealth_gp = pd.DataFrame(index=range(1), columns=iterations)

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
        abs_wealthrl= None,
        abs_wealthgp= None,
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

        if abs_wealthrl:
            self.abs_series_wealth_rl.loc[0, str(it)] = np.mean(abs_wealthrl)
            self.abs_series_wealth_gp.loc[0, str(it)] = np.mean(abs_wealthgp)
        


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

        if 'Cash' in str(gin.query_parameter('%ENV_CLS')):
            self.abs_series_wealth_rl.to_parquet(
                os.path.join(
                    self.savedpath,
                    "AbsWealth_OOS_{}_{}.parquet.gzip".format(
                        format_tousands(self.N_test), self.tag
                    ),
                ),
                compression="gzip",
            )
            self.abs_series_wealth_gp.to_parquet(
                os.path.join(
                    self.savedpath,
                    "AbsWealth_OOS_{}_GP.parquet.gzip".format(format_tousands(self.N_test)),
                ),
                compression="gzip",
            )
