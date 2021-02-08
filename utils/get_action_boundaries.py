# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 14:38:05 2020

@author: aless
"""
import pdb
from utils.MarketEnv import MarketEnv
from tqdm import tqdm
import numpy as np


def get_action_boundaries(
    HalfLife,
    Startholding,
    sigma,
    CostMultiplier,
    kappa,
    N_train,
    discount_rate,
    f_param,
    f_speed,
    returns,
    factors,
    qts=[0.01, 0.99],
    min_n_actions=True,
):

    env = MarketEnv(
        HalfLife,
        Startholding,
        sigma,
        CostMultiplier,
        kappa,
        N_train,
        discount_rate,
        f_param,
        f_speed,
        returns,
        factors,
    )

    CurrOptState = env.opt_reset()
    OptRate, DiscFactorLoads = env.opt_trading_rate_disc_loads()

    cycle_len = len(returns) - 1
    for i in tqdm(iterable=range(cycle_len), desc="Selecting Action boundaries"):
        NextOptState, OptResult = env.opt_step(
            CurrOptState, OptRate, DiscFactorLoads, i
        )
        env.store_results(OptResult, i)
        CurrOptState = NextOptState

    action_quantiles = env.res_df["OptNextAction"].quantile(qts).values

    qt = np.min(np.abs(action_quantiles))
    length = len(str(int(np.round(qt))))
    action = int(np.abs(np.round(qt, -length + 1)))
    if min_n_actions:
        action_ranges = [action, action]
    else:
        action_ranges = [
            action,
            int(action / 2),
        ]  # TODO modify for allowing variables range of actions

    ret_range = float(max(np.abs(returns.min()), returns.max()))

    holding_quantiles = env.res_df["OptNextHolding"].quantile(qts).values
    if np.abs(holding_quantiles[0]) - np.abs(holding_quantiles[1]) < 1000:
        holding_ranges = int(np.abs(np.round(holding_quantiles[0], -2)))
    else:
        holding_ranges = int(np.round(np.min(np.abs(holding_quantiles)), -2))

    return action_ranges, ret_range, holding_ranges
