# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 15:39:12 2020

@author: aless
"""

from typing import Union, Tuple
import pandas as pd
import numpy as np
import gym
from gym.spaces.space import Space
import pdb, os
from utils.math_tools import unscale_action
from utils.common import format_tousands


class ReturnSpace(Space):
    """
    Class used to discretize the space of returns. It inherits from the Space class
    of gym package

    ...

    Attributes
    ----------
    values : np.ndarray
        numpy array containing all the possible values for the space of returns

    Methods
    -------
    contains(x: float) -> bool
        check if a real-valued return is contained in the discretized space
        and return a boolean
    """

    def __init__(self, RT: list):
        self.values = np.arange(-RT[0], RT[0], RT[1])
        super().__init__(self.values.shape, self.values.dtype)

    def contains(self, x: float) -> bool:
        return x in self.values


class HoldingSpace(Space):
    """
    Class used to discretize the space of portfolio holdings. It inherits from the Space class
    of gym package
    ...

    Attributes
    ----------
    values : np.ndarray
        numpy array containing all the possible values for the space of holdings

    Methods
    -------
    contains(x: float or int)-> bool
        check if a real-valued holding is contained in the discretized space
        and return a boolean
    """

    def __init__(self, KLM: list):
        self.values = np.arange(-KLM[2], KLM[2] + 1, KLM[1])
        if 0 not in self.values:
            self.values = np.sort(np.append(self.values, 0))
        super().__init__(self.values.shape, self.values.dtype)

    def contains(self, x: float or int) -> bool:
        return x in self.values


class ActionSpace(Space):
    """
    Class used to discretize the space of action. It inherits from the Space class
    of gym package
    ...

    Attributes
    ----------
    values : np.ndarray
        numpy array containing all the possible values for the actions

    Methods
    -------
    contains(x: float or int)-> bool
        check if an integer action is contained in the discretized space
        and return a boolean
    """

    def __init__(self, KLM: list, zero_action: bool = True, side_only: bool = False):
        if not side_only:
            self.values = np.arange(-KLM[0], KLM[0] + 1, KLM[1])
        else:
            self.values = np.array([-1.0, 0.0, 1.0])
        if not zero_action:
            self.values = self.values[self.values != 0.0]
        super().__init__(self.values.shape, self.values.dtype)

    def contains(self, x: int) -> bool:
        return x in self.values

    def get_n_actions(self, policy_type: str):
        # TODO note that this implementation is valid only for a single action.
        # If we want to do more than one action we should change it
        if policy_type == "continuous":
            return self.values.ndim
        elif policy_type == "discrete":
            return self.values.size


class CreateQTable:
    """
    Class which represents the Q-table to approximate the action-value function
    in the Q-learning algorithm
    ...

    Attributes
    ----------
    rng : np.ndarray
        random number generator with possbily a fixed seed for reproducibility
    ReturnSpace : np.ndarray
        discretized return space
    HoldingSpace : np.ndarray
        discretized holding space
    ActionSpace : np.ndarray
        discretized action space
    tablr : float
        step size for table update (learning rate)
    gamma : float
        discount rate for TD error computation
    Q_space : pd.DataFrame
        current Q-table estimate

    Methods
    -------
    getQvalue(state: np.ndarray)-> np.ndarray
        Get the estimated action-value for each action at the current state

    argmaxQ(state: np.ndarray)-> int
        Get the index position of the action that gives the maximum action-value
        at the current state

    getMaxQ(state: np.ndarray)-> int
        Get the action that gives the maximum action-value at the current state

    chooseAction(state: np.ndarray, epsilon: float)-> int
        Get the index position of the action that gives the maximum action-value
        at the current state or a random action depeding on the epsilon parameter
        of exploration

    chooseGreedyAction(state: np.ndarray)-> int
        Get the index position of the action that gives the maximum action-value
        at the current state

    update(DiscrCurrState: np.ndarray,DiscrNextState: np.ndarray,
           shares_traded: int, Result: dict)
        Perform a Q-table update in correspondence of the current state

    save(savedpath: Union[str, bytes, os.PathLike], N_train: int)-> bool
        Store the current Q-table representation in parquet format
    """

    def __init__(
        self,
        ReturnSpace: ReturnSpace,
        HoldingSpace: HoldingSpace,
        ActionSpace: ActionSpace,
        tablr: float,
        gamma: float,
        seed: int,
    ):
        # generate row index of the dataframe with every possible combination
        # of state space variables
        self.rng = np.random.RandomState(seed)
        self.ReturnSpace = ReturnSpace
        self.HoldingSpace = HoldingSpace
        self.ActionSpace = ActionSpace

        self.tablr = tablr
        self.gamma = gamma

        # generate row index of the dataframe with every possible combination
        # of state space variables
        iterables = [self.ReturnSpace.values, self.HoldingSpace.values]
        State_space = pd.MultiIndex.from_product(iterables)

        # Create dataframe and set properly index and column heading
        Q_space = pd.DataFrame(
            index=State_space, columns=self.ActionSpace.values
        ).fillna(0)
        Q_space.index.set_names(["Return", "Holding"], inplace=True)
        Q_space.columns.set_names(["Action"], inplace=True)
        # initialize the Qvalues for action 0 as slightly greater than 0 so that
        # 'doing nothing' becomes the default action, instead the default action to be the first column of
        # the dataframe.
        Q_space[0] = 0.0000000001

        self.Q_space = Q_space

    def getQvalue(self, state: np.ndarray):
        ret = state[0]
        holding = state[1]
        return self.Q_space.loc[
            (ret, holding),
        ]

    def argmaxQ(self, state: np.ndarray):
        return self.getQvalue(state).idxmax()

    def getMaxQ(self, state: np.ndarray):
        return self.getQvalue(state).max()

    def chooseAction(self, state: np.ndarray, epsilon: float):
        random_action = self.rng.random()
        if random_action < epsilon:
            # pick one action at random for exploration purposes
            dn = self.rng.choice(self.ActionSpace.values)
        else:
            # pick the greedy action
            dn = self.argmaxQ(state)

        return dn

    def chooseGreedyAction(self, state: np.ndarray):
        return self.argmaxQ(state)

    def update(
        self,
        DiscrCurrState: np.ndarray,
        DiscrNextState: np.ndarray,
        shares_traded: int,
        Result: dict,
    ):
        q_sa = self.Q_space.loc[tuple(DiscrCurrState), shares_traded]
        increment = self.tablr * (
            Result["Reward_Q"] + self.gamma * self.getMaxQ(DiscrNextState) - q_sa
        )
        self.Q_space.loc[tuple(DiscrCurrState), shares_traded] = q_sa + increment

    def save(self, savedpath: Union[str, bytes, os.PathLike], N_train: int):
        tmp_cols = self.Q_space.columns
        # convert the column name to string in order to save the dataframe as parquet
        self.Q_space.columns = [str(c) for c in self.Q_space.columns]
        self.Q_space.to_parquet(
            os.path.join(
                savedpath, "QTable{}.parquet.gzip".format(format_tousands(N_train))
            ),
            compression="gzip",
        )
        self.Q_space.columns = tmp_cols


class MarketEnv(gym.Env):
    """
    Custom Market Environment class that follows gym interface. Different agents as
    DQN and Q-learnign can operate within the same envirnoment since there is no way
    for them to affect the asset returns which are simulated in advance.
    ...

    Attributes
    ----------
    HalfLife: Union[int or list or np.ndarray]
        List of HalfLife of mean reversion when the simulated dynamic is driven by
        factors

    Startholding: Union[int or float]
        Initial portfolio holding, usually set at 0

    sigma: float
        Constant volatility for the simulated returns of the asset

    CostMultiplier: float
        Transaction cost parameter which regulates the market liquidity

    discount_rate: float
        Discount rate for the reward function

    kappa: float
        Risk averion parameter

    N_train: int
        Length of simulated experiment

    f_param: Union[float or list or np.ndarray]
        List of factor loadings when the simulated dynamic is driven by
        factors

    f_speed: Union[float or list or np.ndarray]
        List of speed of mean reversion when the simulated dynamic is driven by
        factors

    returns: Union[list or np.ndarray]
        Array of simulated returns

    factors: Union[list or np.ndarray]
        Array of simulated factors when the simulated dynamic is driven by
        factors or lagged factors when it is not the case

    action_limit: int = None
        Action space boundary used in DDPG implementation

    dates: pd.DatetimeIndex = None
        Series of datetime values if real values are used within the environment,
        otherwise it is just a serie of integer number of length N_train

    res_df: pd.Dataframe
        Dataframe which store results of relevant quantities

    Methods
    -------
    reset() -> Tuple[np.ndarray,np.ndarray]
        Get the initial state representation and their associated factors

    step(currState: Union[Tuple or np.ndarray],shares_traded: int,
             iteration: int, tag: str ='DQN') -> Tuple[np.ndarray,dict,np.ndarray]
        Make a step of the environment returning the next state representation,
        a dictionary with relevan measure to store and the factors associated to
        the next state representation

    discretereset() -> np.ndarray
        Discrete counterpart of reset method used for Q-learning

    discretestep(discretecurrState: Union[Tuple or np.ndarray],shares_traded: int,
                 iteration: int) -> Tuple[np.ndarray,np.ndarray]
        Discrete counterpart of step method used for Q-learning

    opt_reset() -> np.ndarray
        Counterpart of reset method used for the benchmark

    opt_step(currOptState: Tuple, OptRate: float,DiscFactorLoads: np.ndarray,
             iteration: int,tag: str = 'Opt') ->  Tuple[np.ndarray,dict]
        Counterpart of step method used for the benchmark

    mv_step(currOptState: Tuple, iteration: int,
            tag: str = 'MV') -> Tuple[np.ndarray,dict]:
        Counterpart of step method used for the Markovitz solution

    store_results(Result:dict, iteration: int)
        Store dictionary of current results to the DataFrame saved as attribute
        of the class

    save_outputs(self, savedpath, test=None, iteration=None, include_dates=False)
        Save the DataFrame saved as attribute of the class in a parquet format

    opt_trading_rate_disc_loads() -> Tuple[float,np.ndarray]
        Compute the trading rate and the discounted factor loading as for
        the optimal benchmark solution

    Private Methods
    -------

    _find_nearest_return(value) -> float
        Get the discretized counterpart of the return (value)

    _find_nearest_holding(value) -> Union[float or int]
        Get the discretized counterpart of the holding (value)

    _totalcost(shares_traded: Union[float or int]) -> Union[float or int]
        Compute transaction cost for the given trade

    _getreward(currState: Tuple[Union[float or int],Union[float or int]],
               nextState: Tuple[Union[float or int],Union[float or int]],
               tag: str) -> dict
        Compute Reward function for the current state and action for DQN or Q-learning
    _get_opt_reward(currState: Tuple[Union[float or int],Union[float or int]],
               nextState: Tuple[Union[float or int],Union[float or int]],
               tag: str) -> dict
        Compute Reward function for the current state and action for the benchmark
    """

    def __init__(
        self,
        HalfLife: Union[int or list or np.ndarray],
        Startholding: Union[int or float],
        sigma: float,
        CostMultiplier: float,
        kappa: float,
        N_train: int,
        discount_rate: float,
        f_param: Union[float or list or np.ndarray],
        f_speed: Union[float or list or np.ndarray],
        returns: Union[list or np.ndarray],
        factors: Union[list or np.ndarray],
        action_limit: int = None,
        dates: pd.DatetimeIndex = None,
    ):

        # super(MarketEnv, self).__init__()
        super().__init__()

        self.HalfLife = HalfLife
        self.Startholding = Startholding
        self.sigma = sigma
        self.CostMultiplier = CostMultiplier
        self.kappa = kappa
        self.N_train = N_train
        self.discount_rate = discount_rate
        self.f_param = f_param
        self.f_speed = f_speed
        self.returns = returns
        self.factors = factors
        self.action_limit = action_limit

        colnames = ["returns"] + ["factor_" + str(hl) for hl in HalfLife]

        res_df = pd.DataFrame(
            np.concatenate(
                [np.array(self.returns).reshape(-1, 1), np.array(self.factors)], axis=1
            ),
            columns=colnames,
        )

        self.dates = dates
        res_df = res_df.astype(np.float32)
        self.res_df = res_df

    def get_state_dim(self):
        state, _ = self.reset()
        return state.shape

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        currState = np.array([self.returns[0], self.Startholding])
        currFactor = self.factors[0]
        return currState, currFactor

    def step(
        self,
        currState: Union[Tuple or np.ndarray],
        shares_traded: int,
        iteration: int,
        tag: str = "DQN",
    ) -> Tuple[np.ndarray, dict, np.ndarray]:

        nextFactors = self.factors[iteration + 1]
        nextRet = self.returns[iteration + 1]
        if tag == "DDPG":
            shares_traded = unscale_action(self.action_limit, shares_traded)

        nextHolding = currState[1] + shares_traded
        nextState = np.array([nextRet, nextHolding], dtype=np.float32)

        Result = self._getreward(currState, nextState, tag)
        # reward scaling
        # if tag == "DDPG":
        #     Result["Reward_{}".format(tag)] = Result["Reward_{}".format(tag)]*0.0001

        return nextState, Result, nextFactors

    def discrete_reset(self) -> np.ndarray:
        discretecurrState = np.array(
            [
                self._find_nearest_return(self.returns[0]),
                self._find_nearest_holding(self.Startholding),
            ]
        )
        return discretecurrState

    def discrete_step(
        self,
        discretecurrState: Union[Tuple or np.ndarray],
        shares_traded: int,
        iteration: int,
    ) -> Tuple[
        np.ndarray, dict, np.ndarray
    ]:  # TODO implement here decoupling if needed
        discretenextRet = self._find_nearest_return(self.returns[iteration + 1])
        discretenextHolding = self._find_nearest_holding(
            discretecurrState[1] + shares_traded
        )
        discretenextState = np.array([discretenextRet, discretenextHolding])
        Result = self._getreward(discretecurrState, discretenextState, "Q")
        return discretenextState, Result

    def opt_reset(self) -> np.ndarray:
        currOptState = np.array(
            [self.returns[0], self.factors[0], self.Startholding], dtype=object
        )
        return currOptState

    def opt_step(
        self,
        currOptState: Tuple,
        OptRate: float,
        DiscFactorLoads: np.ndarray,
        iteration: int,
        tag: str = "Opt",
    ) -> Tuple[np.ndarray, dict]:

        CurrFactors = currOptState[1]
        OptCurrHolding = currOptState[2]

        # Optimal traded quantity between period
        OptNextHolding = (1 - OptRate) * OptCurrHolding + OptRate * (
            1 / (self.kappa * (self.sigma) ** 2)
        ) * np.sum(DiscFactorLoads * CurrFactors)

        nextReturns = self.returns[iteration + 1]
        nextFactors = self.factors[iteration + 1]
        nextOptState = (nextReturns, nextFactors, OptNextHolding)

        OptResult = self._get_opt_reward(currOptState, nextOptState, tag)

        return nextOptState, OptResult

    def mv_step(
        self, currOptState: Tuple, iteration: int, tag: str = "MV"
    ) -> Tuple[np.ndarray, dict]:

        CurrFactors = currOptState[1]

        # Traded quantity as for the Markovitz framework  (Mean-Variance framework)
        OptNextHolding = (1 / (self.kappa * (self.sigma) ** 2)) * np.sum(
            self.f_param * CurrFactors
        )

        nextReturns = self.returns[iteration + 1]
        nextFactors = self.factors[iteration + 1]
        nextOptState = (nextReturns, nextFactors, OptNextHolding)

        OptResult = self._get_opt_reward(currOptState, nextOptState, tag)

        return nextOptState, OptResult

    def store_results(self, Result: dict, iteration: int):

        if iteration == 0:
            for key in Result.keys():
                self.res_df[key] = 0.0
                self.res_df.at[iteration, key] = Result[key]
            self.res_df = self.res_df.astype(np.float32)
        else:
            for key in Result.keys():
                self.res_df.at[iteration, key] = Result[key]

    def save_outputs(self, savedpath, test=None, iteration=None, include_dates=False):

        if not test:
            if include_dates:
                self.res_df["date"] = self.dates
                self.res_df.to_parquet(
                    os.path.join(
                        savedpath,
                        "Results_{}.parquet.gzip".format(format_tousands(self.N_train)),
                    ),
                    compression="gzip",
                )
            else:
                self.res_df.to_parquet(
                    os.path.join(
                        savedpath,
                        "Results_{}.parquet.gzip".format(format_tousands(self.N_train)),
                    ),
                    compression="gzip",
                )
        else:
            if include_dates:
                self.res_df["date"] = self.dates
                self.res_df.to_parquet(
                    os.path.join(
                        savedpath,
                        "TestResults_{}_iteration_{}.parquet.gzip".format(
                            format_tousands(self.N_train), iteration
                        ),
                    ),
                    compression="gzip",
                )
            else:
                self.res_df.to_parquet(
                    os.path.join(
                        savedpath,
                        "TestResults_{}_iteration_{}.parquet.gzip".format(
                            format_tousands(self.N_train), iteration
                        ),
                    ),
                    compression="gzip",
                )

    def opt_trading_rate_disc_loads(self) -> Tuple[float, np.ndarray]:

        # 1 percent annualized discount rate (same rate of Ritter)
        rho = 1 - np.exp(-self.discount_rate / 260)

        # kappa is the risk aversion, CostMultiplier the parameter for trading cost
        num1 = self.kappa * (1 - rho) + self.CostMultiplier * rho
        num2 = np.sqrt(
            num1 ** 2 + 4 * self.kappa * self.CostMultiplier * (1 - rho) ** 2
        )
        den = 2 * (1 - rho)
        a = (-num1 + num2) / den

        OptRate = a / self.CostMultiplier
        DiscFactorLoads = self.f_param / (
            1 + self.f_speed * ((OptRate * self.CostMultiplier) / self.kappa)
        )

        return OptRate, DiscFactorLoads

    # PRIVATE METHODS
    def _find_nearest_return(self, value) -> float:
        array = np.asarray(self.returns_space.values)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    def _find_nearest_holding(self, value) -> Union[float or int]:
        array = np.asarray(self.holding_space.values)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    def _totalcost(self, shares_traded: Union[float or int]) -> Union[float or int]:

        Lambda = self.CostMultiplier * self.sigma ** 2
        quadratic_costs = 0.5 * (shares_traded ** 2) * Lambda

        return quadratic_costs

    def _getreward(
        self,
        currState: Tuple[Union[float or int], Union[float or int]],
        nextState: Tuple[Union[float or int], Union[float or int]],
        tag: str,
    ) -> dict:

        # Remember that a state is a tuple (price, holding)
        # currRet = currState[0]

        nextRet = nextState[0]
        currHolding = currState[1]
        nextHolding = nextState[1]

        shares_traded = nextHolding - currHolding
        GrossPNL = nextHolding * nextRet
        Risk = 0.5 * self.kappa * ((nextHolding ** 2) * (self.sigma ** 2))
        Cost = self._totalcost(shares_traded)
        NetPNL = GrossPNL - Cost
        Reward = GrossPNL - Risk - Cost

        Result = {
            "CurrHolding_{}".format(tag): currHolding,
            "NextHolding_{}".format(tag): nextHolding,
            "Action_{}".format(tag): shares_traded,
            "GrossPNL_{}".format(tag): GrossPNL,
            "NetPNL_{}".format(tag): NetPNL,
            "Risk_{}".format(tag): Risk,
            "Cost_{}".format(tag): Cost,
            "Reward_{}".format(tag): Reward,
        }
        return Result

    def _get_opt_reward(
        self,
        currOptState: Tuple[Union[float or int], Union[float or int]],
        nextOptState: Tuple[Union[float or int], Union[float or int]],
        tag: str,
    ) -> dict:

        # Remember that a state is a tuple (price, holding)
        # currRet = currOptState[0]
        nextRet = nextOptState[0]
        OptCurrHolding = currOptState[2]
        OptNextHolding = nextOptState[2]

        # Traded quantity between period
        OptNextAction = OptNextHolding - OptCurrHolding
        # Portfolio variation
        OptGrossPNL = OptNextHolding * nextRet  # currRet
        # Risk
        OptRisk = 0.5 * self.kappa * ((OptNextHolding) ** 2 * (self.sigma) ** 2)
        # Transaction costs
        OptCost = self._totalcost(OptNextAction)
        # Portfolio Variation including costs
        OptNetPNL = OptGrossPNL - OptCost
        # Compute reward
        OptReward = OptGrossPNL - OptRisk - OptCost

        # Store quantities
        Result = {
            "{}NextAction".format(tag): OptNextAction,
            "{}NextHolding".format(tag): OptNextHolding,
            "{}GrossPNL".format(tag): OptGrossPNL,
            "{}NetPNL".format(tag): OptNetPNL,
            "{}Risk".format(tag): OptRisk,
            "{}Cost".format(tag): OptCost,
            "{}Reward".format(tag): OptReward,
        }

        return Result


# REUCRRENT ENV
class RecurrentMarketEnv(gym.Env):
    """
    Custom Market Environment class that follows gym interface. Different agents as
    DQN and Q-learnign can operate within the same envirnoment since there is no way
    for them to affect the asset returns which are simulated in advance.
    ...

    Attributes
    ----------
    HalfLife: Union[int or list or np.ndarray]
        List of HalfLife of mean reversion when the simulated dynamic is driven by
        factors

    Startholding: Union[int or float]
        Initial portfolio holding, usually set at 0

    sigma: float
        Constant volatility for the simulated returns of the asset

    CostMultiplier: float
        Transaction cost parameter which regulates the market liquidity

    kappa: float
        Risk averion parameter

    N_train: int
        Length of simulated experiment

    f_param: Union[float or list or np.ndarray]
        List of factor loadings when the simulated dynamic is driven by
        factors

    f_speed: Union[float or list or np.ndarray]
        List of speed of mean reversion when the simulated dynamic is driven by
        factors

    returns: Union[list or np.ndarray]
        Array of simulated returns

    factors: Union[list or np.ndarray]
        Array of simulated factors when the simulated dynamic is driven by
        factors or lagged factors when it is not the case

    returns_tens: Union[list or np.ndarray]
        Tensor of simulated returns

    factors_tens: Union[list or np.ndarray]
        Tensor of simulated factors when the simulated dynamic is driven by
        factors or lagged factors when it is not the case

    action_limit: int = None
        Action space boundary used in DDPG implementation

    dates: pd.DatetimeIndex = None
        Series of datetime values if real values are used within the environment,
        otherwise it is just a serie of integer number of length N_train

    res_df: pd.Dataframe
        Dataframe which store results of relevant quantities

    Methods
    -------
    reset() -> Tuple[np.ndarray,np.ndarray]
        Get the initial state representation and their associated factors

    step(currState: Union[Tuple or np.ndarray],shares_traded: int,
             iteration: int, tag: str ='DQN') -> Tuple[np.ndarray,dict,np.ndarray]
        Make a step of the environment returning the next state representation,
        a dictionary with relevan measure to store and the factors associated to
        the next state representation

    discretereset() -> np.ndarray
        Discrete counterpart of reset method used for Q-learning

    discretestep(discretecurrState: Union[Tuple or np.ndarray],shares_traded: int,
                 iteration: int) -> Tuple[np.ndarray,np.ndarray]
        Discrete counterpart of step method used for Q-learning

    opt_reset() -> np.ndarray
        Counterpart of reset method used for the benchmark

    opt_step(currOptState: Tuple, OptRate: float,DiscFactorLoads: np.ndarray,
             iteration: int,tag: str = 'Opt') ->  Tuple[np.ndarray,dict]
        Counterpart of step method used for the benchmark

    mv_step(currOptState: Tuple, iteration: int,
            tag: str = 'MV') -> Tuple[np.ndarray,dict]:
        Counterpart of step method used for the Markovitz solution

    store_results(Result:dict, iteration: int)
        Store dictionary of current results to the DataFrame saved as attribute
        of the class

    save_outputs(self, savedpath, test=None, iteration=None, include_dates=False)
        Save the DataFrame saved as attribute of the class in a parquet format

    opt_trading_rate_disc_loads() -> Tuple[float,np.ndarray]
        Compute the trading rate and the discounted factor loading as for
        the optimal benchmark solution

    Private Methods
    -------

    _find_nearest_return(value) -> float
        Get the discretized counterpart of the return (value)

    _find_nearest_holding(value) -> Union[float or int]
        Get the discretized counterpart of the holding (value)

    _totalcost(shares_traded: Union[float or int]) -> Union[float or int]
        Compute transaction cost for the given trade

    _getreward(currState: Tuple[Union[float or int],Union[float or int]],
               nextState: Tuple[Union[float or int],Union[float or int]],
               tag: str) -> dict
        Compute Reward function for the current state and action for DQN or Q-learning
    _get_opt_reward(currState: Tuple[Union[float or int],Union[float or int]],
               nextState: Tuple[Union[float or int],Union[float or int]],
               tag: str) -> dict
    """

    def __init__(
        self,
        HalfLife: Union[int or list or np.ndarray],
        Startholding: Union[int or float],
        sigma: float,
        CostMultiplier: float,
        kappa: float,
        N_train: int,
        discount_rate: float,
        f_param: Union[float or list or np.ndarray],
        f_speed: Union[float or list or np.ndarray],
        returns: Union[list or np.ndarray],
        factors: Union[list or np.ndarray],
        returns_tens: Union[list or np.ndarray],
        factors_tens: Union[list or np.ndarray],
        action_limit: int = None,
        dates: pd.DatetimeIndex = None,
    ):

        # super(RecurrentMarketEnv, self).__init__()
        super().__init__()

        self.HalfLife = HalfLife
        self.Startholding = Startholding
        self.sigma = sigma
        self.CostMultiplier = CostMultiplier
        self.kappa = kappa
        self.N_train = N_train
        self.discount_rate = discount_rate
        self.f_param = f_param
        self.f_speed = f_speed
        self.returns = np.delete(returns, np.arange(returns_tens.shape[1] - 1))
        self.factors = np.delete(factors, np.arange(returns_tens.shape[1] - 1), axis=0)
        self.returns_tens = returns_tens
        self.factors_tens = factors_tens
        self.action_limit = action_limit

        colnames = ["returns"] + ["factor_" + str(hl) for hl in HalfLife]
        # self.colnames = colnames
        res_df = pd.DataFrame(
            np.concatenate(
                [np.array(self.returns).reshape(-1, 1), np.array(self.factors)], axis=1
            ),
            columns=colnames,
        )

        if dates is not None:
            self.dates = np.delete(dates, np.arange(returns_tens.shape[1] - 1))
        res_df = res_df.astype(np.float32)
        self.res_df = res_df

    def step(
        self,
        currState: Union[Tuple or np.ndarray],
        shares_traded: int,
        iteration: int,
        tag: str = "DQN",
    ):
        nextFactors = self.factors_tens[iteration + 1]
        nextRet = self.returns_tens[iteration + 1]
        if tag == "DDPG":
            shares_traded = shares_traded * self.action_limit

        nextHolding = currState[-1, 1] + shares_traded
        nextHolding = np.append(np.delete(currState[:, 1], 0), nextHolding).reshape(
            -1, 1
        )
        nextState = np.concatenate([nextRet, nextHolding], axis=-1)

        Result = self._getreward(currState, nextState, tag)

        return nextState, Result, nextFactors

    def reset(self):
        ret_shape = self.returns_tens[0].shape
        currState = np.concatenate(
            [self.returns_tens[0], np.zeros(ret_shape) * self.Startholding], axis=-1
        )
        currFactor = self.factors_tens[0]
        return currState, currFactor

    def discrete_step(
        self,
        discretecurrState: Union[Tuple or np.ndarray],
        shares_traded: int,
        iteration: int,
    ):
        discretenextRet = self._find_nearest_return(self.returns[iteration + 1])
        discretenextHolding = self._find_nearest_holding(
            discretecurrState[1] + shares_traded
        )
        discretenextState = np.array([discretenextRet, discretenextHolding])
        Result = self._getreward(discretecurrState, discretenextState, "Q")
        return discretenextState, Result

    def discrete_reset(self):
        discretecurrState = np.array(
            [
                self._find_nearest_return(self.returns[0]),
                self._find_nearest_holding(self.Startholding),
            ]
        )
        return discretecurrState

    def opt_reset(self):
        currOptState = np.array(
            [self.returns[0], self.factors[0], self.Startholding], dtype=object
        )
        return currOptState

    def opt_step(
        self,
        currOptState: Tuple,
        OptRate: float,
        DiscFactorLoads: np.ndarray,
        iteration: int,
        tag: str = "Opt",
    ) -> dict:

        # CurrReturns = currOptState[0]
        CurrFactors = currOptState[1]
        OptCurrHolding = currOptState[2]

        # Optimal traded quantity between period
        OptNextHolding = (1 - OptRate) * OptCurrHolding + OptRate * (
            1 / (self.kappa * (self.sigma) ** 2)
        ) * np.sum(DiscFactorLoads * CurrFactors)

        nextReturns = self.returns[iteration + 1]
        nextFactors = self.factors[iteration + 1]
        nextOptState = (nextReturns, nextFactors, OptNextHolding)

        OptResult = self._get_opt_reward(currOptState, nextOptState, tag)

        return nextOptState, OptResult

    def mv_step(self, currOptState: Tuple, iteration: int, tag: str = "MV") -> dict:

        # CurrReturns = currOptState[0]
        CurrFactors = currOptState[1]
        # OptCurrHolding = currOptState[2]

        # Traded quantity as for the Markovitz framework  (Mean-Variance framework)
        OptNextHolding = (1 / (self.kappa * (self.sigma) ** 2)) * np.sum(
            self.f_param * CurrFactors
        )

        nextReturns = self.returns[iteration + 1]
        nextFactors = self.factors[iteration + 1]
        nextOptState = (nextReturns, nextFactors, OptNextHolding)

        OptResult = self._get_opt_reward(currOptState, nextOptState, tag)

        return nextOptState, OptResult

    def store_results(self, Result: dict, iteration: int):

        if iteration == 0:
            for key in Result.keys():
                self.res_df[key] = 0.0
                self.res_df.at[iteration, key] = Result[key]
            self.res_df = self.res_df.astype(np.float32)
        else:
            for key in Result.keys():
                self.res_df.at[iteration, key] = Result[key]

    def save_outputs(self, savedpath, test=None, iteration=None, include_dates=False):

        if not test:
            if include_dates:
                self.res_df["date"] = self.dates
                self.res_df.to_parquet(
                    os.path.join(
                        savedpath,
                        "Results_{}.parquet.gzip".format(format_tousands(self.N_train)),
                    ),
                    compression="gzip",
                )
            else:
                self.res_df.to_parquet(
                    os.path.join(
                        savedpath,
                        "Results_{}.parquet.gzip".format(format_tousands(self.N_train)),
                    ),
                    compression="gzip",
                )
        else:
            if include_dates:
                self.res_df["date"] = self.dates
                self.res_df.to_parquet(
                    os.path.join(
                        savedpath,
                        "TestResults_{}_iteration_{}.parquet.gzip".format(
                            format_tousands(self.N_train), iteration
                        ),
                    ),
                    compression="gzip",
                )
            else:
                self.res_df.to_parquet(
                    os.path.join(
                        savedpath,
                        "TestResults_{}_iteration_{}.parquet.gzip".format(
                            format_tousands(self.N_train), iteration
                        ),
                    ),
                    compression="gzip",
                )

    def opt_trading_rate_disc_loads(self):

        # 1 percent annualized discount rate (same rate of Ritter)
        rho = 1 - np.exp(-self.discount_rate / 260)

        # kappa is the risk aversion, CostMultiplier the parameter for trading cost
        num1 = self.kappa * (1 - rho) + self.CostMultiplier * rho
        num2 = np.sqrt(
            num1 ** 2 + 4 * self.kappa * self.CostMultiplier * (1 - rho) ** 2
        )
        den = 2 * (1 - rho)
        a = (-num1 + num2) / den

        OptRate = a / self.CostMultiplier

        DiscFactorLoads = self.f_param / (
            1 + self.f_speed * ((OptRate * self.CostMultiplier) / self.kappa)
        )

        return OptRate, DiscFactorLoads

    # PRIVATE METHODS
    def _find_nearest_return(self, value):
        array = np.asarray(self.returns_space.values)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    def _find_nearest_holding(self, value):
        array = np.asarray(self.holding_space.values)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    def _totalcost(self, shares_traded: Union[float or int]) -> Union[float or int]:

        Lambda = self.CostMultiplier * self.sigma ** 2
        quadratic_costs = 0.5 * (shares_traded ** 2) * Lambda

        return quadratic_costs

    def _getreward(
        self,
        currState: Tuple[Union[float or int], Union[float or int]],
        nextState: Tuple[Union[float or int], Union[float or int]],
        tag: str,
    ) -> dict:

        # Remember that a state is a tuple (price, holding)
        # currRet = currState[-1, 0]
        nextRet = nextState[-1, 0]
        currHolding = currState[-1, 1]
        nextHolding = nextState[-1, 1]

        shares_traded = nextHolding - currHolding
        GrossPNL = nextHolding * nextRet
        Risk = 0.5 * self.kappa * ((nextHolding ** 2) * (self.sigma ** 2))
        Cost = self._totalcost(shares_traded)
        NetPNL = GrossPNL - Cost
        Reward = GrossPNL - Risk - Cost

        Result = {
            "CurrHolding_{}".format(tag): currHolding,
            "NextHolding_{}".format(tag): nextHolding,
            "Action_{}".format(tag): shares_traded,
            "GrossPNL_{}".format(tag): GrossPNL,
            "NetPNL_{}".format(tag): NetPNL,
            "Risk_{}".format(tag): Risk,
            "Cost_{}".format(tag): Cost,
            "Reward_{}".format(tag): Reward,
        }
        return Result

    def _get_opt_reward(
        self,
        currOptState: Tuple[Union[float or int], Union[float or int]],
        nextOptState: Tuple[Union[float or int], Union[float or int]],
        tag: str,
    ) -> dict:

        # Remember that a state is a tuple (price, holding)
        # currRet = currOptState[0]
        nextRet = nextOptState[0]
        OptCurrHolding = currOptState[2]
        OptNextHolding = nextOptState[2]

        # Traded quantity between period
        OptNextAction = OptNextHolding - OptCurrHolding
        # Portfolio variation
        OptGrossPNL = OptNextHolding * nextRet  # currRet
        # Risk
        OptRisk = 0.5 * self.kappa * ((OptNextHolding) ** 2 * (self.sigma) ** 2)
        # Transaction costs
        OptCost = self._totalcost(OptNextAction)
        # Portfolio Variation including costs
        OptNetPNL = OptGrossPNL - OptCost
        # Compute reward
        OptReward = OptGrossPNL - OptRisk - OptCost

        # Store quantities
        Result = {
            "{}NextAction".format(tag): OptNextAction,
            "{}NextHolding".format(tag): OptNextHolding,
            "{}GrossPNL".format(tag): OptGrossPNL,
            "{}NetPNL".format(tag): OptNetPNL,
            "{}Risk".format(tag): OptRisk,
            "{}Cost".format(tag): OptCost,
            "{}Reward".format(tag): OptReward,
        }

        return Result
