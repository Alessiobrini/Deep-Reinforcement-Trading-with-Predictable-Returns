from typing import Union, Tuple
import pandas as pd
import numpy as np
import pdb, os
import gym
import gin
from utils.math_tools import unscale_action
from utils.common import format_tousands


@gin.configurable()
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
        factors: Union[list or np.ndarray] = None,
        action_limit: int = None,
        inp_type: str = "ret",
        cost_type: str = 'quadratic',
        cm1: float = 2.89E-4,
        cm2: float = 7.91E-4,
        dates: pd.DatetimeIndex = None,
    ):

        # super(MarketEnv, self).__init__()
        super().__init__()

        self.HalfLife = HalfLife
        self.Startholding = Startholding
        self.sigma = sigma
        self.CostMultiplier = CostMultiplier
        self.cm1 = cm1
        self.cm2 = cm2
        self.kappa = kappa
        self.N_train = N_train
        self.discount_rate = discount_rate
        self.f_param = f_param
        self.f_speed = f_speed
        self.returns = returns
        self.factors = factors
        self.action_limit = action_limit
        self.inp_type = inp_type
        self.cost_type = cost_type

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
        if self.inp_type == "ret" or self.inp_type == "alpha":
            currState = np.array([self.returns[0], self.Startholding])
            currFactor = self.factors[0]
            return currState, currFactor
        elif self.inp_type == "f" or self.inp_type == "alpha_f":
            currState = np.append(self.factors[0], self.Startholding)
            currRet = self.returns[0]
            return currState, currRet

    def step(
        self,
        currState: Union[Tuple or np.ndarray],
        shares_traded: int,
        iteration: int,
        tag: str = "DQN",
    ) -> Tuple[np.ndarray, dict, np.ndarray]:

        nextFactors = self.factors[iteration + 1]
        nextRet = self.returns[iteration + 1]

        nextHolding = currState[-1] + shares_traded
        if self.inp_type == "ret":
            nextState = np.array([nextRet, nextHolding], dtype=np.float32)
        elif self.inp_type == "f":
            nextState = np.append(nextFactors, nextHolding)

        Result = self._getreward(currState, nextState, iteration, tag)

        return nextState, Result, nextFactors

    def MV_res_step(
        self,
        currState: Union[Tuple or np.ndarray],
        shares_traded: int,
        iteration: int,
        tag: str = "DQN",
    ) -> Tuple[np.ndarray, dict, np.ndarray]:

        CurrHolding = currState[-1]
        if self.inp_type == 'alpha':
            curr_alpha = currState[0]
            # Traded quantity as for the Markovitz framework  (Mean-Variance framework)
            OptNextHolding = (1 / (self.kappa * (self.sigma) ** 2)) * curr_alpha
        else:
            CurrFactors = self.factors[iteration]
            # Traded quantity as for the Markovitz framework  (Mean-Variance framework)
            OptNextHolding = (1 / (self.kappa * (self.sigma) ** 2)) * np.sum(
                self.f_param * CurrFactors
            )
            nextFactors = self.factors[iteration + 1]
        # Compute optimal markovitz action
        MV_action = OptNextHolding - CurrHolding

        nextRet = self.returns[iteration + 1]
        nextHolding = currState[-1] + MV_action * (1 - shares_traded)
        if self.inp_type == "ret" or self.inp_type == "alpha":
            nextState = np.array([nextRet, nextHolding], dtype=np.float32)
        elif self.inp_type == "f" or self.inp_type == "alpha_f":
            nextState = np.append(nextFactors, nextHolding)
        
        Result = self._getreward(
            currState, nextState, iteration, tag, res_action=shares_traded
        )

        return nextState, Result

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
        Result = self._getreward(discretecurrState, discretenextState, iteration, "Q")
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

        OptCurrHolding = currOptState[-1]
        CurrFactors = currOptState[1]
        # Optimal traded quantity between period
        OptNextHolding = (1 - OptRate) * OptCurrHolding + OptRate * (
            1 / (self.kappa * (self.sigma) ** 2)
        ) * np.sum(DiscFactorLoads * CurrFactors)
        nextFactors = self.factors[iteration + 1]
        nextReturns = self.returns[iteration + 1]
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
        if self.cost_type == 'quadratic':
            Lambda = self.CostMultiplier * self.sigma ** 2
            cost = 0.5 * (shares_traded ** 2) * Lambda
        elif self.cost_type == 'nondiff':
            #Kyle-Obizhaeva formulation
            # p, v = 40, 1E+6 
            Lambda = self.cm2 * self.sigma ** 2
            quadcost = 0.5 * (shares_traded ** 2) * Lambda
            cost = self.cm1*np.abs(shares_traded) + quadcost

        return cost

    def _getreward(
        self,
        currState: Tuple[Union[float or int], Union[float or int]],
        nextState: Tuple[Union[float or int], Union[float or int]],
        iteration: int,
        tag: str,
        res_action: float = None,
    ) -> dict:

        # Remember that a state is a tuple (price, holding)
        # currRet = currState[0]

        nextRet = self.returns[iteration + 1]
        currHolding = currState[-1]
        nextHolding = nextState[-1]

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
        if res_action:
            Result["ResAction_{}".format(tag)] = res_action
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
        OptCurrHolding = currOptState[-1]
        OptNextHolding = nextOptState[-1]

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
