from typing import Union, Tuple
import pandas as pd
import numpy as np
import pdb, os
import gym
import gin
import sys
import torch
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
        HalfLife: Union[int , list , np.ndarray],
        Startholding: Union[int , float],
        sigma: float,
        CostMultiplier: float,
        kappa: float,
        N_train: int,
        discount_rate: float,
        f_param: Union[float , list , np.ndarray],
        f_speed: Union[float , list , np.ndarray],
        returns: Union[list , np.ndarray],
        factors: Union[list , np.ndarray] = None,
        action_limit: int = None,
        inp_type: str = "ret",
        cost_type: str = 'quadratic',
        cm1: float = 2.89E-4,
        cm2: float = 7.91E-4,
        reward_type: str = 'mean_var',
        dates: pd.DatetimeIndex = None,
        cash : int = None,
        multiasset: bool = False,
        corr: int = None,
        inputs: list = None,
        mv_penalty : bool = False,
        mv_penalty_coef : float = None,
        daily_volume : float = None,
        daily_price : float = None,
        time_dependent : bool = False
    ):

        # super(MarketEnv, self).__init__()
        super().__init__()

        self.HalfLife = HalfLife
        self.Startholding = Startholding
        self.sigma = sigma
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
        self.reward_type = reward_type
        self.multiasset = multiasset
        self.corr = corr
        self.cash = cash
        self.inputs = inputs
        self.mv_penalty = mv_penalty
        self.mv_penalty_coef = mv_penalty_coef
        self.daily_volume = daily_volume
        self.daily_price = daily_price
        self.time_dependent = time_dependent

        
        if cost_type == 'nondiff':
            self.CostMultiplier = self.cm2/(0.01*self.daily_price*self.daily_volume * self.sigma**2)
        elif cost_type == 'quadratic':
            self.CostMultiplier = CostMultiplier
        
        if multiasset:
            colnames = (["returns" + str(hl) for hl in HalfLife] + 
            ["factor_" + str(h) for hl in HalfLife for h in hl])
            
            res_df = pd.DataFrame(
                np.concatenate(
                    [np.array(self.returns), np.array(self.factors)], axis=1
                ),
                columns=colnames,
            )
            self.n_assets = len(HalfLife)
            self.n_factors = len(HalfLife[0])
            
            # Initialize all the names for the columns
            cols=pd.Series(res_df.columns)
            for dup in cols[cols.duplicated()].unique(): 
                cols[cols[cols == dup].index.values.tolist()] = [dup + '.' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
            res_df.columns=cols
            # create names of the new columns
            names1 = ['CurrHolding_PPO', 'NextHolding_PPO', 'Action_PPO', 'OptNextAction', 'OptNextHolding', 'MVNextHolding']
            names1 = [n+'_{}'.format(i) for n in names1 for i in range(self.n_assets)]
            names2 = ['GrossPNL_PPO', 'NetPNL_PPO', 'Risk_PPO', 'Cost_PPO', 
             'Reward_PPO', 'TradedAmount_PPO', 'Cash_PPO', 'Wealth_PPO',
             'OptGrossPNL', 'OptNetPNL', 'OptRisk', 'OptCost', 'OptReward', 
             'OptTradedAmount', 'OptCash', 'OptWealth', 'MVReward', 'MVWealth']
            names = names1 + names2
            res_df = res_df.reindex(columns=list(res_df.columns)+names)
            
            self.currholding_rl, self.nextholding_rl, self.action_rl, self.optaction, self.optholding = [],[],[],[],[]
            (self.grosspnl_rl, self.netpnl_rl, self.risk_rl, self.cost_rl, self.reward_rl, 
             self.tradedamount_rl, self.cash_rl, self.wealth_rl) = [],[],[],[],[],[],[],[]
            (self.grosspnl_opt, self.netpnl_opt, self.risk_opt, self.cost_opt, 
            self.reward_opt, self.tradedamount_opt, self.cash_opt, self.wealth_opt) = [],[],[],[],[],[],[],[]
            self.mvholding, self.mvreward, self.mvwealth = [],[],[]

            if self.cash:
    
                self.holding_ts = [[self.Startholding]*self.n_assets]
                self.cash_ts = [cash]
                self.traded_amount = 0.0
                self.costs = 0.0
            
            if isinstance(self.corr,float):
                cov_matrix = np.eye(self.n_assets,self.n_assets) * self.sigma**2
                cov_matrix[cov_matrix==0] = self.corr * self.sigma**2 
            elif isinstance(self.corr,list):
                cov_matrix = np.zeros((self.n_assets,self.n_assets))
                cov_matrix[np.triu_indices(cov_matrix.shape[0], k = 1)] = np.array(self.corr) * self.sigma**2
                cov_matrix = cov_matrix + cov_matrix.T
                cov_matrix[np.where(cov_matrix==0)] = self.sigma**2
            if np.allclose(cov_matrix, cov_matrix.T):
                self.cov_matrix = cov_matrix
            else:
                print('Created a Covariance matrix which is not symmetric!')
                sys.exit()

        else:
            
            colnames = ["returns"] + ["factor_" + str(hl) for hl in HalfLife]

            res_df = pd.DataFrame(
                np.concatenate(
                    [np.array(self.returns).reshape(-1, 1), np.array(self.factors)], axis=1
                ),
                columns=colnames,
            )

            self.n_assets = 1
            self.n_factors = len(HalfLife)

            if cash:
                self.cash = cash
                self.holding_ts = [self.Startholding]
                self.cash_ts = [cash]
                self.traded_amount = 0.0
                self.costs = 0.0

        self.dates = dates
        res_df = res_df.astype(np.float32)
        self.res_df = res_df
        

    def get_state_dim(self):
        state = self.reset()
        return state.shape

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        
        if self.inp_type == "ret" or self.inp_type == "alpha":
            if self.time_dependent:
                currState = np.array([self.returns[0],len(self.returns) - 2, self.Startholding])
            else:
                currState = np.array([self.returns[0], self.Startholding])
            return currState
        elif self.inp_type == "f" or self.inp_type == "alpha_f":
            if self.time_dependent:
                currState = np.append(self.factors[0],[len(self.returns) - 2, self.Startholding])
            else:
                currState = np.append(self.factors[0], self.Startholding)
            return currState

    def step(
        self,
        currState: Union[Tuple , np.ndarray],
        shares_traded: int,
        iteration: int,
        tag: str = "DQN",
    ) -> Tuple[np.ndarray, dict, np.ndarray]:
        # This is the only environment in which we can run tests with alpha decay inputs in a model free setting
        # It is not implemented in the enviroment with cash because we focused on Res RL setting

        nextFactors = self.factors[iteration + 1]
        nextRet = self.returns[iteration + 1]

        nextHolding = currState[-1] + shares_traded
        
        if self.inp_type == "ret" or self.inp_type == "alpha":
            if self.time_dependent:
                nextState = np.array([nextRet,len(self.returns) - 2 - (iteration+1), nextHolding], dtype=np.float32)
            else:
                nextState = np.array([nextRet, nextHolding], dtype=np.float32)
        elif self.inp_type == "f" or self.inp_type == "alpha_f":
            if self.time_dependent:
                nextState = np.append(nextFactors,[len(self.returns) - 2 - (iteration+1), nextHolding])
            else:
                nextState = np.append(nextFactors, nextHolding)
        
        Result = self._getreward(currState, nextState, iteration, tag)

        return nextState, Result, nextFactors

    def MV_res_step(
        self,
        currState: Union[Tuple, np.ndarray],
        shares_traded: int,
        iteration: int,
        output_action: bool = False,
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
        if output_action:
            return MV_action

        nextRet = self.returns[iteration + 1]
        nextHolding = currState[-1] + MV_action * (1 - shares_traded)
        if self.inp_type == "ret" or self.inp_type == "alpha":
            if self.time_dependent:
                nextState = np.array([nextRet,len(self.returns) - 2 - (iteration+1), nextHolding], dtype=np.float32)
            else:
                nextState = np.array([nextRet, nextHolding], dtype=np.float32)
        elif self.inp_type == "f" or self.inp_type == "alpha_f":
            if self.time_dependent:
                nextState = np.append(nextFactors,[len(self.returns) - 2 - (iteration+1), nextHolding])
            else:
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
        discretecurrState: Union[Tuple , np.ndarray],
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
                if isinstance(Result[key],list) or isinstance(Result[key],np.ndarray):
                    for i,k in enumerate(Result[key]):
                        self.res_df[key+'_{}'.format(i)] = 0.0
                        self.res_df.at[iteration, key+'_{}'.format(i)] = k
                else:
                    self.res_df[key] = 0.0
                    if isinstance(Result[key],torch.Tensor):
                        self.res_df.at[iteration, key] = Result[key].item()
                    else:
                        self.res_df.at[iteration, key] = Result[key]
            self.res_df = self.res_df.astype(np.float32)
        else:

            for key in Result.keys():
                if isinstance(Result[key],list) or isinstance(Result[key],np.ndarray):
                    name = key
                    suffixes = ['_{}'.format(i) for i in range(len(Result[key]))] 
                    names = [name + s for s in suffixes]
                    self.res_df.loc[iteration,names] = Result[key]
                else:
                    if isinstance(Result[key],torch.Tensor):
                        self.res_df.at[iteration, key] = Result[key].item()
                    else:
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

    def _find_nearest_holding(self, value) -> Union[float , int]:
        array = np.asarray(self.holding_space.values)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    def _totalcost(self, shares_traded: Union[float , int]) -> Union[float , int]:
        if self.cost_type == 'quadratic':
            Lambda = self.CostMultiplier * self.sigma ** 2
            cost = 0.5 * (shares_traded ** 2) * Lambda
        elif self.cost_type == 'nondiff':
            #Kyle-Obizhaeva formulation
            p, v = self.daily_price, self.daily_volume
            quadcost =  shares_traded**2 / (0.01*p*v)
            # Lambda = self.cm2 * self.sigma ** 2
            # quadcost = 0.5 * (shares_traded ** 2) * Lambda
            cost = self.cm1*np.abs(shares_traded) + self.cm2 * quadcost

        return cost

    def _getreward(
        self,
        currState: Tuple[Union[float , int], Union[float , int]],
        nextState: Tuple[Union[float , int], Union[float , int]],
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
        Cost = self._totalcost(shares_traded)
        NetPNL = GrossPNL - Cost
        if self.reward_type == 'mean_var':
            Risk = 0.5 * self.kappa * ((nextHolding ** 2) * (self.sigma ** 2))
            # Reward = NetPNL - 0.5 * self.kappa * NetPNL**2  #
            Reward = GrossPNL - Risk - Cost 
        elif self.reward_type == 'cara':
            Reward = (1 - np.e**(-self.kappa*NetPNL))/self.kappa

        if self.mv_penalty:
            # if self.inp_type == 'alpha':
            #     next_alpha = nextState[0]
            #     # Traded quantity as for the Markovitz framework  (Mean-Variance framework)
            #     MVNextHolding = (1 / (self.kappa * (self.sigma) ** 2)) * next_alpha
            # else:
            #     NextFactors = self.factors[iteration + 1]
            #     # Traded quantity as for the Markovitz framework  (Mean-Variance framework)
            #     MVNextHolding = (1 / (self.kappa * (self.sigma) ** 2)) * np.sum(
            #         self.f_param * NextFactors
            #     )
            
            # penalty = self.mv_penalty_coef * (MVNextHolding - nextHolding)**2
            penalty = self.mv_penalty_coef * (currHolding - nextHolding)**2

            Reward -= penalty

        Result = {
            "CurrHolding_{}".format(tag): currHolding,
            "NextHolding_{}".format(tag): nextHolding,
            "Action_{}".format(tag): shares_traded,
            "GrossPNL_{}".format(tag): GrossPNL,
            "NetPNL_{}".format(tag): NetPNL,
            "Cost_{}".format(tag): Cost,
            "Reward_{}".format(tag): Reward,
        }
        if res_action:
            Result["ResAction_{}".format(tag)] = res_action
        
        if self.reward_type == 'mean_var': 
            Result["Risk_{}".format(tag)] = Risk

        if self.mv_penalty:
            Result["Penalty_{}".format(tag)] = penalty

        return Result

    def _get_opt_reward(
        self,
        currOptState: Tuple[Union[float , int], Union[float , int]],
        nextOptState: Tuple[Union[float , int], Union[float , int]],
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
        OptRisk = 0.5 * self.kappa * ((OptNextHolding ** 2) * (self.sigma ** 2))
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


    def _get_inputs(self,reset,iteration=None):

        # could be rewritten better https://stackoverflow.com/questions/25211924/check-every-condition-in-python-if-else-even-if-one-evaluates-to-true
        input_list = []
        input_type= self.inputs
        if reset:
            if self.inp_type == "ret" or self.inp_type == "alpha":
                input_list.extend(list(self.returns[0]))
            elif self.inp_type == "f" or self.inp_type == "alpha_f":
                input_list.extend(list(self.factors[0]))

            if "sigma" in input_type:
                input_list.append(self.sigma**2)
            if "corr" in input_type:
                if isinstance(self.corr,float):
                    input_list.append(self.corr)
                elif isinstance(self.corr,list):
                    input_list.extend(self.corr)
            if "holding" in input_type:
                input_list.extend([self.Startholding]*self.n_assets)
            # entire arrays
            if "cash" in input_type:
                input_list.append(self.cash)
        else:
            if self.inp_type == "ret" or self.inp_type == "alpha":
                input_list.extend(list(self.returns[iteration+1]))
            elif self.inp_type == "f" or self.inp_type == "alpha_f":
                input_list.extend(list(self.factors[iteration+1]))

            if "sigma" in input_type:
                input_list.append(self.sigma**2)
            if "corr" in input_type:
                if isinstance(self.corr,float):
                    input_list.append(self.corr)
                elif isinstance(self.corr,list):
                    input_list.extend(self.corr)
            if "holding" in input_type:
                input_list.extend(self.holding_ts[iteration+1])
            # entire arrays
            if "cash" in input_type:
                input_list.append(self.cash_ts[iteration+1])

        return input_list


@gin.configurable()
class CashMarketEnv(MarketEnv):

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.inp_type == "ret" or self.inp_type == "alpha":
            currState = np.array([self.returns[0], self.Startholding, self.cash])
            return currState
        elif self.inp_type == "f" or self.inp_type == "alpha_f":
            currState = np.append(self.factors[0], [self.Startholding, self.cash])
            return currState


    def step(
        self,
        currState: Union[Tuple , np.ndarray],
        action: float,
        iteration: int,
        tag: str = "DQN",
    ) -> Tuple[np.ndarray, dict, np.ndarray]:
        
        nextFactors = self.factors[iteration + 1]
        nextRet = self.returns[iteration + 1]
        
        # buy/sell here
        if action > 0:
            shares_traded = self._buy(index=iteration, cash=self.cash_ts[iteration], action=action)
        elif action < 0:
            shares_traded = self._sell(index=iteration, holding=self.holding_ts[iteration], action=action)
        else:
            shares_traded = 0.0
            self.costs, self.traded_amount = 0.0, 0.0

        # update rules
        nextHolding = (1+nextRet) * self.holding_ts[iteration] + shares_traded
        self.holding_ts.append(nextHolding)
        nextCash = self.cash_ts[iteration] + self.traded_amount
        self.cash_ts.append(nextCash)

        if self.inp_type == "ret":
            nextState = np.array([nextRet, nextHolding, nextCash], dtype=np.float32)
        elif self.inp_type == "f":
            nextState = np.append(nextFactors, [nextHolding, nextCash])

        Result = self._getreward(currState, nextState, iteration, tag)

        return nextState, Result, nextFactors

    def MV_res_step(
        self,
        currState: Union[Tuple , np.ndarray],
        shares_traded: int,
        iteration: int,
        tag: str = "DQN",
    ) -> Tuple[np.ndarray, dict, np.ndarray]:

        CurrHolding = self.holding_ts[iteration]
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

        nextHolding = (1+nextRet) * CurrHolding + MV_action * (1 - shares_traded)
        self.holding_ts.append(nextHolding)
        nextCash = self.cash_ts[iteration] + self.traded_amount
        self.cash_ts.append(nextCash)

        if self.inp_type == "ret" or self.inp_type == "alpha":
            nextState = np.array([nextRet, nextHolding,nextCash], dtype=np.float32)
        elif self.inp_type == "f" or self.inp_type == "alpha_f":
            nextState = np.append(nextFactors, [nextHolding,nextCash])

        Result = self._getreward(
            iteration, tag, res_action=shares_traded
        )

        return nextState, Result


    def opt_reset(self) -> np.ndarray:
        currOptState = [self.factors[0], self.Startholding, self.cash]
        return currOptState

    def opt_step(
        self,
        currOptState: Tuple,
        OptRate: float,
        DiscFactorLoads: np.ndarray,
        iteration: int,
        tag: str = "Opt",
    ) -> Tuple[np.ndarray, dict]:

        CurrFactors = currOptState[0]
        OptCurrHolding = currOptState[1]
        # Optimal traded quantity between period
        OptNextHolding = (1 - OptRate) * OptCurrHolding + OptRate * (
            1 / (self.kappa * (self.sigma) ** 2)
        ) * np.sum(DiscFactorLoads * CurrFactors)

        action = OptNextHolding - OptCurrHolding

        # buy/sell here
        OptNextHolding = self._opt_trade(
            index=iteration, state=currOptState, action=action
        )

        nextCash = currOptState[-1] + self.traded_amount
        nextReturn = self.returns[iteration + 1]
        nextFactors = self.factors[iteration + 1]
        nextOptState = [nextFactors, OptNextHolding, nextCash]

        OptResult = self._get_opt_reward(
            currOptState, nextOptState, nextReturn, iteration, tag
        )

        return nextOptState, OptResult


    def _getreward(
        self,
        iteration: int,
        tag: str,
        res_action: float = None,
    ) -> dict:

        nextRet = self.returns[iteration+1]
        currHolding = self.holding_ts[iteration]
        nextHolding = self.holding_ts[iteration+1] 
        nextCash = self.cash_ts[iteration+1] 

        shares_traded = nextHolding - currHolding
        NetPNL = nextHolding * nextRet - self.costs
        Risk = 0.5 * self.kappa * ((nextHolding ** 2) * (self.sigma ** 2))
        Reward = NetPNL - Risk
        nextWealth = nextHolding + nextCash

        Result = {
            "CurrHolding_{}".format(tag): currHolding,
            "NextHolding_{}".format(tag): nextHolding,
            "Action_{}".format(tag): shares_traded,
            "GrossPNL_{}".format(tag): NetPNL + self.costs,
            "NetPNL_{}".format(tag): NetPNL,
            "Risk_{}".format(tag): Risk,
            "Cost_{}".format(tag): self.costs,
            "Reward_{}".format(tag): Reward,
            "TradedAmount_{}".format(tag): self.traded_amount,
            "Cash_{}".format(tag): nextCash,
            "Wealth_{}".format(tag): nextWealth,
        }

        self.costs, self.traded_amount = 0.0, 0.0
        
        if isinstance(res_action, float):
            Result["ResAction_{}".format(tag)] = res_action
        return Result

    def _sell(self, index: int, holding: np.ndarray, action: float):

        currholding = holding

        if currholding > 0.0:
            # Sell only if current asset is > 0
            shares_traded = min(abs(action), currholding)
            self.costs = self._totalcost(shares_traded)
            self.traded_amount = shares_traded - self.costs
            return -shares_traded

        else:
            shares_traded = 0.0

            self.costs, self.traded_amount = 0.0, 0.0


        return -shares_traded

    def _buy(self, index: int, cash: np.ndarray, action: float):

        max_tradable_amount = cash
        shares_traded = min(max_tradable_amount, action)

        self.costs = self._totalcost(shares_traded)
        self.traded_amount = -shares_traded - self.costs


        return shares_traded

    def _get_opt_reward(
        self,
        currOptState: Tuple[Union[float , int], Union[float , int]],
        nextOptState: Tuple[Union[float , int], Union[float , int]],
        nextReturn: float,
        iteration: int,
        tag: str,
    ) -> dict:

        OptCurrHolding = currOptState[1]
        OptNextHolding = nextOptState[1]
        CurrOptcash = currOptState[-1]
        NextOptcash = nextOptState[-1]

        # Traded quantity between period
        OptNextAction = OptNextHolding - OptCurrHolding
        # Portfolio variation
        OptNetPNL = OptNextHolding * nextReturn - self.costs
        # Risk
        OptRisk = 0.5 * self.kappa * ((OptNextHolding) ** 2 * (self.sigma) ** 2)
        # Compute reward
        OptReward = OptNetPNL - OptRisk
        

        nextWealth = OptNextHolding + NextOptcash #self.prices[iteration + 1] * 

        # Store quantities
        Result = {
            "{}NextAction".format(tag): OptNextAction,
            "{}NextHolding".format(tag): OptNextHolding,
            "{}GrossPNL".format(tag): OptNetPNL + self.costs,
            "{}NetPNL".format(tag): OptNetPNL,
            "{}Risk".format(tag): OptRisk,
            "{}Cost".format(tag): self.costs,
            "{}Reward".format(tag): OptReward,
            "{}TradedAmount".format(tag): self.traded_amount,
            "{}Cash".format(tag): NextOptcash,
            "{}Wealth".format(tag): nextWealth,
        }

        self.costs, self.traded_amount = 0.0, 0.0

        return Result

    def _opt_trade(self, index: int, state: np.ndarray, action: float):

        if action > 0.0:

            max_tradable_amount = state[-1]
            shares_traded = min(max_tradable_amount, action)

            self.costs = self._totalcost(shares_traded)
            self.traded_amount = -shares_traded - self.costs

            return shares_traded + state[1]* (1 + self.returns[index+1])

        elif action < 0.0:

            currholding = state[1]

            if currholding > 0.0:
                # Sell only if current asset is > 0
                shares_traded = min(abs(action), currholding)

                self.costs = self._totalcost(shares_traded)
                self.traded_amount = shares_traded - self.costs

                return -shares_traded + state[1]* (1 + self.returns[index+1])

            else:
                shares_traded = 0.0
                self.costs, self.traded_amount = 0.0, 0.0

                return shares_traded + state[1]* (1 + self.returns[index+1])

        else:
            shares_traded = 0.0
            self.costs, self.traded_amount = 0.0, 0.0

            return shares_traded + state[1]* (1 + self.returns[index+1])



@gin.configurable()
class ShortCashMarketEnv(CashMarketEnv):

    def _sell(self, index: int, holding: np.ndarray, action: float):

        shares_traded = action
        self.costs = self._totalcost(shares_traded)
        # absolute quantity because now you can sell short and shares_traded can be negative
        self.traded_amount = np.abs(shares_traded) - self.costs

        return shares_traded

    def _buy(self, index: int, cash: np.ndarray, action: float):

        max_tradable_amount = cash 
        shares_traded = min(max_tradable_amount, action)

        self.costs = self._totalcost(shares_traded)
        self.traded_amount = -shares_traded- self.costs

        return shares_traded

    def _opt_trade(self, index: int, state: np.ndarray, action: float):

        if action > 0.0:
            max_tradable_amount = state[-1] 

            shares_traded = min(max_tradable_amount, action)

            self.costs = self._totalcost(shares_traded)
            self.traded_amount = -shares_traded - self.costs

            return shares_traded + state[1]* (1 + self.returns[index+1])

        elif action < 0.0:
            # one could insert a stop to consider cost of short selling
            shares_traded = -abs(action)

            self.costs = self._totalcost(shares_traded)
            self.traded_amount = np.abs(shares_traded) - self.costs
            
            return shares_traded + state[1]* (1 + self.returns[index+1])
        else:
            shares_traded = 0.0
            self.costs, self.traded_amount = 0.0, 0.0

            return shares_traded + state[1]* (1 + self.returns[index+1])

@gin.configurable()
class MultiAssetCashMarketEnv(CashMarketEnv):

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        input_list = self._get_inputs(reset=True)
        currState = np.array(input_list).reshape(-1,)  # long vector
        return currState


    def step(
        self,
        currState: Union[Tuple , np.ndarray],
        action: float,
        iteration: int,
        tag: str = "DQN",
    ) -> Tuple[np.ndarray, dict, np.ndarray]:
        # TODO Adapt to multi asset. I adapted only Mv_res step
        nextFactors = self.factors[iteration + 1]
        nextRet = self.returns[iteration + 1]
        
        # buy/sell here
        if action > 0:
            shares_traded = self._buy(index=iteration, cash=self.cash_ts[iteration], action=action)
        elif action < 0:
            shares_traded = self._sell(index=iteration, holding=self.holding_ts[iteration], action=action)
        else:
            shares_traded = 0.0
            self.costs, self.traded_amount = 0.0, 0.0

        # update rules
        nextHolding = (1+nextRet) * self.holding_ts[iteration] + shares_traded
        self.holding_ts.append(nextHolding)
        nextCash = self.cash_ts[iteration] + self.traded_amount
        self.cash_ts.append(nextCash)

        if self.inp_type == "ret":
            nextState = np.array([nextRet, nextHolding, nextCash], dtype=np.float32)
        elif self.inp_type == "f":
            nextState = np.append(nextFactors, [nextHolding, nextCash])

        Result = self._getreward(currState, nextState, iteration, tag)

        return nextState, Result, nextFactors

    def MV_res_step(
        self,
        currState: Union[Tuple , np.ndarray],
        shares_traded: int,
        iteration: int,
        tag: str = "DQN",
    ) -> Tuple[np.ndarray, dict, np.ndarray]:
        
        CurrHolding = np.array(self.holding_ts[iteration])
        if self.inp_type == 'alpha':
            curr_alpha = np.array(currState[:self.n_assets])
            # Traded quantity as for the Markovitz framework  (Mean-Variance framework)
            OptNextHolding = np.dot(np.linalg.inv(self.cov_matrix** self.kappa), curr_alpha)
        else:
            CurrFactors = self.factors[iteration].reshape(self.n_assets,self.n_factors)
            # Traded quantity as for the Markovitz framework  (Mean-Variance framework)
            OptNextHolding = np.dot(np.linalg.inv(self.cov_matrix* self.kappa), np.dot(
                CurrFactors,self.f_param[0]
            )) 
            nextFactors = self.factors[iteration + 1]
        # Compute optimal markovitz action
        MV_action = OptNextHolding - CurrHolding
        MV_res_action = MV_action * (1-shares_traded)

        # buy/sell here
        res_shares_traded = []
        for i,a in enumerate(MV_res_action):
            if a > 0:
                trade = self._buy(
                    index=iteration, cash=currState[-1], action=a
                )
            elif a < 0:
                trade = self._sell(
                    index=iteration, holding=currState[-1-self.n_assets+i], action=a
                )
            else:
                trade = 0.0
                self.costs, self.traded_amount = 0.0, 0.0
            res_shares_traded.append(trade)
        res_shares_traded = np.array(res_shares_traded)
        

        nextRet = self.returns[iteration + 1]

        nextHolding = (1+nextRet) * CurrHolding + res_shares_traded
        self.holding_ts.append(nextHolding)
        nextCash = self.cash_ts[iteration] + self.traded_amount
        self.cash_ts.append(nextCash)


        input_list = self._get_inputs(reset=False,iteration=iteration)
        nextState = np.array(input_list).reshape(-1,)  # long vector
            
        Result = self._getreward(
            iteration, tag, res_action=res_shares_traded
        )

        return nextState, Result


    def opt_reset(self) -> np.ndarray:
        currOptState = list(self.factors[0]) + [self.Startholding]*self.n_assets + [self.cash]
        return currOptState

    def opt_step(
        self,
        currOptState: Tuple,
        OptRate: float,
        DiscFactorLoads: np.ndarray,
        iteration: int,
        tag: str = "Opt",
    ) -> Tuple[np.ndarray, dict]:
        
        CurrFactors = self.factors[iteration].reshape(self.n_assets,self.n_factors)
        OptCurrHolding = np.array(currOptState[-1-self.n_assets:-1])
        # Optimal traded quantity between period
        
        DiscFactors = CurrFactors/ (1+self.f_speed * ((OptRate * self.CostMultiplier) / self.kappa))
        OptNextHolding = np.dot(np.linalg.inv(self.cov_matrix* self.kappa), np.dot(
            DiscFactors,self.f_param[0]
        ))
        
        action = OptNextHolding - OptCurrHolding

        # buy/sell here
        OptTrades = []
        for i,a in enumerate(action):
            opt_t = self._opt_trade(
                index=iteration, holding=currOptState[-1-self.n_assets+i], cash=currOptState[-1], action=a
            )
            OptTrades.append(opt_t)
        OptNextHolding = np.array(OptTrades)  + OptCurrHolding * (1 + self.returns[iteration+1])
        
        nextCash = currOptState[-1] + self.traded_amount
        nextReturn = self.returns[iteration + 1]
        nextFactors = self.factors[iteration + 1]
        nextOptState = list(nextFactors) + list(OptNextHolding) + [nextCash]
        
        OptResult = self._get_opt_reward(
            currOptState, nextOptState, nextReturn, iteration, tag
        )
        
        return nextOptState, OptResult

    def mv_step(
        self, currOptState: Tuple, iteration: int, tag: str = "MV"
    ) -> Tuple[np.ndarray, dict]:

        CurrFactors = self.factors[iteration].reshape(self.n_assets,self.n_factors)
        OptCurrHolding = np.array(currOptState[-1-self.n_assets:-1])

        # Traded quantity as for the Markovitz framework  (Mean-Variance framework)
        OptNextHolding = np.dot(np.linalg.inv(self.cov_matrix* self.kappa), np.dot(
            CurrFactors,self.f_param[0]
        ))

        action = OptNextHolding - OptCurrHolding

        # buy/sell here
        OptTrades = []
        for i,a in enumerate(action):
            opt_t = self._opt_trade(
                index=iteration, holding=currOptState[-1-self.n_assets+i], cash=currOptState[-1], action=a
            )
            OptTrades.append(opt_t)
        OptNextHolding = np.array(OptTrades)  + OptCurrHolding * (1 + self.returns[iteration+1])
        
        nextCash = currOptState[-1] + self.traded_amount
        nextReturn = self.returns[iteration + 1]
        nextFactors = self.factors[iteration + 1]
        nextOptState = list(nextFactors) + list(OptNextHolding) + [nextCash]

        OptResult = self._get_opt_reward(
            currOptState, nextOptState, nextReturn, iteration, tag
        )

        return nextOptState, OptResult

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
        

        DiscFactorLoads = self.f_param 

        return OptRate, DiscFactorLoads

    def _getreward(
        self,
        iteration: int,
        tag: str,
        res_action: float = None,
    ) -> dict:

        nextRet = self.returns[iteration+1]
        currHolding = self.holding_ts[iteration]
        nextHolding = self.holding_ts[iteration+1] 
        nextCash = self.cash_ts[iteration+1] 

        shares_traded = nextHolding - currHolding
        NetPNL = np.dot(nextHolding,nextRet) - self.costs
        Risk = 0.5 * self.kappa * np.dot(np.dot(nextHolding.T,self.cov_matrix),nextHolding)
        Reward = NetPNL - Risk
        nextWealth = nextHolding.sum() + nextCash

        Result = {
            "CurrHolding_{}".format(tag): currHolding,
            "NextHolding_{}".format(tag): nextHolding,
            "Action_{}".format(tag): shares_traded,
            "GrossPNL_{}".format(tag): NetPNL + self.costs,
            "NetPNL_{}".format(tag): NetPNL,
            "Risk_{}".format(tag): Risk,
            "Cost_{}".format(tag): self.costs,
            "Reward_{}".format(tag): Reward,
            "TradedAmount_{}".format(tag): self.traded_amount,
            "Cash_{}".format(tag): nextCash,
            "Wealth_{}".format(tag): nextWealth,
        }

        self.costs, self.traded_amount = 0.0, 0.0

        if isinstance(res_action, float):
            Result["ResAction_{}".format(tag)] = res_action
        return Result

    def _sell(self, index: int, holding: np.ndarray, action: float):

        currholding = holding

        if currholding > 0.0:
            # Sell only if current asset is > 0
            shares_traded = min(abs(action), currholding)
            self.costs += self._totalcost(shares_traded)
            self.traded_amount += shares_traded - self.costs
            return -shares_traded

        else:
            shares_traded = 0.0

            self.costs, self.traded_amount = 0.0, 0.0


        return -shares_traded

    def _buy(self, index: int, cash: np.ndarray, action: float):

        max_tradable_amount = cash
        shares_traded = min(max_tradable_amount, action)

        self.costs += self._totalcost(shares_traded)
        self.traded_amount += -shares_traded - self.costs


        return shares_traded

    def _get_opt_reward(
        self,
        currOptState: Tuple[Union[float , int], Union[float , int]],
        nextOptState: Tuple[Union[float , int], Union[float , int]],
        nextReturn: float,
        iteration: int,
        tag: str,
    ) -> dict:
        
        OptCurrHolding = np.array(currOptState[-1-self.n_assets:-1])
        OptNextHolding = np.array(nextOptState[-1-self.n_assets:-1])
        CurrOptcash = currOptState[-1]
        NextOptcash = nextOptState[-1]

        # Traded quantity between period
        OptNextAction = OptNextHolding - OptCurrHolding
        # Portfolio variation
        OptNetPNL = np.dot(OptNextHolding, nextReturn) - self.costs
        # Risk
        OptRisk = 0.5 * self.kappa * np.dot(np.dot(OptNextHolding.T,self.cov_matrix),OptNextHolding)
        # Compute reward
        OptReward = OptNetPNL - OptRisk

        nextWealth = OptNextHolding.sum() + NextOptcash #self.prices[iteration + 1] * 
        
        # Store quantities
        Result = {
            "{}NextAction".format(tag): OptNextAction,
            "{}NextHolding".format(tag): OptNextHolding,
            "{}GrossPNL".format(tag): OptNetPNL + self.costs,
            "{}NetPNL".format(tag): OptNetPNL,
            "{}Risk".format(tag): OptRisk,
            "{}Cost".format(tag): self.costs,
            "{}Reward".format(tag): OptReward,
            "{}TradedAmount".format(tag): self.traded_amount,
            "{}Cash".format(tag): NextOptcash,
            "{}Wealth".format(tag): nextWealth,
        }

        self.costs, self.traded_amount = 0.0, 0.0

        return Result

    def _opt_trade(self, index: int, holding: float, cash: float, action: float):
        
        if action > 0.0:

            max_tradable_amount = cash
            shares_traded = min(max_tradable_amount, action)

            self.costs += self._totalcost(shares_traded)
            self.traded_amount += -shares_traded - self.costs

            return shares_traded

        elif action < 0.0:

            currholding = holding

            if currholding > 0.0:
                # Sell only if current asset is > 0
                shares_traded = min(abs(action), currholding)

                self.costs = self._totalcost(shares_traded)
                self.traded_amount = shares_traded - self.costs

                return -shares_traded

            else:
                shares_traded = 0.0
                self.costs, self.traded_amount = 0.0, 0.0

                return shares_traded

        else:
            shares_traded = 0.0
            self.costs, self.traded_amount = 0.0, 0.0

            return shares_traded


    def store_results(self, Result: dict, iteration: int):
        if iteration == self.N_train:
            if 'PPO' in list(Result.keys())[0]:
                self.res_df = self.res_df.iloc[:-2,:]
                self.res_df[self.res_df.filter(like='CurrHolding_PPO').columns] = np.array(self.currholding_rl)
                self.res_df[self.res_df.filter(like='NextHolding_PPO').columns] = np.array(self.nextholding_rl)
                self.res_df[self.res_df.filter(like='Action_PPO').columns] = np.array(self.action_rl)
                self.res_df['GrossPNL_PPO'] = self.grosspnl_rl
                self.res_df['NetPNL_PPO'] = self.netpnl_rl
                self.res_df['Risk_PPO'] = self.risk_rl
                self.res_df['Cost_PPO'] = self.cost_rl
                self.res_df['Reward_PPO'] = self.reward_rl
                self.res_df['TradedAmount_PPO'] = self.tradedamount_rl
                self.res_df['Cash_PPO'] = self.cash_rl
                self.res_df['Wealth_PPO'] = self.wealth_rl
            elif 'MV' in list(Result.keys())[0]:
                self.res_df[self.res_df.filter(like='MVNextHolding').columns] = np.array(self.mvholding)
                self.res_df['MVReward'] = self.mvreward
                self.res_df['MVWealth'] = self.mvwealth
            elif 'Opt' in list(Result.keys())[0]:
                self.res_df[self.res_df.filter(like='OptNextAction').columns] = np.array(self.optaction)
                self.res_df[self.res_df.filter(like='OptNextHolding').columns] = np.array(self.optholding)
                self.res_df['OptGrossPNL'] = self.grosspnl_opt
                self.res_df['OptNetPNL'] = self.netpnl_opt
                self.res_df['OptRisk'] = self.risk_opt
                self.res_df['OptCost'] = self.cost_opt
                self.res_df['OptReward'] = self.reward_opt
                self.res_df['OptTradedAmount'] = self.tradedamount_opt
                self.res_df['OptCash'] = self.cash_opt
                self.res_df['OptWealth'] = self.wealth_opt
        else:
            if 'PPO' in list(Result.keys())[0]:
                self.currholding_rl.append(Result['CurrHolding_PPO'])
                self.nextholding_rl.append(Result['NextHolding_PPO'])
                self.action_rl.append(Result['Action_PPO'])
                self.grosspnl_rl.append(Result['GrossPNL_PPO'])
                self.netpnl_rl.append(Result['NetPNL_PPO'])
                self.risk_rl.append(Result['Risk_PPO'])
                self.cost_rl.append(Result['Cost_PPO'])
                self.reward_rl.append(Result['Reward_PPO'])
                self.tradedamount_rl.append(Result['TradedAmount_PPO'])
                self.cash_rl.append(Result['Cash_PPO'])
                self.wealth_rl.append(Result['Wealth_PPO'])
            elif 'MV' in list(Result.keys())[0]:
                self.mvholding.append(Result['MVNextHolding'])
                self.mvreward.append(Result['MVReward'])
                self.mvwealth.append(Result['MVWealth'])
            elif 'Opt' in list(Result.keys())[0]:
                self.optaction.append(Result['OptNextAction'])
                self.optholding.append(Result['OptNextHolding'])
                self.grosspnl_opt.append(Result['OptGrossPNL'])
                self.netpnl_opt.append(Result['OptNetPNL'])
                self.risk_opt.append(Result['OptRisk'])
                self.cost_opt.append(Result['OptCost']) 
                self.reward_opt.append(Result['OptReward'])
                self.tradedamount_opt.append(Result['OptTradedAmount'])
                self.cash_opt.append(Result['OptCash'])
                self.wealth_opt.append(Result['OptWealth'])



@gin.configurable()
class ShortMultiAssetCashMarketEnv(MultiAssetCashMarketEnv):


    def _sell(self, index: int, holding: np.ndarray, action: float):
 
        shares_traded = action
        self.costs += self._totalcost(shares_traded)
        # absolute quantity because now you can sell short and shares_traded can be negative
        self.traded_amount += np.abs(shares_traded) - self.costs

        return shares_traded


    def _opt_trade(self, index: int, holding: float, cash: float, action: float):
        
        if action > 0.0:

            max_tradable_amount = cash
            shares_traded = min(max_tradable_amount, action)

            self.costs += self._totalcost(shares_traded)
            self.traded_amount += -shares_traded - self.costs

            return shares_traded

        elif action < 0.0:

            # one could insert a stop to consider cost of short selling
            shares_traded = action

            self.costs += self._totalcost(shares_traded)
            self.traded_amount += np.abs(shares_traded) - self.costs

            return shares_traded

        else:
            shares_traded = 0.0
            self.costs, self.traded_amount = 0.0, 0.0

            return shares_traded