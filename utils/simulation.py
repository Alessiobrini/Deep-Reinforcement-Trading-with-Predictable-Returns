# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 11:32:29 2020

@author: aless
"""
from typing import Tuple, Union
import numpy as np
from tqdm import tqdm
import pdb, sys
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import pandas as pd
import seaborn
import seaborn as sns
import gin
from utils.tools import CalculateLaggedSharpeRatio, RunModels

seaborn.set_style("darkgrid")
plt.rcParams["figure.figsize"] = (20.0, 10.0)
plt.rcParams["savefig.dpi"] = 90
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 14
from arch.univariate import (
    ARX,
    ConstantMean,
    GARCH,
    EGARCH,
    ConstantVariance,
    HARCH,
    FIGARCH,
)
from arch.univariate import Normal, StudentsT, SkewStudent, GeneralizedError

@gin.configurable()
class DataHandler():
    def __init__(self,datatype: str, 
                N_train:int,
                rng: object, 
                factor_lb: Union[list or None] = None):

        self.datatype = datatype
        self.N_train = N_train
        self.rng = rng
        self.factor_lb = factor_lb 

    def generate_returns(self):

        if self.datatype != 'garch':
            # REMARK if you do t_studmfit and you want to compare the training over the same
            # series wrt t_stud, you need to simulate a series of lenght N_train + factor_lb[-1]
            # This is currently not implemented here
            self.returns, self.factors, self.f_speed = return_sampler_GP(N_train=self.N_train,
                                                               rng=self.rng)


        elif self.datatype == "garch":
            self.returns, self.params = return_sampler_garch(N_train=self.N_train)


        else:
            print("Datatype to simulate is not correct")
            sys.exit()

    def estimate_parameters(self):

        if self. datatype == 't_stud_mfit':
            # look at the remark above: use .loc[factor_lb[-1] :] if you want the same series of t_stud
            # this method doesn't require to fit the true parameters
            df = pd.DataFrame(
                data=np.concatenate([self.returns.reshape(-1, 1), self.factors], axis=1)
            )

            y, X = df[df.columns[0]], df[df.columns[1:]]
            
            params_meanrev, fitted_ous = RunModels(y, X, mr_only=True)

        else:
            df = CalculateLaggedSharpeRatio(
                self.returns, self.factor_lb, nameTag=self.datatype, seriestype="return"
            )

            y, X = df[df.columns[0]], df[df.columns[1:]]

            params_retmodel, params_meanrev, fitted_retmodel, fitted_ous = RunModels(y, X)

            sigma_fit = df.iloc[:, 0].std()
            gin.bind_parameter('%SIGMA',sigma_fit)

            f_param_fit = params_retmodel["params"]
            gin.bind_parameter('%F_PARAM',f_param_fit)

            sigmaf_fit = X.std().values
            gin.bind_parameter('%SIGMAF',sigmaf_fit)

        self.f_speed = np.abs(np.array([*params_meanrev.values()]).ravel())
        gin.bind_parameter('%HALFLIFE',np.around(np.log(2) / self.f_speed_fit, 2))

        self.returns = df.iloc[:, 0].values
        self.factors = df.iloc[:, 1:].values
        


@gin.configurable()
def return_sampler_GP(
    N_train: int,
    sigmaf: Union[float or list or np.ndarray],
    f_param: Union[float or list or np.ndarray],
    sigma: Union[float or list or np.ndarray],
    HalfLife: Union[int or list or np.ndarray],
    rng: np.random.mtrand.RandomState = None,
    offset: int = 2,
    uncorrelated: bool = False,
    t_stud: bool = False,
    degrees: int = 8,
    vol: str = "omosk",
    disable_tqdm: bool = False,
) -> Tuple[
    Union[list or np.ndarray], Union[list or np.ndarray], Union[list or np.ndarray]
]:
    """
    Generates financial returns driven by mean-reverting factors.

    Parameters
    ----------
    N_train : int
        Length of the experiment

    sigmaf : Union[float or list or np.ndarray]
        Volatilities of the mean reverting factors

    f_param: Union[float or list or np.ndarray]
        Factor loadings of the mean reverting factors

    sigma: Union[float or list or np.ndarray]
        volatility of the asset return (additional noise other than the intrinsic noise
                                        in the factors)

    HalfLife: Union[int or list or np.ndarray]
        HalfLife of mean reversion to simulate factors with different speeds

    rng: np.random.mtrand.RandomState
        Random number generator for reproducibility

    offset: int = 2
        Amount of additional observation to simulate

    uncorrelated: bool = False
        Boolean to regulate if the simulated factor are correlated or not

    t_stud : bool = False
        Bool to regulate if Student\'s t noises are needed

    degrees : int = 8
        Degrees of freedom for Student\'s t noises

    vol: str = 'omosk'
        Choose between 'omosk' and 'eterosk' for the kind of volatility
    Returns
    -------
    realret: Union[list or np.ndarray]
        Simulated series of returns
    factors: Union[list or np.ndarray]
        Simulated series of factors
    f_speed: Union[list or np.ndarray]
        Speed of mean reversion computed form HalfLife argument
    """

    # use samplesize +2 because when iterating the algorithm is necessary to
    # have one observation more (the last space representation) and because
    # we want be able to plot insample operation every tousand observation.
    # Therefore we don't want the index ending at 999 instead of 1000

    # Generate stochastic factor component and compute speed of mean reversion
    # simulate the single factor according to OU process
    # select proper speed of mean reversion and initialization point
    # it is faster to increase the size of a python list than a numpy array
    # therefore we convert later the list
    # https://www.jmp.com/en_us/statistics-knowledge-portal/t-test/t-distribution.html#:~:text=The%20shape%20of%20the%20t,%E2%80%9D%20than%20the%20z%2Ddistribution.

    lambdas = np.around(np.log(2) / HalfLife, 4)

    f0 = np.zeros(shape=(len(lambdas),))

    if vol == "omosk":
        if t_stud:
            if uncorrelated:
                eps = rng.standard_t(degrees, (N_train + offset, len(HalfLife)))
            else:
                eps = rng.standard_t(degrees, (N_train + offset))
        else:
            if uncorrelated:
                eps = rng.randn(N_train + offset, len(HalfLife))
            else:
                eps = rng.randn(N_train + offset)

        f = []

        # possibility of triple noise
        for i in tqdm(
            iterable=range(N_train + offset),
            desc="Simulating Factors",
            disable=disable_tqdm,
        ):
            # multiply makes the hadamard (componentwise) product
            # if we want to add different volatility for different factors we could
            # add multiply also the the second part of the equation
            f1 = np.multiply((1 - lambdas), f0) + np.multiply(sigmaf, eps[i])
            f.append(f1)
            f0 = f1

    elif vol == "heterosk":
        volmodel = GARCH(p=1, q=1)
        # these factors, if multiple, are uncorrelated by default because the noise is constructed one by one
        if len(sigmaf) > 1:

            eps = []
            for i in range(len(sigmaf)):
                om = sigmaf[i] ** 2  # same vol as original GP experiments
                alph = 0.05
                b = 1 - alph - om
                garch_p = np.array([om, alph, b])

                e = volmodel.simulate(garch_p, N_train + offset, rng.randn)[0]
                eps.append(e.reshape(-1, 1))

            eps = np.concatenate(eps, axis=1)
        else:

            om = sigmaf[0] ** 2  # same vol as original GP experiments
            alph = 0.05
            b = 1 - alph - om
            garch_p = np.array([om, alph, b])

            eps = volmodel.simulate(garch_p, N_train + offset, rng.randn)[0]

        f = []
        # possibility of triple noise
        for i in tqdm(
            iterable=range(N_train + offset),
            desc="Simulating Factors",
            disable=disable_tqdm,
        ):
            # multiply makes the hadamard (componentwise) product
            # if we want to add different volatility for different factors we could
            # add multiply also the the second part of the equation
            f1 = np.multiply((1 - lambdas), f0) + eps[i]
            f.append(f1)
            f0 = f1
    else:
        print('Choose proper volatility setting')
        sys.exit()

    factors = np.vstack(f)
    if vol == "omosk":
        if t_stud:
            u = rng.standard_t(degrees, N_train + offset)
        else:
            u = rng.randn(N_train + offset)

        realret = np.sum(f_param * factors, axis=1) + sigma * u

    elif vol == "heterosk":
        volmodel = GARCH(p=1, q=1)
        om = sigma ** 2  # same vol as original GP experiments
        alph = 0.05
        b = 1 - alph - om
        garch_p = np.array([om, alph, b])

        u = volmodel.simulate(garch_p, N_train + offset, rng.randn)[0]

        realret = np.sum(f_param * factors, axis=1) + sigma * u
    else:
        print('Choose proper volatility setting')
        sys.exit()
    f_speed = lambdas

    return realret.astype(np.float32), factors.astype(np.float32), f_speed

@gin.configurable()
def return_sampler_garch(
    N_train: int,
    mean_process: str = "Constant",
    lags_mean_process: int = None,
    vol_process: str = "GARCH",
    distr_noise: str = "normal",
    seed: int = None,
    seed_param: int = None,
    p_arg: list = None,
) -> Tuple[np.ndarray, pd.Series]:
    # https://stats.stackexchange.com/questions/61824/how-to-interpret-garch-parameters
    # https://arch.readthedocs.io/en/latest/univariate/introduction.html
    # https://arch.readthedocs.io/en/latest/univariate/volatility.html
    # https://github.com/bashtage/arch/blob/master/arch/univariate/volatility.py
    """
    Generates financial returns driven by mean-reverting factors.

    Parameters
    ----------
    N_train: int
        Length of the experiment

    mean_process: str
        Mean process for the returns. It can be 'Constant' or 'AR'

    lags_mean_process: int
        Order of autoregressive lag if mean_process is AR

    vol_process: str
        Volatility process for the returns. It can be 'GARCH', 'EGARCH', 'TGARCH',
        'ARCH', 'HARCH', 'FIGARCH' or 'Constant'. Note that different volatility
        processes requires different parameter, which are hard coded. If you want to
        pass them explicitly, use p_arg.

    distr_noise: str
        Distribution for the unpredictable component of the returns. It can be
        'normal', 'studt', 'skewstud' or 'ged'. Note that different distributions
        requires different parameter, which are hard coded. If you want to
        pass them explicitly, use p_arg.

    seed: int
        Seed for experiment reproducibility

    seed_param: int
        Seed for drawing randomly the parameters needed for the simulation. The
        ranges provided are obtained as average lower and upper bounds of several
        GARCH-type model fitting on real financial time-series.

    p_arg: pd.Series
        Pandas series of parameters that you want to pass explicitly.
        They need to be passed in the right order. Check documentation of the
        arch python package (https://arch.readthedocs.io/en/latest/index.html) for more details.
    Returns
    -------
    simulations['data'].values: np.ndarray
        Simulated series of returns
    p: pd.Series
        Series  of parameters used for simulation
    """
    names = []
    vals = []

    if seed_param is None : seed_param = seed

    rng = np.random.RandomState(seed_param)

    # choose mean process
    if mean_process == "Constant":
        model = ConstantMean(None)
        names.append("const")
        if seed_param:
            vals.append(rng.uniform(0.01, 0.09))
        else:
            vals.append(0.0)

    elif mean_process == "AR":
        model = ARX(None, lags=lags_mean_process)
        names.append("const")
        vals.append(0.0)
        if seed_param:
            for i in range(lags_mean_process):
                names.append("lag{}".format(i))
                vals.append(rng.uniform(-0.09, 0.09))
        else:
            for i in range(lags_mean_process):
                names.append("lag{}".format(i))
                vals.append(0.9)

    else:
        return print("This mean process doesn't exist or it's not available.")
        sys.exit()

    # choose volatility process
    if vol_process == "GARCH":
        model.volatility = GARCH(p=1, q=1)
        names.extend(["omega", "alpha", "beta"])
        if seed_param:
            om = rng.uniform(0.03, 0.1)
            alph = rng.uniform(0.05, 0.1)
            b = rng.uniform(0.86, 0.92)
            garch_p = np.array([om, alph, b]) / (np.array([om, alph, b]).sum())
        else:
            om = 0.01
            alph = 0.05
            b = 0.94
            garch_p = np.array([om, alph, b])
        vals.extend(list(garch_p))

    elif vol_process == "ARCH":
        model.volatility = GARCH(p=1, q=0)

        names.extend(["omega", "alpha"])
        if seed_param:
            om = rng.uniform(1.4, 4.0)
            alph = rng.uniform(0.1, 0.6)
        else:
            om = 0.01
            alph = 0.4
        garch_p = np.array([om, alph])
        vals.extend(list(garch_p))

    elif vol_process == "HARCH":
        model.volatility = HARCH(lags=[1, 5, 22])

        names.extend(["omega", "alpha[1]", "alpha[5]", "alpha[22]"])
        if seed_param:
            om = rng.uniform(1.2, 0.5)
            alph1 = rng.uniform(0.01, 0.1)
            alph5 = rng.uniform(0.05, 0.3)
            alph22 = rng.uniform(0.4, 0.7)
        else:
            om = 0.01
            alph1 = 0.05
            alph5 = 0.15
            alph22 = 0.5
        garch_p = np.array([om, alph1, alph5, alph22])
        vals.extend(list(garch_p))

    elif vol_process == "FIGARCH":
        model.volatility = FIGARCH(p=1, q=1)

        names.extend(["omega", "phi", "d", "beta"])
        if seed_param:
            om = rng.uniform(0.05, 0.03)
            phi = rng.uniform(0.1, 0.35)
            d = rng.uniform(0.3, 0.5)
            beta = rng.uniform(0.4, 0.7)
        else:
            om = 0.01
            phi = 0.2
            d = 0.2
            beta = 0.55
        garch_p = np.array([om, phi, d, beta])
        vals.extend(list(garch_p))

    elif vol_process == "TGARCH":
        model.volatility = GARCH(p=1, o=1, q=1)
        names.extend(["omega", "alpha", "gamma", "beta"])
        if seed_param:
            om = rng.uniform(0.02, 0.15)
            alph = rng.uniform(0.01, 0.07)
            gamma = rng.uniform(0.03, 0.1)
            b = rng.uniform(0.88, 0.94)
        else:
            om = 0.01
            alph = 0.05
            gamma = 0.04
            b = 0.90
        garch_p = np.array([om, alph, gamma, b])
        vals.extend(list(garch_p))

    elif vol_process == "EGARCH":
        model.volatility = EGARCH(p=1, o=1, q=1)
        names.extend(["omega", "alpha", "gamma", "beta"])
        if seed_param:
            om = rng.uniform(0.01, 0.03)
            alph = rng.uniform(0.06, 0.17)
            gamma = rng.uniform(-0.05, -0.02)
            b = rng.uniform(0.97, 0.99)
            garch_p = np.array([om, alph, gamma, b]) / (
                np.array([om, alph, gamma, b]).sum()
            )
        else:
            om = 0.01
            alph = 0.05
            gamma = -0.02
            b = 0.94
            garch_p = np.array([om, alph, gamma, b])
        vals.extend(list(garch_p))

    elif vol_process == "Constant":
        model.volatility = ConstantVariance()
        names.append("sigma_const")
        vals.append(rng.uniform(0.02, 0.05))
    else:
        print("This volatility process doesn't exist or it's not available.")
        sys.exit()

    if distr_noise == "normal":
        model.distribution = Normal(np.random.RandomState(seed))
    elif distr_noise == "studt":
        model.distribution = StudentsT(np.random.RandomState(seed))
        names.append("nu")
        if seed_param:
            vals.append(rng.randint(6.0, 10.0))
        else:
            vals.append(8.0)
    elif distr_noise == "skewstud":
        model.distribution = SkewStudent(np.random.RandomState(seed))
        names.extend(["nu", "lambda"])
        if seed_param:
            vals.extend([rng.uniform(6.0, 10.0), rng.uniform(-0.1, 0.1)])
        else:
            vals.extend([8.0, 0.05])
    elif distr_noise == "ged":
        model.distribution = GeneralizedError(np.random.RandomState(seed))
        names.append("nu")
        if seed_param:
            vals.append(rng.uniform(1.05, 3.0))
        else:
            vals.append(2.0)
    else:
        print("This noise distribution doesn't exist or it's not available.")
        sys.exit()

    p = pd.Series(data=vals, index=names)
    if p_arg:
        p = p_arg
    simulations = model.simulate(p, N_train) / 100

    return simulations["data"].values, p


