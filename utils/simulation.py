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

seaborn.set_style("white")
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
class DataHandler:
    def __init__(
        self,
        datatype: str,
        N_train: int,
        rng: object,
        factor_lb: Union[list, None] = None,
    ):

        self.datatype = datatype
        self.N_train = N_train
        self.rng = rng
        self.factor_lb = factor_lb

    def generate_returns(self, disable_tqdm: bool = False):

        if self.datatype != "garch":
            if self.datatype == "alpha_term_structure":
                self.returns, self.factors, self.f_speed = alpha_term_structure_sampler(N_train=self.N_train, rng=self.rng)
            elif self.datatype == "t_stud_mfit" or self.datatype == "t_stud":
                self.returns, self.factors, self.f_speed = return_sampler_GP(N_train=self.N_train + self.factor_lb[-1], rng=self.rng, disable_tqdm=disable_tqdm)
            else:
                if gin.query_parameter('%MULTIASSET'):
                    self.returns, self.factors, self.f_speed = multi_return_sampler_GP(
                        N_train=self.N_train, rng=self.rng, disable_tqdm=disable_tqdm
                    )
                else:
                    self.returns, self.factors, self.f_speed = return_sampler_GP(
                        N_train=self.N_train, rng=self.rng, disable_tqdm=disable_tqdm
                    )

        elif self.datatype == "garch":
            self.returns, self.params = return_sampler_garch(
                N_train=self.N_train + self.factor_lb[-1] + 2, disable_tqdm=disable_tqdm, rng=self.rng
            )


        else:
            print("Datatype to simulate is not correct")
            sys.exit()

    def estimate_parameters(self):
        if self.datatype == "t_stud_mfit":
            df = pd.DataFrame(
                data=np.concatenate([self.returns.reshape(-1, 1), self.factors], axis=1)
            )

            y, X = df[df.columns[0]].loc[self.factor_lb[-1] :], df[df.columns[1:]].loc[self.factor_lb[-1] :]

            params_meanrev, _ = RunModels(y, X, mr_only=True)

        else:
            df = CalculateLaggedSharpeRatio(
                self.returns, self.factor_lb, nameTag=self.datatype, seriestype="return"
            )

            y, X = df[df.columns[0]], df[df.columns[1:]]

            params_retmodel, params_meanrev, _, _ = RunModels(
                y, X
            )

        self.f_speed = np.abs(np.array([*params_meanrev.values()]).ravel())
        self.returns = df.iloc[:, 0].values
        self.factors = df.iloc[:, 1:].values


@gin.configurable()
def return_sampler_GP(
    N_train: int,
    sigmaf: Union[float , list , np.ndarray],
    f_param: Union[float , list , np.ndarray],
    sigma: Union[float , list , np.ndarray],
    HalfLife: Union[int , list , np.ndarray],
    rng: np.random.mtrand.RandomState = None,
    offset: int = 2,
    uncorrelated: bool = False,
    t_stud: bool = False,
    degrees: int = 8,
    vol: str = "omosk",
    dt: int = 1,
    disable_tqdm: bool = False,
) -> Tuple[
    Union[list , np.ndarray], Union[list , np.ndarray], Union[list , np.ndarray]
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
            # pct = 1.1
            # if i> N_train*pct: # temporary add
            #     f1 = np.multiply((1 - lambdas*2*dt), f0) + np.multiply(np.array(sigmaf)*np.sqrt(dt), eps[i])
            # else:
            f1 = np.multiply((1 - lambdas*dt), f0) + np.multiply(np.array(sigmaf)*np.sqrt(dt), eps[i])
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
            f1 = np.multiply((1 - lambdas*dt), f0) + eps[i]*np.sqrt(dt)
            f.append(f1)
            f0 = f1
    else:
        print("Choose proper volatility setting")
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
        print("Choose proper volatility setting")
        sys.exit()
    f_speed = lambdas

    generate_plot = False
    from utils.common import set_size
    if generate_plot:
        fig,ax = plt.subplots(figsize=set_size(360))
        ax.plot(realret)
        ax.set_xlabel('Time', fontsize=11)
        ax.set_ylabel('Return', fontsize=11)
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)
        fig.tight_layout()
        # plt.show()
    # pdb.set_trace()
    # sys.exit()
    return realret.astype(np.float32), factors.astype(np.float32), f_speed


@gin.configurable()
def multi_return_sampler_GP(
    N_train: int,
    sigmaf: Union[float , list , np.ndarray],
    f_param: Union[float , list , np.ndarray],
    sigma: Union[float , list , np.ndarray],
    HalfLife: Union[int , list , np.ndarray],
    rng: np.random.mtrand.RandomState = None,
    offset: int = 2,
    uncorrelated: bool = False,
    t_stud: bool = False,
    degrees: int = 8,
    vol: str = "omosk",
    dt: int = 1,
    disable_tqdm: bool = False,
) -> Tuple[
    Union[list , np.ndarray], Union[list , np.ndarray], Union[list , np.ndarray]
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

    lambdas = np.around(np.log(2) / HalfLife, 4).reshape(-1,)
    
    f0 = np.zeros(shape=lambdas.shape).reshape(-1,)
    eps = rng.randn(N_train + offset, len(lambdas))
    f = []

    for i in tqdm(
        iterable=range(N_train + offset),
        desc="Simulating Factors",
        disable=disable_tqdm,
    ):
        f1 = np.multiply((1 - lambdas*dt), f0) + np.multiply(np.array(sigmaf)*np.sqrt(dt), eps[i])
        f.append(f1)
        f0 = f1

    factors = np.vstack(f)
    
    # Generate covariance matrix
    correlations = gin.query_parameter("%CORRELATION")
    n_series = len(lambdas)
    cov = np.eye(n_series)
    for i in range(n_series):
        for j in range(i+1, n_series):
            if i < len(correlations) and j < len(correlations)+1:
                cov[i, j] = correlations[i]
                cov[j, i] = correlations[i]  # mirror lower triangular part
            else:
                cov[i, j] = 0.0
        
    #     # Generate correlated random series
    #     mean = np.zeros(n_series)
    #     u = rng.multivariate_normal(mean, cov, N)


    # u = rng.randn(N_train + offset, len(lambdas))
    u = rng.multivariate_normal(np.zeros(len(lambdas)), cov, N_train + offset)
    
    realret = (factors * np.array(f_param).reshape(-1,)) + sigma * u
    f_speed = lambdas

    generate_plot = True
    if generate_plot:
        fig,ax = plt.subplots(figsize=(10,5))
        pd.DataFrame(realret).plot(ax=ax)
        ax.set_title('GP Ret')
        plt.show()
        pdb.set_trace()
    return realret.astype(np.float32), factors.astype(np.float32), f_speed


@gin.configurable()
def return_sampler_garch(
    N_train: int,
    mean_process: str = "Constant",
    lags_mean_process: int = None,
    vol_process: str = "GARCH",
    distr_noise: str = "normal",
    seed: int = None,
    p_arg: list = None,
    disable_tqdm: bool = False,
    rng: np.random.mtrand.RandomState = None,
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

    if not rng:
        rng = np.random.RandomState(None)

    # choose mean process
    if mean_process == "Constant":
        model = ConstantMean(None)
        names.append("const")
        if rng:
            vals.append(rng.uniform(0.01, 0.09))
        else:
            vals.append(0.0)

    elif mean_process == "AR":
        model = ARX(None, lags=lags_mean_process)
        names.append("const")
        vals.append(0.0)
        if rng:
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
        if rng:
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
        if rng:
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
        if rng:
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
        if rng:
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
        if rng:
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
        if rng:
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
        if rng:
            vals.append(rng.randint(6.0, 10.0))
        else:
            vals.append(8.0)
    elif distr_noise == "skewstud":
        model.distribution = SkewStudent(np.random.RandomState(seed))
        names.extend(["nu", "lambda"])
        if rng:
            vals.extend([rng.uniform(6.0, 10.0), rng.uniform(-0.1, 0.1)])
        else:
            vals.extend([8.0, 0.05])
    elif distr_noise == "ged":
        model.distribution = GeneralizedError(np.random.RandomState(seed))
        names.append("nu")
        if rng:
            vals.append(rng.uniform(1.05, 3.0))
        else:
            vals.append(2.0)
    else:
        print("This noise distribution doesn't exist or it's not available.")
        sys.exit()

    p = pd.Series(data=vals, index=names)
    if p_arg:
        p.values[-3:] = p_arg

    simulations = model.simulate(p, N_train) / 100

    return simulations["data"].values, p

# https://courses.lumenlearning.com/waymakercollegealgebra/chapter/exponential-growth-and-decay/
@gin.configurable()
def alpha_term_structure_sampler(    
    N_train: int,
    HalfLife: Union[int , list , np.ndarray],
    initial_alpha: Union[int , list , np.ndarray],
    f_param: Union[int , list , np.ndarray],
    sigma: Union[int , list , np.ndarray]= None,
    sigmaf: Union[int , list , np.ndarray]= None,
    rng: np.random.mtrand.RandomState = None,
    offset: int = 2,
    generate_plot:bool = False,
    multiasset: bool = False,
    double_noise: bool = False,
    fixed_alpha: bool = False):

    # pdb.set_trace()
    tmp_rng = rng
    if fixed_alpha:
        rng=None
        
    if multiasset:
        term_structures = []
        alpha_factor_terms = []
        speeds = []
        
        for i in range(len(HalfLife)):
            init_a = initial_alpha[i]
            hl = HalfLife[i]
            fp = f_param[i]
            if rng:
                posneg = rng.choice([0,1])
                if posneg == 0:
                    init_a = np.array([rng.uniform(-val*(1+0.2),-val*(1-0.2),1) for val in init_a]).reshape(-1,)
                elif posneg == 1:
                    init_a = np.array([rng.uniform(val*(1-0.2),val*(1+0.2),1) for val in init_a]).reshape(-1,)
                if 'truncate' in hl:
                    hl = np.array([rng.uniform(int(N_train * 0.85),int(N_train * 1.25),1) for _ in init_a]).reshape(-1,)
                elif None in hl:
                    hl = np.array([rng.uniform(5,int(N_train * 0.75),1) for _ in init_a]).reshape(-1,)
                else:
                    hl = np.array([rng.uniform(val*(1-0.5),val*(1+0.5),1) for val in hl]).reshape(-1,)

            alpha_n = len(hl)
            f_speed =  np.log(2)/hl
            t = np.arange(0,N_train+offset).repeat(alpha_n).reshape(-1,alpha_n)

            if fixed_alpha:
                rng=tmp_rng
            if double_noise:
                alpha_terms = init_a * np.e**(-f_speed*t) + sigma * rng.normal(size=(len(t),alpha_n))
            else:
                alpha_terms = init_a * np.e**(-f_speed*t)
            if None not in sigmaf:
                # noise_magnitude = np.cumsum(sigmaf*t).reshape(-1,alpha_n)
                noise_magnitude = (np.array(sigmaf) * np.sqrt(t)).reshape(-1,alpha_n)
                noise = noise_magnitude * rng.normal(size=(len(t),alpha_n))
                alpha_terms = alpha_terms + noise
            if fixed_alpha:
                tmp_rng = rng
                rng = None

            if sum(fp) != 1.0:
                print('Factor loadings for term structure do not sum to one.')
                sys.exit()
            alpha_structure = np.sum(np.array(fp)* alpha_terms, axis=1)
 
            term_structures.append(alpha_structure)
            speeds.append(f_speed)
            alpha_factor_terms.append(alpha_terms)

        alpha_structure = np.transpose(np.array(term_structures,dtype='float')) 
        alpha_factor_terms = np.array(alpha_factor_terms,dtype='float')

        if alpha_factor_terms.shape[-1] != 1:
            alpha_terms = np.concatenate((alpha_factor_terms[0,:,:],alpha_factor_terms[1,:,:]),axis=1)
            # alpha_terms = np.transpose(alpha_factor_terms.reshape(-1,alpha_factor_terms.shape[1]))
        else:
            alpha_terms = np.transpose(np.squeeze(alpha_factor_terms))
        f_speed = np.array(speeds, dtype='float')

        # generate_plot=True
        if generate_plot:
            fig,ax = plt.subplots(figsize=(10,5))
            ax.plot(alpha_structure)
            # ax.plot(alpha_terms.sum(axis=1), ls='--')
            ax.set_title('Alpha term structure')
            plt.show()

        return alpha_structure, alpha_terms, f_speed
    else:
        # single asset case where different alpha term structure can be combined in 
        # a unique prediction
        if rng:
            posneg = rng.choice([0,1])
            if posneg == 0:
                initial_alpha = np.array([rng.uniform(-val*(1+0.5),-val*(1-0.5),1) for val in initial_alpha]).reshape(-1,)
            elif posneg == 1:
                initial_alpha = np.array([rng.uniform(val*(1-0.5),val*(1+0.5),1) for val in initial_alpha]).reshape(-1,)

            # if 'truncate' in HalfLife:
            #     HalfLife = np.array([rng.uniform(int(N_train * 0.85),int(N_train * 1.25),1) for _ in initial_alpha]).reshape(-1,)
            # elif None in HalfLife:
            #     HalfLife = np.array([rng.uniform(5,int(N_train * 0.75),1) for _ in initial_alpha]).reshape(-1,)
            # else:
            #     HalfLife = np.array([rng.uniform(val*(1-0.5),val*(1+0.5),1) for val in HalfLife]).reshape(-1,)

        alpha_n = len(HalfLife)
        f_speed =  np.log(2)/HalfLife
        t = np.arange(0,N_train+offset).repeat(alpha_n).reshape(-1,alpha_n)
        if fixed_alpha:
            rng=tmp_rng
        if double_noise:
            alpha_terms = initial_alpha * np.e**(-f_speed*t) + sigma * rng.normal(size=(len(t),alpha_n))
        else:
            alpha_terms = initial_alpha * np.e**(-f_speed*t)

        if None not in sigmaf:
            # noise_magnitude = (np.array(sigmaf) * np.sqrt(t)).reshape(-1,alpha_n)
            # noise = noise_magnitude * rng.normal(size=(len(t),alpha_n))
            noise = np.array(sigmaf) * rng.normal(size=(len(t),alpha_n)) 
            alpha_terms = alpha_terms + noise

        if sum(f_param) != 1.0:
            print('Factor loadings for term structure do not sum to one.')
            sys.exit()
        alpha_structure = np.sum(np.array(f_param)* alpha_terms, axis=1)

        generate_plot = False
        if generate_plot:
            fig,ax = plt.subplots(figsize=(10,5))
            ax.plot(alpha_terms)
            ax.plot(alpha_structure, ls='--')
            # ax.plot(alpha_terms.sum(axis=1), ls='--')
            ax.set_title('Alpha term structure')
            plt.show()

        return alpha_structure, alpha_terms, f_speed

