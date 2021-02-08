# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 15:48:01 2020

@author: aless
"""
import pdb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
class GaussianActionNoise:
    def __init__(self, mu, sigma, rng):
        self.mu = mu
        self.sigma = sigma
        self.rng = rng

    def __call__(self):
        return self.rng.normal(self.mu, self.sigma)

    def __repr__(self):
        return "NormalActionNoise(mu={}, sigma={})".format(self.mu, self.sigma)


# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma, theta, rng, dt=1, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.rng = rng
        self.reset()

    def __call__(self):
        x = (
            self.x_prev
            + self.theta * (self.mu - self.x_prev) * self.dt
            + self.sigma * np.sqrt(self.dt) * self.rng.normal(size=self.mu.shape)
        )
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return "OrnsteinUhlenbeckActionNoise(mu={}, sigma={})".format(
            self.mu, self.sigma
        )


if __name__=='__main__':

    # N = 5000
    # sigma = 0.22
    # sigma_lab = str(sigma)
    # sigma_decay = (sigma- 0.01)/N
    # theta = 0.1
    
    
    # rng = np.random.RandomState(123)
    # OU_process = OrnsteinUhlenbeckActionNoise(mu=np.array([0]), sigma=sigma, theta=theta, x0=0, rng=rng)
    # # rng = np.random.RandomState(123)
    # # G_process = GaussianActionNoise(mu=np.array([0]), sigma=sigma, rng=rng)

    # # noises_g = [G_process() for _ in range(N)]
    # noises_ou = [OU_process() for _ in range(N)]

    # noises_g_decay = []
    # noises_ou_decay = []
    # for _ in range(N):
    #     sigma = max(0.0, sigma - sigma_decay)
        # G_process.sigma = sigma
        # noises_g_decay.append(G_process())
        # OU_process.sigma = sigma
        # noises_ou_decay.append(OU_process())

    # fig = plt.figure()
    # ax = fig.add_subplot()
    # # ax.plot(noises_g, label = 'sigma_{}-{}'.format(sigma_lab, 'G'))
    # plt.plot(noises_ou, label = 'sigma_{}-{}'.format(sigma_lab, 'OU'))
    # # ax.plot(noises_g_decay, label = 'sigma_{}_{}'.format(sigma_lab,'G_decay'))
    # # plt.plot(noises_ou_decay, label = 'sigma_{}_{}'.format(sigma_lab,'OU_decay'), alpha=0.6)
    # ax.legend()

    # df = pd.DataFrame(np.concatenate([noises_g,noises_ou],axis=1),
    #               columns=['G','OU'])
    # df['G_decay'] = [el[0] for el in noises_g_decay]
    # df['OU_decay'] = [el[0] for el in noises_ou_decay]
    # print(df.describe())
    # df.plot()

    N = 5000
    sigma = 0.22
    
    # thetas = np.arange(0.1,1.0,0.2)
    thetas = [0.05 ,0.1, 0.35, 0.7]
    df = pd.DataFrame(index=range(N))
    for t in thetas:
        rng = np.random.RandomState(123)
        OU_process = OrnsteinUhlenbeckActionNoise(mu=np.array([0]), sigma=sigma, theta=t, x0=0, rng=rng)
        # rng = np.random.RandomState(123)
        # G_process = GaussianActionNoise(mu=np.array([0]), sigma=sigma, rng=rng)
    
        # noises_g = [G_process() for _ in range(N)]
        noises_ou = [OU_process() for _ in range(N)]
        # pdb.set_trace()
        df['theta_{}'.format(np.round(t,2))] = np.array(noises_ou)

    df.plot()
    print(df.describe())