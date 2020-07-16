# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 15:48:01 2020

@author: aless
"""
import pdb
import numpy as np
import matplotlib.pyplot as plt

# https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
class GaussianActionNoise():
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __call__(self):
        return np.random.normal(self.mu, self.sigma)

    def __repr__(self):
        return 'NormalActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise():
    def __init__(self, mu, sigma=0.2, theta=.15, dt=1, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)
    
if __name__=='__main__':

    sigma = 0.1
    sigma_decay = (sigma- 0.01)/4000
    # OU_process = OrnsteinUhlenbeckActionNoise(mu=np.array([0]), sigma=sigma, theta=0.05, x0=0)
    G_process = GaussianActionNoise(mu=np.array([0]), sigma=sigma)
    
    # noises_ou = [OU_process() for _ in range(5000)]
    noises_g = [G_process() for _ in range(5000)]
    fig = plt.figure()
    plt.plot(noises_g, label = 'sigma_{}-{}'.format(sigma, 'G'))
    
    noises_g_decay = []
    for _ in range(5000):
        sigma = max(0.0, sigma - sigma_decay)
        G_process.sigma = sigma
        noises_g_decay.append(G_process())
    # plt.plot(noises_ou, label = 'sigma_{}_{}'.format(sigma,'OU'))

    plt.plot(noises_g_decay, label = 'sigma_{}-{}'.format(sigma, 'G_decay'))
    plt.legend()
    plt.show()

    