# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 17:31:42 2020

@author: aless
"""

# There are many different ways of implementing a SumTree in Python.
# The code below uses a class to define the structure,
# and uses recursive functions to both traverse and create the SumTree


import pdb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class PER_buffer:
    # Here we initialize the tree with all nodes = 0, and initialize the data with all values = 0
    def __init__(
        self,
        PER_e,
        PER_a,
        PER_b,
        final_PER_b,
        PER_b_steps,
        PER_b_growth,
        final_PER_a,
        PER_a_steps,
        PER_a_growth,
        max_experiences,
        rng,
        sample_type,
    ):

        self.PER_e = PER_e
        self.PER_a = PER_a
        self.PER_b = PER_b
        self.final_PER_b = final_PER_b
        self.PER_b_steps = PER_b_steps
        self.PER_b_growth = PER_b_growth
        self.final_PER_a = final_PER_a
        self.PER_a_steps = PER_a_steps
        self.PER_a_growth = PER_a_growth
        self.absolute_error_upper = np.power(10, 6)
        # initialize the counter
        self.data_pointer = 0
        self.rng = rng
        self.sample_type = sample_type

        # Number of leaf nodes (final nodes) that contains experiences
        self.max_experiences = max_experiences  # equal to max experiences

        # Generate the tree with all nodes values = 0
        # To understand this calculation (2 * capacity - 1) look at the schema below
        # Remember we are in a binary node (each node has max 2 children)
        # so 2x size of leaf (capacity) - 1 (root node)
        # Parent nodes = capacity - 1
        # Leaf nodes = capacity
        self.tree = np.zeros(2 * max_experiences - 1)

        # Contains the experiences (so the size of data is capacity)
        # self.data = np.zeros(capacity, dtype=object)
        self.experience = {
            "s": np.zeros(max_experiences, dtype=object),
            "a": np.zeros(max_experiences, dtype=object),
            "r": np.zeros(max_experiences, dtype=object),
            "s2": np.zeros(max_experiences, dtype=object),
            "f": np.zeros(max_experiences, dtype=object),
        }

    def add(self, exp):
        """Define add function that will add our priority score in the sumtree leaf and 
        add the experience in data.
        tree:
                    0
                   / \
                  0   0
                 / \ / \
        tree_index  0 0  0  We fill the leaves from left to right"""

        # Find the max priority
        max_priority = np.max(self.tree[-self.max_experiences :])

        # If the max priority = 0 we can't put priority = 0 since this experience will never have a chance to be selected
        # So we use a minimum priority

        if max_priority == 0:
            max_priority = self.absolute_error_upper

        # Look at what index we want to put the experience
        # You can check that len(self.tree) - tree_index when self.data_pointer==0
        # is equal to self.capacity
        tree_index = self.data_pointer + self.max_experiences - 1

        # Update experience
        # self.data[self.data_pointer] = data
        # if len(self.experiences['s']) >= self.max_experiences:
        #     for key in self.experience.keys():
        #         self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key][self.data_pointer] = value

        # Update the leaf
        self.update(tree_index, max_priority)

        # Add 1 to data_pointer
        self.data_pointer += 1

        if (
            self.data_pointer >= self.max_experiences
        ):  # If we're above the capacity, we go back to first index (we overwrite)
            self.data_pointer = 0

    def update(self, tree_index, priority):
        """Create function to update the leaf priority score and propagate the change through tree"""
        # Change = new priority score - former priority score
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        # then propagate the change through tree
        # this method is faster than the recursive loop
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, v):
        parent_index = 0

        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            # If we reach bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        exp_index = leaf_index - self.max_experiences + 1
        self.exp_leaf = {
            key: value[exp_index] for key, value in self.experience.items()
        }

        return leaf_index, self.tree[leaf_index], self.exp_leaf

    @property
    def total_priority(self):
        return self.tree[0]  # Returns the root node

    def sample_batch(self, batch_size):
        # Create a minibatch array that will contains the minibatch
        minibatch = {"s": [], "a": [], "r": [], "s2": [], "f": []}

        # one could use also np zeros
        b_idx = np.empty((batch_size,), dtype=np.int32)

        # Calculate the priority segment
        # Here, as explained in the paper, we divide the Range[0, ptotal] into n ranges
        priority_segment = self.total_priority / batch_size  # priority segment

        for i in range(batch_size):
            # A value is uniformly sample from each range
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = self.rng.uniform(a, b)

            # Experience that correspond to each value is retrieved
            index, priority, data = self.get_leaf(value)

            b_idx[i] = index
            for key, value in data.items():
                minibatch[key].append(value)

        return b_idx, minibatch

    def batch_update(self, tree_idx, abs_errors):

        if self.sample_type == "TDerror" or self.sample_type == "rewards":
            abs_errors += self.PER_e  # convert to abs and avoid 0
        elif self.sample_type == "diffTDerror":
            # variant specified in original paper for stochastic and partially observable env
            abs_diff_error = (
                np.abs(abs_errors - self.tree[tree_idx]) + self.PER_e
            )  # abs value of the difference of abs values
            first_play_idx = np.where(self.tree[tree_idx] == np.power(10, 6))
            if first_play_idx:
                first_priority = abs_errors[first_play_idx] + self.PER_e
                abs_diff_error[first_play_idx] = first_priority
        else:
            print("Sample type for PER not available")
            sys.exit()

        # clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        self.PER_a = min(self.final_PER_a, self.PER_a + self.PER_a_growth)
        if self.sample_type == "TDerror" or "rewards":
            ps = np.power(abs_errors, self.PER_a)
        elif self.sample_type == "diffTDerror":
            ps = np.power(abs_diff_error, self.PER_a)
        else:
            print("Sample type for PER not available")
            sys.exit()

        for ti, p in zip(tree_idx, ps):
            self.update(ti, p)


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


################################################################################
# TEST PART
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
