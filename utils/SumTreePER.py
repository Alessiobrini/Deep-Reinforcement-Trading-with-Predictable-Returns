# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 17:31:42 2020

@author: aless
"""

# There are many different ways of implementing a SumTree in Python.
# The code below uses a class to define the structure,
# and uses recursive functions to both traverse and create the SumTree

import numpy as np
import pdb, sys


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


################################################################################
# TEST PART

# buffer = Memory(100000)
# for _ in range(10000):
#     buffer.store(experience = np.random.randint(0,1000,5))

# b_idx, minibatch = buffer.sample(256)

# tree = SumTree(100000)
# exp = {'s': (0.005,1000), 'a': 50, 'r': 2103, 's2': (0.0002, 1050)}
# tree.add(1.0, exp)
