from gin.config import configurable
from gym.spaces.space import Space
import numpy as np
import gin


@gin.configurable()
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

    def __init__(self, action_range: list, zero_action: bool = True, side_only: bool = False):
        if not side_only:
            self.values = np.round(np.linspace(-action_range[0], action_range[0], action_range[1]),2)
        else:
            self.values = np.array([-1.0, 0.0, 1.0])
        if not zero_action:
            self.values = self.values[self.values != 0.0]

        self.action_range=action_range
        self.zero_action = zero_action
        self.side_only = side_only
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

@gin.configurable()
class ResActionSpace(Space):
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

    def __init__(self, action_range:list, zero_action: bool = True, side_only: bool = False):

        self.values = np.round(np.linspace(-action_range[0], action_range[1], action_range[2]),2)
        if not zero_action:
            self.values = self.values[self.values != 0.0]
        
        self.action_range=action_range
        self.zero_action = zero_action
        self.side_only = side_only
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