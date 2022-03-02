"""
The MIT License (MIT)

Copyright 2020 Siemens AG, Technical University of Berlin

Authors: Phillip Swazinna (Earlier Version: Ludwig Winkler)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import gym
import numpy as np

from industrial_benchmark_python.IDS import IDS


class IBGym(gym.Env):
    """
    OpenAI Gym Wrapper for the industrial benchmark
    """

    def __init__(
        self,
        setpoint,
        reward_type="classic",
        action_type="continuous",
        observation_type="classic",
        reset_after_timesteps=1000,
        n_past_timesteps=30
    ):
        """
        Initializes the underlying environment, seeds numpy and initializes action / observation spaces
        as well as other necessary variables
        :param setpoint: determines behavior of industrial benchmark
        :param reward_type: classic / delta - determines whether absolute or change in reward is returned
        :param action_type: discrete / continuous
        :param observation_type: classic / include_past - determines wether single or N state frames used as observation
        :param reset_after_timesteps: how many timesteps can the environment run without resetting
        :param init_seed: seed for numpy to make environment behavior reproducible
        :param n_past_timesteps: if observation type is include_past, this determines how many state frames are used
        """

        # IB environment parameter
        self.setpoint = setpoint

        # Used to determine whether to return the absolute value or the relative change in the cost function
        self.reward_function = reward_type

        # Used to set an arbitrary limit of how many time steps the environment can take before resetting
        self.reset_after_timesteps = reset_after_timesteps

        # Define what actions and observations can look like
        self.action_type = action_type  # discrete or continuous
        self.observation_type = observation_type  # classic or include_past
        self.n_past_timesteps = n_past_timesteps  # if past should be included - how many steps?

        # variables that will change over the course of a trajectory - only initialized here
        self.IB = None  # the actual IDS Object -> real environment

        # Defining the action space
        if self.action_type == "discrete":  # Discrete action space with three values per steering (3^3 = 27)
            self.action_space = gym.spaces.Discrete(27)

            # A list of all possible discretized actions
            self.env_action = [[v, g, s] for s in [-1, 0, 1] for g in [-1, 0, 1] for v in [-1, 0, 1]]
        elif self.action_type == "continuous":  # Continuous action space for each steering [-1,1]
            self.action_space = gym.spaces.Box(np.array([-1, -1, -1]), np.array([1, 1, 1]), dtype=float)
        else:
            raise ValueError('Invalid action_type. action_space can either be "discrete" or "continuous"')

        # Defining the observation space -> single frame: [setpoint, velocity, gain, shift, fatigue, consumption]
        single_low = np.array([0, 0, 0, 0, 0, 0])
        single_high = np.array([100, 100, 100, 100, 1000, 1000])

        if self.observation_type == "classic":  # classic only has the current state frame
            self.observation_space = gym.spaces.Box(low=single_low, high=single_high, dtype=float)
        elif self.observation_type == "include_past":  # time embedding: state contains also past N state frames
            low = np.hstack([single_low] * self.n_past_timesteps)
            high = np.hstack([single_high] * self.n_past_timesteps)
            self.observation_space = gym.spaces.Box(low=low, high=high)
        else:
            raise ValueError('Invalid observation_type. observation_type can either be "classic" or "include_past"')

    def step(self, action):
        """
        performs one step in the environment by taking the specified action and returning the resulting observation
        :param action: the action to be taken
        :return: the new observation
        """
        if self.IB is None:
            raise ValueError("Environment must be reset before first step")

        # keep the current action around for potential rendering
        self.last_action = action
        last_reward = self.reward

        # Executing the action and saving the observation
        if self.action_type == "discrete":
            self.IB.step(self.env_action[action])  # for discrete actions, we expect the action's index
        elif self.action_type == "continuous":
            self.IB.step(action)  # in the continuous case, we expect the entire three dimensional action

        # update observation representation
        return_observation = self._update_observation()

        # Calculating both the relative reward (improvement or decrease) and updating the absolute reward
        delta_reward = self.reward - last_reward  # positive when improved

        # Stopping condition
        self.env_steps += 1
        done = self.env_steps >= self.reset_after_timesteps

        # Two reward functions are available:
        # 'classic' which returns the original cost and
        # 'delta' which returns the change in the cost function w.r.t. the previous cost
        if self.reward_function == "classic":
            return_reward = self.reward
        elif self.reward_function == "delta":
            return_reward = delta_reward
        else:
            raise ValueError(
                f"Invalid reward function specification '{self.reward_function}'. 'classic' for the original cost function"
                " or 'delta' for the change in the cost fucntion between steps."
            )

        info = self._markovian_state()  # entire markov state - not all info is visible in observations
        return return_observation, return_reward, done, info

    @property
    def reward(self) -> float:
        return -self.IB.state["cost"]

    def reset(self, seed: int = None) -> np.ndarray:
        """
        resets environment
        :return: first observation of fresh environment
        """
        self.IB = IDS(self.setpoint, inital_seed=seed)

        self.observation = None
        self.last_action = None  # contains the action taken in the last step
        # used to set the self.done variable - If larger than self.reset_after_timesteps, the environment resets
        self.env_steps = 0

        return self._update_observation()

    def render(self, mode="human"):
        """
        prints the current reward, state, and last action taken
        :param mode: not used, needed to overwrite the abstract method though
        """
        if mode == "human":
            print("Reward:", self.reward, "State (v,g,s):", self.IB.visibleState()[1:4], "Action: ", self.last_action)
        else:
            raise ValueError(f"Invalid mode '{mode}'")

    def _update_observation(self) -> np.ndarray:
        """
        gets the new observation from the IDS environment and updates own representation as part of the step method
        :return: the new observation representation
        """

        # when the observation type is classic, an observation consists of a single state frame
        if self.observation_type == "classic":
            return self.IB.visibleState()[:-2]

        # when the observation type is include_past, an observation consists of self.n_past_timesteps state frames
        if self.observation_type == "include_past":
            # when the env has just been reset, observation is an empty list. Otherwise it containes
            # self.n_past_timesteps state frames, and we remove the oldest so that we have room for a new one
            if self.observation is not None:
                # insert new observation at the beginning
                self.observation = [self.IB.visibleState()[:-2]] + self.observation[:-1]
            else:
                # when the env has just been created, there aren't self.n_past_timesteps state frames available yet
                # thus, we repeat the oldes (only) state frame self.n_past_timesteps times
                self.observation = [self.IB.visibleState()[:-2] for _ in range(self.n_past_timesteps)]
            return np.hstack(self.observation)  # return observation is a single flattened numpy.ndarray

        raise ValueError('Invalid observation_type. observation_type can either be "classic" or "include_past"')

    def _markovian_state(self):
        """
        get the entire markovian state for debugging purposes
        :return: markov state as a dctionary
        """
        return {
            "setpoint": self.IB.state["p"],
            "velocity": self.IB.state["v"],
            "gain": self.IB.state["g"],
            "shift": self.IB.state["h"],
            "fatigue": self.IB.state["f"],
            "consumption": self.IB.state["c"],
            "op_cost_t0": self.IB.state["o"][0],
            "op_cost_t1": self.IB.state["o"][1],
            "op_cost_t2": self.IB.state["o"][2],
            "op_cost_t3": self.IB.state["o"][3],
            "op_cost_t4": self.IB.state["o"][4],
            "op_cost_t5": self.IB.state["o"][5],
            "op_cost_t6": self.IB.state["o"][6],
            "op_cost_t7": self.IB.state["o"][7],
            "op_cost_t8": self.IB.state["o"][8],
            "op_cost_t9": self.IB.state["o"][9],
            "ml1": self.IB.state["gs_domain"],
            "ml2": self.IB.state["gs_sys_response"],
            "ml3": self.IB.state["gs_phi_idx"],
            "hv": self.IB.state["hv"],
            "hg": self.IB.state["hg"],
        }
