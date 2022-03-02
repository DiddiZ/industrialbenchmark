import unittest
from pathlib import Path

import numpy as np
import gym

from industrial_benchmark_python.IDS import IDS
from industrial_benchmark_python.IBGym import IBGym


class TestIB(unittest.TestCase):

    def all_states(self, env):
        return [
            env.state["o"][0], env.state["o"][1], env.state["o"][2], env.state["o"][3], env.state["o"][4], env.state["o"][5],
            env.state["o"][6], env.state["o"][7], env.state["o"][8], env.state["o"][9], env.state["coc"], env.state["hg"], env.state["hv"],
            env.state["he"], env.state["gs_domain"], env.state["gs_sys_response"], env.state["gs_phi_idx"], env.state["ge"],
            env.state["ve"], env.state["MC"], env.state["c"], env.state["p"], env.state["v"], env.state["g"], env.state["h"],
            env.state["f"], env.state["fb"], env.state["oc"], env.state["reward"]
        ]

    def test_example(self):
        trajectories = 10
        T = 1000  # perform 1000 actions/ steps

        # generate different values of setpoint
        p = [88, 32, 40, 80, 97, 78, 95, 84, 54, 69]

        # perform 1000 actions per trajectory
        for i in range(trajectories):
            # generate IB with fixed seed. If no seed is given, IB is generated with random values
            env = IDS(p[i], inital_seed=1005 + i)

            markovStates = np.empty((T, 29))
            for t in range(T):
                at = 2 * np.random.rand(3) - 1
                # perform action
                env.step(at)
                markovStates[t] = self.all_states(env)

            # test if test files and original files are equal
            np.testing.assert_array_almost_equal(
                np.genfromtxt(Path(__file__).parent / f"test_data/markovStates{i}.csv", delimiter=","),
                markovStates,
            )


class TestIBGym(unittest.TestCase):

    def test_action_space_discrete(self):
        env = IBGym(50, action_type="discrete")

        self.assertEqual(env.action_space, gym.spaces.Discrete(27))
        self.assertEqual(len(env.env_action), 27)
        self.assertListEqual(env.env_action[0], [-1, -1, -1])
        self.assertListEqual(env.env_action[-1], [1, 1, 1])

    def test_action_space_continuous(self):
        env = IBGym(50, action_type="continuous")

        np.testing.assert_array_equal(env.action_space.low, [-1, -1, -1])
        np.testing.assert_array_equal(env.action_space.high, [1, 1, 1])

    def test_observations_classic(self):
        env = IBGym(50, observation_type="classic")

        obs = env.reset()
        self.assertEqual(obs.shape, (6, ))

        obs, _, _, _ = env.step([1, 1, 1])
        self.assertEqual(obs.shape, (6, ))

    def test_observations_include_past(self):
        env = IBGym(50, observation_type="include_past", n_past_timesteps=30)

        obs = env.reset()
        self.assertEqual(obs.shape, (6 * 30, ))
        np.testing.assert_array_equal(obs, np.tile(obs[:6], (30, 1)).flatten())

        obs, _, _, _ = env.step([1, 1, 1])
        self.assertEqual(obs.shape, (6 * 30, ))
        np.testing.assert_array_equal(obs, np.concatenate([obs[:6], np.tile(obs[6:12], (29, 1)).flatten()]))
