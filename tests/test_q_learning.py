import gym
import numpy as np


def test_q_learning_slots():
    """
    Tests that the Q-learning implementation successfully finds the slot
    machine with the largest expected reward.
    """
    from code import QLearning

    np.random.seed(0)

    env = gym.make('SlotMachines-v0', n_machines=10, mean_range=(-10, 10), std_range=(5, 10))
    env.seed(0)
    means = np.array([m.mean for m in env.machines])

    agent = QLearning(epsilon=0.2, discount=0)
    state_action_values, rewards = agent.fit(env, steps=10000)

    assert state_action_values.shape == (1, 10)
    assert len(rewards) == 100

    assert np.argmax(means) == np.argmax(state_action_values)


def test_q_learning_frozen_lake():
    """
    Tests that the MultiArmedBandit implementation successfully finds the slot
    machine with the largest expected reward.
    """
    from code import QLearning

    np.random.seed(0)

    env = gym.make('FrozenLake-v0')
    env.seed(0)

    agent = QLearning(epsilon=0.2, discount=0.95)
    state_action_values, rewards = agent.fit(env, steps=10000)

    state_values = np.mean(state_action_values, axis=1)

    assert state_action_values.shape == (16, 4)
    assert len(rewards) == 100

    assert np.allclose(state_values[np.array([5, 7, 11, 12])], np.zeros(4))
    assert np.all(state_values[np.array([0, 1, 2, 3, 4, 6, 8, 9, 10, 13, 14])] > 0)
