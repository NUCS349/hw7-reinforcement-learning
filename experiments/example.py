import gym
from src import MultiArmedBandit

print('Starting example experiment')

env = gym.make('FrozenLake-v0')
agent = MultiArmedBandit()
action_values, rewards = agent.fit(env)

print('Finished example experiment')
