# Coding (5 points)

Your task is to implement two reinforcement learning algorithms:

1.  Multi-armed bandits (in `code/multi_armed_bandits.py`)
1.  Q-Learning (in `code/q_learning.py`)

Your goal is to pass the test suite (contained in `tests/`). Once the tests are passed, you
may move on to the next part - reporting your results.

Your grade for this section is defined by the autograder. If it says you got an 80/100,
you get 4 points here. Suggested order for passing test_cases:

1. test_bandit_slots
2073. test_bandit_frozen_lake
3. test_q_learning_slots
4. test_q_learning_frozen_lake


# Free-response questions (5 points)
To answer some of these questions, you will have to write extra code (that is not covered by the test cases). You may include your experiments in new files in the `experiments` directory. See `experiments/example.py` for an example. You can run any experiments you create within this directory with `python -m experiments.<experiment_name>`. For example, `python -m experiments.example` runs the example experiment.

## 1. (1 point) Reinforcement Learning vs Supervised Learning
   - a. (0.25 points) Describe at least two differences between reinforcement learning and supervised learning.
   - b. (0.5 points) While inefficient, it is possible to formulate any supervised learning problem as a reinforcement learning problem. Describe how to perform this conversion from supervised learning to reinforcement learning. Specifically, describe the states, actions, policy, and reward function.

   ** Could this be made more specific? Pick a problem like handwritten digits on mnist with a perceptron? **

   - c. (0.25 points) Reinforcement learning can be thought of as a type of _active_ learning, in which the learner influences the subsequent sequence of examples used during training. Active learning can also be used in supervised learning to increase the speed of training by selecting more difficult examples to train on. Briefly describe how to incorporate active learning within a supervised learning algorithm. Specifically, describe how you would determine the "difficulty" of an example and how you would increase the impact of difficult examples within the learning process.

## 2. (1 point) Bandits vs Q-Learning
Set up an experiment to train both the `MultiArmedBandit` and `QLearning` models on the `SlotMachines` environment. Use the default values for `epsilon` and `discount`.
   - a. (0.25 points)  Train 10  `MultiArmedBandit` learners, each for 10,000 steps. Average the rewards arrays learned by the 10 learners. Plot the averaged `rewards` array.
   
   - b. (0.25 points) Repeat what you did for `MultiArmedBandit`, but this time for `QLearning.` Plot the averaged  `QLearning rewards` array ** on the same plot ** as the `MultiArmedBandit` rewards array. Make sure to label each line in your plot with its associated model. 
   
   Include your plot and answer the following questions:
      - 1. (0.25 points) Why is it important that we average over multiple independent trainings for the same learner? How does this affect the variance of the observed reward?
      - 2. (0.25 points) How does the perforamce of the two learners differ on the `SlotMachines` environment?

   - b. (0.5 points) Repeat the experiment performed in parts a and b, but this time use the `FrozenLake-v0` environment. Include your plot and answer the following questions:
      - 1. (0.25 points) How does the performance of the two learners differ on the `FrozenLake-v0` environment?
      - 2. (0.25 points) Look at the plot of the average rewards for `FrozenLake-v0`. From this,   identify which learner underfits and describe how that underfitting occurs via the inductive bias of that learner.

      ** QUESTION: What do you want them to learn? To detect underfitting by looking at a rewards function? To think about model capacity vs the problem? **

## 3. (1.5 points) Exploration vs Exploitation
   - a. (0.75 points) Setup an experiment to train the `QLearning` model on the `FrozenLake-v0` environment for values of $\epsilon \in [0, 0.001, 0.01, 0.1, 0.5]$. For each value, train the `QLearning` 10 times over 10000 steps with the default discount rate of 0.95. See `test_q_learning` for an example of how to run your model on an OpenAI Gym Environment. For each value of epsilon, plot the averaged `rewards` arrays on one plot. Label each line in your plot with its associated $\epsilon$ value. Include your plots and answer the following questions:
      - 1. (0.25 points) For which value of $\epsilon$ did `QLearning` achieve its maximum reward on the `FrozenLake-v0` environment?
      - 2. (0.25 points) For which value of $\epsilon$ did the average reward increase the fastest at the start? For which value did it increase the slowest?
      - 3. (0.25 points) Given your answers to the previous questions, should values of $\epsilon$ increase or decrease as the agent improves? What does this say about the tradeoff between exploration and exploitation over time within the training process?
   - b. (0.75 points) Based on your answers to part a of this question, implement the `_adaptive_epsilon()` function in `q_learning.py` to update the value of $\epsilon$ as training progresses. You may use any initial value of $\epsilon$. Train the `QLearning` learner 10 times over 10000 steps with the default discount rate of 0.95. Plot the averaged `rewards` array as in part a. Include your plot and answer the following questions:
      - 1. (0.25 points) Describe the update function you implemented in `_adaptive_epsilon`. Justify why you chose this function (and your initial value) based on your answers to part a.
      - 2. (0.25 points) Compare your results to your plot from part a. How does your average reward compare to the average reward for static values of $\epsilon$?
      - 3. (0.25 points) Consider the following modification to the `FrozenLake-v0` environment: every $n$ steps, the location of the holes and the goal move to random locations. Assume the goal is still accessible. How should $\epsilon$ change over time to accommodate the stochasticity in this environment?

## 4. (0.5 points) On-Policy vs Off-Policy
   - a. (0.25 points) Describe in your own words the difference between on-policy and off-policy learners. Provide an example of each.
   - b. (0.25 points) Consider the following environment: your agent is placed next to a cliff and must get to the goal. The shortest path to the goal is to move along the edge of the cliff. There is also a longer path to the goal that requires the agent to first move away from the cliff, and then towards the goal. The reward for reaching the goal is 100 points, and the reward for falling of the cliff is -1000 points. Assume we use an $\epsilon$-greedy policy for exploration. If we would like to learn the shortest path, should we use an on-policy or off-policy algorithm? Explain why.

## 5. (1 point) Markov Decision Processes
   - a. (0.25 points) Consider the following environment: you are designing a trash-collecting robot. The robot is rewarded 1 point for each piece of trash it collects, and 10 points for each piece of trash it places in the trash bin. Assume we modify the environment so that there is a 10% chance at each time step of the robot suddenly failing. If the robot fails, it can no longer increase its reward. How does this modification affect the optimal behavior of the robot? Describe how to modify the MDP formulation to account for these spurious failures.
   - b. (0.75 points) Consider the following environment, in which circles represent states, double circles represent the goal, and arrows represent actions. Assume you start at state 0. Assume all actions are deterministic. Rewards are 0 for all states except the goal, which has a reward of 1.
      ![Markov Decision Process](https://github.com/NUCS349/hw8-reinforcement-learning-maxrmorrison/blob/master/images/mdp.png "Markov Decision Process")
      - 1. (0.25 points) What are the optimal state-values and state-action-values for this environment?
      - 2. (0.25 points) What is the optimal policy for this environment?
      - 3. (0.25 points) Assume we introduce a discount factor of 0.95 into our value functions. Determine the new values of the optimal value functions as well as the optimal policy. Describe the effect of the discount factor on the optimal policy.
