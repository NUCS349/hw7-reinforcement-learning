# Coding (5 points)

Your task is to implement two reinforcement learning algorithms:

1.  Multi-armed bandits (in `code/multi_armed_bandits.py`)
1.  Q-Learning (in `code/q_learning.py`)

Your goal is to pass the test suite (contained in `tests/`). Once the tests are passed, you
may move on to the next part - reporting your results.

Your grade for this section is defined by the autograder. If it says you got an 80/100,
you get 4 points here. Suggested order for passing test_cases:

1. test_bandit_slots
2. test_bandit_frozen_lake
3. test_q_learning_slots
4. test_q_learning_frozen_lake
5. test_q_learning_deterministic


# Free-response questions (5 points)
To answer some of these questions, you will have to write extra code (that is not covered by the test cases). You may include your experiments in new files in the `experiments` directory. See `experiments/example.py` for an example. You can run any experiments you create within this directory with `python -m experiments.<experiment_name>`. For example, `python -m experiments.example` runs the example experiment.

## 1. (0.5 points) Tic-Tac-Toe
Here we will formulate Tic-Tac-Toe as an environment in which we can train a reinforcement learning agent. You will play as X's, and your opponent will be O's. Two-player games such as Tic-Tac-Toe are often modeled using *game theory*, in which we try and predict the moves of our opponent as well. For simplicity, we ignore the modeling of the opponent moves and treat our opponent's actions as a source of randomness within the environment. Assume you always go first.
   - a. (0.25 points) What are the states and actions within the Tic-Tac-Toe reinforcement learning environment? How does the current state affect the actions you can take?
   - b. (0.25 points) Design a reward function for teaching a reinforcement learning agent to play optimally in the Tic-Tac-Toe environment. Your reward function should specify a reward value for each of the 3 possible ways that a game can end (win, loss, or draw) as well as a single reward value for actions that do not result in the end of the game (e.g., your starting move). For actions that do not end the game, should reward be given to your agent before or after your opponent plays?

## 2. (1.5 points) Bandits vs Q-Learning
Here we will setup an experiment to train both the `MultiArmedBandit` and `QLearning` models on the `SlotMachines` environment. Use the default values for `epsilon` and `discount`.
   - a. (0.25 points) Train 10 `MultiArmedBandit` learners, each for 10,000 steps. We will refer to each of the 10 independent trainings as one *trial*. Create one plot with 3 lines on it. Plot as the first line the `rewards` array (the second return value from the `fit` function you implemented in the code) from the first trial. For the second line, average the `rewards` arrays learned in the first 5 independent trials. The resulting averaged `rewards` array should be the element-wise average over the first 5 trials of `MultiArmedBandit` and should be of length 100. For your third line, repeat what you did to create the second line, but this time use all 10 trials. Label each line on your plot with the number of trials that were averaged over to create the line.
   - b. (0.25 points) Now train 10 `QLearning` learners, each for 10,000 steps, on the `SlotMachines` environment. Plot the averaged  `QLearning` `rewards` array over all 10 trials that used `QLearning` and the averaged `MultiArmedBandit` `rewards` array over all 10 trials that used `MultiArmedBandit` **on the same plot**. Make sure to label each line in your plot with its associated learner.
   - c. (0.25 points) Look at your plot from question 2.a. Why is it important that we average over multiple independent trials for the same learner? What happened to the jaggedness of the line (the variance) as the number of trials increased?
   - d. (0.25 points) Look at your plot from question 2.b. How does the reward obtained by the two learners differ on the `SlotMachines` environment? Does one learner appear to be significantly better (i.e., obtain higher reward) than the other?
   - e. (0.5 points) Replicate your plot from question 2.b, but this time train both learners for 10 trials on the `FrozenLake-v0` environment. Include your plot and answer the following questions:
      - 1. (0.25 points) How does the reward obtained by the two learners differ on the `FrozenLake-v0` environment? Does one learner appear to be significantly better (i.e., obtain higher reward) than the other?
      - 2. (0.25 points) The best action to take within the `FrozenLake-v0` is state-dependent. For example, there are some states in which moving to the right will cause you to fall into the water. In these states, moving right is a bad choice of action. Look at your plot from question 2.e. You should observe that one of your learners performed poorly on `FrozenLake-v0`. Identify which learner was unable to learn the environment (i.e., underfitting) and describe why that underfitting occurs due to the limited hypothesis space of that learner.

## 3. (1.5 points) Exploration vs Exploitation
   - a. (0.75 points) Setup an experiment to train the `QLearning` model on the `FrozenLake-v0` environment for values of $\epsilon \in [0, 0.001, 0.01, 0.1, 0.5]$. For each value, train 10 `QLearning` learners, each for 10,000 steps (i.e., `steps=10000`), with the default discount rate of 0.95. See `test_q_learning` for an example of how to run your model on an OpenAI Gym Environment. For each value of epsilon, plot the averaged `rewards` arrays on one plot. Label each line in your plot with its associated $\epsilon$ value. Include your plot and answer the following questions:

      - 1. (0.25 points) For which value of $\epsilon$ is the averaged `rewards` of the `QLearning` learner maximized on the `FrozenLake-v0` environment? Note: we are asking for the value of epsilon that produces the largest reward value during training throughout your plot.
      - 2. (0.25 points) For which value of $\epsilon$ did the averaged `rewards` increase the fastest at the start? For which value did it increase the slowest?
      - 3. (0.25 points) Given your answers to the previous questions, should values of $\epsilon$ increase or decrease as the agent improves? What does this say about the tradeoff between exploration and exploitation over time within the training process? *An aside: note that we are only evaluated $\epsilon$ during the training process. In practice, when choosing a "best" value of epsilon, it is important to evaluate which value of epsilon maximizes reward during prediction. You can evaluate this by running your predict function for, e.g., 1000 episodes for each value of epsilon and determining which value of epsilon produced the highest average reward. For simplicity, we are omitting this step.*

   - b. (0.75 points) Based on your answers to question 3.a, implement the `_adaptive_epsilon()` function in `q_learning.py` to update the value of $\epsilon$ as training progresses. You may use any initial value of $\epsilon$. Train 10 `QLearning` learners, each for 10,000 steps, with the default discount rate of 0.95. Plot the averaged `rewards` array for your adaptive epsilon **on the same plot** that you created in question 3.a. Include your plot and answer the following questions:

      - 1. (0.25 points) Describe the update function you implemented in `_adaptive_epsilon`. Justify why you chose this function (and your initial value) based on your answers to part a.
      - 2. (0.25 points) Compare your results to your plot from part a. How does your average reward compare to the average reward for static (non-adaptive) values of $\epsilon$?
      - 3. (0.25 points) Consider the following modification to the `FrozenLake-v0` environment: every $n$ steps, the location of the holes and the goal move to random locations. Assume the goal is still accessible. How should $\epsilon$ change over time to accommodate the stochasticity in this environment?

## 4. (0.5 points) On-Policy vs Off-Policy
   - a. (0.25 points) Describe in your own words the difference between on-policy and off-policy learners. Provide an example of each.
   - b. (0.25 points) Consider the following environment: your agent is placed next to a cliff and must get to the goal. The shortest path to the goal is to move along the edge of the cliff. There is also a longer path to the goal that requires the agent to first move away from the cliff, and then towards the goal. The reward for reaching the goal is 100 points, and the reward for falling of the cliff is -1000 points. Assume we use an $\epsilon$-greedy policy for exploration. If we would like to learn the shortest path, should we use an on-policy or off-policy algorithm? Explain why. Note: reading chapter 6 of Sutton & Barto will help you answer this question.

## 5. (1 point) Markov Decision Processes
   - a. (0.25 points) Consider the following environment: you are designing a trash-collecting robot. The robot is rewarded 1 point for each piece of trash it collects, and 10 points for each piece of trash it places in the trash bin. Assume we modify the environment so that there is a 10% chance at each time step of the robot suddenly failing. If the robot fails, it can no longer increase its reward. How does this modification affect the optimal behavior of the robot? Describe how to modify the MDP formulation to account for these spurious failures.
   - b. (0.75 points) Consider the following environment represented as a directed graph, in which circles represent states, double circles represent the goal, and arrows represent actions. Assume you start at state 0. Assume all actions are deterministic. Rewards are 0 for all states except the goal, which has a reward of 1.

      ![Markov Decision Process](https://github.com/NUCS349/hw5-reinforcement-learning/blob/master/images/mdp.png "Markov Decision Process")

      - 1. (0.25 points) What are the optimal state-values and state-action-values for this environment?
      - 2. (0.25 points) What is the optimal policy for this environment?
      - 3. (0.25 points) Assume we introduce a discount factor of 0.95 into our value functions. Determine the new values of the optimal value functions as well as the optimal policy. Describe the effect of the discount factor on the optimal policy.
