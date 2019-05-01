import numpy as np


class QLearning:
    """
    QLearning reinforcement learning agent.

    Arguments:
      epsilon - (float) The probability of randomly exploring the action space
        rather than exploiting the best action.
      discount - (float) The discount factor. Controls the perceived value of
        future reward relative to short-term reward.
      adaptive - (bool) Whether to use an adaptive policy for setting
        values of epsilon during training
    """

    def __init__(self, epsilon=0.01, discount=0.95, adaptive=False):
        self.epsilon = epsilon
        self.discount = discount
        self.adaptive = adaptive

    def fit(self, env, steps=1000):
        """
        Trains an agent using Q-Learning on an OpenAI Gym Environment.

        See page 131 of Sutton and Barto's book Reinformcement Learning for
        pseudocode (http://incompleteideas.net/book/RLbook2018.pdf).
        Initialize your parameters as all zeros. For the step size (alpha), use
        1 / N, where N is the number of times the current action has been
        performed in the current state. Use an epsilon-greedy policy for action
        selection.

        See (https://gym.openai.com/) for examples of how to use the OpenAI
        Gym Environment interface.

        Hints:
          - Use env.action_space.n and env.observation_space.n to get the
            number of available actions and states, respectively.
          - Remember to reset your environment at the end of each episode.
          - If all values of a np.array are equal, np.argmax deterministically
            returns 0.
          - In order to avoid non-deterministic tests, use only np.random for
            random number generation.

        Arguments:
          env - (Env) An OpenAI Gym environment with discrete actions and
            observations. See the OpenAI Gym documentation for example use
            cases (https://gym.openai.com/docs/).
          steps - (int) The number of actions to perform within the environment
            during training.

        Returns:
          state_action_value - (np.array) The values assigned by the algorithm
            to each state-action pair as a 2D numpy array. The dimensionality
            of the numpy array should be S x A, where S is the number of
            states in the environment and A is the number of possible actions.
          rewards - (np.array) A 1D sequence of averaged rewards of length 100.
            Let s = np.floor(steps / 100), then rewards[0] should contain the
            average reward over the first s steps, rewards[1] should contain
            the average reward over the next s steps, etc.
        """
        raise NotImplementedError()

    def _get_epsilon(self, progress):
        """
        Retrieves the current value of epsilon. Should be called by the fit
        function during each step.

        Arguments:
            progress - (float) A value between 0 and 1 that indicates the
                training progess. Equivalent to current_step / steps.
        """
        return self._adaptive_epsilon(progress) if self.adaptive else self.epsilon

    def _adaptive_epsilon(self, progress):
        """
        An adaptive policy for epsilon-greedy reinforcement learning. Returns
        the current epsilon value given the learner's progress. This allows for
        the amount of exploratory vs exploitatory behavior to change over time.

        See free response question 3 for instructions on how to implement this
        function.

        Arguments:
            progress - (float) A value between 0 and 1 that indicates the
                training progess. Equivalent to current_step / steps.
        """
        raise NotImplementedError()
