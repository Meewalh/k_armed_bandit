import numpy as np

import bandit_automaton


class LearningAgent:
    def __init__(self, number_levers: int, epsilon: int):
        # at the beginning all estimates are 0
        self.estimates = np.zeros(number_levers)
        self.times_lever_used = np.zeros(number_levers)

        assert 1 >= epsilon >= 0
        self.epsilon = epsilon

    def train_agent(self, automaton: bandit_automaton, steps: int):
        used_leavers = np.zeros(steps)
        for i in range(steps):

            # choose if this will be an exploration step
            if np.random.rand() <= self.epsilon:
                max_pos = np.random.random_integers(0, self.estimates.size - 1)
            else:
                max_pos = np.argmax(self.estimates)

            # choose action
            used_leavers[i] = max_pos
            value_this_round = automaton.use_lever(max_pos)
            self.times_lever_used[max_pos] += 1
            self.estimates[max_pos] = self.estimates[max_pos] + \
                                      (1 / self.times_lever_used[max_pos]) * \
                                      (value_this_round - self.estimates[max_pos])
        return used_leavers
