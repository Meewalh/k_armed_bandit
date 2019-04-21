import numpy as np


class BanditAutomaton:
    def __init__(self, k):
        assert isinstance(k, int)
        self.true_values = np.zeros(k)
        for i in range(0, k):
            # unit variance (=1), mean is zero
            self.true_values[i] = np.random.normal(0, 1, 1)
        self.highest_lever = np.argmax(self.true_values)  # this is needed for the evaluation of the agent's performance

    def use_lever(self, k: int):
        return np.random.normal(self.true_values[k], 1, 1)
