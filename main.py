from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

import bandit_automaton
import learning_agent


def count_occurrences(array, value):
    counter = 0
    for x in array:
        if x == value:
            counter += 1
    return counter


def used_correct(array, value):
    correct = np.zeros(array.size)
    for i in range(0, array.size):
        if value == array[i]:
            correct[i] = 1
    return correct


# all parameters that can be changed:
number_levers: int = 10  # this is the parameter k
number_steps: int = 1000
number_iterations: int = 100
epsilon_values = [0.01, 0.05, 0.1, 0.25] # the performance is plotted for each given epsilon value

time_begin = datetime.now()

for current_epsilon in epsilon_values:
    results = np.zeros(number_steps)

    for iteration in range(0, number_iterations):
        # print("Starting iteration", iteration)
        # create automaton and agent for the current iteration
        automaton = bandit_automaton.BanditAutomaton(number_levers)
        agent = learning_agent.LearningAgent(number_levers, current_epsilon)

        # train the agent and save the results
        used_leavers = agent.train_agent(automaton, number_steps)
        temp = used_correct(used_leavers, automaton.highest_lever)
        results = results + temp

    plt.plot(results / number_iterations, label=str(current_epsilon))

time_end = datetime.now()

print("Time needed: ", (time_end - time_begin))

plt.xlabel("Steps")
plt.ylabel("% optimal action")
plt.legend(loc='upper left')
plt.show()
