import numpy as np
import matplotlib.pyplot as plt
import gym
import random
import time
from IPython.display import clear_output
from matplotlib.ticker import MultipleLocator

env = gym.make("MountainCar-v0")

action_space_size = env.action_space.n
state_space_size = np.round((env.observation_space.high - env.observation_space.low) *
                            np.array([10, 100]), 0).astype(int) + 1

q_table = np.zeros((state_space_size[0], state_space_size[1], action_space_size))

total_episodes = 10000
alpha = 0.1
gamma = 0.9
epsilon = 1
max_epsilon = 1
min_epsilon = 0
decay_rate = 0.001


def epsilon_greedy_policy(state, eps, Q, state_adj):
    if np.random.random() < 1 - eps:
        action = np.argmax(Q[state_adj[0], state_adj[1]])
    else:
        action = np.random.randint(0, env.action_space.n)
    return action


def updateQValue(reward, Q, state_adj, state2_adj, action, eps):
    # Maximum policy
    delta = alpha * (reward + gamma * np.mean(Q[state2_adj[0], state2_adj[1]]) -
                     Q[state_adj[0], state_adj[1], action])
    Q[state_adj[0], state_adj[1], action] += delta
    return Q


def updateByExpected(reward, Q, state_adj, state2_adj, action, eps):
    # Expected value
    predict = Q[state_adj[0], state_adj[1], action]
    expected_q = 0
    q_max = np.max(Q[state2_adj[0], state2_adj[1]])
    greedy_actions = 0

    for i in range(action_space_size):
        if Q[state2_adj[0], state2_adj[1]][i] == q_max:
            greedy_actions += 1

    non_greedy_action_probability = eps / action_space_size
    greedy_action_probability = ((1 - eps) / greedy_actions) + non_greedy_action_probability

    for i in range(action_space_size):
        if Q[state2_adj[0], state2_adj[1]][i] == q_max:
            expected_q += Q[state2_adj[0], state2_adj[1]][i] * greedy_action_probability
        else:
            expected_q += Q[state2_adj[0], state2_adj[1]][i] * non_greedy_action_probability

    target = reward + gamma * expected_q
    Q[state_adj[0], state_adj[1], action] += alpha * (target - predict)
    return Q


# Define Q-learning function
def QLearning(episodes, policy='max'):
    global epsilon
    # Initialize Q table
    Q = q_table

    # Initialize variables to track rewards and steps
    reward_list = []
    ave_reward_list = []
    step_list = []
    steps_list = []
    ave_step_list = []
    eps_list = []

    # Run Q learning algorithm
    for episode in range(episodes):
        # Initialize parameters
        done = False
        tot_reward, reward, steps = 0, 0, 0
        state = env.reset()

        # Discretize state
        state_adj = (state - env.observation_space.low) * np.array([10, 100])
        state_adj = np.round(state_adj, 0).astype(int)

        while not done:
            # Determine next action - epsilon greedy strategy
            action = epsilon_greedy_policy(state, epsilon, Q, state_adj)

            # Get next state and reward
            state2, reward, done, info = env.step(action)

            # Discretize state2
            state2_adj = (state2 - env.observation_space.low) * np.array([10, 100])
            state2_adj = np.round(state2_adj, 0).astype(int)

            # Allow for terminal states
            if done and state2[0] >= 0.5:
                Q[state_adj[0], state_adj[1], action] = reward

            # Adjust Q value for current state
            else:
                if policy == 'max':
                    Q = updateQValue(reward, Q, state_adj, state2_adj, action, epsilon)
                elif policy == 'expected':
                    Q = updateByExpected(reward, Q, state_adj, state2_adj, action, epsilon)

            # Update variables
            tot_reward += reward
            state_adj = state2_adj
            steps += 1

        if steps == 200:
            eps_list.append(epsilon)
        else:
            eps_list.append(0)

        # Decay epsilon
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

        # Track rewards, step and epsilon
        reward_list.append(tot_reward)
        step_list.append(steps)
        steps_list.append(steps)

        if (episode + 1) % 100 == 0:
            ave_reward = np.mean(reward_list)
            ave_step = np.mean(step_list)
            ave_reward_list.append(ave_reward)
            ave_step_list.append(ave_step)
            reward_list, step_list = [], []

        if (episode + 1) % 100 == 0:
            print(
                'Episode {} Average Reward: {} Average Steps: {} epsilon: {}'.format(episode + 1, ave_reward, ave_step,
                                                                                     epsilon))

    env.close()

    return ave_reward_list, steps_list, eps_list


def plotData(data, title1='', title2='', y1=-200, y2=-110, ep=100):
    plt.plot(ep * (np.arange(len(data)) + 1), data)
    plt.xlabel('Episodes')
    plt.ylabel(title1)
    plt.title(title2)
    plt.ylim(y1, y2)
    plt.show()


def multi_plot_data(data, names, title1='', title2='', y1=-200, y2=-110):
    x = np.arange(len(data[0]))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i, y in enumerate(data):
        plt.plot(x, y, '-', markersize=2, label=names[i])
    plt.legend(loc='upper right', prop={'size': 16}, numpoints=10)
    plt.xlabel('Episodes')
    plt.ylabel(title1)
    ax.set_title(title2)
    plt.ylim(y1, y2)
    plt.show()


def experiment1():
    rewards, steps, Q = QLearning(total_episodes)
    plotData(rewards, 'Average Reward', 'Average Reward vs. Episodes')


def experiment2():
    global alpha
    alpha = 0.09
    rewards, steps, Q = QLearning(total_episodes)
    plotData(rewards, 'Average Reward', 'Q-Learning with  alpha = {}'.format(alpha))


def experiment3():
    global gamma
    gammas = [0.2, 0.6]
    data, names = [], []
    for d in gammas:
        gamma = d
        rewards, steps, Q = QLearning(total_episodes)
        data.append(rewards)
        names.append(d)
    multi_plot_data(data, names, 'Gamma', 'Q-Learning with Different Gamma')


def experiment4():
    global epsilon, max_epsilon, decay_rate, gamma
    epsilon, max_epsilon, decay_rates, gamma = 0.5, 0.5, [0.001, 0.01], 0.99
    data, names = [], []
    for dr in decay_rates:
        decay_rate = dr
        rewards, _, eps_list = QLearning(total_episodes)
        data.append(eps_list)
        names.append(dr)
        # plotData(rewards, 'Average Reward', 'Q-Learning with  epsilon = {} and decay rate = {}'.format(0.5, dr))
    multi_plot_data(data, names, 'Epsilon', 'Epsilon vs. Episodes', y1=0, y2=1)


def experiment5():
    global gamma
    gamma = 0.99
    rewards, steps, _ = QLearning(total_episodes)
    plotData(steps, 'Average Steps', 'Average steps per episode', y1=80, y2=200, ep=1)


def experiment6():
    rewards, _, eps_list = QLearning(total_episodes)
    plotData(eps_list, 'Epsilon', 'Epsilon vs. Episodes', y1=0, y2=1, ep=1)


if __name__ == "__main__":
    experiments = {
        1:
            experiment1,
        2:
            experiment2,
        3:
            experiment3,
        4:
            experiment4,
        5:
            experiment5,
        6:
            experiment6
    }
    e = input("Enter experiment number:")
    experiment = experiments.get(int(e))
    experiment()
