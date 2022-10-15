import numpy as np
import matplotlib.pyplot as plt

dueling_arr = np.loadtxt("avd_reward_dqn.txt")
double_arr = np.loadtxt("avd_reward_double_dqn.txt")
normal_arr = np.loadtxt("avd_reward_normal_dqn.txt")

dueling_arr1 = np.mean(dueling_arr.reshape(-1, 500), axis=1)
double_arr1 = np.mean(double_arr.reshape(-1, 500), axis=1)
normal_arr1 = np.mean(normal_arr.reshape(-1, 500), axis=1)

plt.plot(dueling_arr, label='Dueling DQN')
plt.plot(double_arr, label='Double DQN')
plt.plot(normal_arr, label='DQN')

plt.xlabel('Every 500 Episode')
plt.ylabel('Average Reward')
plt.legend()
plt.show()