import plotly.express as px


import numpy as np
import matplotlib.pyplot as plt

dueling_arr = np.loadtxt("avd_reward_dqn.txt")
double_arr = np.loadtxt("avd_reward_double_dqn.txt")
normal_arr = np.loadtxt("avd_reward_normal_dqn.txt")

# dueling_arr = np.mean(dueling_arr.reshape(-1, 500), axis=1)
# double_arr = np.mean(double_arr.reshape(-1, 500), axis=1)
# normal_arr = np.mean(normal_arr.reshape(-1, 500), axis=1)
# Creating the Figure instance
fig = px.line(y=[dueling_arr, double_arr, normal_arr], labels={"value": "Dueling DQN", "variable": "Algorithm"})
fig.show()
# printing the figure instance
# print(fig)