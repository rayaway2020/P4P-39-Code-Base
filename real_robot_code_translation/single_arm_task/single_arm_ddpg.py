
import torch
import torch.nn as nn
import torch.optim as optim
from single_arm_env import SingleArmEnv
import time
import matplotlib.pyplot as plt


from collections import deque
import random
import numpy as np


class MemoryClass:
    def __init__(self, replay_max_size):
        self.replay_max_size = replay_max_size
        self.replay_buffer = deque(maxlen=replay_max_size)  # buffer of experiences to sample during training

    def replay_buffer_add(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.replay_buffer.append(experience)

    def sample_experience(self, batch_size):
        state_batch      = []
        action_batch     = []
        reward_batch     = []
        next_state_batch = []
        done_batch       = []

        batch = random.sample(self.replay_buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.replay_buffer)


class CriticDDPG(nn.Module):
    def __init__(self, input_size, hidden_size, num_actions):
        super(CriticDDPG, self).__init__()

        self.h_linear_1 = nn.Linear(in_features=input_size,     out_features=hidden_size[0])
        self.h_linear_2 = nn.Linear(in_features=hidden_size[0], out_features=hidden_size[1])
        self.h_linear_3 = nn.Linear(in_features=hidden_size[1], out_features=hidden_size[2])
        self.h_linear_4 = nn.Linear(in_features=hidden_size[2], out_features=hidden_size[3])
        self.h_linear_5 = nn.Linear(in_features=hidden_size[3], out_features=num_actions)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)   # Concatenates the seq tensors in the given dimension
        x = torch.relu(self.h_linear_1(x))
        x = torch.relu(self.h_linear_2(x))
        x = torch.relu(self.h_linear_3(x))
        x = torch.relu(self.h_linear_4(x))
        x = self.h_linear_5(x)                  # No activation function here
        return x


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Actor, self).__init__()

        self.h_linear_1 = nn.Linear(in_features=input_size,     out_features=hidden_size[0])
        self.h_linear_2 = nn.Linear(in_features=hidden_size[0], out_features=hidden_size[1])
        self.h_linear_3 = nn.Linear(in_features=hidden_size[1], out_features=hidden_size[2])
        self.h_linear_4 = nn.Linear(in_features=hidden_size[2], out_features=hidden_size[3])
        self.h_linear_5 = nn.Linear(in_features=hidden_size[3], out_features=output_size)

    def forward(self, state):
        x = torch.relu(self.h_linear_1(state))
        x = torch.relu(self.h_linear_2(x))
        x = torch.relu(self.h_linear_3(x))
        x = torch.relu(self.h_linear_4(x))
        x = torch.tanh(self.h_linear_5(x))
        return x



class DDPGagent_SingleArm:

    def __init__(self, env, actor_learning_rate=1e-4, critic_learning_rate=1e-5, gamma=0.99,
                 max_memory_size=10_000, tau=0.005, batch_size=64):

        # -------- Hyper-parameters --------------- #
        self.max_memory_size = max_memory_size
        self.batch_size = batch_size

        self.update_interaction = 1  # number of updates of NN per Episode

        self.gamma = gamma  # discount factor
        self.tau = tau
        self.hidden_size_critic = [64, 64, 64, 10]
        self.hidden_size_actor  = [32, 32, 32, 10]

        # -------- Parameters --------------- #
        self.num_states  = 6  # todo check this value
        self.num_actions = 2

        # ---------- Initialization and build the networks ----------- #
        # Main networks
        self.actor  = Actor(self.num_states, self.hidden_size_actor, self.num_actions)
        self.critic = CriticDDPG(self.num_states + self.num_actions, self.hidden_size_critic, self.num_actions)

        # Target networks
        self.actor_target  = Actor(self.num_states, self.hidden_size_actor, self.num_actions)
        self.critic_target = CriticDDPG(self.num_states + self.num_actions, self.hidden_size_critic, self.num_actions)

        # Initialization of the target networks as copies of the original networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

        self.memory = MemoryClass(max_memory_size)

    def get_action(self, state):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)  # numpy to a tensor with shape [1,3]
        action = self.actor.forward(state_tensor)
        action = action.cpu().data.numpy()
        return action[0]

    def get_action_exploration(self):
        act_m1 = np.clip(random.uniform(-1, 1), -1, 1)
        act_m2 = np.clip(random.uniform(-1, 1), -1, 1)
        action_vector = np.array([act_m1, act_m2])
        return action_vector


    def add_experience_memory(self, state, action, reward, next_state, done):
        # Save experience in memory
        self.memory.replay_buffer_add(state, action, reward, next_state, done)


    def step_training(self):
        # check, if enough samples are available in memory
        if self.memory.__len__() <= self.batch_size:
            return

        # update the networks every N times
        for it in range(self.update_interaction):

            states, actions, rewards, next_states, dones = self.memory.sample_experience(self.batch_size)

            states = np.array(states)
            actions = np.array(actions)
            rewards = np.array(rewards).reshape(-1, 1)
            dones = np.array(dones).reshape(-1, 1)
            next_states = np.array(next_states)

            states = torch.FloatTensor(states)
            actions = torch.FloatTensor(actions)
            rewards = torch.FloatTensor(rewards)
            dones = torch.FloatTensor(dones)
            next_states = torch.FloatTensor(next_states)

            # ----------------------------------- Calculate the loss ----- #
            Q_vals = self.critic.forward(states, actions)

            with torch.no_grad():
                next_actions = self.actor_target.forward(next_states)  # Note this is from actor-target
                next_Q_vales = self.critic_target.forward(next_states, next_actions)
                Q_target = rewards + (self.gamma * next_Q_vales)

            loss = nn.MSELoss()
            critic_loss = loss(Q_vals, Q_target)

            # ------- calculate the actor loss
            actor_loss = - self.critic.forward(states, self.actor.forward(states)).mean()

            # ------------------------------------- Update main networks --------------- #
            # Actor step Update
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Critic step Update
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # ------------------------------------- Update target networks --------------- #
            # update the target networks using tao "soft updates"
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    def save_model(self):
        torch.save(self.actor.state_dict(), 'weights/ddpg_actor_single_arm.pth')
        torch.save(self.critic.state_dict(), 'weights/ddpg_critic_single_arm.pth')
        print("models has been saved...")


def main_run():

    num_episodes      = 1_000
    episode_horizont  = 7
    batch_size        = 32

    rewards     = []
    avg_rewards = []

    env   = SingleArmEnv()
    agent = DDPGagent_SingleArm(env, batch_size=batch_size)

    epsilon       = 1.0
    epsilon_min   = 0.01
    epsilon_decay = 0.003


    for episode in range(1, num_episodes+1):
        env.reset_env()
        episode_reward = 0

        if epsilon > epsilon_min:
            epsilon *= (1 - epsilon_decay)
        explore_probability = epsilon


        for step in range(1, episode_horizont+1):

            print(f"-------Episode:{episode} Step:{step}---------")

            state = env.state_space_funct()


            if np.random.random() <= explore_probability:
                action = agent.get_action_exploration()
                print("Exploration Stage")
            else:
                action = agent.get_action(state)
                print("Action_learned-------->:", action)
                noise  = np.random.normal(0, scale=0.08, size=2)
                action = action + noise
                action = np.clip(action, -1, 1)

            env.env_step(action)
            next_state = env.state_space_funct()
            reward, done = env.calculate_reward()
            agent.add_experience_memory(state, action, reward, next_state, done)
            episode_reward += reward
            if done:
                break

        agent.step_training()
        print(f"******* -----Episode {episode} Ended----- ********* ")
        print("Episode total reward:", episode_reward)

        rewards.append(episode_reward)
        avg_rewards.append(np.mean(rewards[-100:]))

    agent.save_model()

    plt.plot(rewards)
    plt.plot(avg_rewards)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    #plt.savefig("DDPG.png")
    plt.show()


if __name__ == '__main__':
    main_run()
