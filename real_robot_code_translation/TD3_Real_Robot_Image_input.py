
'''
TD3 RL algorithm using images as state-space input
task = move the cube
'''

import torch
import torch.nn as nn
import torch.optim as optim
import random

from main_rl_env import RL_ENV
from Memory import MemoryClass
from Networks import Actor_Img, Critic_Img

import numpy as np
import matplotlib.pyplot as plt

import cv2
from collections import deque


class TD3agentimage(object):

    def __init__(self, env, batch_size=32, max_memory_size=50_000, gamma=0.99, critic_learning_rate=1e-3,
                 actor_learning_rate=1e-4, tau=0.005):

        # -------- Hyper-parameters --------------- #
        self.gamma = gamma  # discount factor
        self.max_memory_size = max_memory_size
        self.batch_size = batch_size
        self.update_interaction = 6
        self.policy_freq_update = 2
        self.tau = tau
        self.update_counter = 0

        # -------- Parameters --------------- #
        self.action_dim = 4

        # ------------- Initialization memory --------------------- #
        self.memory = MemoryClass(self.max_memory_size)

        # ---------- Initialization and build the networks ----------- #
        # Main Networks
        self.actor     = Actor_Img(self.action_dim)
        self.critic_q1 = Critic_Img(self.action_dim)
        self.critic_q2 = Critic_Img(self.action_dim)

        # Target networks
        self.actor_target     = Actor_Img(self.action_dim)
        self.critic_target_q1 = Critic_Img(self.action_dim)
        self.critic_target_q2 = Critic_Img(self.action_dim)

        # Initialization of the target networks as copies of the original networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target_q1.parameters(), self.critic_q1.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target_q2.parameters(), self.critic_q2.parameters()):
            target_param.data.copy_(param.data)

        self.actor_optimizer    = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer_1 = optim.Adam(self.critic_q1.parameters(), lr=critic_learning_rate)
        self.critic_optimizer_2 = optim.Adam(self.critic_q2.parameters(), lr=critic_learning_rate)


    def get_action(self, state):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor.forward(state_tensor)
            action = action.detach()
            action = action.numpy()  # tensor to numpy
            self.actor.train()
        return action[0]

    def get_action_exploration(self):
        act_m1 = np.clip(random.uniform(-1, 1), -1, 1)
        act_m2 = np.clip(random.uniform(-1, 1), -1, 1)
        act_m3 = np.clip(random.uniform(-1, 1), -1, 1)
        act_m4 = np.clip(random.uniform(-1, 1), -1, 1)
        action_vector = np.array([act_m1, act_m2, act_m3, act_m4])
        return action_vector

    def add_experience_memory(self, state, action, reward, next_state, done):
        # Save experience in memory
        self.memory.replay_buffer_add(state, action, reward, next_state, done)

    def step_training(self):
        # check, if enough samples are available in memory
        if self.memory.__len__() <= self.batch_size:
            return

        self.update_counter += 1

        # update the networks every N times
        for it in range(self.update_interaction):

            states, actions, rewards, next_states, dones = self.memory.sample_experience(self.batch_size)

            states = np.array(states)
            actions = np.array(actions)
            rewards = np.array(rewards).reshape(-1, 1)
            dones = np.array(dones).reshape(-1, 1)
            next_states = np.array(next_states)

            states  = torch.FloatTensor(states)
            actions = torch.FloatTensor(actions)
            rewards = torch.FloatTensor(rewards)
            dones   = torch.FloatTensor(dones)
            next_states = torch.FloatTensor(next_states)

            #print(states.size())       # --> [Batch_size, 4, 64, 64]
            #print(next_states.size())  # --> [Batch_size, 4, 64, 64]
            #print(actions.size())      # --> [Batch_size, 3]
            #print(rewards.size())      # --> [Batch_size, 1]
            #print(dones.size())        # --> [Batch_size, 1]

            # ------- compute the target action
            next_actions = self.actor_target.forward(next_states)

            # add noise also here, paper mention this
            next_actions = next_actions.detach().numpy()  # tensor to numpy
            next_actions = next_actions + (np.random.normal(0, scale=0.2, size=self.action_dim))
            next_actions = np.clip(next_actions, -1, 1)
            next_actions = torch.FloatTensor(next_actions)

            # compute next targets values
            next_Q_vales_q1 = self.critic_target_q1.forward(next_states, next_actions)
            next_Q_vales_q2 = self.critic_target_q2.forward(next_states, next_actions)

            q_min = torch.minimum(next_Q_vales_q1, next_Q_vales_q2)

            Q_target = rewards + (self.gamma * (1 - dones) * q_min).detach()

            loss = nn.MSELoss()

            Q_vals_q1 = self.critic_q1.forward(states, actions)
            Q_vals_q2 = self.critic_q2.forward(states, actions)

            critic_loss_1 = loss(Q_vals_q1, Q_target)
            critic_loss_2 = loss(Q_vals_q2, Q_target)

            # Critic step Update
            self.critic_optimizer_1.zero_grad()
            critic_loss_1.backward()
            self.critic_optimizer_1.step()

            self.critic_optimizer_2.zero_grad()
            critic_loss_2.backward()
            self.critic_optimizer_2.step()

            # TD3 updates the policy (and target networks) less frequently than the Q-function
            if self.update_counter % self.policy_freq_update == 0:

                # ------- calculate the actor loss
                actor_loss = - self.critic_q1.forward(states, self.actor.forward(states)).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # ------------------------------------- Update target networks --------------- #
                # update the target networks using tao "soft updates"
                for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                    target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

                for target_param, param in zip(self.critic_target_q1.parameters(), self.critic_q1.parameters()):
                    target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

                for target_param, param in zip(self.critic_target_q2.parameters(), self.critic_q2.parameters()):
                    target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))


    def save_model(self):
        torch.save(self.actor.state_dict(), 'weights/td3_actor_cube_image_t_1.pth')
        torch.save(self.critic_q1.state_dict(), 'weights/td3_critic_1_cube_img_t_1.pth')
        torch.save(self.critic_q2.state_dict(), 'weights/td3_critic_2_cube_img_t_1.pth')
        print("models has been saved...")

    def load_model(self):
        self.actor.load_state_dict(torch.load('weights/td3_actor_cube_image.pth'))
        self.critic_q1.load_state_dict(torch.load('weights/td3_critic_1_cube_img.pth'))
        self.critic_q2.load_state_dict(torch.load('weights/td3_critic_2_cube_img.pth'))
        print("models has been loaded...")


def crop_normalize_observation(observation):

    crop_image = observation[140:240, 70:250]
    resized = cv2.resize(crop_image, (64, 64), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    norm_image = cv2.normalize(gray, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    cv2.imshow("input", norm_image)
    cv2.imshow("State visual", crop_image)
    cv2.waitKey(10)

    return norm_image


def main_run():

    num_episodes = 10_000
    episode_horizont = 6
    batch_size = 64

    rewards     = []
    avg_rewards = []

    env   = RL_ENV()
    agent = TD3agentimage(env, batch_size=batch_size)

    success_counter = 0
    step_counter = 0


    for episode in range(1, num_episodes+1):

        env.reset_env()

        initial_frame = env.graphical_state_space_funct()
        initial_frame = crop_normalize_observation(initial_frame)

        state_frame_queue = deque([initial_frame] * 3, maxlen=3)
        episode_reward = 0
        done = False

        for step in range(1, episode_horizont + 1):
            step_counter += 1
            print(f"-------Episode:{episode} Step:{step}---------")

            current_state = np.array(state_frame_queue)  # (4,64,64)

            if step_counter < 1500:
                print("exploration")
                action = agent.get_action_exploration()
            else:
                action = agent.get_action(current_state)
                action = action + (np.random.normal(0, scale=0.1, size=4))
                action = np.clip(action, -1, 1)
                print("action learned:", action)

            env.env_step(action)

            next_frame_state = env.graphical_state_space_funct()
            next_frame_state = crop_normalize_observation(next_frame_state)

            state_frame_queue.append(next_frame_state)
            next_state = np.array(state_frame_queue)

            reward, done = env.calculate_reward_continuous()
            agent.add_experience_memory(current_state, action, reward, next_state, done)

            episode_reward += reward
            if done:
                success_counter += 1
                print("Success Counter:", success_counter)
                break

        agent.step_training()  # Update the NNs
        print(f"******* -----Episode {episode} Ended-----********* ")
        print("Episode total reward:", episode_reward)

        rewards.append(episode_reward)
        avg_rewards.append(np.mean(rewards[-100:]))

    agent.save_model()
    env.close_env()  # close the usb port and disable torquecle

    # save point to plot later
    np.savetxt('data_to_plot/rewards_t1_imag.txt', rewards)
    np.savetxt('data_to_plot/avd_reward_t1_imag.txt', avg_rewards)

    plt.plot(rewards)
    plt.plot(avg_rewards)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig("TD3_image_input_t1.png")
    plt.show()


def test_mode():

    num_episodes = 50
    episode_horizont = 6
    rewards     = []
    env   = RL_ENV()
    agent = TD3agentimage(env, batch_size=64)
    agent.load_model()


    for episode in range(num_episodes):
        env.reset_env()
        initial_frame = env.graphical_state_space_funct()
        initial_frame = crop_normalize_observation(initial_frame)
        state_frame_queue = deque([initial_frame] * 3, maxlen=3)
        episode_reward = 0

        for step in range(1, episode_horizont + 1):

            print(f"-------Episode:{episode} Step:{step}---------")

            current_state = np.array(state_frame_queue)
            action = agent.get_action(current_state)
            action = np.clip(action, -1, 1)
            env.env_step(action)

            next_frame_state = env.graphical_state_space_funct()
            next_frame_state = crop_normalize_observation(next_frame_state)
            state_frame_queue.append(next_frame_state)
            next_state = np.array(state_frame_queue)
            reward, done = env.calculate_reward()

            episode_reward += reward

            if done:
                break

        print(f"******* -----Episode {episode} Ended-----********* ")
        print("Episode total reward:", episode_reward)

        rewards.append(episode_reward)

    plt.plot(rewards)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()


if __name__ == '__main__':
    main_run()
    #test_mode()
