"""
TD3 RL algorithm using images as state-space input
task = move the cube
Version 2, using full aruco marks
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from Memory import MemoryClass
from main_rl_env_translation_v2 import RL_ENV
from Networks import Actor_Img, Critic_Img


torch.set_num_threads(5)  # set number of processes --> CPU no sure yet if it's correct


class TD3_Agent_Img:

    def __init__(self):

        self.gamma      = 0.99
        self.tau        = 0.005
        self.batch_size = 32
        self.max_memory_size = 10_000

        self.actor_learning_rate  = 7e-5
        self.critic_learning_rate = 1e-3

        self.G = 10
        self.update_counter = 0
        self.policy_freq_update = 2

        self.num_actions = 4

        # ------------- Initialization memory --------------------- #
        self.memory = MemoryClass(replay_max_size_env=self.max_memory_size)

        # ---------- Initialization and build the networks ----------- #
        # Main Networks
        self.actor     = Actor_Img(self.num_actions)
        self.critic_q1 = Critic_Img(self.num_actions)
        self.critic_q2 = Critic_Img(self.num_actions)

        # Target networks
        self.actor_target     = Actor_Img(self.num_actions)
        self.critic_target_q1 = Critic_Img(self.num_actions)
        self.critic_target_q2 = Critic_Img(self.num_actions)

        # Initialization of the target networks as copies of the original networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target_q1.parameters(), self.critic_q1.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target_q2.parameters(), self.critic_q2.parameters()):
            target_param.data.copy_(param.data)

        self.actor_optimizer    = optim.Adam(self.actor.parameters(),     lr=self.actor_learning_rate)
        self.critic_optimizer_1 = optim.Adam(self.critic_q1.parameters(), lr=self.critic_learning_rate)
        self.critic_optimizer_2 = optim.Adam(self.critic_q2.parameters(), lr=self.critic_learning_rate)


    def get_action_from_policy(self, state):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)  # torch.Size([1, 1, 64, 64])
        self.actor.eval()
        with torch.no_grad():
            action = self.actor.forward(state_tensor)
            action = action.detach()
            action = action.numpy()  # tensor to numpy
            self.actor.train()
        return action[0]

    def add_real_experience_memory(self, state, action, reward, next_state, done):
        self.memory.replay_buffer_environment_add(state, action, reward, next_state, done)

    def step_training(self):
        if self.memory.len_env_buffer() <= self.batch_size:
            return

        for it in range(1, self.G + 1):

            self.update_counter += 1

            states, actions, rewards, next_states, dones = self.memory.sample_experience_from_env(self.batch_size)

            states  = np.array(states)
            actions = np.array(actions)
            rewards = np.array(rewards).reshape(-1, 1)
            dones   = np.array(dones).reshape(-1, 1)
            next_states = np.array(next_states)

            states  = torch.FloatTensor(states)
            actions = torch.FloatTensor(actions)
            rewards = torch.FloatTensor(rewards)
            dones   = torch.FloatTensor(dones)
            next_states = torch.FloatTensor(next_states)

            #print(states.size())       # --> [Batch_size, 1, 64, 64]
            #print(next_states.size())  # --> [Batch_size, 1, 64, 64]
            #print(actions.size())      # --> [Batch_size, 4]
            #print(rewards.size())      # --> [Batch_size, 1]
            #print(dones.size())        # --> [Batch_size, 1]

            # ------- compute the target action
            next_actions = self.actor_target.forward(next_states)

            # add noise also here, paper mention this
            next_actions = next_actions.detach().numpy()  # tensor to numpy
            next_actions = next_actions + (np.random.normal(0, scale=0.25, size=self.num_actions))
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
        torch.save(self.actor.state_dict(),     'models/TD3/image_representation/2td3_actor_image.pth')
        torch.save(self.critic_q1.state_dict(), 'models/TD3/image_representation/2td3_critic_1_image.pth')
        torch.save(self.critic_q2.state_dict(), 'models/TD3/image_representation/2td3_critic_2_image.pth')
        print("models has been saved...")

    def load_model(self):
        self.actor.load_state_dict(torch.load    ('models/TD3/image_representation/2td3_actor_image.pth'))
        self.critic_q1.load_state_dict(torch.load('models/TD3/image_representation/2td3_critic_1_image.pth'))
        self.critic_q2.load_state_dict(torch.load('models/TD3/image_representation/2td3_critic_2_image.pth'))
        print("models has been loaded...")


def crop_normalize_observation(observation):
    crop_image = observation[110:325, 0:350]
    resized    = cv2.resize(crop_image, (128, 128), interpolation=cv2.INTER_AREA)
    gray       = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    norm_image = cv2.normalize(gray, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    cv2.imshow("State visual Completed", observation)
    cv2.imshow("State visual", norm_image)
    cv2.waitKey(10)

    return norm_image


def run_training(env, episodes, horizont, agent):
    mode = f"Training TD3 Image Representation"
    rewards     = []
    avg_rewards = []
    for episode in range(episodes):
        env.reset_env()
        episode_reward = 0
        for step in range(horizont):

            state, _ = env.graphical_state_space_function()
            state = crop_normalize_observation(state)  # (64, 64)
            state = np.expand_dims(state, axis=0)  # (1, 64, 64)

            action = agent.get_action_from_policy(state)
            noise  = np.random.normal(0, scale=0.15, size=4)
            action = action + noise
            action = np.clip(action, -1, 1)

            env.env_step(action)

            next_state, img_raw = env.graphical_state_space_function()
            next_state = crop_normalize_observation(next_state)
            next_state = np.expand_dims(next_state, axis=0)

            reward, done    = env.calculate_reward()
            episode_reward += reward

            agent.add_real_experience_memory(state, action, reward, next_state, done)
            env.env_render(image=img_raw, episode=episode, step=step, done=done, mode=mode)

            agent.step_training()

            if done:
                break

        rewards.append(episode_reward)
        avg_rewards.append(np.mean(rewards[-100:]))

        print(f"******* -----Episode {episode} Ended-----********* ")
        print("Episode total reward:", episode_reward)

    agent.save_model()
    np.savetxt(f'results/TD3/image_representation/2TD3_rewards_img.txt', rewards)
    np.savetxt(f'results/TD3/image_representation/2TD3_avg_reward_img.txt', avg_rewards)

    plt.figure(1, figsize=(20, 10))
    plt.plot(rewards)
    plt.plot(avg_rewards)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig(f"results/TD3/image_representation/2TD3_ima.png")



def run_exploration(env, episodes, horizont, agent):
    mode = "Exploration"
    for episode in range(1, episodes+1):
        env.reset_env()
        for step in range(1, horizont+1):
            state, _ = env.graphical_state_space_function()
            state = crop_normalize_observation(state)  # (64, 64)
            state = np.expand_dims(state, axis=0)  # (1, 64, 64)
            action = env.generate_sample_act()
            env.env_step(action)
            next_state, img_raw = env.graphical_state_space_function()
            next_state = crop_normalize_observation(next_state)
            next_state = np.expand_dims(next_state, axis=0)
            reward, done = env.calculate_reward()
            agent.add_real_experience_memory(state, action, reward, next_state, done)
            env.env_render(image=img_raw, episode=episode, step=step, done=done, mode=mode)
            if done:
                break
    print(f"******* -----{episodes} for exploration ended-----********* ")


def main_run():
    env = RL_ENV(usb_index='/dev/ttyUSB0', camera_index=0)
    agent = TD3_Agent_Img()

    num_exploration_episodes  = 800
    num_episodes_training     = 3000
    episode_horizont          = 5

    run_exploration(env, num_exploration_episodes, episode_horizont, agent)
    run_training(env, num_episodes_training, episode_horizont, agent)


if __name__ == "__main__":
    main_run()
