"""
    - Algorithm: Soft Actor Critic
    - Task: Move the cube
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

import numpy as np
import matplotlib.pyplot as plt

from Memory import MemoryClass
from main_rl_env_translation_v2 import RL_ENV
from Networks import ValueNetworkSAC, SoftQNetworkSAC, PolicyNetworkSAC


use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")
torch.set_num_threads(5)  # set number of processes --> CPU no sure yet if it's correct


class SAC_Agent:

    def __init__(self):

        # ------- Hyper-parameters --------
        self.gamma = 0.99
        self.tau   = 0.005
        self.batch_size = 32

        self.G = 10
        self.max_memory_size_env = 20_000

        self.value_lr  = 1e-3
        self.soft_q_lr = 1e-3
        self.policy_lr = 1e-4

        self.hidden_dim = [64, 32]
        #self.hidden_dim = 256

        self.num_states  = 16
        self.num_actions = 4

        # ------------- Initialization memory --------------------- #
        self.memory = MemoryClass(self.max_memory_size_env)

        # ---------- Initialization and build the networks for SAC----------- #
        self.value_net = ValueNetworkSAC(self.num_states, self.hidden_dim).to(device)

        self.soft_q_net1 = SoftQNetworkSAC(self.num_states, self.num_actions, self.hidden_dim).to(device)
        self.soft_q_net2 = SoftQNetworkSAC(self.num_states, self.num_actions, self.hidden_dim).to(device)

        self.policy_net = PolicyNetworkSAC(self.num_states, self.num_actions, self.hidden_dim).to(device)


        # Target Network
        self.value_net_target = ValueNetworkSAC(self.num_states, self.hidden_dim).to(device)

        # Initialization of the target network for SAC we have one target network only
        for target_param, param in zip(self.value_net_target.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        self.value_optimizer   = optim.Adam(self.value_net.parameters(),   lr=self.value_lr)
        self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=self.soft_q_lr)
        self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=self.soft_q_lr)
        self.policy_optimizer  = optim.Adam(self.policy_net.parameters(),  lr=self.policy_lr)


    def get_action_from_policy(self, state):
        #state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        self.policy_net.eval()
        with torch.no_grad():
            mean, log_std = self.policy_net.forward(state)
            std    = log_std.exp()
            normal = Normal(0, 1)
            z      = normal.sample().to(device)
            action = torch.tanh(mean + std * z)

            action = action.detach().cpu().numpy()
            #action = action.cpu().detach().numpy()
            self.policy_net.train()
        return action[0]


    def add_real_experience_memory(self, state, action, reward, next_state, done):
        self.memory.replay_buffer_environment_add(state, action, reward, next_state, done)

    def calculate_policy_net_values(self, state):
        epsilon = 1e-6
        mean, log_std = self.policy_net.forward(state)
        std = log_std.exp()
        normal   = Normal(0, 1)
        z        = normal.sample()
        action   = torch.tanh(mean + std * z.to(device))
        log_prob = Normal(mean, std).log_prob(mean + std * z.to(device)) - torch.log(1 - action.pow(2) + epsilon)
        return action, log_prob, z, mean, log_std


    def step_training(self):
        if self.memory.len_env_buffer() <= self.batch_size:
            return
        else:
            self.update_weights()

    def update_weights(self):

        for it in range(1, self.G + 1):

            states, actions, rewards, next_states, dones = self.memory.sample_experience_from_env(self.batch_size)

            states  = np.array(states)
            actions = np.array(actions)
            rewards = np.array(rewards).reshape(-1, 1)
            dones   = np.array(dones).reshape(-1, 1)
            next_states = np.array(next_states)

            states  = torch.FloatTensor(states).to(device)
            actions = torch.FloatTensor(actions).to(device)
            rewards = torch.FloatTensor(rewards).to(device)
            dones   = torch.FloatTensor(dones).to(device)
            next_states = torch.FloatTensor(next_states).to(device)

            predicted_q_value1 = self.soft_q_net1(states, actions)
            predicted_q_value2 = self.soft_q_net2(states, actions)
            predicted_value    = self.value_net(states)

            new_action, log_prob, epsilon, mean, log_std = self.calculate_policy_net_values(states)

            # Training Q Function
            target_value   = self.value_net_target(next_states)
            target_q_value = rewards + (1 - dones) * self.gamma * target_value

            loss_criterion_1 = nn.MSELoss()
            loss_criterion_2 = nn.MSELoss()

            q_value_loss1 = loss_criterion_1(predicted_q_value1, target_q_value.detach())
            q_value_loss2 = loss_criterion_2(predicted_q_value2, target_q_value.detach())

            self.soft_q_optimizer1.zero_grad()
            q_value_loss1.backward()
            self.soft_q_optimizer1.step()

            self.soft_q_optimizer2.zero_grad()
            q_value_loss2.backward()
            self.soft_q_optimizer2.step()

            # Training Value Function
            value_criterion_loss = nn.MSELoss()
            predicted_new_q_value = torch.min(self.soft_q_net1(states, new_action), self.soft_q_net2(states, new_action))
            target_value_func = predicted_new_q_value - log_prob

            value_loss = value_criterion_loss(predicted_value, target_value_func.detach())

            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

            # Training Policy Function
            policy_loss = (log_prob - predicted_new_q_value).mean()
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            for target_param, param in zip(self.value_net_target.parameters(), self.value_net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
                

    def save_rl_model(self, style="MFRL"):
        torch.save(self.value_net.state_dict(),   f"models/SAC/SAC_value_net_{style}.pth")
        torch.save(self.policy_net.state_dict(),  f'models/SAC/SAC_policy_net_{style}.pth')
        torch.save(self.soft_q_net1.state_dict(), f'models/SAC/SAC_soft_q_net_1_{style}.pth')
        torch.save(self.soft_q_net2.state_dict(), f'models/SAC/SAC_soft_q_net_2_{style}.pth')
        print(f"models for {style} has been saved...")


def plot_reward_curves(rewards, avg_rewards, number=2):
    plt.figure(number, figsize=(20, 10))
    plt.plot(rewards)
    plt.plot(avg_rewards)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig(f"results/SAC/SAC_MFRL.png")
    #plt.show()


def run_exploration(env, episodes, horizont, agent):
    mode = "Exploration"
    for episode in range(1, episodes+1):
        env.reset_env()
        for step in range(1, horizont+1):
            state, _ = env.state_space_function()
            action   = env.generate_sample_act()
            env.env_step(action)
            next_state, image_state = env.state_space_function()
            reward, done = env.calculate_reward()
            agent.add_real_experience_memory(state, action, reward, next_state, done)
            env.env_render(image=image_state, episode=episode, step=step, done=done, mode=mode)
            if done:
                break
    print(f"******* -----{episodes} for exploration ended-----********* ")


def run_training(env, episodes, horizont, agent):
    mode        = f"Training SAC MFRL"
    rewards     = []
    avg_rewards = []

    for episode in range(episodes):
        env.reset_env()
        episode_reward = 0
        for step in range(horizont):
            state, _ = env.state_space_function()
            action   = agent.get_action_from_policy(state)
            action   = np.clip(action, -1, 1)

            env.env_step(action)

            next_state, image_state = env.state_space_function()
            reward, done = env.calculate_reward()
            episode_reward += reward

            agent.add_real_experience_memory(state, action, reward, next_state, done)
            env.env_render(image=image_state, episode=episode, step=step, done=done, mode=mode)

            agent.step_training()

            if done:
                break

        rewards.append(episode_reward)
        avg_rewards.append(np.mean(rewards[-100:]))

        print(f"******* -----Episode {episode} Ended-----********* ")
        print("Episode total reward:", episode_reward)

    agent.save_rl_model()
    plot_reward_curves(rewards, avg_rewards, number=2)
    np.savetxt(f'results/SAC/SAC_rewards.txt', rewards)
    np.savetxt(f'results/SAC/SAC_avg_reward.txt', avg_rewards)


def main_run():
    env   = RL_ENV()
    agent = SAC_Agent()

    num_exploration_episodes  = 600
    num_episodes_training     = 3000
    episode_horizont          = 5

    run_exploration(env, num_exploration_episodes, episode_horizont, agent)
    run_training(env, num_episodes_training, episode_horizont, agent)


if __name__ == '__main__':
    main_run()

