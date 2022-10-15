"""
Author: David Valencia
Date:   16/ 05 /2022
Update:

Describer:

"""
import time
import torch
import torch.optim as optim
import torch.nn as nn

from Memory import MemoryClass
from main_rl_env import RL_ENV
from Networks import Critic, Actor

import numpy as np
import random
import matplotlib.pyplot as plt


import copy
class OUNoise:
    """Ornstein-Uhlenbeck process."""
    def __init__(self, size=4, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.randn() for i in range(len(x))])
        self.state = x + dx
        return self.state


class Noise(object):
    # to encourage action exploration

    def __init__(self, action_space=4, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = 4
        self.low  = 300
        self.high = 700
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action_oup(self, action, t=0):
        # Ornstein-Uhlenbeck Process to add noise to the action output
        ou_state    = self.evolve_state()
        self.sigma  = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        tran_action = action + ou_state
        tran_action = (tran_action-(-1)) * (self.high - self.low) / (1 - (-1)) + self.low  # to manually put the output
                                                                                      # in the operation servo step area
        return tran_action.tolist()

    def get_action_gaussian(self, action):
        # Gaussian Normal Distribution to add noise to the action output
        gauss_noise = np.random.normal(0, scale=0.1, size=self.action_dim)
        tran_action = action + gauss_noise
        tran_action = np.clip(tran_action, -1, 1)
        #return tran_action.tolist()
        return tran_action


class DDPGagent:

    def __init__(self, env, actor_learning_rate=1e-3, critic_learning_rate=1e-4, gamma=0.99,
                 max_memory_size=50_000, tau=0.005, batch_size=32):

        # -------- Hyper-parameters --------------- #
        self.max_memory_size = max_memory_size
        self.batch_size = batch_size

        self.update_interaction = 10  # number of updates of NN per Episode

        self.gamma = gamma  # discount factor
        self.tau   = tau
        self.hidden_size_critic = [32, 32, 32]
        self.hidden_size_actor  = [64, 32, 32]

        # -------- Parameters --------------- #
        self.num_states  = 12
        self.num_actions = 4

        # ---------- Initialization and build the networks ----------- #
        # Main networks
        self.actor  = Actor(self.num_states, self.hidden_size_actor, self.num_actions)
        self.critic = Critic(self.num_states + self.num_actions, self.hidden_size_critic, self.num_actions)

        # Target networks
        self.actor_target  = Actor(self.num_states, self.hidden_size_actor, self.num_actions)
        self.critic_target = Critic(self.num_states + self.num_actions, self.hidden_size_critic, self.num_actions)

        # Initialization of the target networks as copies of the original networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # optimizers
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

        # ------------- Initialization memory --------------------- #
        self.memory = MemoryClass(max_memory_size)

    def get_action(self, state):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)  # numpy to a tensor with shape [1,12]
        self.actor.eval()
        with torch.no_grad():
            action = self.actor.forward(state_tensor)
            action = action.detach()
            action = action.numpy()  # tensor to numpy
        self.actor.train()
        return action[0]

    def get_action_exploration(self):
        #act_m1 = random.randint(300, 700)
        #act_m2 = random.randint(300, 700)
        #act_m3 = random.randint(300, 700)
        #act_m4 = random.randint(300, 700)

        act_m1 = np.clip(random.uniform(-1, 1), -1, 1)
        act_m2 = np.clip(random.uniform(-1, 1), -1, 1)
        act_m3 = np.clip(random.uniform(-1, 1), -1, 1)
        act_m4 = np.clip(random.uniform(-1, 1), -1, 1)

        action_vector = np.array([act_m1, act_m2, act_m3, act_m4])
        return action_vector


    def add_experience_memory(self, state, action, reward, next_state, done):
        # Save experience in memory
        self.memory.replay_buffer_add(state, action, reward, next_state, done)


    def step_training(self, episod):

        # check, if enough samples are available in memory
        if self.memory.__len__() <= self.batch_size:
            return
        # update the networks every N times
        for it in range(self.update_interaction):

            states, actions, rewards, next_states, dones = self.memory.sample_experience(self.batch_size)
            states  = np.array(states)
            actions = np.array(actions)
            rewards = np.array(rewards).reshape(-1, 1)
            dones   = np.array(dones).reshape(-1, 1)
            next_states = np.array(next_states)

            states  = torch.FloatTensor(states)
            actions = torch.FloatTensor(actions)
            rewards = torch.FloatTensor(rewards)
            dones = torch.FloatTensor(dones)
            next_states = torch.FloatTensor(next_states)

            # ----------------------------------- Calculate the loss ----- #
            Q_vals = self.critic.forward(states, actions)

            # ------- calculate the critic loss
            next_actions = self.actor_target.forward(next_states)  # Note this is from actor-target

            # next_Q_vales = self.critic_target.forward(next_states, next_actions.detach())
            # Q_target = rewards + self.gamma * next_Q_vales

            next_Q_vales = self.critic_target.forward(next_states, next_actions)
            Q_target = rewards + (self.gamma * next_Q_vales).detach()

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


    def save_checkpoint(self, episode, accumulate_rewards):

        torch.save({'episode':           episode,
                    'actor.state_dict':  self.actor.state_dict(),
                    'critic.state_dict': self.critic.state_dict(),
                    'memory_batch':      self.memory.replay_buffer,
                    'rewards':           accumulate_rewards,
                    }, 'weights/checkpoint.pt')



    def save_model(self):
        torch.save(self.actor.state_dict(), 'weights/ddpg_actor.pth')
        torch.save(self.critic.state_dict(), 'weights/ddpg_critic.pth')
        print("models has been saved...")

    def load_model(self):
        self.actor.load_state_dict(torch.load('weights/ddpg_actor.pth'))
        self.critic.load_state_dict(torch.load('weights/ddpg_critic.pth'))
        print("models has been loaded...")



def main_run():

    num_episodes      = 10000
    episode_horizont  = 5
    update_interval   = episode_horizont / 3
    batch_size        = 64
    save_period       = 5  # When to save a checkpoint
    start = time.time()
    time_logs = []
    rewards           = []
    avg_rewards       = []

    env   = RL_ENV()
    agent = DDPGagent(env, batch_size=batch_size)
    noise = Noise()

    epsilon       = 1.0
    epsilon_min   = 0.01
    epsilon_decay = 0.001

    for episode in range(num_episodes):
        env.reset_env()
        noise.reset()
        done    = False
        episode_reward = 0
        time.sleep(2.0)

        if epsilon > epsilon_min:
            epsilon *= (1 - epsilon_decay)
        explore_probability = epsilon

        for step in range(1, episode_horizont+1):
            print(f"-------Episode:{episode + 1} Step:{step}---------")
            state, _ = env.state_space_funct()
            if np.random.random() <= explore_probability:
                action = agent.get_action_exploration()
                print("Exploration Stage")
            else:
                print("Exploitation Stage")
                action = agent.get_action(state)
                print("Action_learned-------->:", action)
                action = noise.get_action_gaussian(action) 

            env.env_step(action)
            next_state, image_state = env.state_space_funct()
            reward, done = env.calculate_reward_continuous()
            print("Step Reward:", reward)

            agent.add_experience_memory(state, action, reward, next_state, done)
            episode_reward += reward

            env.env_render(image_state, done, step)
            if done:
                break
            #env.render()
            time.sleep(0.4)

            #if step % update_interval == 0:
            agent.step_training(episode)  # Update the NNs
            #print("updating network")

        print(f"******* -----Episode {episode+1} Ended----- ********* ")
        print("Episode total reward:", episode_reward)
        rewards.append(episode_reward)
        avg_rewards.append(np.mean(rewards[-100:]))

        if episode % 500 == 0 and episode != 0:
            i = str(episode / 500)
            now = time.time()
            specific = (now - start) / 3600
            time_logs.append(specific)
            np.savetxt('time-logs_ddpg.txt', time_logs)

            agent.save_model()
            np.savetxt('rewards_ddpg.txt', rewards)
            np.savetxt('avd_reward_ddpg.txt', avg_rewards)
            plt.plot(rewards)
            plt.plot(avg_rewards)
            plt.title('DDPG')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.savefig("ddpg" + i + ".png")


        #if episode % save_period == 0:
            #agent.save_checkpoint(episode, rewards)

    agent.save_model()
    env.close_env()  # close the usb port and disable torque

    np.savetxt('data_to_plot/rewards_DDPG.txt', rewards)
    np.savetxt('data_to_plot/avd_reward_DDPG.txt', avg_rewards)

    plt.plot(rewards)
    plt.plot(avg_rewards)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig("DDPG.png")
    plt.show()



def test_run():
    TEST_EPISODES = 10
    episode_horizont = 10

    env   = RL_ENV()
    agent = DDPGagent(env, batch_size=64)
    noise = Noise()
    agent.load_model()

    for episode in range(TEST_EPISODES):
        env.reset_env()
        done    = False
        episode_reward = 0
        time.sleep(3.0)

        for step in range(episode_horizont):
            print(f"-------Episode:{episode + 1} Step:{step + 1}---------")
            state  = env.state_space_funct()
            action  = agent.get_action(state)
            env.env_step(action)  # take the action
            reward, done = env.calculate_reward_continuous()
            next_state = env.state_space_funct()
            agent.add_experience_memory(state, action, reward, next_state, done)
            episode_reward += reward
            if done:
                break
            time.sleep(0.4)
        print("Episode total reward:", episode_reward)



if __name__ == '__main__':
    main_run()
    #test_run()
