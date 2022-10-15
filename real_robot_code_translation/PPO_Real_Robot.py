from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from main_rl_env import RL_ENV
import time
import random

gamma = 0.9
seed = 0


torch.manual_seed(seed)

TrainingRecord = namedtuple('TrainingRecord', ['ep', 'reward'])
Transition = namedtuple('Transition', ['s', 'a', 'a_log_p', 'r', 's_'])


class ActorPPO(nn.Module):

    def __init__(self):
        super(ActorPPO, self).__init__()
        self.fc = nn.Linear(12, 64)
        self.mu_head = nn.Linear(64, 4)
        self.sigma_head = nn.Linear(64, 4)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        mu = torch.tanh(self.mu_head(x))
        sigma = F.softplus(self.sigma_head(x))
        return (mu, sigma)


class CriticPPO(nn.Module):

    def __init__(self):
        super(CriticPPO, self).__init__()
        self.fc = nn.Linear(12, 32)
        self.v_head = nn.Linear(32, 4)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        state_value = self.v_head(x)
        return state_value


class Agent():

    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_epoch = 10
    buffer_capacity, batch_size = 14, 32

    def __init__(self):
        self.training_step = 0
        self.anet = ActorPPO().float()
        self.cnet = CriticPPO().float()
        self.buffer = []
        self.counter = 0

        self.optimizer_a = optim.Adam(self.anet.parameters(), lr=1e-4)
        self.optimizer_c = optim.Adam(self.cnet.parameters(), lr=3e-4)

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            (mu, sigma) = self.anet(state)
        dist = Normal(mu, sigma)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        # action.clamp(-2.0, 2.0)
        action.clamp(-1.0, 1.0)
        action = action.detach()
        action = action.numpy()  # tensor to numpy
        # print(action)
        action_log_prob = action_log_prob.detach()
        action_log_prob = action_log_prob.numpy()
        print(action_log_prob)
        return action[0], action_log_prob[0]

    def get_action_exploration(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            (mu, sigma) = self.anet(state)
        dist = Normal(mu, sigma)
        act_m1 = np.clip(random.uniform(-1, 1), -1, 1)
        act_m2 = np.clip(random.uniform(-1, 1), -1, 1)
        act_m3 = np.clip(random.uniform(-1, 1), -1, 1)
        act_m4 = np.clip(random.uniform(-1, 1), -1, 1)
        action_vector = np.array([act_m1, act_m2, act_m3, act_m4])
        action_log_prob = dist.log_prob(torch.from_numpy(action_vector))
        return action_vector, action_log_prob

    def get_value(self, state):

        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            state_value = self.cnet(state)
        return state_value.item()

    def save_param(self):
        torch.save(self.anet.state_dict(), 'ppo_anet_params.pkl')
        torch.save(self.cnet.state_dict(), 'ppo_cnet_params.pkl')

    def store(self, transition):
        self.buffer.append(transition)
        self.counter += 1
        return self.counter % self.buffer_capacity == 0

    def update(self):
        self.training_step += 1

        s = torch.tensor([t.s for t in self.buffer], dtype=torch.float)
        a = torch.tensor([t.a for t in self.buffer], dtype=torch.float).view(-1, 1)
        r = torch.tensor([t.r for t in self.buffer], dtype=torch.float).view(-1, 1)
        s_ = torch.tensor([t.s_ for t in self.buffer], dtype=torch.float)
        print("-----------THIS IS MEMORY-------------")
        print(self.buffer)
        print("-------------MEMORY ENDS HERE------------")
        print([t.a_log_p for t in self.buffer])
        old_action_log_probs = torch.tensor(
            [t.a_log_p for t in self.buffer], dtype=torch.float64).view(-1, 1)

        r = (r - r.mean()) / (r.std() + 1e-5)
        with torch.no_grad():
            target_v = r + gamma * self.cnet(s_)

        adv = (target_v - self.cnet(s)).detach()

        for _ in range(self.ppo_epoch):
            for index in BatchSampler(
                    SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, False):

                (mu, sigma) = self.anet(s[index])
                dist = Normal(mu, sigma)
                action_log_probs = dist.log_prob(a[index])
                ratio = torch.exp(action_log_probs - old_action_log_probs[index])

                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv[index]
                action_loss = -torch.min(surr1, surr2).mean()

                self.optimizer_a.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.anet.parameters(), self.max_grad_norm)
                self.optimizer_a.step()

                value_loss = F.smooth_l1_loss(self.cnet(s[index]), target_v[index])
                self.optimizer_c.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.cnet.parameters(), self.max_grad_norm)
                self.optimizer_c.step()

        del self.buffer[:]


def main():
    start = time.time()

    env = RL_ENV()
    agent = Agent()
    running_q = 0
    rewards = []
    avg_rewards = []
    step_counter = 0
    training_records = []
    time_logs = []
    running_reward = 0
    for i_ep in range(10000):
        running_reward = 0
        env.reset_env()
        state, _ = env.state_space_funct()
        num_done = 0
        step_counter += 1
        print('Episode {}'.format(i_ep))

        for t in range(7):
            # angles_steps = get_angles_steps()
            #
            # action, action_index = agent.select_action(state, angles_steps)
            step_counter += 1

            if step_counter <= 700:
                action, action_log_prob = agent.get_action_exploration(state)
            else:
                action, action_log_prob = agent.select_action(state)
            # state_, reward, done, _ = env.env_step([action])
            env.env_step(action)
            next_state, image_state = env.state_space_funct()
            reward, done = env.calculate_reward_continuous()
            if agent.store(Transition(state, action, action_log_prob, (reward + 8) / 8, next_state)):
                agent.update()
            env.env_render(image_state, done, t)
            running_reward += reward
            state = next_state

        # running_reward = running_reward * 0.9 + score * 0.1
        training_records.append(TrainingRecord(i_ep, running_reward))
            # env.render()
        # if running_reward > -200:
        #     print("Solved! Moving average score is now {}!".format(running_reward))
        #     env.close()
        #     agent.save_param()
        #     with open('log/ppo_training_records.pkl', 'wb') as f:
        #         pickle.dump(training_records, f)
        #     break


        print("-----------------Running Reward: --------------------")
        print(running_reward)
        rewards.append(running_reward)
        avg_rewards.append(np.mean(rewards[-100:]))

        if i_ep % 10 == 0:
            print('Ep {}\tAverage score: {:.2f}\tAverage Q: {:.2f}'.format(
                i_ep, running_reward, running_q))

        if i_ep % 100 == 0 and i_ep != 0:
            now = time.time()
            specific = (now - start) / 3600
            time_logs.append(specific)
            np.savetxt('time-logs.txt', time_logs)

            i = str(i_ep / 100)
            agent.save_param()
            np.savetxt('rewards_ppo.txt', rewards)
            np.savetxt('avd_reward_ppo.txt', avg_rewards)
            plt.plot(rewards)
            plt.plot(avg_rewards)
            plt.title('ppo')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.savefig("ppo" + i + ".png")

    agent.save_param()

    plt.plot(rewards)
    plt.plot(avg_rewards)
    np.savetxt('rewards_ppo.txt', rewards)
    np.savetxt('avd_reward_ppo.txt', avg_rewards)
    plt.title('ppo')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig("ppo.png")
    plt.show()


    plt.plot([r.ep for r in training_records], [r.reward for r in training_records])
    plt.title('PPO')
    plt.xlabel('Episode')
    plt.ylabel('Moving averaged episode reward')
    plt.savefig("ppo.png")
    plt.show()


if __name__ == '__main__':
    main()
