import argparse
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
from motor_utilities import Motor
from main_rl_env_translation_v2 import RL_ENV
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import heapq
import time

parser = argparse.ArgumentParser(description='Solve the Robotic control single arm problem with DQN')
parser.add_argument('--gamma', type=float, default=0.9, metavar='G', help='discount factor (default: 0.9)')
parser.add_argument('--num_actions', type=int, default=8, metavar='N', help='discretize action space (default: 8)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--log-interval',type=int,default=10,metavar='N',help='interval between training status logs (default: 10)')
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

TrainingRecord = namedtuple('TrainingRecord', ['ep', 'reward'])
Transition = namedtuple('Transition', ['s', 'a', 'r', 's_'])
torch.set_num_threads(4)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(16, 200)
        # self.fc2 = nn.Linear(256,128)
        self.a_head = nn.Linear(200, args.num_actions)
        self.v_head = nn.Linear(200, 1)

    def forward(self, x):
        # x = self.fc1(x)
        x = F.tanh(self.fc1(x))
        a = self.a_head(x) - self.a_head(x).mean(1, keepdim=True)
        v = self.v_head(x)
        action_scores = a + v
        return action_scores


class Memory():

    data_pointer = 0
    isfull = False

    def __init__(self, capacity):
        self.memory = np.empty(capacity, dtype=object)
        self.capacity = capacity

    def update(self, transition):
        self.memory[self.data_pointer] = transition
        self.data_pointer += 1
        if self.data_pointer == self.capacity:
            self.data_pointer = 0
            self.isfull = True

    def sample(self, batch_size):
        return np.random.choice(self.memory, batch_size)


class Agent():

    action_list = [0, 1, 2, 3, 4, 5, 6, 7]
    max_grad_norm = 0.5
    min_step = 300
    max_step = 700
    step_size = 30

    def __init__(self):
        self.training_step = 0
        self.epsilon = 1
        self.eval_net, self.target_net = Net().float(), Net().float()
        self.memory = Memory(20000)
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=1e-3)
        # self.position = [0, 0, 0, 0]

    def select_action(self, state, angle_steps):
        state = torch.from_numpy(state).float().unsqueeze(0)

        #put all valid options into an array
        options = []
        for i in range(4):
            if (angle_steps[i] <= self.max_step - self.step_size and angle_steps[i] >= self.min_step + self.step_size):
                options.append(2 * i)
                options.append(2 * i + 1)
            elif (angle_steps[i] < self.min_step + self.step_size and angle_steps[i] >= self.min_step):
                options.append(2 * i + 1)
            elif (angle_steps[i] > self.max_step - self.step_size and angle_steps[i] <= self.max_step):
                options.append(2 * i)

        if np.random.random() < self.epsilon:
            # print("now exploring!")
            if (len(options) != 0):
                action_index = np.random.choice(options)
            else:
                print("No action to be taken, impossible here")
                action_index = np.random.randint(8)
        else:
            # print("now finding the best!")
            probs = self.eval_net(state)
            list = probs.detach().numpy()[0].tolist()
            sorted = heapq.nlargest(len(list), list)
            for i in sorted:
                if list.index(i) in options:
                    action_index = list.index(i)
                    break
            # action_index = probs.max(1)[1].item()
            print("--------------ACTION INDEX------------")
            print(state)
            print(probs)
            print(action_index)
            print("--------------ACTION INDEX ENDS------------")

        # action_chosen = self.action_list[action_index]
        if action_index == 0:
            angle_steps[0] -= self.step_size
            if angle_steps [0] < self.min_step:
                angle_steps[0] = self.min_step
        elif action_index == 1:
            angle_steps[0] += self.step_size
            if angle_steps[0] > self.max_step:
                angle_steps[0] = self.max_step
        elif action_index == 2:
            angle_steps[1] -= self.step_size
            if angle_steps[1] < self.min_step:
                angle_steps[1] = self.min_step
        elif action_index == 3:
            angle_steps[1] += self.step_size
            if angle_steps[1] > self.max_step:
                angle_steps[1] = self.max_step
        elif action_index == 4:
            angle_steps[2] -= self.step_size
            if angle_steps[2] < self.min_step:
                angle_steps[2] = self.min_step
        elif action_index == 5:
            angle_steps[2] += self.step_size
            if angle_steps[2] > self.max_step:
                angle_steps[2] = self.max_step
        elif action_index == 6:
            angle_steps[3] -= self.step_size
            if angle_steps[3] < self.min_step:
                angle_steps[3] = self.min_step
        elif action_index == 7:
            angle_steps[3] += self.step_size
            if angle_steps[3] > self.max_step:
                angle_steps[3] = self.max_step

        return angle_steps, action_index

    def select_action_test(self, state, angle_steps):
        state = torch.from_numpy(state).float().unsqueeze(0)
        options = []
        for i in range(4):
            if (angle_steps[i] <= self.max_step - self.step_size and angle_steps[i] >= self.min_step + self.step_size):
                options.append(2 * i)
                options.append(2 * i + 1)
            elif (angle_steps[i] < self.min_step + self.step_size and angle_steps[i] >= self.min_step):
                options.append(2 * i + 1)
            elif (angle_steps[i] > self.max_step - self.step_size and angle_steps[i] <= self.max_step):
                options.append(2 * i)
        probs = self.eval_net(state)
        list = probs.detach().numpy()[0].tolist()
        sorted = heapq.nlargest(len(list), list)
        for i in sorted:
            if list.index(i) in options:
                action_index = list.index(i)
                break
        # action_index = probs.max(1)[1].item()
        print("--------------ACTION INDEX------------")
        print(state)
        print(probs)
        print(action_index)
        print("--------------ACTION INDEX ENDS------------")
        if action_index == 0:
            angle_steps[0] -= self.step_size
            if angle_steps [0] < self.min_step:
                angle_steps[0] = self.min_step
        elif action_index == 1:
            angle_steps[0] += self.step_size
            if angle_steps[0] > self.max_step:
                angle_steps[0] = self.max_step
        elif action_index == 2:
            angle_steps[1] -= self.step_size
            if angle_steps[1] < self.min_step:
                angle_steps[1] = self.min_step
        elif action_index == 3:
            angle_steps[1] += self.step_size
            if angle_steps[1] > self.max_step:
                angle_steps[1] = self.max_step
        elif action_index == 4:
            angle_steps[2] -= self.step_size
            if angle_steps[2] < self.min_step:
                angle_steps[2] = self.min_step
        elif action_index == 5:
            angle_steps[2] += self.step_size
            if angle_steps[2] > self.max_step:
                angle_steps[2] = self.max_step
        elif action_index == 6:
            angle_steps[3] -= self.step_size
            if angle_steps[3] < self.min_step:
                angle_steps[3] = self.min_step
        elif action_index == 7:
            angle_steps[3] += self.step_size
            if angle_steps[3] > self.max_step:
                angle_steps[3] = self.max_step

        return angle_steps, action_index

    def save_param(self):
        torch.save(self.eval_net.state_dict(), 'dqn_net_params.pkl')
    def load_param(self):
        self.eval_net.load_state_dict(torch.load('dqn_net_params.pkl'))

    def store_transition(self, transition):
        self.memory.update(transition)

    # def reset(self):
    #     self.position = [0, 0, 0, 0]

    def update(self):

        self.training_step += 1

        transitions = self.memory.sample(32)
        s = torch.tensor([t.s for t in transitions], dtype=torch.float)
        a = torch.tensor([t.a for t in transitions], dtype=torch.long).view(-1, 1)
        r = torch.tensor([t.r for t in transitions], dtype=torch.float).view(-1, 1)
        s_ = torch.tensor([t.s_ for t in transitions], dtype=torch.float)

        # natural dqn
        # q_eval = self.eval_net(s).gather(1, a)
        # with torch.no_grad():
        #     q_target = r + args.gamma * self.target_net(s_).max(1, keepdim=True)[0]

        # double dqn
        with torch.no_grad():
            a_ = self.eval_net(s_).max(1, keepdim=True)[1]
            q_target = r + args.gamma * self.target_net(s_).gather(1, a_)

        q_eval = self.eval_net(s).gather(1, a)

        self.optimizer.zero_grad()
        loss = F.smooth_l1_loss(q_eval, q_target)
        loss.backward()
        nn.utils.clip_grad_norm_(self.eval_net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        if self.training_step % 200 == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        self.epsilon = max(self.epsilon * 0.999, 0.01)

        return q_eval.mean().item()


def main():
    start = time.time()

    env = RL_ENV()
    agent = Agent()
    running_q = 0
    rewards = []
    avg_rewards = []

    time_logs = []
    motor = Motor()
    for i_ep in range(10000):
        env.reset_env()
        running_reward = 0
        #state, ini_image = env.state_space_function()
        num_done = 0
        print('Episode {}'.format(i_ep))


        for t in range(40):
            state, _ = env.state_space_function()
            angles_steps = motor.get_angles_steps()
            action, action_index = agent.select_action(state, angles_steps)
            print(action)
            crash = env.env_step_discrete(action)
            next_state, image_state = env.state_space_function()
            reward, done = env.calculate_reward_discrete()
            env.env_render(image_state, done, t)
            if done and (t == 0 or t == 1 or t == 2 or t ==3 or t == 4):
                continue
            if crash == 1 and not done:
                reward = 2.0 * reward
            running_reward += reward

            agent.store_transition(Transition(state, action_index, reward, next_state))
            print("This is reward in this step")
            print(reward)

            # Policy Update
            if agent.memory.isfull:
                for i in range(0, 4):
                    q = agent.update()
                running_q = 0.99 * running_q + 0.01 * q

            # Make sure the target point is not touched at the very start of the episode when returning to origin.
            if done and (t != 0 or t != 1 or t != 2 or t !=3 or t != 4):
                num_done += 1
            else:
                num_done = 0

            if num_done == 1:
                break
        print("-----------------Running Reward: --------------------")
        print(running_reward)
        rewards.append(running_reward)
        avg_rewards.append(np.mean(rewards[-100:]))

        # print(running_reward)
        if i_ep % args.log_interval == 0:
            print('Ep {}\tAverage score: {:.2f}\tAverage Q: {:.2f}'.format(
                i_ep, running_reward, running_q))

        if i_ep % 250 == 0 and i_ep != 0:
            now = time.time()
            specific = (now - start) / 3600
            time_logs.append(specific)
            np.savetxt('time-logs.txt', time_logs)

            i = str(i_ep / 250)
            agent.save_param()
            np.savetxt('rewards_dqn.txt', rewards)
            np.savetxt('avd_reward_dqn.txt', avg_rewards)
            plt.plot(rewards)
            plt.plot(avg_rewards)
            plt.title('DQN')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.savefig("dqn" + i + ".png")

    agent.save_param()

    plt.plot(rewards)
    plt.plot(avg_rewards)
    np.savetxt('rewards_dqn.txt', rewards)
    np.savetxt('avd_reward_dqn.txt', avg_rewards)
    plt.title('DQN')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig("dqn.png")
    plt.show()

def test():
    env = RL_ENV()
    agent = Agent()
    agent.load_param()
    rewards = []
    avg_rewards = []
    motor = Motor()
    success = 0

    for i_ep in range(100):
        env.reset_env()
        running_reward = 0
        num_done = 0

        print('Episode {}'.format(i_ep))
        for t in range(40):
            state, _ = env.state_space_function()
            angles_steps = motor.get_angles_steps()
            action, action_index = agent.select_action_test(state, angles_steps)
            print(action)
            crash = env.env_step_discrete(action)
            next_state, image_state = env.state_space_function()
            reward, done = env.calculate_reward_discrete()
            if done and (t == 0 or t == 1 or t == 2 or t ==3 or t == 4):
                continue
            if crash == 1 and not done:
                reward = 1.5*reward
            running_reward += reward
            print("This is reward in this step", reward)
            env.env_render(image_state, done, t)

            # Make sure the target point is not touched at the very start of the episode when returning to origin.
            if done and (t != 0 or t != 1 or t != 2 or t !=3 or t != 4):
                num_done += 1
            else:
                num_done = 0
            if num_done == 1:
                success += 1
                break
        print("Running Reward: ")
        print(running_reward)
        rewards.append(running_reward)
        avg_rewards.append(np.mean(rewards[-20:]))

    plt.plot(rewards)
    plt.plot(avg_rewards)
    plt.title('DQN')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()
    print("The success rate over 100 epis are:")
    print(success)

if __name__ == '__main__':
    # main()
    test()
