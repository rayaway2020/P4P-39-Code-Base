import argparse
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
from motor_ultilities import get_angles_steps
from main_rl_env import RL_ENV
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import heapq
import time
import torch.autograd as autograd

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
Transition = namedtuple('Transition', ['s', 'a', 'r', 's_', 'd'])


N_Atoms = 51
V_MIN = -10.
V_MAX = 10.
V_RANGE = np.linspace(V_MIN, V_MAX, N_Atoms)
V_STEP = ((V_MAX-V_MIN)/(N_Atoms-1))

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # self.fc = nn.Sequential(
        #     nn.Linear(12, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, args.num_actions * N_Atoms)
        # )
        self.fc = nn.Linear(12, 200)
        self.a_head = nn.Linear(200, args.num_actions * N_Atoms)
        self.softmax = nn.Softmax(dim=1)
        self.support = torch.arange(V_MIN, V_MAX + V_STEP, V_STEP)
        # self.softmax = nn.Softmax(dim=1)
        # self.fc = nn.Linear(12, 200)
        # self.a_head = nn.Linear(200, args.num_actions * N_Atoms)
        # self.v_head = nn.Linear(200, 1)

    def forward(self, x):
        # print(x)
        batch_size = x.size()[0]
        x = self.fc(x)
        # print(x)
        # x = self.a_head(x)
        dist = self.a_head(x).view(batch_size, -1, N_Atoms)
        probs = self.softmax(dist)
        Qvals = torch.sum(probs * self.support, dim=2)
        return dist, Qvals

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
        self.memory = Memory(100)
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=1e-3)
        self.value_range = torch.FloatTensor(V_RANGE)  # (N_ATOM)
        # self.position = [0, 0, 0, 0]

    def select_action(self, state, angle_steps):
        state = torch.from_numpy(state).float().unsqueeze(0)
        # print("epsilon is now:")
        # print(self.epsilon)


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
            print("This is state size" + str(state.size()))
            dist, probs = self.eval_net.forward(state)
            # action = np.argmax(qvals.detach().numpy())
            # return action
            # print("now finding the best!")
            # probs = self.eval_net(state)
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
        # for i in range(4):
        #     self.position[i] = (angle_steps[i] - (-1)) * (700 - 300) / (1 - (-1)) + 300
        # return self.position, action_index
        # return self.action_list[action_index], action_index

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
        batch_size = 32
        transitions = self.memory.sample(batch_size)
        s = torch.tensor([t.s for t in transitions], dtype=torch.float)
        a = torch.tensor([t.a for t in transitions], dtype=torch.long).view(-1, 1)
        r = torch.tensor([t.r for t in transitions], dtype=torch.float).view(-1, 1)
        s_ = torch.tensor([t.s_ for t in transitions], dtype=torch.float)
        d = torch.tensor([t.d for t in transitions], dtype=torch.float)

        # natural dqn
        # q_eval = self.eval_net(s).gather(1, a)
        curr_dist, q_eval = self.eval_net.forward(s)
        next_dist, next_qvals = self.eval_net.forward(s_)
        next_actions = torch.max(next_qvals, 1)[1]
        next_dist = self.eval_net.softmax(next_dist)
        optimal_dist = next_dist[range(batch_size), next_actions]
        # q_eval = self.eval_net(s)
        # mb_size = q_eval.size(0)
        print(optimal_dist)
        size = r.size(0)
        m = torch.zeros(batch_size, N_Atoms)

        for sample_idx in range(size):
            reward = r[sample_idx]

            for atom in range(N_Atoms):
                # compute projection of Tz_j
                Tz_j = reward + args.gamma * self.eval_net.support[atom]
                Tz_j = torch.clamp(Tz_j, V_MIN, V_MAX)
                b_j = (Tz_j - V_MIN) / V_STEP
                l = torch.floor(b_j).long().item()
                u = torch.ceil(b_j).long().item()

                # distribute probability of Tz_j
                m[sample_idx][l] = m[sample_idx][l] + optimal_dist[sample_idx][atom] * (u - b_j)
                m[sample_idx][u] = m[sample_idx][u] + optimal_dist[sample_idx][atom] * (b_j - l)

        # print(m)
        projection = m
        print("This is projection: -----------------")
        print(projection)
        loss = -(torch.sum(optimal_dist * (torch.log(optimal_dist) - torch.log(projection))))
        loss = loss.mean()
        # q_eval = torch.stack([q_eval[i].index_select(0, a[i]) for i in range(mb_size)]).squeeze(1)
        # q_target = np.zeros((mb_size, N_Atoms))  # (m, N_ATOM)
        # q_next = self.target_net(s_).detach() # (m, N_ACTIONS, N_ATOM)
        # # next value mean
        # q_next_mean = torch.sum(q_next * self.value_range.view(1, 1, -1), dim=2) # (m, N_ACTIONS)
        # best_actions = q_next_mean.argmax(dim=1) # (m)
        # q_next = torch.stack([q_next[i].index_select(0, best_actions[i]) for i in range(mb_size)]).squeeze(1)
        # q_next = q_next.data.cpu().numpy() # (m, N_ATOM)
        # # with torch.no_grad():
        # #     q_target = r + args.gamma * self.target_net(s_).max(1, keepdim=True)[0]
        #
        # # double dqn
        # # with torch.no_grad():
        # #     a_ = self.eval_net(s_).max(1, keepdim=True)[1]
        # #     q_target = r + args.gamma * self.target_net(s_).gather(1, a_)
        # #
        # # q_eval = self.eval_net(s).gather(1, a)
        #
        # # categorical projection
        # '''
        # next_v_range : (z_j) i.e. values of possible return, shape : (m, N_ATOM)
        # next_v_pos : relative position when offset of value is V_MIN, shape : (m, N_ATOM)
        # '''
        # # we vectorized the computation of support and position
        # next_v_range = np.expand_dims(r, 1) + args.gamma * np.expand_dims(1. - d, 1) * np.expand_dims(self.value_range.data.cpu().numpy(), 0)
        # # clip for categorical distribution
        # next_v_range = np.clip(next_v_range, V_MIN, V_MAX)
        # # calc relative position of possible value
        # next_v_pos = (next_v_range - V_MIN) / V_STEP
        # # get lower/upper bound of relative position
        # lb = np.floor(next_v_pos).astype(int)
        # ub = np.ceil(next_v_pos).astype(int)
        # # we didn't vectorize the computation of target assignment.
        # # print("this is 32")
        # # print(lb[32, 0])
        # for i in range(mb_size):
        #     for j in range(mb_size):
        #         # print(i)
        #         # print(j)
        #         # print("------------THIS IS Q TARGET----------")
        #         # print(q_target)
        #         # print("------------THIS IS LB----------")
        #         # print(lb[0,30])
        #         # print("------------THIS IS Q NEXT----------")
        #         # print(q_next)
        #         # print("------------THIS IS UB----------")
        #         # print(ub)
        #         # calc prob mass of relative position weighted with distance
        #         q_target[i, lb[i, j]] += (q_next * (ub - next_v_pos))[i, j]
        #         q_target[i, ub[i, j]] += (q_next * (next_v_pos - lb))[i, j]
        #
        # q_target = torch.FloatTensor(q_target)
        #
        # # calc huber loss, dont reduce for importance weight
        # loss = q_target * (- torch.log(q_eval + 1e-8))  # (m , N_ATOM)
        # loss = torch.mean(loss)

        # backprop loss
        self.optimizer.zero_grad()
        loss.backward()
        # self.optimizer.step()
        # self.optimizer.zero_grad()
        # loss = F.smooth_l1_loss(q_eval, q_target)
        # loss.backward()
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

    for i_ep in range(10000):
        env.reset_env()
        # agent.reset()
        running_reward = 0
        state, _ = env.state_space_funct()
        num_done = 0
        print('Episode {}'.format(i_ep))


        for t in range(40):
            angles_steps = get_angles_steps()

            action, action_index = agent.select_action(state, angles_steps)
            print(action)
            crash = env.env_step_discrete(action)

            next_state, image_state = env.state_space_funct()
            reward, done = env.calculate_reward_discrete()

            if done and t == 0:
                continue
            #state_, reward, done, _ = env.env_step(action)
            if crash == 1:
                reward = 2*reward
            running_reward += reward
            env.env_render(image_state, done, t)
            agent.store_transition(Transition(state, action_index, reward, next_state, float(done)))
            print("This is reward in this step")
            print(reward)

            state = next_state
            if agent.memory.isfull:
                q = agent.update()
                # print("Running Q Updated")
                running_q = 0.99 * running_q + 0.01 * q

            if done and t != 0:
                num_done += 1
            # elif done and t == 0:
            #     reward -= 500
            #     num_done = 0
            else:
                num_done = 0

            if num_done == 1:
                break
        # running_reward = running_reward * 0.9 + score * 0.1
        # training_records.append(TrainingRecord(i_ep, running_reward))
        print("-----------------Running Reward: --------------------")
        print(running_reward)
        rewards.append(running_reward)
        avg_rewards.append(np.mean(rewards[-100:]))

        # print(running_reward)
        if i_ep % args.log_interval == 0:
            print('Ep {}\tAverage score: {:.2f}\tAverage Q: {:.2f}'.format(
                i_ep, running_reward, running_q))

        if i_ep % 500 == 0 and i_ep != 0:
            now = time.time()
            specific = (now - start) / 3600
            time_logs.append(specific)
            np.savetxt('time-logs.txt', time_logs)

            i = str(i_ep / 500)
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

    for i_ep in range(10):
        env.reset_env()
        running_reward = 0
        state, _ = env.state_space_funct()
        num_done = 0

        print('Episode {}'.format(i_ep))
        for t in range(40):
            angles_steps = get_angles_steps()
            action, action_index = agent.select_action_test(state, angles_steps)
            print(action)
            crash = env.env_step_discrete(action)
            next_state, image_state = env.state_space_funct()
            reward, done = env.calculate_reward_discrete()
            if done and t == 0:
                continue
            #state_, reward, done, _ = env.env_step(action)
            if crash == 1:
                reward = 2*reward
            running_reward += reward
            print("This is reward in this step", reward)
            env.env_render(image_state, done, t)
            # agent.store_transition(Transition(state, action_index, reward, next_state))
            state = next_state

            if done and t != 0:
                num_done += 1
            # elif done and t == 0:
            #     reward -= 500
            #     num_done = 0
            else:
                num_done = 0

            if num_done == 1:
                break
            # if agent.memory.isfull:
            # agent.update()
            #     # print("Running Q Updated")
            #     running_q = 0.99 * running_q + 0.01 * q

        # running_reward = running_reward * 0.9 + score * 0.1
        # training_records.append(TrainingRecord(i_ep, running_reward))
        print("RUnning Reward: ")
        print(running_reward)
        rewards.append(running_reward)
        avg_rewards.append(np.mean(rewards[-2:]))

    plt.plot(rewards)
    plt.plot(avg_rewards)
    plt.title('DQN')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()

if __name__ == '__main__':
    main()
    # test()
