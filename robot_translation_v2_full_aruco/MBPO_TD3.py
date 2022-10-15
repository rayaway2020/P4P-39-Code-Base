"""
Description:
            MBPO, model based policy optimization with TD3
            Real Robot Gripper
            Task move the cube
            Using aruco markers for all the detection and measures
"""

import torch
import torch.nn as nn
import torch.optim as optim

import random
import numpy as np
import matplotlib.pyplot as plt

from Memory import MemoryClass
from main_rl_env_translation_v2 import RL_ENV
from Networks import Actor_NN, Critic_NN, ModelNet_probabilistic_transition


class TD3_Agent:

    def __init__(self, env):

        self.env = env
        self.gamma      = 0.99
        self.tau        = 0.005

        self.batch_size = 32
        self.batch_size_model = 32

        self.M = 50
        self.G = 5

        self.update_counter     = 0
        self.policy_freq_update = 2

        self.max_memory_size_env   = 10_000
        self.max_memory_size_model = 50_000

        self.actor_learning_rate      = 1e-4
        self.critic_learning_rate     = 1e-3

        self.transition_learning_rate = 0.003

        self.hidden_size_critic = [64, 64, 32]
        self.hidden_size_actor  = [64, 64, 32]

        self.hidden_size_network_model  = [32, 32, 32]

        self.num_actions = 4
        self.num_states  = 16
        self.num_states_training = 14  # 16 in total but remove the target point from the state for training trans model

        # to store the loss values
        self.loss_transition_model_1 = []
        self.loss_transition_model_2 = []
        self.loss_transition_model_3 = []
        self.loss_transition_model_4 = []
        self.loss_transition_model_5 = []

        # to choose the "imagine data"
        self.epsilon       = 1
        self.epsilon_min   = 0.001
        self.epsilon_decay = 0.005

        # ------------- Initialization memory --------------------- #
        self.memory = MemoryClass(self.max_memory_size_env, self.max_memory_size_model)

        # ---------- Initialization and build the networks for TD3----------- #
        # Main networks
        self.actor     = Actor_NN(self.num_states, self.hidden_size_actor, self.num_actions)
        self.critic_q1 = Critic_NN(self.num_states + self.num_actions, self.hidden_size_critic, self.num_actions)
        self.critic_q2 = Critic_NN(self.num_states + self.num_actions, self.hidden_size_critic, self.num_actions)

        # Target networks
        self.actor_target     = Actor_NN(self.num_states, self.hidden_size_actor, self.num_actions)
        self.critic_target_q1 = Critic_NN(self.num_states + self.num_actions, self.hidden_size_critic, self.num_actions)
        self.critic_target_q2 = Critic_NN(self.num_states + self.num_actions, self.hidden_size_critic, self.num_actions)

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

        # ---------- Initialization and build the networks for Model Learning ----------- #
        self.pdf_transition_model_1 = ModelNet_probabilistic_transition(self.num_states_training + self.num_actions,
                                                                        self.hidden_size_network_model)
        self.pdf_transition_model_2 = ModelNet_probabilistic_transition(self.num_states_training + self.num_actions,
                                                                        self.hidden_size_network_model)
        self.pdf_transition_model_3 = ModelNet_probabilistic_transition(self.num_states_training + self.num_actions,
                                                                        self.hidden_size_network_model)
        self.pdf_transition_model_4 = ModelNet_probabilistic_transition(self.num_states_training + self.num_actions,
                                                                        self.hidden_size_network_model)
        self.pdf_transition_model_5 = ModelNet_probabilistic_transition(self.num_states_training + self.num_actions,
                                                                        self.hidden_size_network_model)

        self.pdf_transition_1_optimizer = optim.Adam(self.pdf_transition_model_1.parameters(),
                                                     lr=self.transition_learning_rate)
        self.pdf_transition_2_optimizer = optim.Adam(self.pdf_transition_model_2.parameters(),
                                                     lr=self.transition_learning_rate)
        self.pdf_transition_3_optimizer = optim.Adam(self.pdf_transition_model_3.parameters(),
                                                     lr=self.transition_learning_rate)
        self.pdf_transition_4_optimizer = optim.Adam(self.pdf_transition_model_4.parameters(),
                                                     lr=self.transition_learning_rate)
        self.pdf_transition_5_optimizer = optim.Adam(self.pdf_transition_model_5.parameters(),
                                                     lr=self.transition_learning_rate)

    def get_action_from_policy(self, state):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor.forward(state_tensor)
            action = action.detach()
            action = action.numpy()  # tensor to numpy
            self.actor.train()
        return action[0]

    def add_real_experience_memory(self, state, action, reward, next_state, done):
        self.memory.replay_buffer_environment_add(state, action, reward, next_state, done)

    def add_imagined_experience_memory(self, state, action, reward, next_state, done):
        self.memory.replay_buffer_model_add(state, action, reward, next_state, done)

    def epsilon_greedy_function_update(self):
        # this is used for choose the sample from memory when dream samples are generated
        if self.epsilon > self.epsilon_min:
            self.epsilon *= (1 - self.epsilon_decay)

    def step_training(self, style="MFRL", data_type='env_data'):
        if style == "MFRL":
            if self.memory.len_env_buffer() <= self.batch_size:
                return
            else:
                self.update_weights(style)

        elif style == "MBRL":
            if self.memory.len_env_buffer() and self.memory.len_model_buffer() <= self.batch_size:
                return
            else:
                #self.epsilon_greedy_function_update()
                self.update_weights(style, data_type)

    def update_weights(self, style, data_type):
        for it in range(1, self.G + 1):

            self.update_counter += 1
            if style == "MFRL":
                states, actions, rewards, next_states, dones = self.memory.sample_experience_from_env(self.batch_size)

            elif style == "MBRL":
                # I did this because the model predict bad things at the beginning, so I choose and update the policy
                # from previous real experiences and then move to the "dream" experiences using epsilon greedy.
                #if np.random.random() <= self.epsilon:
                    #states, actions, rewards, next_states, dones = self.memory.sample_experience_from_env(self.batch_size)
                    #print("sample from env data set")
                #else:
                    #print("sample from model data set")
                    #states, actions, rewards, next_states, dones = self.memory.sample_experience_from_model(self.batch_size)

                if data_type == 'env_data':
                    states, actions, rewards, next_states, dones = self.memory.sample_experience_from_env(self.batch_size)
                elif data_type == 'model_data':
                    states, actions, rewards, next_states, dones = self.memory.sample_experience_from_model(self.batch_size)
                else:
                    print("no data to sample")
                    return

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

                # ------- compute the target action
                next_actions = self.actor_target.forward(next_states)

                # add noise also here, paper mention this
                next_actions = next_actions.detach().numpy()  # tensor to numpy
                next_actions = next_actions + (np.random.normal(0, scale=0.2, size=self.num_actions))
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

    def sample_from_data_manually(self, batch_size, data):
        state_batch      = []
        action_batch     = []
        reward_batch     = []
        next_state_batch = []
        done_batch       = []

        batch = random.sample(data, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def transition_model_learn(self):

        if self.memory.len_env_buffer() <= self.batch_size_model:
           return

        else:
            # maybe a for loop here could help to improve the model learning

            states, actions, rewards, next_states, dones = self.memory.sample_experience_from_env(self.batch_size_model)

            # these values for states and next states include the target point but it needs to be removed
            states  = np.array(states)
            actions = np.array(actions)
            next_states = np.array(next_states)

            # Remove the target point from state and next state because there is no point in predict
            # the target value since it randomly changes in the environment.
            # target point here 2 values

            states      = states[:, :-2]
            next_states = next_states[:, :-2]

            states      = torch.FloatTensor(states)
            actions     = torch.FloatTensor(actions)
            next_states = torch.FloatTensor(next_states)

            distribution_probability_model_1 = self.pdf_transition_model_1.forward(states, actions)
            distribution_probability_model_2 = self.pdf_transition_model_2.forward(states, actions)
            distribution_probability_model_3 = self.pdf_transition_model_3.forward(states, actions)
            distribution_probability_model_4 = self.pdf_transition_model_4.forward(states, actions)
            distribution_probability_model_5 = self.pdf_transition_model_5.forward(states, actions)

            # calculate the loss
            loss_neg_log_likelihood_1 = - distribution_probability_model_1.log_prob(next_states)
            loss_neg_log_likelihood_2 = - distribution_probability_model_2.log_prob(next_states)
            loss_neg_log_likelihood_3 = - distribution_probability_model_3.log_prob(next_states)
            loss_neg_log_likelihood_4 = - distribution_probability_model_4.log_prob(next_states)
            loss_neg_log_likelihood_5 = - distribution_probability_model_5.log_prob(next_states)

            loss_neg_log_likelihood_1 = torch.mean(loss_neg_log_likelihood_1)
            loss_neg_log_likelihood_2 = torch.mean(loss_neg_log_likelihood_2)
            loss_neg_log_likelihood_3 = torch.mean(loss_neg_log_likelihood_3)
            loss_neg_log_likelihood_4 = torch.mean(loss_neg_log_likelihood_4)
            loss_neg_log_likelihood_5 = torch.mean(loss_neg_log_likelihood_5)

            self.pdf_transition_model_1.train()
            self.pdf_transition_1_optimizer.zero_grad()
            loss_neg_log_likelihood_1.backward()
            self.pdf_transition_1_optimizer.step()

            self.pdf_transition_model_2.train()
            self.pdf_transition_2_optimizer.zero_grad()
            loss_neg_log_likelihood_2.backward()
            self.pdf_transition_2_optimizer.step()

            self.pdf_transition_model_3.train()
            self.pdf_transition_3_optimizer.zero_grad()
            loss_neg_log_likelihood_3.backward()
            self.pdf_transition_3_optimizer.step()

            self.pdf_transition_model_4.train()
            self.pdf_transition_4_optimizer.zero_grad()
            loss_neg_log_likelihood_4.backward()
            self.pdf_transition_4_optimizer.step()

            self.pdf_transition_model_5.train()
            self.pdf_transition_5_optimizer.zero_grad()
            loss_neg_log_likelihood_5.backward()
            self.pdf_transition_5_optimizer.step()

            self.loss_transition_model_1.append(loss_neg_log_likelihood_1.item())
            self.loss_transition_model_2.append(loss_neg_log_likelihood_2.item())
            self.loss_transition_model_3.append(loss_neg_log_likelihood_3.item())
            self.loss_transition_model_4.append(loss_neg_log_likelihood_4.item())
            self.loss_transition_model_5.append(loss_neg_log_likelihood_5.item())

            print("Loss:", loss_neg_log_likelihood_1.item(), loss_neg_log_likelihood_2.item(),
                           loss_neg_log_likelihood_3.item(), loss_neg_log_likelihood_4.item(),
                           loss_neg_log_likelihood_5.item())

    def generate_dream_samples(self):

        if self.memory.len_env_buffer() <= self.batch_size:
            return

        else:
            for _ in range(1, self.M + 1):

                state, _, _, _, _ = self.memory.sample_experience_from_env(self.batch_size)

                state = np.array(state)  # --> (batch_size, 16)
                state_tensor = torch.FloatTensor(state)

                target_point = state[:, -2:]  # just target point only --> (sample_batch, 2)

                # Remove the target point from state
                state_input = state[:, :-2]  # (sample_batch, 14 values)
                state_input_tensor = torch.Tensor(state_input)  # torch.Size([sample_batch, 14])

                # generate sample batch size of actions from currently policy

                with torch.no_grad():
                    self.actor.eval()
                    action = self.actor.forward(state_tensor)  # this takes the 16 values tensor
                    action = action.detach().numpy()
                    action = action + (np.random.normal(0, scale=0.10, size=4))
                    action = np.clip(action, -1, 1)
                    self.actor.train()

                action_tensor = torch.FloatTensor(action)  # torch.Size([sample_batch, 4])

                # predict and generate new "dream" samples
                with torch.no_grad():
                    self.pdf_transition_model_1.eval()
                    function_generated_1 = self.pdf_transition_model_1.forward(state_input_tensor, action_tensor)
                    predicted_state_1    = function_generated_1.sample()       # torch.Size([sample_batch, 14])
                    predicted_state_1    = predicted_state_1.detach().numpy()  # (sample_batch, 14)

                with torch.no_grad():
                    self.pdf_transition_model_2.eval()
                    function_generated_2 = self.pdf_transition_model_2.forward(state_input_tensor, action_tensor)
                    predicted_state_2    = function_generated_2.sample()  # torch.Size([sample_batch, 14])
                    predicted_state_2    = predicted_state_2.detach().numpy()  # (sample_batch, 14)

                with torch.no_grad():
                    self.pdf_transition_model_3.eval()
                    function_generated_3 = self.pdf_transition_model_3.forward(state_input_tensor, action_tensor)
                    predicted_state_3    = function_generated_3.sample()  # torch.Size([sample_batch, 14])
                    predicted_state_3    = predicted_state_3.detach().numpy()  # (sample_batch, 14)

                with torch.no_grad():
                    self.pdf_transition_model_4.eval()
                    function_generated_4 = self.pdf_transition_model_4.forward(state_input_tensor, action_tensor)
                    predicted_state_4    = function_generated_4.sample()  # torch.Size([sample_batch, 14])
                    predicted_state_4    = predicted_state_4.detach().numpy()  # (sample_batch, 14)

                with torch.no_grad():
                    self.pdf_transition_model_5.eval()
                    function_generated_5 = self.pdf_transition_model_5.forward(state_input_tensor, action_tensor)
                    predicted_state_5    = function_generated_5.sample()  # torch.Size([sample_batch, 14])
                    predicted_state_5    = predicted_state_5.detach().numpy()  # (sample_batch, 14)

                next_state_imagined = np.mean(np.array([predicted_state_1, predicted_state_2, predicted_state_3,
                                                        predicted_state_4, predicted_state_5]), axis=0)

                # calculate the reward and distance based on the prediction and input not using a model-reward
                cube_position   = next_state_imagined[:, -2:]  # (sample_batch, 2)
                target_position = target_point  # (sample_batch, 2 )

                imagined_distance_cube_goal = np.linalg.norm(cube_position - target_position, axis=1)  # (batch, )

                for single_state, single_action, single_next_state, single_distance_dif, single_target in zip(state,
                                                                                                              action,
                                                                                                              next_state_imagined,
                                                                                                              imagined_distance_cube_goal,
                                                                                                              target_position):
                    if single_distance_dif <= 10:
                        done     = True
                        reward_d = np.float64(500)
                    else:
                        done = False
                        reward_d = -single_distance_dif

                    full_next_state = np.append(single_next_state, single_target)

                    state_to_save      = single_state
                    next_state_to_save = full_next_state
                    action_to_save     = single_action
                    reward_to_save      = reward_d

                    # todo posible punto de fallo, revisar en caso de algo no salga bien
                    self.add_imagined_experience_memory(state_to_save, action_to_save, reward_to_save, next_state_to_save, done)

    def save_rl_model(self, style):
        torch.save(self.actor.state_dict(),     f"models/TD3/TD3_actor_{style}.pth")
        torch.save(self.critic_q1.state_dict(), f'models/TD3/TD3_critic_1_{style}.pth')
        torch.save(self.critic_q2.state_dict(), f'models/TD3/TD3_critic_2_{style}.pth')
        print(f"models for {style} has been saved...")

    def load_rl_model(self, style):
        self.actor.load_state_dict(torch.load(f"models/TD3/TD3_actor_{style}.pth"))
        self.critic_q1.load_state_dict(torch.load(f'models/TD3/TD3_critic_1_{style}.pth'))
        self.critic_q2.load_state_dict(torch.load(f'models/TD3/TD3_critic_2_{style}.pth'))
        print(f"models for {style} has been loaded...")

    def save_transition_models(self):
        torch.save(self.pdf_transition_model_1.state_dict(), f"models/TD3/TD3_transition_model_1.pth")
        torch.save(self.pdf_transition_model_2.state_dict(), f"models/TD3/TD3_transition_model_2.pth")
        torch.save(self.pdf_transition_model_3.state_dict(), f"models/TD3/TD3_transition_model_3.pth")
        torch.save(self.pdf_transition_model_4.state_dict(), f"models/TD3/TD3_transition_model_4.pth")
        torch.save(self.pdf_transition_model_5.state_dict(), f"models/TD3/TD3_transition_model_5.pth")
        print(f"models for transitions has been saved...")

    def load_transition_models(self):
        self.pdf_transition_model_1.load_state_dict(torch.load(f"models/TD3/TD3_transition_model_1.pth"))
        self.pdf_transition_model_2.load_state_dict(torch.load(f"models/TD3/TD3_transition_model_2.pth"))
        self.pdf_transition_model_3.load_state_dict(torch.load(f"models/TD3/TD3_transition_model_3.pth"))
        self.pdf_transition_model_4.load_state_dict(torch.load(f"models/TD3/TD3_transition_model_4.pth"))
        self.pdf_transition_model_5.load_state_dict(torch.load(f"models/TD3/TD3_transition_model_5.pth"))
        print(f"transitions models loaded...")


def plot_reward_curves(rewards, avg_rewards, style, number=2):
    np.savetxt(f'results/TD3/TD3_rewards_{style}.txt', rewards)
    np.savetxt(f'results/TD3/TD3_avg_reward_{style}.txt', avg_rewards)

    plt.figure(number, figsize=(20, 10))
    plt.plot(rewards)
    plt.plot(avg_rewards)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig(f"results/TD3/TD3_{style}.png")
    #plt.show()


def save_training_model_curves(agent):
    '''
    plt.figure(1, figsize=(20, 10))
    plt.subplot(311)
    plt.plot(np.array(agent.loss_transition_model_1))
    plt.subplot(312)
    plt.plot(np.array(agent.loss_transition_model_2))
    plt.subplot(313)
    plt.plot(np.array(agent.loss_transition_model_3))
    plt.savefig(f"results/model_learning_curves.png")
    #plt.show()
    '''
    np.savetxt(f'results/TD3/TD3_model_1_loss.txt', agent.loss_transition_model_1)
    np.savetxt(f'results/TD3/TD3_model_2_loss.txt', agent.loss_transition_model_2)
    np.savetxt(f'results/TD3/TD3_model_3_loss.txt', agent.loss_transition_model_3)
    np.savetxt(f'results/TD3/TD3_model_4_loss.txt', agent.loss_transition_model_4)
    np.savetxt(f'results/TD3/TD3_model_5_loss.txt', agent.loss_transition_model_5)



def run_exploration(env, episodes, horizont, agent):
    mode = "Exploration"
    vector_experiences = []
    for episode in range(1, episodes+1):
        env.reset_env()
        for step in range(1, horizont+1):
            state, _ = env.state_space_function()
            action   = env.generate_sample_act()
            env.env_step(action)
            next_state, image_state = env.state_space_function()
            reward, done = env.calculate_reward()
            agent.add_real_experience_memory(state, action, reward, next_state, done)

            #experience_to_save = (state, action, reward, next_state, done)
            #vector_experiences.append(experience_to_save)

            env.env_render(image=image_state, episode=episode, step=step, done=done, mode=mode)
            if done:
                break
    print(f"******* -----{episodes} for exploration ended-----********* ")

    #np.save("exploration_experiences_record.npy", vector_experiences)  # Save exploration record
    #print("save exploration values")


def run_training(env, episodes, horizont, agent, style):
    mode        = f"Training TD3 {style}"
    rewards     = []
    avg_rewards = []

    for episode in range(1, episodes+1):
        env.reset_env()
        episode_reward = 0

        for step in range(1, horizont+1):

            state, _ = env.state_space_function()
            action   = agent.get_action_from_policy(state)
            noise    = np.random.normal(0, scale=0.15, size=4)
            action   = action + noise
            action   = np.clip(action, -1, 1)

            env.env_step(action)

            next_state, image_state = env.state_space_function()
            reward, done    = env.calculate_reward()
            episode_reward += reward
            agent.add_real_experience_memory(state, action, reward, next_state, done)
            env.env_render(image=image_state, episode=episode, step=step, done=done, mode=mode)

            if style == "MFRL":
                agent.step_training(style)

            elif style == "MBRL":
                #if step % 3 == 0:
                    #print("training the model")
                    #agent.transition_model_learn()

                agent.transition_model_learn()
                agent.generate_dream_samples()

                if episode <= 50:
                    data_type = 'env_data'
                    print('working with env data')
                    agent.step_training(style, data_type)
                else:
                    data_type = 'model_data'
                    print("working with model data")
                    agent.step_training(style, data_type)

            if done:
                break

        rewards.append(episode_reward)
        avg_rewards.append(np.mean(rewards[-100:]))

        print(f"******* -----Episode {episode} Ended-----********* ")
        print("Episode total reward:", episode_reward)

        if episode % 100 == 0:
            plot_reward_curves(rewards, avg_rewards, style, number=1)
            agent.save_transition_models()
            agent.save_rl_model(style)


    if style == "MBRL":
        save_training_model_curves(agent)
        agent.save_transition_models()
        plot_reward_curves(rewards, avg_rewards, style, number=1)
        agent.save_rl_model(style)

    elif style == "MFRL":
        plot_reward_curves(rewards, avg_rewards, style, number=2)
        agent.save_rl_model(style)
        np.savetxt(f'results/TD3/TD3_rewards_{style}.txt', rewards)
        np.savetxt(f'results/TD3/TD3_avg_reward_{style}.txt', avg_rewards)

    print(f"******* -----{episodes} episodes for training ended-----********* ")


def run_testing(env, horizont, agent, style):

    agent.load_rl_model(style)
    mode        = f"Testing {style}"
    rewards     = []
    episodes_test = 100

    for episode in range(1, episodes_test+1):
        env.reset_env()
        episode_reward = 0

        for step in range(horizont):
            print(f"-------Episode:{episode + 1} Step:{step + 1}---------")
            state, _ = env.state_space_function()
            action   = agent.get_action_from_policy(state)
            env.env_step(action)
            next_state, image_state = env.state_space_function()
            reward, done    = env.calculate_reward()
            episode_reward += reward
            env.env_render(image=image_state, episode=episode, step=step, done=done, mode=mode)

            if done:
                break

        print("Episode total reward:", episode_reward)
        rewards.append(episode_reward)

    plt.figure(3, figsize=(20, 10))
    plt.plot(rewards)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()


def evaluate_model_transition(agent, env):

    agent.load_transition_models()

    for episode in range(1, 10):
        env.reset_env()

        for step in range(1, 6):
            state, _ = env.state_space_function()
            action = env.generate_sample_act()
            env.env_step(action)
            next_state, _ = env.state_space_function()

            state = np.array(state)
            state_input  = state[:-2]  # Remove the target point from state
            state_tensor = torch.FloatTensor(state_input)
            state_tensor = state_tensor.unsqueeze(0)  # torch.Size([1, 14])

            action_tensor = torch.FloatTensor(action)
            action_tensor = action_tensor.unsqueeze(0)  # torch.Size([1, 4])

            agent.pdf_transition_model_1.eval()
            with torch.no_grad():
                function_generated_1 = agent.pdf_transition_model_1.forward(state_tensor, action_tensor)
                predicted_state_1    = function_generated_1.sample()
                predicted_state_1    = predicted_state_1.detach().numpy()

            agent.pdf_transition_model_2.eval()
            with torch.no_grad():
                function_generated_2 = agent.pdf_transition_model_2.forward(state_tensor, action_tensor)
                predicted_state_2    = function_generated_2.sample()
                predicted_state_2    = predicted_state_2.detach().numpy()

            agent.pdf_transition_model_3.eval()
            with torch.no_grad():
                function_generated_3 = agent.pdf_transition_model_3.forward(state_tensor, action_tensor)
                predicted_state_3    = function_generated_3.sample()
                predicted_state_3    = predicted_state_3.detach().numpy()

            agent.pdf_transition_model_4.eval()
            with torch.no_grad():
                function_generated_4 = agent.pdf_transition_model_4.forward(state_tensor, action_tensor)
                predicted_state_4    = function_generated_4.sample()
                predicted_state_4    = predicted_state_4.detach().numpy()

            agent.pdf_transition_model_5.eval()
            with torch.no_grad():
                function_generated_5 = agent.pdf_transition_model_5.forward(state_tensor, action_tensor)
                predicted_state_5    = function_generated_5.sample()
                predicted_state_5    = predicted_state_5.detach().numpy()

            next_state_predicted = np.mean(np.array([predicted_state_1[0], predicted_state_2[0], predicted_state_3[0],
                                                     predicted_state_4[0], predicted_state_5[0]]), axis=0)

            print("true:")
            print(next_state[:-2])

            print("prediction:")
            print(next_state_predicted)
            print("-------------")


def main_run(style="MFRL"):

    env   = RL_ENV()
    agent = TD3_Agent(env)

    num_exploration_episodes  = 500
    num_episodes_training     = 1000
    episode_horizont          = 6

    run_exploration(env, num_exploration_episodes, episode_horizont, agent)
    run_training(env, num_episodes_training, episode_horizont, agent, style)
    evaluate_model_transition(agent, env)

    # run_testing(env, episode_horizont, agent, style)

if __name__ == "__main__":
    #model = "MBRL"
    model = "MFRL"
    main_run(model)


