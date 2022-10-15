

import torch
import torch.nn as nn
import torch.optim as optim

import random
import numpy as np

from Networks import ModelNet_probabilistic_transition, ModelNet_deterministic_transition



class model_learning_test:

    def __init__(self):

        self.transition_learning_rate  = 0.01
        self.hidden_size_network_model = [64, 64, 32]

        self.num_actions         = 4
        self.num_states_training = 14

        self.pdf_transition_model_1 = ModelNet_probabilistic_transition(self.num_states_training + self.num_actions, self.hidden_size_network_model)
        self.pdf_transition_model_2 = ModelNet_probabilistic_transition(self.num_states_training + self.num_actions, self.hidden_size_network_model)
        self.pdf_transition_model_3 = ModelNet_probabilistic_transition(self.num_states_training + self.num_actions, self.hidden_size_network_model)
        self.pdf_transition_model_4 = ModelNet_probabilistic_transition(self.num_states_training + self.num_actions, self.hidden_size_network_model)
        self.pdf_transition_model_5 = ModelNet_probabilistic_transition(self.num_states_training + self.num_actions, self.hidden_size_network_model)

        self.pdf_transition_1_optimizer = optim.Adam(self.pdf_transition_model_1.parameters(), lr=self.transition_learning_rate)
        self.pdf_transition_2_optimizer = optim.Adam(self.pdf_transition_model_2.parameters(), lr=self.transition_learning_rate)
        self.pdf_transition_3_optimizer = optim.Adam(self.pdf_transition_model_3.parameters(), lr=self.transition_learning_rate)
        self.pdf_transition_4_optimizer = optim.Adam(self.pdf_transition_model_4.parameters(), lr=self.transition_learning_rate)
        self.pdf_transition_5_optimizer = optim.Adam(self.pdf_transition_model_5.parameters(), lr=self.transition_learning_rate)




        self.deterministic_model_1 = ModelNet_deterministic_transition(self.num_states_training + self.num_actions, self.hidden_size_network_model)
        self.deterministic_1_optimizer  = optim.Adam(self.deterministic_model_1.parameters(),  lr=self.transition_learning_rate)


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
        data = np.load("exploration_experiences_record.npy", allow_pickle=True)
        data = list(data)
        inter_batch = 32

        model_mode = "probabilistic"
        #model_mode = "deterministic"


        for _ in range(1, 5):
            
            input_state_minibatch  = []
            output_state_minibatch = []
            states, actions, rewards, next_states, dones = self.sample_from_data_manually(inter_batch, data)

            states  = np.array(states)
            actions = np.array(actions)
            rewards = np.array(rewards).reshape(-1, 1)
            dones   = np.array(dones).reshape(-1, 1)
            next_states = np.array(next_states)

            for state in states:
                input_state = state[:-2]
                input_state_minibatch.append(input_state)

            for next_state in next_states:
                output_state = next_state[:-2]
                output_state_minibatch.append(output_state)

            states      = np.array(input_state_minibatch)
            next_states = np.array(output_state_minibatch)

            states  = torch.FloatTensor(states)
            actions = torch.FloatTensor(actions)
            rewards = torch.FloatTensor(rewards)
            dones   = torch.FloatTensor(dones)
            next_states = torch.FloatTensor(next_states)

            if model_mode == "probabilistic":
                
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

                print("Loss:", loss_neg_log_likelihood_1.item(), loss_neg_log_likelihood_2.item(),
                      loss_neg_log_likelihood_3.item(), loss_neg_log_likelihood_4.item(),
                      loss_neg_log_likelihood_5.item())
                
                
            elif model_mode == "deterministic":
                prediction_delta = self.deterministic_model_1.forward(states, actions)
                prediction_state = prediction_delta + states

                loss = nn.MSELoss()
                prediction_loss  = loss(prediction_state, next_states)

                self.deterministic_model_1.train()
                self.deterministic_1_optimizer.zero_grad()
                prediction_loss.backward()
                self.deterministic_1_optimizer.step()
                print(prediction_loss.item())


    def generate_dream_samples(self):

        data = np.load("exploration_experiences_record.npy", allow_pickle=True)
        data = list(data)
        state, actions, _, next_state_true, _ = self.sample_from_data_manually(32, data)

        state = np.array(state)
        state = state[:, :-2]  # 14 values
        state_tensor = torch.FloatTensor(state)  # torch.Size([batch_size, 14])

        actions = np.array(actions)
        actions = torch.FloatTensor(actions)  # torch.Size([batch_size, 4])

        '''
        state_input  = state[:-2]  # 14 values
        state_tensor = torch.FloatTensor(state_input)
        state_tensor = state_tensor.unsqueeze(0)  # torch.Size([1, 14])
        
        actions = np.array(actions)
        actions = torch.FloatTensor(actions)
         '''

        model_mode = "probabilistic"

        if model_mode == "probabilistic":
            self.pdf_transition_model_1.eval()
            with torch.no_grad():
                function_generated_1 = self.pdf_transition_model_1.forward(state_tensor, actions)
                predicted_state_1 = function_generated_1.sample()  # torch.Size([32, 14])
                predicted_state_1 = predicted_state_1.detach().numpy()  # (32, 14)

                #predicted_state_1 = torch.reshape(predicted_state_1, (1, 14))  # torch.Size([1, 15])
                #predicted_state_1 = predicted_state_1.detach().numpy()
                #print(predicted_state_1[0])
                #print(next_state_true[0][:-1])

                for generated_sample in predicted_state_1:
                    print(generated_sample.shape)

        elif model_mode == "deterministic":
            self.deterministic_model_1.eval()
            with torch.no_grad():
                function_generated_1 = self.deterministic_model_1.forward(state_tensor, actions)
                predicted_state = function_generated_1
                predicted_state = predicted_state + state_tensor
                predicted_state = predicted_state.detach().numpy()
                print(predicted_state[0])
                print(type(predicted_state))
                




process = model_learning_test()
process.transition_model_learn()
#process.generate_dream_samples()
