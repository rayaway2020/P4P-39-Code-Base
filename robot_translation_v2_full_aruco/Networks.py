
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN_Net(nn.Module):

    def __init__(self, input_size, num_actions):
        super(DQN_Net, self).__init__()

        self.fc1 = nn.Linear(input_size, 256)
        self.a_head = nn.Linear(256, num_actions)
        self.v_head = nn.Linear(256, 1)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        a = self.a_head(x) - self.a_head(x).mean(1, keepdim=True)
        v = self.v_head(x)
        action_scores = a + v
        return action_scores

class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class DQN_Img_Net(nn.Module):

    def __init__(self, input_size, num_actions):
        super(DQN_Img_Net, self).__init__()
        self.encoder_critic_1 = nn.Sequential(
            nn.Conv2d(1, input_size, 5, 2, padding=2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(input_size),
            torch.nn.Conv2d(input_size, 32, 5, 2, padding=2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(32, 64, 5, 2, padding=2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(64, 128, 5, 4, padding=2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.Conv2d(128, 256, 5, 2, padding=2),
            Flatten(),  ## output: 256  1024
        )

        self.fc1 = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

    def forward(self, x):
        x1 = x
        x1 = self.encoder_critic_1(x1)
        x1 = torch.cat([x1], dim=1)
        x1 = self.fc1(x1)
        return x1

class Duelling_DQN_Net(nn.Module):

    def __init__(self, input_size, num_actions):
        self.input_size = input_size
        self.num_actions = num_actions
        super(Duelling_DQN_Net, self).__init__()

        self.fc1_adv = nn.Linear(in_features=input_size, out_features=256)
        self.fc1_val = nn.Linear(in_features=input_size, out_features=256)

        self.fc2_adv = nn.Linear(in_features=256, out_features=num_actions)
        self.fc2_val = nn.Linear(in_features=256, out_features=1)

        self.relu = nn.ReLU()


    def forward(self, x):
        adv = self.relu(self.fc1_adv(x))
        val = self.relu(self.fc1_val(x))
        adv = self.fc2_adv(adv)
        val = self.fc2_val(val).expand(x.size(0), self.num_actions)
        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.num_actions)
        return x


class Duelling_DQN_Img_Net(nn.Module):

    def __init__(self, input_size, num_actions):
        super(Duelling_DQN_Img_Net, self).__init__()
        self.input_size = input_size
        self.num_actions = num_actions
        self.encoder_critic_1 = nn.Sequential(
            nn.Conv2d(1, input_size, 5, 2, padding=2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(16),
            torch.nn.Conv2d(input_size, 32, 5, 2, padding=2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(32, 64, 5, 2, padding=2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(64, 128, 5, 4, padding=2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.Conv2d(128, 256, 5, 2, padding=2),
            Flatten(),
        )

        self.fc2_adv = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )

        self.fc2_val = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x1 = x
        x1 = self.encoder_critic_1(x1)
        x1 = torch.cat([x1], dim=1)
        adv = self.fc2_adv(x1)
        val = self.fc2_val(x1).expand(x1.size(0), self.num_actions)
        x1 = val + adv - adv.mean(1).unsqueeze(1).expand(x1.size(0), self.num_actions)
        return x1


class Critic_NN(nn.Module):
    def __init__(self, input_size, hidden_size, num_actions):
        super(Critic_NN, self).__init__()
        self.h_linear_1 = nn.Linear(in_features=input_size,     out_features=hidden_size[0])
        self.h_linear_2 = nn.Linear(in_features=hidden_size[0], out_features=hidden_size[1])
        self.h_linear_3 = nn.Linear(in_features=hidden_size[1], out_features=hidden_size[2])
        self.h_linear_4 = nn.Linear(in_features=hidden_size[2], out_features=num_actions)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)   # Concatenates the seq tensors in the given dimension
        x = torch.relu(self.h_linear_1(x))
        x = torch.relu(self.h_linear_2(x))
        x = torch.relu(self.h_linear_3(x))
        x = self.h_linear_4(x)                  # No activation function here
        return x


class Actor_NN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Actor_NN, self).__init__()
        self.h_linear_1 = nn.Linear(in_features=input_size,     out_features=hidden_size[0])
        self.h_linear_2 = nn.Linear(in_features=hidden_size[0], out_features=hidden_size[1])
        self.h_linear_3 = nn.Linear(in_features=hidden_size[1], out_features=hidden_size[2])
        self.bn1 = nn.BatchNorm1d(hidden_size[2])
        self.h_linear_4 = nn.Linear(in_features=hidden_size[2], out_features=output_size)

    def forward(self, state):
        x = torch.relu(self.h_linear_1(state))
        x = torch.relu(self.h_linear_2(x))
        x = torch.relu(self.bn1(self.h_linear_3(x)))
        x = torch.tanh(self.h_linear_4(x))
        return x
# ------------------------------------------------------------------------#

# -------------------Networks for SAC ------------------------------------#


class ValueNetworkSAC(nn.Module):
    def __init__(self, state_dim, hidden_dim, init_w=3e-3):
        super(ValueNetworkSAC, self).__init__()

        self.linear1 = nn.Linear(state_dim, hidden_dim[0])
        self.linear2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.linear3 = nn.Linear(hidden_dim[1], 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = torch.relu(self.linear1(state))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class SoftQNetworkSAC(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(SoftQNetworkSAC, self).__init__()
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size[0])
        self.linear2 = nn.Linear(hidden_size[0], hidden_size[0])
        self.linear3 = nn.Linear(hidden_size[1], 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class PolicyNetworkSAC(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetworkSAC, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(num_inputs, hidden_size[0])
        self.linear2 = nn.Linear(hidden_size[0], hidden_size[1])

        self.mean_linear = nn.Linear(hidden_size[1], num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(hidden_size[1], num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = torch.relu(self.linear1(state))
        x = torch.relu(self.linear2(x))

        mean    = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std
# ----------------------------------------------------------------------------#
# ----------------------------------------------------------------------------#


# -----------------Networks for Image Representation -------------------------#
class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Actor_Img(nn.Module):
    def __init__(self, action_dim):
        super(Actor_Img, self).__init__()
        self.latent_dim = 1024

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 5, 2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 32, 5, 2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, 5, 2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, 5, 4, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, 5, 2, padding=2),  ## output size: [1, 256, 2, 2]
            Flatten(),  # size: [1, 1024]
        )

        self.fc1 = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, action_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc1(x)
        return x



class Critic_Img(nn.Module):
    def __init__(self, action_dim):
        super(Critic_Img, self).__init__()
        self.latent_dim = 1024

        self.encoder_critic_1 = nn.Sequential(
            nn.Conv2d(1, 16, 5, 2, padding=2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(16),
            torch.nn.Conv2d(16, 32, 5, 2, padding=2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(32, 64, 5, 2, padding=2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(64, 128, 5, 4, padding=2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.Conv2d(128, 256, 5, 2, padding=2),
            Flatten(),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(self.latent_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x, a):
        x1 = x
        x1 = self.encoder_critic_1(x1)
        x1 = torch.cat([x1, a], dim=1)
        x1 = self.fc1(x1)
        return x1


# -----------------------------------Networks for Model Learning --------------------------------------------#
# ------------------------------------------------------------------------------------------------------------
class ModelNet_probabilistic_transition(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ModelNet_probabilistic_transition, self).__init__()

        self.number_mixture_gaussians = 3

        self.initial_shared_layer = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, hidden_size[0], bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size[0]),
            nn.Linear(hidden_size[0], hidden_size[1], bias=True),
            nn.ReLU(),
        )

        self.phi_layer = nn.Sequential(
            nn.BatchNorm1d(hidden_size[1]),
            nn.Linear(hidden_size[1], hidden_size[2], bias=True),
            nn.ReLU(),
            nn.Linear(hidden_size[2], 14 * self.number_mixture_gaussians),
            nn.Softmax()
        )

        self.mean_layer = nn.Sequential(
            nn.BatchNorm1d(hidden_size[1]),
            nn.Linear(hidden_size[1], hidden_size[2], bias=True),
            nn.ReLU(),
            nn.Linear(hidden_size[2], 14 * self.number_mixture_gaussians)
        )

        self.std_layer = nn.Sequential(
            nn.BatchNorm1d(hidden_size[1]),
            nn.Linear(hidden_size[1], hidden_size[2], bias=True),
            nn.ReLU(),
            nn.Linear(hidden_size[2], 14 * self.number_mixture_gaussians),
            nn.Softplus()
        )

        nn.init.xavier_normal_(self.initial_shared_layer[1].weight.data)
        nn.init.ones_(self.initial_shared_layer[1].bias.data)
        nn.init.xavier_normal_(self.initial_shared_layer[4].weight.data)
        nn.init.ones_(self.initial_shared_layer[4].bias.data)

        nn.init.xavier_normal_(self.mean_layer[1].weight.data)
        nn.init.ones_(self.mean_layer[1].bias.data)
        nn.init.xavier_normal_(self.mean_layer[3].weight.data)
        nn.init.ones_(self.mean_layer[3].bias.data)

        nn.init.xavier_normal_(self.std_layer[1].weight.data)
        nn.init.ones_(self.std_layer[1].bias.data)
        nn.init.xavier_normal_(self.std_layer[3].weight.data)
        nn.init.ones_(self.std_layer[3].bias.data)

        nn.init.xavier_normal_(self.phi_layer[1].weight.data)
        nn.init.ones_(self.phi_layer[1].bias.data)
        nn.init.xavier_normal_(self.phi_layer[3].weight.data)
        nn.init.ones_(self.phi_layer[3].bias.data)


    def forward(self, state, action):
        x   = torch.cat([state, action], dim=1)  # Concatenates the seq tensors in the given dimension
        x   = self.initial_shared_layer(x)

        u   = self.mean_layer(x)
        std = torch.clamp(self.std_layer(x), min=0.001)
        phi = self.phi_layer(x)

        u   = torch.reshape(u,   (-1, 14, self.number_mixture_gaussians))
        std = torch.reshape(std, (-1, 14, self.number_mixture_gaussians))
        phi = torch.reshape(phi, (-1, 14, self.number_mixture_gaussians))

        mix        = torch.distributions.Categorical(phi)
        norm_distr = torch.distributions.Normal(u, std)

        #comp = torch.distributions.Independent(norm_distr, 1)
        gmm = torch.distributions.MixtureSameFamily(mix, norm_distr)

        return gmm

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------



class ModelNet_deterministic_transition(nn.Module):

    def __init__(self, input_size, hidden_size):

        super(ModelNet_deterministic_transition, self).__init__()

        self.initial_shared_layer = nn.Sequential(

            nn.Linear(input_size, hidden_size[0], bias=True),
            nn.BatchNorm1d(hidden_size[0]),
            nn.ReLU(),

            nn.Linear(hidden_size[0], hidden_size[1], bias=True),
            nn.BatchNorm1d(hidden_size[1]),
            nn.ReLU(),
        )

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size[1], hidden_size[2], bias=True),
            #nn.BatchNorm1d(hidden_size[2]),
            nn.ReLU(),
            nn.Linear(hidden_size[2], 14)
        )

    def forward(self, state, action):
        x   = torch.cat([state, action], dim=1)  # Concatenates the seq tensors in the given dimension
        x   = self.initial_shared_layer(x)
        x   = self.output_layer(x)
        return x