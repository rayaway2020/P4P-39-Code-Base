import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Critic, self).__init__()

        self.h_linear_1 = nn.Linear(in_features=input_size,     out_features=hidden_size[0])
        self.bn1 = nn.BatchNorm1d(hidden_size[0])
        self.h_linear_2 = nn.Linear(in_features=hidden_size[0], out_features=hidden_size[1])
        self.h_linear_3 = nn.Linear(in_features=hidden_size[1], out_features=output_size)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)   # Concatenates the seq tensors in the given dimension
        x = torch.relu(self.bn1(self.h_linear_1(x)))
        x = torch.relu(self.h_linear_2(x))
        x = self.h_linear_3(x)                  # No activation function here
        return x


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Actor, self).__init__()

        self.h_linear_1 = nn.Linear(in_features=input_size,     out_features=hidden_size[0])
        self.bn1 = nn.BatchNorm1d(hidden_size[0])
        self.h_linear_2 = nn.Linear(in_features=hidden_size[0], out_features=hidden_size[1])
        self.h_linear_3 = nn.Linear(in_features=hidden_size[1], out_features=output_size)


    def forward(self, state):
        x = torch.relu(self.bn1(self.h_linear_1(state)))
        x = torch.relu(self.h_linear_2(x))
        x = torch.tanh(self.h_linear_3(x))
        #x = torch.sigmoid(self.h_linear_4(x)) * 700
        return x

# ---------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------


class CriticTD3(nn.Module):
    def __init__(self, input_size, hidden_size, num_actions):
        super(CriticTD3, self).__init__()

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


class ActorTD3(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ActorTD3, self).__init__()

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

# ---------------------------------------------Image State Space NN ---------------------------------------#
latent_dim = 256


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Actor_Img(nn.Module):
    def __init__(self, action_dim):
        super(Actor_Img, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 5, 2, padding=2),
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

            nn.Conv2d(128, 256, 5, 2, padding=2),  ## output size: [256, 1, 1]
            Flatten(),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(latent_dim, 30),
            nn.BatchNorm1d(30),
            nn.ReLU(),
            nn.Linear(30, action_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc1(x)
        return x


class Critic_Img(nn.Module):

    def __init__(self, action_dim):
        super(Critic_Img, self).__init__()

        self.encoder_critic_1 = nn.Sequential(
            nn.Conv2d(3, 16, 5, 2, padding=2),
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
            Flatten(),  ## output: 256
        )

        self.fc1 = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 30),
            nn.ReLU(),
            nn.Linear(30, 1),
        )


    def forward(self, x, a):
        x1 = x
        x1 = self.encoder_critic_1(x1)
        x1 = torch.cat([x1, a], dim=1)
        x1 = self.fc1(x1)

        return x1