import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size) # first layer
        self.linear2 = nn.Linear(hidden_size, output_size) # output layer

    def forward(self, x):
        x = F.relu(self.linear1(x)) # relu for some non-linearity
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path) # make folder if it doesn't exist
            os.makedirs(model_folder_path) # make folder if it doesn't exist
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name) # save the weights
        torch.save(self.state_dict(), file_name) # save the weights


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr) # adam optimizer
        self.criterion = nn.MSELoss() # mean squared error
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr) # adam optimizer
        self.criterion = nn.MSELoss() # mean squared error

    def train_step(self, state, action, reward, next_state, gameOver):
        # convert everything to tensors just in case
        # convert everything to tensors just in case
        state = np.array(state, dtype=float)
        state = torch.tensor(state, dtype=torch.float)
        next_state = np.array(next_state, dtype=float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            gameOver = [gameOver]  # make it a list of length 1

        # predict Q values with the current state
        # predict Q values with the current state
        pred = self.model(state)
        target = pred.clone()
        for i in range(len(gameOver)):
            newQ = reward[i]
            if not gameOver[i]:
                newQ = reward[i] + self.gamma * torch.max(self.model(next_state[i])) # bellman equation
            target[i][torch.argmax(action[i]).item()] = newQ # update only the action taken
            newQ = reward[i] + self.gamma * torch.max(self.model(next_state[i])) # bellman equation
            target[i][torch.argmax(action[i]).item()] = newQ # update only the action taken

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred) # how far off were we?
        loss = self.criterion(target, pred) # how far off were we?
        loss.backward()
        self.optimizer.step() # do the update

        self.optimizer.step() # do the update