# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 03:37:02 2023

@author: chun
"""

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# Create NN architecture
class Network(nn.Module):
    
    def __init__(self, input_size, output_size):
        super(Network, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, 30)
        self.fc3 = nn.Linear(30, output_size)
    
    def forward(self, state):
        X = F.relu(self.fc1(state))
        Y = F.relu(self.fc2(X))
        q_values = self.fc3(Y)
        return q_values

# Experience replay
class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
    
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            self.memory.popleft()
    
    def sample(self, batch_size):
        # if list = ((s1, a1, r1), (s2, a2, r2)), then zip(*list) = ((s1, s2), (a1, a2), (r1, r2))
        # Since that would could form the (states, actions, rewards) format for us.
        samples = zip(*random.sample(self.memory, batch_size))
        
        # Convert torch tensor into Pytorch variable(containing tensor and gradient)
        return map(lambda x: Variable(torch.cat(x, 0)), samples)

# Deep Q learning
class Dqn():
    
    def __init__(self, input_size, output_size, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, output_size)
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        # fake dimension for pytorch seeing state as a batch
        # ex. x = torch.tensor([1, 2, 3, 4])
        # torch.unsqueeze(x, 0)
        # >>> tensor([[ 1,  2,  3,  4]])
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
    
    def select_action(self, state):
        # State should be a torch variable (convert form torch tensor)
        # When the t para is higher, model has more chance to pick the highest propobility action
        # If t para is 0, actions is random
        probs = F.softmax(self.model(Variable(state, volatile = True)) * 75) # temperature para = 75
        action = probs.multinomial()
        return action.data[0, 0]
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        # The output is the selected action's Q value
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        # Get the max Q value from the next_state
        next_outputs =self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma * next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        # Reinitialize optimizer in each iteration
        self.optimizer.zero_grad()
        td_loss.backward(retain_graph = True) # imporve training performance
        # Update the weight
        self.optimizer.step()
    
    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        # torch.LongTensor convert int value to torch tensor
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action

    def score(self):
        return sum(self.reward_window) / (len(self.reward_window) + 1.0)
    
    def save(self):
        torch.save({'state_dict' : self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict()},
                   'last_brain.pth')
        print("Model saved.")
        
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> Loading checkpoint...")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("Model loaded!")
        else:
            print("No checkpoint found...")
        






            
        
        
        
        
        

