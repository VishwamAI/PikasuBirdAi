import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class ModelTrainer(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(ModelTrainer, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculate the size of the output from the last convolutional layer
        conv_out_size = self._get_conv_out(input_shape)

        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, n_actions)

        self.optimizer = optim.Adam(self.parameters())
        self.loss_fn = nn.MSELoss()

    def _get_conv_out(self, shape):
        o = self.conv1(torch.zeros(1, *shape))
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def predict(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        return self.forward(state)

    def update(self, states, actions, rewards, next_states, dones):
        states = torch.FloatTensor(np.array(states))
        next_states = torch.FloatTensor(np.array(next_states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        current_q_values = self.forward(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.forward(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * 0.99 * next_q_values

        loss = self.loss_fn(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()