# Section 5: Rapid Prototyping and Iteration
# Using reinforcement learning (RL) with PyTorch for agent optimization in construction simulation.
# Prototype: Train an agent to optimize resource allocation in projects.

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Simple RL environment for construction resource allocation
class ConstructionEnv:
    def __init__(self):
        self.state = np.array([10, 5])  # [budget left, tasks left]
        self.action_space = [0, 1]  # 0: allocate low, 1: allocate high

    def step(self, action):
        reward = 0
        if action == 1:
            self.state[0] -= 3
            self.state[1] -= 2
            reward += 2 if self.state[1] > 0 else -1
        else:
            self.state[0] -= 1
            self.state[1] -= 1
            reward += 1
        done = self.state[1] <= 0 or self.state[0] <= 0
        return self.state, reward, done

# Policy network
class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 2)

    def forward(self, x):
        return torch.softmax(self.fc(x), dim=-1)

policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=0.01)

# Train (prototype iteration)
env = ConstructionEnv()
for episode in range(100):
    state = torch.tensor(env.state, dtype=torch.float32)
    probs = policy(state)
    action = np.random.choice([0, 1], p=probs.detach().numpy())
    next_state, reward, done = env.step(action)
    loss = -torch.log(probs[action]) * reward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if done:
        print(f"Episode {episode}: Final state {next_state}, Reward {reward}")
        env = ConstructionEnv()  # Reset for iteration
