import math
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torch.distributions import Categorical

from rlturtle.exception import OffScreenException, UserConditionException
from rlturtle.rlturtle import RLTurtle

FORWARD_10 = 0
LEFT_90 = 1
RIGHT_90 = 2

class TurtleEnvironment():
    
    def __init__(self):
        self.tortoise = RLTurtle(width=200, height=200)
        self.tortoise.turtle.screensize(140,140)
        self.tortoise.turtle.setup(200,200)
        self.tortoise.turtle.delay(0)
        self.tortoise.turtle.pensize(3)
        def self_collision_test(x, y, deg, canvas):
            if int(canvas[int(y)][int(x)]) != 0:
                return True
            return False
        self.tortoise.add_condition(self_collision_test)
        self.reset()
    
    def position(self):
        return self.x, self.y
    
    def heading(self):
        return self.deg
        
    def step(self, action):
        broke = False
        if action != 0 and self.last_action != 0:
            return self.tortoise.canvas, -1, True, None
        self.last_action = action
        reward = 0
        if action == FORWARD_10:
            try:
                self.tortoise.forward(10)
            except OffScreenException:
                return self.tortoise.canvas, self.cum_reward, True, None
            except UserConditionException:
                return self.tortoise.canvas, self.cum_reward, True, None
            self.cum_reward += 2
            reward = 1
        if action == LEFT_90:
            self.tortoise.left(90)
            self.cum_reward += 1
            reward = 1
        if action == RIGHT_90:
            self.tortoise.right(90)
            self.cum_reward += 1
            reward = 1
        if broke == True:
            reward = self.cum_reward
        return self.tortoise.canvas, reward, True if broke == True else False, None
    
    def reset(self):
        self.cum_reward = torch.zeros(1)
        self.last_action = 0
        self.tortoise.reset()
        return self.tortoise.canvas

env = TurtleEnvironment()

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.conv1 = nn.Conv2d(3, 36, kernel_size=2, stride=1, padding=2)
        self.conv2 = nn.Conv2d(36, 1, kernel_size=2, stride=1, padding=1)
        self.hidden_1 = nn.Linear(41616, 1000)
        self.hidden_2 = nn.Linear(1000, 200)
        self.dropout = nn.Dropout(p=0.2)
        self.hidden_3 = nn.Linear(200, 30)
        self.interim1 = nn.Linear(30, 20)
        self.interim2 = nn.Linear(30, 20)
        self.action_head = nn.Linear(20, 3)
        self.value_head = nn.Linear(20, 1)
        self.saved_actions = []
        self.rewards = []
        self.eps_start = 0.6
        self.eps_final = 0.01
        self.eps_decay = 10000.

    def forward(self, x):
        x = x.reshape(1,3,200,200).cuda()
        x = torch.sigmoid(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.hidden_1(x.reshape(1, -1)))
        x = F.relu(self.hidden_2(x))
        x = self.dropout(x)
        x = F.relu(self.hidden_3(x))
        action_scores = self.action_head(F.relu(self.interim1(x)))
        state_values = torch.tanh(self.value_head(F.relu(self.interim2(x))))
        return F.softmax(action_scores, dim=-1), state_values
    
    def eps_by_frame(self, frame_idx):
        return self.eps_final + (self.eps_start - self.eps_final) * math.exp(-1. * frame_idx / self.eps_decay)

    def act(self, state, last_action, episode):
        probs, state_value = self.forward(state)
        if random.random() < self.eps_by_frame(episode):
            action = torch.FloatTensor([random.randrange(0, 3)]).cuda()
            m = Categorical(probs)
            self.saved_actions.append((m.log_prob(action), state_value))
            return action.item()
        else:
            m = Categorical(probs)
            action = m.sample()
            self.saved_actions.append((m.log_prob(action), state_value))
            return action.item()

model = Policy()
model = model.cuda()
optimizer = optim.Adam(model.parameters())

env.reset()

gamma = 0.99
log_interval = 100

def finish_episode(episode):
    R = 0
    saved_actions = model.saved_actions
    policy_losses = []
    value_losses = []
    rewards = []
    for r in model.rewards[::-1]:
        R = (r + (gamma * R)) 
        epsilon = (1 * (torch.rand(1) - 0.5) * r / 100) * (1 / episode)
        R += epsilon
        rewards.insert(0, R)
    rewards = torch.FloatTensor(rewards)
    rewards = rewards.cuda()
    
    shuf = list(zip(saved_actions, rewards))
    random.shuffle(shuf)
    
    for (log_prob, value), r in shuf:
        reward = r - value.item()
        policy_losses.append(-log_prob * reward)
        value_losses.append(F.smooth_l1_loss(value.squeeze(), torch.tensor([r]).cuda()))
    optimizer.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    loss.backward()
    optimizer.step()
    del model.rewards[:]
    del model.saved_actions[:]

running_reward = 0
for episode in range(0, 500000):
    state = env.reset()
    oldest_state = state.clone()
    older_state = state.clone()
    old_state = state.clone()
    reward = 0
    done = False
    msg = None
    last_action = 0.
    while not done:
        action = model.act(torch.cat((state, env.tortoise.prev_frames[0], env.tortoise.prev_frames[1])), episode)
        last_action = action
        oldest_state = older_state.clone()
        older_state = old_state.clone()
        old_state = state.clone()
        state, reward, done, msg = env.step(action)
        model.rewards.append(reward)
        if done:
            break
    running_reward = running_reward * (1 - 1/log_interval) + reward * (1/log_interval)
    finish_episode(episode + 1)
    if episode % log_interval == 0:
        print("{}".format(running_reward))