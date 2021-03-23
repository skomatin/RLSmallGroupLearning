"""
This file implements the REINFORCE algorithm on the CartPole environment.
This starter code was obtained from https://towardsdatascience.com/learning-reinforcement-learning-reinforce-with-pytorch-5e8ad7fc7da0
We modified it significantly to enable parallel training of actors.
"""

import numpy as np
import matplotlib.pyplot as plt
import gym
import sys
import copy

import torch
from torch import nn
from torch import optim

print(sys.version)
print(torch.__version__)
print(torch.version.cuda)

def gen(listIn = [1,2,3,4,5]):
    for i in listIn:
        yield i

def discount_rewards(rewards, gamma=0.99):
    r = np.array([gamma**i * rewards[i]
                  for i in range(len(rewards))])
    # Reverse the array direction for cumsum and then
    # revert back to the original order
    r = r[::-1].cumsum()[::-1]
    return r - r.mean()

class policy_estimator():
    def __init__(self, env):
        self.n_inputs = env.observation_space.shape[0]
        self.n_outputs = env.action_space.n

        print((nn.Linear(self.n_inputs, 16), nn.ReLU(), nn.Linear(16, self.n_outputs),nn.Softmax(dim=-1)))

        # Define network
        self.network = nn.Sequential(
            nn.Linear(self.n_inputs, 16),
            nn.ReLU(),
            nn.Linear(16, self.n_outputs),
            nn.Softmax(dim=-1))

    def predict(self, state):
        action_probs = self.network(torch.FloatTensor(state))
        return action_probs


v = 0
def reinforce(env, policy_estimator, num_episodes=1000,
              batch_size=10, gamma=0.8):

    # Set up lists to hold results
    total_rewards = []
    batch_rewards = []
    batch_actions = []
    batch_states = []
    batch_counter = 1

    # Define optimizer
    optimizer = optim.Adam(policy_estimator.network.parameters(), lr=0.01)

    action_space = np.arange(env.action_space.n)
    for ep in range(num_episodes):
        s_0 = env.reset()
        states = []
        rewards = []
        actions = []
        complete = False
        while complete == False:
            # Get actions and convert to numpy array
            action_probs = policy_estimator.predict(s_0).detach().numpy()
            action = np.random.choice(action_space, p=action_probs)
            s_1, r, complete, _ = env.step(action)

            states.append(s_0)
            rewards.append(r)
            actions.append(action)
            s_0 = s_1

            # If complete, batch data
            if complete:
                batch_rewards.extend(discount_rewards(rewards, gamma))
                batch_states.extend(states)
                batch_actions.extend(actions)
                batch_counter += 1
                total_rewards.append(sum(rewards))

                # If batch is complete, update network
                if batch_counter == batch_size:
                    if batch_size >5:
                        batch_size -= 1
                    optimizer.zero_grad()
                    state_tensor = torch.FloatTensor(batch_states)
                    reward_tensor = torch.FloatTensor(batch_rewards)
                    # Actions are used as indices, must be LongTensor
                    action_tensor = torch.LongTensor(batch_actions)

                    # Calculate loss
                    logprob = torch.log(policy_estimator.predict(state_tensor))
                    selected_logprobs = reward_tensor * logprob[np.arange(len(action_tensor)), action_tensor]
                    loss = -selected_logprobs.mean()
                    # print("\nloss:\n", loss)
                    # print("\nloss+loss:\n", loss+loss)
                    # print("\nloss*2:\n", loss*2)
                    # print("\nloss+0:\n", loss+1)

                    # Calculate gradients
                    a = copy.deepcopy(list(policy_estimator.network.parameters()))
                    loss.backward()# Apply gradients
                    optimizer.step()
                    b = list(policy_estimator.network.parameters())
                    deltaPolicyEstimator = [ a-b for (a,b) in zip(a,b) ]




                    batch_rewards = []
                    batch_actions = []
                    batch_states = []
                    batch_counter = 1

                # Print running average
                print("\rEp: {} Average of last 10: {:.2f}".format(
                    ep + 1, np.mean(total_rewards[-10:])), end="")

    return total_rewards


env = gym.make('CartPole-v0')
s = env.reset()
pe = policy_estimator(env)
# print(pe.predict(s))
# print(pe.network(torch.FloatTensor(s)))


rewards = reinforce(env, pe)
window = 10
smoothed_rewards = [np.mean(rewards[i-window:i+1]) if i > window
                    else np.mean(rewards[:i+1]) for i in range(len(rewards))]

plt.figure(figsize=(12,8))
# plt.plot(rewards)
plt.plot(smoothed_rewards)
plt.ylabel('Total Rewards')
plt.xlabel('Episodes')
plt.show()