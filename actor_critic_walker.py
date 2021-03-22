"""
This file contains the implementation of the plain actor critic model tested with Bipedal Walker environment.
This bulk of the code has been obtained from https://github.com/philtabor/Youtube-Code-Repository/tree/master/ReinforcementLearning/PolicyGradient/actor_critic
and modified to implement it on the Bipedal Walker environment.
"""

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import gym

def plotLearning(scores, filename, x=None, window=5):
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-window):(t+1)])
    if x is None:
        x = [i for i in range(N)]
    plt.ylabel('Score')
    plt.xlabel('Game')
    plt.plot(x, running_avg)
    plt.savefig(filename)

class GenericNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims,
                 n_actions):
        super(GenericNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        # self.device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')
        # self.to(self.device)

    def forward(self, observation):
        state = T.tensor(observation, dtype=T.float)#.to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class Agent(object):
    def __init__(self, alpha, beta, input_dims, gamma=0.99, n_actions=2,
                 layer1_size=256, layer2_size=256, n_outputs=1):
        self.gamma = gamma
        self.log_probs = None
        self.n_outputs = n_outputs
        self.actor = GenericNetwork(alpha, input_dims, layer1_size,
                                           layer2_size, n_actions=n_actions)
        self.critic = GenericNetwork(beta, input_dims, layer1_size,
                                            layer2_size, n_actions=1)

    def choose_action(self, observation):
        n_actions = 4
        action_params = self.actor.forward(observation)#.to(self.actor.device)
        mu = action_params[:n_actions]
        sigma = action_params[n_actions:]
        sigma = T.exp(sigma)
        action_probs = T.distributions.Normal(mu, sigma)
        probs = action_probs.sample(sample_shape=T.Size([self.n_outputs]))
        self.log_probs = action_probs.log_prob(probs)#.to(self.actor.device)
        action = T.tanh(probs)

        return action

    def learn(self, state, reward, new_state, done):
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()

        critic_value_ = self.critic.forward(new_state)
        critic_value = self.critic.forward(state)
        reward = T.tensor(reward, dtype=T.float)#.to(self.actor.device)
        delta = ((reward + self.gamma*critic_value_*(1-int(done))) - \
                                                                critic_value)

        actor_loss = -self.log_probs * delta
        critic_loss = delta**2

        (actor_loss + critic_loss).backward(T.FloatTensor([1.0, 1.0, 1.0, 1.0]).reshape((1,4)))

        self.actor.optimizer.step()
        self.critic.optimizer.step()

if __name__ == '__main__':
    agent = Agent(alpha=0.000005, beta=0.00001, input_dims=[24], gamma=0.99,
                  n_actions=8, layer1_size=256, layer2_size=256, n_outputs=1)

    # env = gym.make('MountainCarContinuous-v0')
    env = gym.make('BipedalWalker-v3')
    score_history = []
    num_episodes = 100
    for i in range(num_episodes):
        #env = wrappers.Monitor(env, "tmp/mountaincar-continuous-trained-1",
        #                        video_callable=lambda episode_id: True, force=True)
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = np.array(agent.choose_action(observation).reshape(4,))
            observation_, reward, done, info = env.step(action)
            agent.learn(observation, reward, observation_, done)
            observation = observation_
            score += reward
        score_history.append(score)
        print('episode: ', i,'score: %.2f' % score)
    filename = 'bipedal-continuous-old-actor-critic-alpha000005-256x256fc-100games.png'
    plotLearning(score_history, filename=filename, window=20)