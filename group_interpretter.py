"""
This file contains the implementation of the Group Interpreter Algorithm applied to
the Cartpole environment.

Implementation details:
Each actor is setup to be a separate Process and therefore allowed to run simultaneously.
We have setup pipe connections between each actor and also included a leader actor which controls the
initiation and termination of all the learners. The leader's presence does not affect the other learner's processing.
The user is essentially the leader since they can control the processes by inputting commands.
"""

from utils import *
import multiprocessing
from multiprocessing import Process, Pipe
import os
import time
import numpy as np
import sys
import matplotlib.pyplot as plt
import gym
import torch
from torch import nn
from torch import optim
import copy


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

        # Define network
        self.network = nn.Sequential(
            nn.Linear(self.n_inputs, 16),
            nn.ReLU(),
            nn.Linear(16, self.n_outputs),
            nn.Softmax(dim=-1))

    def predict(self, state):
        action_probs = self.network(torch.FloatTensor(state))
        return action_probs


def predict(self, state):
    action_probs = self.network(torch.FloatTensor(state))
    return action_probs


class Actor(Process):
    def __init__(self, idNum, channels):
        Process.__init__(self)
        self.id = idNum
        self.leaderChannel = channels[0]
        self.channels = channels[1:]        #All actors that this actor is communicating with
        self.keepRunning = True

        #Zeros Params is used to set all network parameters to 0
        self.zeroParams = [a.detach()*0 for a in list(copy.deepcopy(list(pe.network.parameters())))]

        self.deltaParams = self.zeroParams       #Initialize delta parameters to 0
        self.paramsRecieved = 0
        self.ne=TOTAL_NUM_EPISODES

        self.policy_estimator = pe
        self.groupAction = 0
        self.action_probs = 0
        self.performance = 1
        self.total_rewards = []

    def run(self):
        # Main run loop for the process
        while(self.keepRunning == True):
            # While user did not press quit, keep polling the leader and other learners
            self.pollLeader()
            self.pollLearners()


    def print(self, inputMessage, message2 = '', message3 = '', inID = ""):
        if inID == self.id or "":
            print("\nlearner{}:".format(self.id), inputMessage, message2, message3)

    def inquire(self, message):
        """
        All inquires will be sent through here, but the actions will be covered in pollLearners
        :param message: Message used to inquire other learners
        :return:
        """

        for channel in self.channels:
            channel.send(message)

    def pollLeader(self):
        """
        The leaderChannel is the messages typed into the console during run time. The user is the leadeChannel
        :return:
        """
        if(self.leaderChannel.poll()):
            command = self.leaderChannel.recv()
            self.print(command)
            if command == "quit\n":
                self.keepRunning = False
            elif command == "getID\n":
                self.inquire("id")
            elif command == "rewards\n":
                rewards = self.reinforce(env, pe, ne=self.ne)
                self.leaderChannel.send([self.id ,rewards])
            elif command == "inquireRewards":
                self.print("sendingRewards")
                self.leaderChannel.send([self.id ,self.total_rewards])
            else:
                try:
                    exec(command)
                except:
                    print("there is an error with that python command")

    def pollLearners(self, variables = ""):
        """
        This will handle the inquires sent by each learner
        Both sending response and handling incoming responses are hangled here
        :return:
        """

        for channel in self.channels:
            if(channel.poll()):
                message = channel.recv()

                # deltaParams: adds the self.deltaParams. This is to be used in the Learn by Watching Method
                if message[0] == "deltaParams":
                    self.deltaParams = [a+b for (a,b) in zip(self.deltaParams,message[1])]
                    self.paramsRecieved = self.paramsRecieved +1

                if message[0] == "inquireAction":
                    channel.send(["respondAction", np.random.choice(message[1], p=self.action_probs)*self.performance])
                if message[0] == "respondAction":
                    self.groupAction += message[1]
                    self.paramsRecieved += 1


    def reinforce(self, env, policy_estimator, ne=100,batch_size=10, gamma=0.99):
        # print("running reinforce")
        # Set up lists to hold results
        total_rewards = []
        batch_rewards = []
        batch_actions = []
        batch_states = []
        batch_counter = 1

        # Define optimizer
        optimizer = optim.Adam(self.policy_estimator.network.parameters(), lr=ADAM_OPTIMIZER_LR)

        #For the Cartpole problem, action_space.n is 2
        action_space = np.arange(env.action_space.n)
        for ep in range(ne):
            self.pollLeader()
            s_0 = env.reset()       #Reset the environment and get starting state
            states = []
            rewards = []
            actions = []
            complete = False        #Keeps track of whether the episode has completed running
            while complete == False:
                # Get actions and remove its connection to the DCG and convert it to numpy array
                self.action_probs = self.policy_estimator.predict(s_0).detach().numpy()

                #Randomly select an action given the probabiities for each action
                action = np.random.choice(action_space, p=self.action_probs)

                # Inquire group to determine what action this learner should take
                for channel in self.channels:
                    channel.send(["inquireAction", action_space])
                self.groupAction = action*0
                self.paramsRecieved = 0

                # wait until all learners have responded
                while(self.paramsRecieved<len(self.channels)):
                    self.pollLearners()
                    self.pollLeader()
                # self.print(self.performance, inID = 1)
                gamma = .9 + .1/(1+abs(self.groupAction - action))

                # if self.id == 1:
                #     gamma = .99
                # self.print(gamma, inID = 1)


                #Perform the action and get the next state and reward
                s_1, r, complete, _ = env.step(action)

                #Add completed states and corresponding reward and action taken to main list
                states.append(s_0)
                rewards.append(r)
                actions.append(action)

                #Set next state to current state variable
                s_0 = s_1

                #If the episode has ended, add rewards, states, and action info to batch arrays
                if complete:
                    batch_rewards.extend(discount_rewards(rewards, gamma))
                    batch_states.extend(states)
                    batch_actions.extend(actions)
                    batch_counter += 1
                    self.total_rewards.append(sum(rewards))

                    # If batch is complete, update network
                    if batch_counter == batch_size:
                        #Reset optimizer gradient to 0
                        optimizer.zero_grad()
                        state_tensor = torch.FloatTensor(batch_states)
                        reward_tensor = torch.FloatTensor(batch_rewards)

                        # Actions are used as indices, must be LongTensor
                        action_tensor = torch.LongTensor(batch_actions)

                        # Calculate loss
                        logprob = torch.log(
                            self.policy_estimator.predict(state_tensor))
                        selected_logprobs = reward_tensor * \
                            logprob[np.arange(len(action_tensor)), action_tensor]
                        loss = -selected_logprobs.mean()

                        # Calculate gradients
                        loss.backward()
                        optimizer.step()

                        #Reset all batch arrays
                        batch_rewards = []
                        batch_actions = []
                        batch_states = []
                        batch_counter = 1
                        self.performance = np.mean(self.total_rewards[-10:])

                    # Print running average
                    # print(ep)
                    if self.id == 1:
                        print("\rEp: {} Average of last 10: {:.2f}".format(ep + 1, np.mean(self.total_rewards[-10:])), end="")

        return self.total_rewards

def get_input(stdin, channels, n):
    """
    Process user input and take appropriate action
    :param stdin: user input stream
    :param channels: list of actors
    :param n: total number of actors
    :return:
    """

    global show
    show = False

    for line in iter(stdin.readline, ''):
        if line == "show\n":
            print("", "Show Toggled", show)
            if show == True:
                show = False
            elif show == False:
                show = True
            print(show)
        if line == "poll\n":
            #Plot results for each actor
            plt.figure(figsize=(12,8))
            print("figureMade")
            for channel in channels:
                if(channel.poll()):
                    rewards = channel.recv()[1]
                    window = 10
                    smoothed_rewards = [np.mean(rewards[i-window:i+1]) if i > window else np.mean(rewards[:i+1]) for i in range(len(rewards))]
                    # plt.plot(rewards)
                    plt.plot(smoothed_rewards)
                    plt.ylabel('Total Rewards')
                    plt.xlabel('Episodes')
            plt.show()
        elif line == "plot\n":
            plt.figure(figsize = (12,8))
            plt.plot()
            for channel in channels:
                channel.send("inquireRewards")
            for channel in channels:
                check = channel.poll(5)
                print(check)
                if(check):
                    message = channel.recv()
                    rewards = message[1]
                    if show == True:
                        plt.plot(rewards, label = str(message[0]))
                    window = 10
                    smoothed_rewards = [np.mean(rewards[i-window:i+1]) if i > window else np.mean(rewards[:i+1]) for i in range(len(rewards))]
                    plt.plot(smoothed_rewards, label = str(message[0])+ "smoothed" )
        else:
            #Send command to each learner
            for channel in channels:
                channel.send(line)
        time.sleep(.3)

        #Stop if at least one actor has stopped
        if(len(multiprocessing.active_children())<n):
            break
    stdin.close()

#Initialize Cartpole Gym environment
env = gym.make('CartPole-v0')
s = env.reset()

#Initialize policy estimator nentwork and define neural network architecture
pe = policy_estimator(env)

if __name__ == '__main__':
    # #Initialize Cartpole Gym environment
    # env = gym.make('CartPole-v0')


    # #Initialize policy estimator nentwork and define neural network architecture
    # pe = policy_estimator(env)

    n = NUM_LEARNERS                                   #Number of learners
    n = n+1                                 #Add 1 learner for the leader learner

    #Setup an array of pipe connections between each learner
    pipeArr = np.empty([n,n-1], dtype = multiprocessing.connection.Connection)
    plist = []                              #List of actors

    #Setup pipe array
    for i in range(n):
        for j in range(n-1-i):
            pipeArr[i,i+j], pipeArr[i+j+1,i] = Pipe()  ##(1+i/10+j/100, 2+i/10+j/100)
        if i!=0:
            #Setup actor for every non-leader learner
            a = Actor(i,pipeArr[i])
            plist.append(a)
            a.start()

    print("type 'rewards' to start the process")
    print("'quit' at any time to terminated the script")
    print("other inputs will be run as python code by each actor")

    #Allow user's command to control the program
    get_input(sys.stdin, pipeArr[0], n-1)

    #Stop each actor process
    for actor in plist:
        actor.close()

    #Close all connections
    np.vectorize(multiprocessing.connection.Connection.close)(pipeArr)