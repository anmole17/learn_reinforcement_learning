import numpy as np
from mnest.Entities import Brain
import random
import bisect
import csv

## DQL 
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F
from DQN_input_aa_code import *

def weighted_choice_b(weights):
    totals = []
    running_total = 0

    for w in weights:
        running_total += w
        totals.append(running_total)

    rnd = random.random() * running_total
    return bisect.bisect_right(totals, rnd)

class DQN(nn.Module):
    def __init__(self, in_states, out_actions,n1=170, n2=85, n3=85):
        super().__init__()
        # Define network layers
        self.layer1 = nn.Linear(in_states, n1)   # first fully connected layer
        self.layer2 = nn.Linear(n1, n2)
        self.layer3 = nn.Linear(n2, n3)
        #self.layer4 = nn.Linear(85, 42)
        self.out = nn.Linear(n3, out_actions) # ouptut layer w
    def forward(self, x):
        #x=x.to(torch.float32)
        x = torch.flatten(x)
        #print(x.dtype)
        x = F.relu(self.layer1(x)) # Apply rectified linear unit (ReLU) activation
        #print(x.dtype)
        x = F.relu(self.layer2(x))
        #print(x.dtype)
        x = F.relu(self.layer3(x))   
        # #print(x.dtype)    
        # x = F.relu(self.layer4(x))   
        #print(x.dtype)   
        return self.out(x)

class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)
    
    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)

class Brain_v2(Brain):
    def __init__(self, brain_type: str, exploration_type:str, sim_name:str, action_list: list, learning_rate=0.1, learning_decay=0.0,
                 min_learning_rate=0.1,  discounted_return=0.9,
                 # epsion decay para
                 exploration_rate=0.5, exploration_decay=0.001, min_exploration=0, exploration_decay_per_episodes=1000,
                 # softmax para
                 temperature=0, temperature_decay=0.25,temperature_decay_per_episodes=250, min_temperature=0.25, 
                 ##DQN parameters
                 network_sync_rate=10000,replay_memory_size = 10000,mini_batch_size = 1000):

        self.brain_type = brain_type
        self.action_list = action_list

        # choose 'epsilon_decay' or 'softmax' for 'Q-Table' brain type; else put 'Deep-Q'
        self.exploration_type = exploration_type 
        self.sim_name=sim_name
        # Learning Parameters
        self.learning_rate = learning_rate  # Alpha
        self.learning_decay = learning_decay # alpha decay
        self.min_learning_rate = min_learning_rate
        self.discounted_return = discounted_return  # Gamma or Lambda

        # epsilon greedy
        self.exploration_rate = exploration_rate  # Epsilon
        self.exploration_decay = exploration_decay  # Epsilon Decay
        self.min_exploration = min_exploration
        

        # softmax
        self.temperature = temperature # beta
        self.temperature_decay = temperature_decay
        self.temperature_decay_per_episodes = temperature_decay_per_episodes
        self.exploration_decay_per_episodes = exploration_decay_per_episodes
        self.min_temperature=min_temperature

        #DQN
        self.step_count=0
        self.network_sync_rate=network_sync_rate
        self.replay_memory_size = replay_memory_size
        self.mini_batch_size = mini_batch_size

        if self.brain_type == 'Q-Table':
            self.q_table = {}
        elif self.brain_type == 'Deep-Q':
            self.loss_fn = nn.MSELoss()
            self.num_states = 17*20
            num_actions=len(action_list)
            self.memory = ReplayMemory(self.replay_memory_size)
            # Create policy and target network. Number of nodes in the hidden layer can be adjusted.
            self.policy_dqn = DQN(in_states=self.num_states, out_actions=num_actions, n1=170, n2=85, n3=85)
            self.target_dqn = DQN(in_states=self.num_states, out_actions=num_actions, n1=170, n2=85, n3=85)
            # Make the target and policy networks the same (copy weights/biases from one network to the other)
            self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
            # print('Policy (random, before training):')
            # self.print_dqn(self.policy_dqn)
            self.optimizer = torch.optim.Adam(self.policy_dqn.parameters(), lr=self.learning_rate)

        else:
            print('There seems to be some mistake on the brain type.')

    def learn(self, state_observed: str, action_taken: int, next_state: str, reward_earned: float):
        """

        :param next_state:
        :param state_observed:
        :param action_taken:
        :param reward_earned:
        :return:
        """
        if self.brain_type == 'Q-Table':
            # There is a possibility that the new state does not exist in the q table.
            if next_state not in self.q_table:
                self.add_state(next_state)

            values_state_observed = self.q_table[state_observed]
            values_next_state = self.q_table[next_state]

            learned_value = reward_earned + self.discounted_return * (max(values_next_state))
            new_value = ((1 - self.learning_rate) * values_state_observed[action_taken] +
                         self.learning_rate * learned_value)
            
            values_next_state[action_taken] = new_value

            if self.learning_rate> self.min_learning_rate:
                self.learning_rate -= self.learning_decay
            print(f'Learning rate :: {self.learning_rate}')

        elif self.brain_type == 'Deep-Q':
            #print("DQL learn start")
            self.memory.append((state_observed, action_taken, next_state, reward_earned))
            self.step_count+=1
            #print(len(self.memory))

            if len(self.memory)>self.mini_batch_size:
                #print("DQL learn : mini batch start")
                mini_batch = self.memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, self.policy_dqn, self.target_dqn)        

                # Copy policy network to target network after a certain number of steps
                if self.step_count > self.network_sync_rate:
                    self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
                    torch.save(self.policy_dqn.state_dict(), f'{self.sim_name}.pt')
                    self.step_count=0
                #print("DQL learn : mini batch end")
            #print("DQL learn : end")
        else:
            print('There seems to be some mistake on the brain type.')
            pass
    
    def predict_action(self, state: str, episode:int):
        """
        This function takes in a state and predicts an action based on the brain.
        :param state: The state for which the action is to be predicted
        :return: a number between 0 and number of actions to act as an index.
        """
        if self.brain_type == 'Q-Table':
            if episode % 1000 == 0:
                try: 
                    file = open(f'{self.sim_name}/q_table.txt', 'wt') 
                    file.write(str(self.q_table)) 
                    file.close() 
                except: 
                    print("Unable to write to file")
            if self.exploration_type =='epsilon_decay':
                # Checking Exploration vs Exploitation.
                if np.random.random() < self.exploration_rate:
                    # Explore
                    action = np.random.randint(len(self.action_list))

                    # We have to store the state anyway
                    if state not in self.q_table:
                        self.add_state(state)

                else:
                    # Exploit
                    if state in self.q_table:
                        q_values = self.q_table[state]  # q_values for that state
                        predict_list = np.where(q_values == max(q_values))[0]  # list of all indices with max q_values
                        action = np.random.choice(predict_list)
                        # print('State_found')
                    else:
                        self.add_state(state)
                        action = np.random.randint(len(self.action_list))
                        # action = 0
                        # print('New_state')

                # Decaying exploration_rate
                # if self.exploration_rate > self.min_exploration:
                #     self.exploration_rate -= self.exploration_decay
                if (self.exploration_rate> self.min_exploration) and (episode % self.exploration_decay_per_episodes == 0):
                    if episode != 0:
                        self.exploration_rate -= self.exploration_decay
                #print(f'Temperature :: {self.temperature}')
                print(f'Exploration rate :: {self.exploration_rate}')
            
            elif self.exploration_type =='softmax':
                if state in self.q_table:
                    q_values = self.q_table[state] 
                    total = sum([np.exp(val / self.temperature) for val in q_values]) #denominator
                    probs = [np.exp(val / self.temperature) / total for val in q_values]

                    action=weighted_choice_b(probs)

                
                else: # new state : action is random
                    self.add_state(state)
                    action = np.random.randint(len(self.action_list))
                
                if (self.temperature > self.min_temperature) and (episode % self.temperature_decay_per_episodes == 0):
                    if episode != 0:
                        self.temperature -= self.temperature_decay
                print(f'Temperature :: {self.temperature}')

            else:
                action = None
                print('There seems to be some mistake on the exploration_type.')

        elif self.brain_type == 'Deep-Q':
            #print("DQL predict action start")
            if np.random.random() < self.exploration_rate:
                    # select random action
                    action = np.random.randint(len(self.action_list))
                    print("action predicted by random")
            else:
                # select best action            
                with torch.no_grad():
                    action = self.policy_dqn(self.state_to_dqn_input(state)).argmax().item()
                    print("action predicted by DQN")
                    print(action)
            #print("DQL predict action done")
            # for epsilon greedy
            if (self.exploration_rate> self.min_exploration) and (episode % self.exploration_decay_per_episodes == 0):
                    if episode != 0:
                        self.exploration_rate -= self.exploration_decay
            print(f'Exploration rate :: {self.exploration_rate}')


        else:
            action = None
            print('There seems to be some mistake on the brain type.')
            # print(len(self.q_table), self.q_table.keys())
            # print(action)
        #print("predict action done")
        return action
    
    def optimize(self, mini_batch, policy_dqn, target_dqn):
        #print("inside optimize")

        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward in mini_batch:
            #print("minibatch start")
        # Calculate target q value 
            with torch.no_grad():
                target = torch.FloatTensor(
                    reward + self.discounted_return * target_dqn(self.state_to_dqn_input(new_state)).max()
                )

            # Get the current set of Q values
            current_q = policy_dqn(self.state_to_dqn_input(state))
            current_q_list.append(current_q)

            # Get the target set of Q values
            target_q = target_dqn(self.state_to_dqn_input(state)) 
            # Adjust the specific action to the target that was just calculated
            target_q[action] = target
            target_q_list.append(target_q)
            #print("minibatch end")
                
        # Compute loss for the whole minibatch
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        #print("optimize optimize done")
        return "Done"
    
    def state_to_dqn_input(self, state:str)->torch.Tensor:
        #print("state to DQN start")
        input_tensor=one_hot_coding(state)
        input_tensor=torch.from_numpy(input_tensor).float().unsqueeze(-1)
        #print("state to DQN done")
        return input_tensor