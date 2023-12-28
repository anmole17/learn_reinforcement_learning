import numpy as np 
import gym 
import random
import time 
from IPython.display import clear_output

env = gym.make("FrozenLake-v1")

action_space_size= env.action_space.n
state_space_size=env.observation_space.n

q_table=np.zeros((state_space_size, action_space_size))

# print(q_table)

# num_episodes = 10000
# max_steps_per_episodes = 100
# learning_rate = 0.1
# discount_rate = 0.99
# exploration_rate = 1
# max_exploration_rate = 1
# min_exploration_rate = 0.01
# exploration_decay_rate = 0.01
# max at 22000 :  0.7270000000000005

# num_episodes = 100000
# max_steps_per_episodes = 1000
# learning_rate = 0.1
# discount_rate = 0.99
# exploration_rate = 1
# max_exploration_rate = 1
# min_exploration_rate = 0.001
# exploratoion_decay_rate = 0.0001
# # max at 93000 :  0.8400000000000006

# num_episodes = 100000
# max_steps_per_episodes = 1000

num_episodes = 1000000
max_steps_per_episodes = 1000
learning_rate = 0.2
discount_rate = 0.99
exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploratoion_decay_rate = 0.00001
random_seed=1729

# the one we are using : for pembro and PD1
# learning_rate = 0.3
# discount_rate = 0.90

# exploration_rate = 0.5
# max_exploration_rate = 0.5
# min_exploration_rate = 0.5
# exploratoion_decay_rate = 0.001
# max at epi: 69000 :  0.06800
print("num_episodes = ",num_episodes,
"\nmax_steps_per_episodes =", max_steps_per_episodes,
"\nlearning_rate =", learning_rate,
"\ndiscount_rate = ",discount_rate,
"\nexploration_rate = ",exploration_rate,
"\nmax_exploration_rate = ",max_exploration_rate,
"\nmin_exploration_rate = ",min_exploration_rate,
"\nexploratoion_decay_rate = ",exploratoion_decay_rate,
"\nrandom_seed = ", random_seed)

np.random.seed(random_seed)
rewards_all_episodes =[]

for episode in range(num_episodes):
    state = env.reset()[0]
    done = False
    rewards_current_episode = 0

    for step in range(max_steps_per_episodes):
        exploration_rate_threshold = random.uniform(0,1)
        if exploration_rate_threshold > exploration_rate:
            # exploitation
            action = np.argmax(q_table[state,:])
        else:
            # exploration
            action = env.action_space.sample()
        new_state, reward, done, truncated, info = env.step(action)
        q_table[state,action] = q_table[state,action] * (1 - learning_rate) + \
            learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))
        state = new_state
        rewards_current_episode += reward

        if done == True:
            break

    exploration_rate = min_exploration_rate + \
        (max_exploration_rate - min_exploration_rate) * np.exp(-exploratoion_decay_rate*episode)
    rewards_all_episodes.append(rewards_current_episode)

rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes), num_episodes/1000) 
count = 1000

print("***Average reward per thousand episodes*** \n")
for r in rewards_per_thousand_episodes:
    print(count, ": ", str(sum(r/1000)))
    count += 1000

