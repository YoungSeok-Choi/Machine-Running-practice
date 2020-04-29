import copy
import numpy as np
import matplotlib.pyplot as plt
import time

class Environment:
    cliff = -3;
    road = -1;
    sink = -2;
    goal = 2
    goal_position = [2, 3]
    reward_list = [[road, road, road, road], [road, road, sink, road], [road, road, road, goal]]
    reward_list1 = [['road', 'road', 'road', 'road'], ['road', 'road', 'sink', 'road'],
                    ['road', 'road', 'road', 'goal']]

    def __init__(self):
        self.reward = np.asarray(self.reward_list)

    def move(self, agent, action):
        done = False
        new_pos = agent.pos + agent.action[action]

        if self.reward_list1[agent.pos[0]][agent.pos[1]] == 'goal':
            reward = self.goal
            observation = agent.set_pos(agent.pos)
            done = True
        elif new_pos[0] < 0 or new_pos[0] >= self.reward.shape[0] or new_pos[1] < 0 or new_pos[1] >= self.reward.shape[
            1]:
            reward = self.cliff
            observation = agent.set_pos(agent.pos)
            done = True
        else:
            observation = agent.set_pos(new_pos)
            reward = self.reward[observation[0], observation[1]]
        return observation, reward, done


class Agent:
    action = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])
    select_action_pr = np.array([0.25, 0.25, 0.25, 0.25])

    def __init__(self, initial_position):
        self.pos = initial_position

    def set_pos(self, position):
        self.pos = position
        return self.pos

    def get_pos(self):
        return self.pos

def show_v_table(v_table, env):
    for i in range(env.reward.shape[0]):
        print('+-----------------' * env.reward.shape[1], end='')
        print('+')
        for k in range(3):
            print('|', end='')
            for j in range(env.reward.shape[1]):
                if k == 0:
                    print('          |', end='')
                if k == 1:
                    print(' {0:8.2f} |'.format(v_table[i, j]), end='')
                if k == 2:
                    print('          |', end='')
            print()
        print('+-----------------' * env.reward.shape[1], end='')
        print('+')

def show_policy(policy, env):
    for i in range(env.reward.shape[0]):
        print('+----------' * env.reward.shape[1], end='');
        print('+');
        print('|', end='')
        for j in range(env.reward.shape[1]):
            if env.reward_list1[i][j] != 'goal':
                if policy[i, j] == 0:
                    print('   ↑   |', end='')
                elif policy[i, j] == 1:
                    print('   →   |', end='')
                elif policy[i, j] == 2:
                    print('   ↓   |', end='')
                elif policy[i, j] == 3:
                    print('   ←   |', end='')
            else:
                print('   *   |', end='')
        print()

def policy_extraction(env, agent, v_table, optimal_policy):
    gamma = 0.9
    for i in range(env.reward.shape[0]):
        for j in range(env.reward.shape[1]):
            temp = -1e+10
            for action in range(len(agent.action)):
                agent.set_pos([i, j])
                observation, reward, done = env.move(agent, action)
                if temp < reward + gamma * v_table[observation[0], observation[1]]:
                    optimal_policy[i, j] = action
                    temp = reward + gamma * v_table[observation[0], observation[1]]
    return optimal_policy


np.random.seed(0)
env = Environment()
initial_position = np.array([0,0])
agent = Agent(initial_position)
gamma = 0.9

V = np.zeros((env.reward.shape[0], env.reward.shape[1]))
max_episode = 10000
max_step = 100
alpha = 0.01
print('TD(0) 예측 시작')

for epi in range(max_episode):
    delta = 0
    i = 0;j = 0
    agent.set_pos([i,j])
    for k in range(max_step):
        pos = agent.get_pos()
        action = np.random.randint(0,len(agent.action))
        observation, reward, done = env.move(agent, action)
        V[pos[0],pos[1]] += alpha*(reward + gamma*V[observation[0], observation[1]] - V[pos[0],pos[1]])

        if done == True:
            break
print('V(s)')
show_v_table(np.round(V,2), env)
policy = np.zeros((env.reward.shape[0], env.reward.shape[1]))
policy = policy_extraction(env, agent, V, policy)
show_policy(policy, env)