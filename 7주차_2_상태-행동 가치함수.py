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

def action_value_function(env, agent, act, G, max_step, now_step):
    gamma = 0.9

    if env.reward_list1[agent.pos[0]][agent.pos[1]] == 'goal':
        return env.goal
    if max_step == now_step:
        observation, reward, done = env.move(agent, act)
        G += agent.select_action_pr[act]*reward
        return G
    else:
        pos1 = agent.get_pos()
        observation, reward, done = env.move(agent, act)
        G += agent.select_action_pr[act]*reward

        if done == True:
            if observation[0]<0 or observation[0]>=env.reward.shape[0] or observation[1]<0 or observation[1]>=env.reward.shape[1]:
                agent.set_pos(pos1)

        pos1 = agent.get_pos()

        for i in range(len(agent.action)):
            agent.set_pos(pos1)
            next_v = action_value_function(env, agent, i, 0, max_step, now_step+1)
            G += agent.select_action_pr[i]*gamma*next_v
        return G

def show_q_table(q_table, env):
    for i in range(env.reward.shape[0]):
        print('+-------' * env.reward.shape[1], end='');
        print('+')
        for k in range(3):
            print('|', end='')
            for j in range(env.reward.shape[1]):
                if k == 0:
                    print('{0:10.2f}    '.format(q_table[i, j, 0]), end='')
                if k == 1:
                    print('{0:6.2f}  {1:6.2f} |'.format(q_table[i, j, 3], q_table[i, j, 1]), end='')
                if k == 2:
                    print('{0:10.2f}    '.format(q_table[i, j, 2]), end='')
            print()
    print('+----------' * env.reward.shape[1], end='');
    print('+')


def show_q_table_arrow(q_table, env):
    for i in range(env.reward.shape[0]):
        print('+----------' * env.reward.shape[1], end='');
        print('+')
        for k in range(3):
            print('|', end='')
            for j in range(env.reward.shape[1]):
                if k == 0:
                    if np.max(q[i, j, :]) == q[i, j, 0]:
                        print('   ↑   |', end='')
                    else:
                        print('       |', end='')
                if k == 1:
                    if np.max(q[i, j, :]) == q[i, j, 1] and np.max(q[i, j, :]) == q[i, j, 3]:
                        print('  ← →  |', end='')
                    elif np.max(q[i, j, :]) == q[i, j, 1]:
                        print('    →  |', end='')
                    elif np.max(q[i, j, :]) == q[i, j, 3]:
                        print('  ←    |', end='')
                    else:
                        print('       |', end='')
                if k == 2:
                    if np.max(q[i, j, :]) == q[i, j, 2]:
                        print('   ↓   |', end='')
                    else:
                        print('       |', end='')
            print()
    print('+----------' * env.reward.shape[1], end='');
    print('+')


env = Environment()
initial_position = np.array([0, 0])
agent = Agent(initial_position)
np.random.seed(0)
max_step_number = 8

for max_step in range(max_step_number):
    print('max_step = {}'.format(max_step))
    q_table = np.zeros((env.reward.shape[0], env.reward.shape[1], len(agent.action)))
    for i in range(env.reward.shape[0]):
        for j in range(env.reward.shape[1]):
            for action in range(len(agent.action)):
                agent.set_pos([i, j])
                q_table[i, j, action] = action_value_function(env, agent, action, 0, max_step, 0)

    q = np.round(q_table, 2)
    print('\nQ-table')
    show_q_table(q, env)
    print('\n정책')
    show_q_table_arrow(q, env)