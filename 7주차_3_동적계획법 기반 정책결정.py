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

def policy_evaluation(env, agent, v_table, policy):
    while True:
        delta = 0
        temp_v = copy.deepcopy(v_table)
        for i in range(env.reward.shape[0]):
            for j in range(env.reward.shape[1]):
                agent.set_pos([i,j])
                action = policy[i,j]
                observation, reward, done = env.move(agent, action)
                v_table[i,j] = reward + gamma*v_table[observation[0], observation[1]]
            delta = np.max([delta, np.max(np.abs(temp_v - v_table))])
            if delta < 0.000001:
                break
    return v_table,  delta

def policy_improvement(env, agent, v_table, policy):
    policyStable = True # 기존엔 pilicy
    for i in range(env.reward.shape[0]):
        for j in range(env.reward.shape[1]):
            old_action = policy[i,j]
            temp_action = 0
            temp_value = -1e+10

            for action in range(len(agent.action)):
                agent.set_pos([i,j])
                observation, reward, done = env.move(agent, action)
                if temp_value < reward + gamma*v_table[observation[0], observation[1]]:
                    temp_action = action
                    temp_value = reward + gamma*v_table[observation[0], observation[1]]
                if old_action != temp_action:
                    policyStable = False # 위에어 오류랑 킹리적 갓심
                policy[i,j] = temp_action
    return policy, policyStable

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


np.random.seed(0)
env = Environment()
initial_position = np.array([0,0])
agent = Agent(initial_position)
gamma = 0.0

v_table = np.random.rand(env.reward.shape[0], env.reward.shape[1])
policy = np.random.randint(0, 4, (env.reward.shape[0], env.reward.shape[1]))

max_iter_number = 20000
for iter_number in range(max_iter_number):
    v_table, delta = policy_evaluation(env, agent, v_table, policy)
    policy, policyStable = policy_improvement(env, agent, v_table, policy)
    show_v_table(v_table, env)
    show_policy(policy, env)

    if policyStable == True:
        break