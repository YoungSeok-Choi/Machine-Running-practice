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

def generate_episode_with_policy(env, agent, first_visit, policy):
    gamma = 0.09
    episode = []
    visit = np.zeros((env.reward.shape[0], env.reward.shape[1], len(agent.action)))

    i = 0; j = 0
    agent.set_pos([i,j])
    G = 0; step = 0; max_step = 100

    for k in range(max_step):
        pos = agent.get_pos()

        action = np.random.choice(range(0,len(agent.action)), p=policy[pos[0],pos[1],:])
        observation, reward, done = env.move(agent, action)
        if first_visit:
            if visit[pos[0], pos[1], action] == 0:
                G += gamma**(step)*reward
                visit[pos[0], pos[1], action] = 1
                step += 1
                episode.append((pos, action, reward))
        else:
            G + gamma**(step)*reward
            step += 1
            episode.append((pos, action, reward))
        if done == True:
            break
    return i, j, G, episode

np.random.seed(0)
env = Environment()
initial_position = np.array([0,0])
agent = Agent(initial_position)

#Q_visit = np.zeros((env.reward.shape[0], env.reward.shape[1], len(agent.action)))
#optimal_a = np.zeros((env.reward.shape[0], env.reward.shape[1]))
#Q_table = np.random.rand(env.reward.shape[0], env.reward.shape[1], len(agent.action))
#policy = np.zeros((env.reward.shape[0], env.reward.shape[1], len(agent.action)))

max_episode = 10000
first_visit = True
gamma = 0.09
for epi in range(max_episode):
    x,y,G, episode = generate_episode_with_policy(env, agent, first_visit, policy)

for step_num in range(len(episode)):
    G = 0
    i = episode[step_num][0][0]
    j = episode[step_num][0][1]
    action = episode[step_num][1]
    Q_visit[i,j, action] += 1
    for step, k in enumerate(episode[step_num:]):
        G += gamma**(step)*k[2]

    Q_table[i,j,action] += 1/Q_visit[i,j,action]*(G-Q_table[i,j,action])
    for i in range(env.reward.shape[0]):
        for f in range(env.reward.shape[1]):
            optimal_a[i,j] = np.argmax(Q_table[i,j,:])

    epsilon = 1 - epi/max_episode
    for i in range(env.reward.shape[0]):
        for j in range(env.reward.shape[1]):
            for k in range(len(agent.action)):
                if optimal_a[i,j] == k:
                    policy[i,j,k] = 1 - epsilon + epsilon/len(agent.action)
                else:
                    policy[i,j,k] = epsilon/len(agent.action)
print('최종 Q(s,a)'); show_q_table(Q_table, env)
print('최종 정책'); show_q_table(policy, env)
print('최종 optimal_a'); show_policy(optimal_a, env)