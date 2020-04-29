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