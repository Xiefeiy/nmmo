import argparse
import collections
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

import pufferlib
import pufferlib.emulation
import pufferlib.models

import nmmo
from nmmo.entity.entity import EntityState

from model import QMixNet, QNet


class TeamAgent():

    def __init__(self, env, args):

        self.args = args
        self.mixnet = QMixNet(env, args.input_size, args.hidden_size, args.task_size, args.member_num)
        self.agents = []
        for i in range(args.member_num):
            agent = Agent(env, args)
            self.agents.append(agent)

    def take_action(self, team_flat_observations):
        team_actions = []
        for i in range(self.args.member_num):
            team_actions.append(self.agents[i].take_action(team_flat_observations[i]))

        return team_actions

    def update(self, data):
        #! TODO



        return None





class Agent():
    def __init__(self, env, args):

        self.args = args
        self.net = QNet(env, args.input_size, args.hidden_size, args.task_size)

    def take_action(self, flat_observations):
        actions_prob = self.net(flat_observations)
        actions = []
        for p in actions_prob:
            dist = torch.distributions.Categorical(torch.softmax(p, dim=-1))
            action = dist.sample()
            actions.append(action.item())

        return np.array(actions, dtype=np.int32)


# class TeamReplayBuffer:
#     def __init__(self, capacity, n_agents):
#         self.buffers = []
#         self.n_agents = n_agents
#         for i in range(n_agents):
#             self.buffers.append(collections.deque(maxlen=capacity))
#
#     def add(self, traj, agent_id):
#         self.buffers[agent_id].append(traj)
#
#     def sample(self, batch_size, max_seq_len):
#         idx = [i for i in range(self.size())]
#         traj_idx = random.sample(idx, batch_size)
#         team_data = {'center':{'states':[]},'agents':[]}
#         team_max_traj_lens = -1
#         for i in self.n_agents:
#             traj_lens = []
#             for j in traj_idx:
#                 traj_lens.append(len(self.buffers[i][j]['state']))
#             traj_lens = np.array(traj_lens, dtype=np.int32)
#             max_traj_lens = min(np.max(traj_lens), max_seq_len)
#             team_max_traj_lens = max(team_max_traj_lens, max_traj_lens)
#
#         for i in self.n_agents:
#             data = {'state': [], 'action': [], 'next_state': [], 'reward': [], 'done': []}
#             for j in traj_idx:
#                 traj_len = len(self.buffers[i][j]['state'])
#                 for k in data.keys():
#                     pad_shape = self.buffers[i][j][k].shape(-1)
#                     pad = np.zeros_like(self.buffers[i][j][k][0])
#                     data[k].append()
#
#
#         for i in traj_idx:
#             for k in data.keys():
#
#
#
#         return np.array(state), action, reward, np.array(next_state), done
#
#     def size(self):
#         return len(self.buffer)

class TeamReplayBuffer:
    def __init__(self, capacity, n_agents):
        self.buffers = []
        self.n_agents = n_agents
        for i in range(n_agents):
            self.buffers.append(collections.deque(maxlen=capacity))

    def add(self, traj, agent_id):
        self.buffers[agent_id].append(traj)

    def size(self):
        return len(self.buffers[0])

    def sample(self, batch_size):
        traj_num = batch_size // 4
        idx = [i for i in range(self.size())]
        traj_idx = random.sample(idx, batch_size)
        team_data = {'center': [], 'agents': []}
        datas = []
        data = {'state': [], 'action': [], 'next_state': [], 'reward': [], 'done': []}
        for i in range(self.n_agents):
            datas.append(data.copy())
        for i in traj_idx:
            min_traj_len = 1e+5
            for j in range(self.n_agents):
                traj_len = len(self.buffers[j][i]['state'])
                min_traj_len = min(min_traj_len, traj_len)
            transition_idx = [k for k in range(min_traj_len)]
            transition_idx = random.sample(transition_idx, 4)
            for j in range(self.n_agents):
                for t in transition_idx:
                    for key in datas[j].keys():
                        datas[j][key].append(self.buffers[j][i][key][t])

        for i in range(self.n_agents):
            for key in datas[i].keys():
                datas[i][key] = np.array(datas[i][key])

        center_data = []
        state_shape = self.buffers[0][0]['state'].shape(-1)
        for i in range(batch_size):
            center_state = np.zeros(shape=(self.n_agents, state_shape), dtype=np.float32)
            for j in range(self.n_agents):
                center_state[j] = datas[j]['state'][i]

            center_state = center_state.reshape(-1)
            center_data.append(center_state)

        team_data['center'] = center_data
        team_data['agents'] = datas

        return team_data
