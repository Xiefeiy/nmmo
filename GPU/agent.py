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


def orthogonal_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class TeamAgent():

    def __init__(self, env, args):

        self.args = args
        self.mixnet = QMixNet(env, args.input_size, args.hidden_size, args.task_size, args.member_num, args.device).to(device=args.device)
        self.mixnet.apply(orthogonal_init)
        self.target_mixnet = QMixNet(env, args.input_size, args.hidden_size, args.task_size, args.member_num, args.device).to(device=args.device)
        self.target_mixnet.load_state_dict(self.mixnet.state_dict())
        self.team_optimizer = torch.optim.Adam(self.mixnet.parameters(), lr=args.lr)
        self.agents = []
        self.cnt = 0
        for i in range(args.member_num):
            agent = Agent(env, args)
            self.agents.append(agent)

    def take_action(self, team_flat_observations):
        team_actions = []
        for i in range(self.args.member_num):
            team_actions.append(self.agents[i].take_action(team_flat_observations[i]))

        return team_actions

    def update(self, data):
        center_data = data['center']
        agent_datas = data['agents']

        team_q_values = torch.zeros(size=(self.args.batch_size, 0), dtype=torch.float32).to(self.args.device)
        team_q_targets = torch.zeros(size=(self.args.batch_size, 0), dtype=torch.float32).to(self.args.device)

        for i in range(self.args.member_num):
            obs = torch.tensor(agent_datas[i]['state'], dtype=torch.float32).to(self.args.device)

            next_obs = torch.tensor(agent_datas[i]['next_state'], dtype=torch.float32).to(self.args.device)

            # print(f"next_obs = {next_obs}")

            actions = torch.tensor(agent_datas[i]['action'], dtype=torch.float32).to(self.args.device)

            # print(f"actions = {actions}")
            q_values = self.agents[i].get_q_value(obs, actions)
            q_targets = self.agents[i].get_q_target(next_obs)

            team_q_values = torch.cat((team_q_values, q_values), dim=-1)
            team_q_targets = torch.cat((team_q_targets, q_targets), dim=-1)

        team_obs = torch.tensor(center_data['state'], dtype=torch.float32).to(self.args.device)
        team_next_obs = torch.tensor(center_data['next_state'], dtype=torch.float32).to(self.args.device)
        team_rewards = torch.tensor(center_data['reward'], dtype=torch.float32).to(self.args.device)
        team_rewards = torch.sum(team_rewards, dim=-1).unsqueeze(-1)

        # print(f"team_q_values = {team_q_values}")
        # print(f"team_q_targets = {team_q_targets}")
        q_tot = self.mixnet(team_obs, team_q_values)
        q_tot_target = self.target_mixnet(team_next_obs, team_q_targets) # (batch_size, 1)


        targets = team_rewards + self.args.gamma * q_tot_target

        # print(f"targets = {targets}")

        # print(f"q_tot = {q_tot}")
        # print(f"q_tot_target = {q_tot_target}")

        loss = torch.mean(F.mse_loss(q_tot, targets))

        print(f"loss = {loss}")

        # if loss.detach().cpu().numpy() > 1e+4:
        #     print(f"q_tot = {q_tot}")
        #     print(f"q_tot_target = {q_tot_target}")

        self.team_optimizer.zero_grad()
        for i in range(self.args.member_num):
            self.agents[i].optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.mixnet.parameters(),10, 2)
        for i in range(self.args.member_num):
            torch.nn.utils.clip_grad_norm_(self.agents[i].net.parameters(),10,2)

        # for param in self.mixnet.parameters():
        #     print(f"team_grad = {param.grad}")

        self.team_optimizer.step()
        for i in range(self.args.member_num):
            # for param in self.agents[i].net.parameters():
            #     print(f"agent_grad = {param.grad}")
            self.agents[i].optimizer.step()

        if self.cnt % self.args.target_update == 0:
            self.target_mixnet.load_state_dict(self.mixnet.state_dict())

        self.cnt += 1

        return loss.detach().cpu().numpy()







class Agent():
    def __init__(self, env, args):

        self.args = args
        self.net = QNet(env, args.input_size, args.hidden_size, args.task_size, args.device).to(device=args.device)
        self.net.apply(orthogonal_init)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr)

    def take_action(self, flat_observations):
        actions_prob = self.net(flat_observations)
        actions = []
        for p in actions_prob:
            dist = torch.distributions.Categorical(torch.softmax(p,dim=-1))
            action = dist.sample()
            actions.append(action.item())

        return np.array(actions, dtype=np.int32)

    def get_q_value(self, flat_observations, actions):
        # flat_obs (batch_size, obs_shape)
        # actions (batch_size, action_dim)
        raw_q_value = self.net(flat_observations)

        # print(f"raw_q_value = {raw_q_value[-1]}")

        q_value = torch.zeros(size=(flat_observations.shape[0], 1), dtype=torch.float32).to(self.args.device)
        for i in range(len(raw_q_value)):

            for j in range(self.args.batch_size):
                action = int(actions[j][i])
                # print(f"action = {action}, raw_q_value[i][j] = {raw_q_value[i][j]}")
                if raw_q_value[i][j][action] != -1e+9:
                    q_value[j][0] += raw_q_value[i][j][action]
                    # print(f"raw_q_value[i][j][action] = {raw_q_value[i][j][action]}")

        return q_value

    def get_q_target(self, flat_observations):
        # flat_obs (batch_size, obs_shape)
        raw_q_value = self.net(flat_observations)
        q_target = torch.zeros(size=(flat_observations.shape[0], 1), dtype=torch.float32).to(self.args.device)
        for i in range(len(raw_q_value)):
            for j in range(self.args.batch_size):

                target = torch.max(raw_q_value[i][j]).item()
                if target != -1e+9:
                    q_target[j][0] += target
                # print(f"target = {target}")


        return q_target



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
    def __init__(self, capacity, n_agents, device):
        self.buffers = []
        self.n_agents = n_agents
        for i in range(n_agents):
            self.buffers.append(collections.deque(maxlen=capacity))
        self.device = device

    def add(self, traj, agent_id):
        self.buffers[agent_id].append(traj)

    def size(self):
        return len(self.buffers[0])

    def sample(self, batch_size):
        traj_num = batch_size // 4
        idx = [i for i in range(self.size())]
        traj_idx = random.sample(idx, traj_num)
        team_data = {'center': {'state': [], 'action': [], 'next_state': [], 'reward': [], 'done': []}, 'agents': []}
        datas = []
        for i in range(self.n_agents):
            datas.append({'state': [], 'action': [], 'next_state': [], 'reward': [], 'done': []})
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

        for key in team_data['center'].keys():

            center_datas = []
            if key == 'reward' or key == 'done':
                data_shape = 1
            else:
                data_shape = self.buffers[0][0][key][0].shape[-1]
            for i in range(batch_size):
                center_data = np.zeros(shape=(self.n_agents, data_shape), dtype=np.float32)
                for j in range(self.n_agents):
                    center_data[j] = datas[j][key][i]

                center_data = center_data.reshape(-1)
                center_datas.append(center_data)

            team_data['center'][key] = np.array(center_datas)

        team_data['agents'] = datas

        return team_data
