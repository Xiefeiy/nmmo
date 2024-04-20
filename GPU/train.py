import nmmo

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from model import QMixNet, QNet
from agent import TeamAgent, Agent, TeamReplayBuffer
import config
import environment
import pufferlib
import pufferlib.emulation
import pufferlib.models
from nmmo.entity.entity import EntityState
import pickle

def setup_env(args):

    print("-------setup env-------")
    env_args = config.create_config(config.Config)

    env_args.num_agents = args.team_num * args.member_num
    env_args.local_mode = True      # If True, then only one env is ...; hard code here (temporarily)
    if env_args.local_mode:
        env_args.num_envs = 1
        env_args.num_buffers = 1
        env_args.use_serial_vecenv = True
        env_args.rollout_batch_size = 2**10

    print(f'env_args = {env_args}')

    env_creator = environment.make_env_creator(env_args)
    env = env_creator()

    print("-------setup env finished-------")
    return env

def init_agents(env, args):

    print("-------init agents-------")
    teams = []
    for i in range(args.team_num):
        team = TeamAgent(env, args)
        teams.append(team)

    print("-------init agents finished-------")
    return teams

def init_buffers(args):

    team_buffers = []
    for i in range(args.team_num):
        team_buffer = TeamReplayBuffer(args.buffer_capacity, args.member_num, args.device)
        team_buffers.append(team_buffer)

    return team_buffers

def train(args):

    env = setup_env(args)

    teams = init_agents(env, args)

    team_buffers = init_buffers(args)

    update_cnt = 0
    update_team_idx = 0             # suppose we only train the Team-0 (temporarily)

    for i in range(args.episode_num):
        dones_mask = {k+1:False for k in range(args.team_num * args.member_num)}
        episode_done = False
        obs = env.reset()
        step = 0

        all_trajs = {k+1:{'state':[], 'action':[], 'next_state':[], 'reward':[], 'done': []} for k in range(args.team_num * args.member_num)}

        while not episode_done:
            actions = {}

            for j in range(args.team_num * args.member_num):
                if dones_mask[j+1] == True:
                    continue
                
                team_idx = j // 8
                agent_idx = j % 8
                mod_obs = obs[j+1]
                mod_obs = mod_obs[np.newaxis, :]
                actions[j+1] = teams[team_idx].agents[agent_idx].take_action(mod_obs)

            next_obs, rewards, dones, infos = env.step(actions)
            step += 1

            for j in range(args.team_num * args.member_num):
                if dones_mask[j+1] == True:
                    continue
                if j >= 8:
                    continue
                all_trajs[j + 1]['state'].append(np.array(obs[j + 1]))
                all_trajs[j + 1]['action'].append(np.array(actions[j + 1]))
                all_trajs[j + 1]['next_state'].append(np.array(next_obs[j + 1]))
                all_trajs[j + 1]['reward'].append(np.array(rewards[j + 1]))
                all_trajs[j + 1]['done'].append(np.array(dones[j + 1]))
            dones_mask = dones

            episode_done = True
            for k in dones_mask.keys():
                if not dones_mask[k]:
                    episode_done = False
                    break

            obs = next_obs

        print(f"num_episode = {i}, step = {step}")

        for j in range(args.team_num * args.member_num):
            team_idx = j // 8
            if team_idx >= 1:
                continue
            agent_idx = j % 8
            team_buffers[team_idx].add(all_trajs[j+1], agent_idx)
            # print(f"team_buffers[team_idx].buffers[0][0]['reward'][0].shape = {team_buffers[team_idx].buffers[0][0]['reward'][0]}")

        if team_buffers[0].size() > args.minimal_size:
            for k in range(args.update_times):
                
                team_datas = []
                for j in range(args.team_num):
                    if j >= 1:
                        continue
                    team_data = team_buffers[j].sample(args.batch_size)
                    team_datas.append(team_data)


                teams[update_team_idx].update(team_datas[update_team_idx])

            update_cnt += 1
            # if update_cnt % args.update_change == 0:
            #     update_team_idx += 1
            #     update_team_idx %= args.team_num

            if update_cnt % args.save_interval == 0:
                torch.save(teams[update_team_idx].mixnet.state_dict(), f'./model/team_{update_cnt}.pth')
                torch.save(teams[update_team_idx].agents[0].net.state_dict(), f'./model/agent_{update_cnt}.pth')

            if update_cnt % args.para_update_interval == 0:
                print("update other teams paramaters")
                for team_idx in range(args.team_num):
                    if team_idx == 0:
                        continue
                    teams[team_idx].mixnet = torch.load(teams[0].mixnet.state_dict())
                    teams[team_idx].target_mixnet = torch.load(teams[0].target_mixnet.state_dict())
                    for agent_idx in range(args.member_num):
                        teams[team_idx].agents[agent_idx].net = torch.load(teams[0].agents[agent_idx].net.state_dict())

def bug_de(args):
    env = setup_env(args)

    teams = init_agents(env, args)

    team_buffers = init_buffers(args)

    with open('./data.pkl', 'rb') as f:
        team_datas = pickle.load(f)

    teams[0].update(team_datas[0])



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--team_num', type=int, default=8)
    parser.add_argument('--member_num', type=int, default=8)        # members each team
    parser.add_argument('--episode_num', type=int, default=10000)     # Originally 10000; hard code temporarily
    parser.add_argument('--input_size', type=int, default=256)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--task_size', type=int, default=4096)
    parser.add_argument('--buffer_capacity', type=int, default=200)         # How many trajectories are kept in the buffer. Out of this capacity, the trajs are first-in-first-out (FIFO), leaving only the most recent ones.
    parser.add_argument('--minimal_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--gamma', type=float, default=0.98)
    parser.add_argument('--target_update', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--save_interval', type=int, default=50)     # 50
    parser.add_argument('--update_change', type=int, default=16)
    parser.add_argument('--update_times', type=int, default=8)
    parser.add_argument('--para_update_interval', type=int, default=100)        # 100


    args = parser.parse_args()

    # bug_de(args)
    train(args)


