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

def setup_env(args):

    print("-------setup env-------")
    env_args = config.create_config(config.Config)


    env_args.local_mode = True
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
        team_buffer = TeamReplayBuffer(args.buffer_capacity, args.member_num)
        team_buffers.append(team_buffer)

    return team_buffers

def train(args):

    env = setup_env(args)

    teams = init_agents(env, args)

    team_buffers = init_buffers(args)

    for i in range(args.episode_num):
        dones_mask = {k+1:False for k in range(args.team_num * args.member_num)}
        episode_done = False
        obs = env.reset()
        step = 0
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

            dones_mask = dones

            episode_done = True
            for k in dones_mask.keys():
                if not dones_mask[k]:
                    episode_done = False

            obs = next_obs

        print(f"num_episode = {i}, step = {step}")

        #! TODO
        # interact with environment and save traj to buffer

        #! TODO
        # update by MA2QL



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--team_num', type=int, default=8)
    parser.add_argument('--member_num', type=int, default=8)
    parser.add_argument('--episode_num', type=int, default=10000)
    parser.add_argument('--input_size', type=int, default=256)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--task_size', type=int, default=4096)
    parser.add_argument('--buffer_capacity', type=int, default=5000)


    args = parser.parse_args()

    train(args)


