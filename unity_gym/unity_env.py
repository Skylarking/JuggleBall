import numpy
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import numpy as np
from ppo import PPO
import RollerParms
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import os, shutil
import torch
import gym
from gym import spaces

class RollerEnv(gym.Env):
    def __init__(self, env_file, seed=1, worker_id=0, time_scale=10):
        super().__init__()
        self.channel = EngineConfigurationChannel()
        self.channel.set_configuration_parameters(time_scale=time_scale)
        self.env = UnityEnvironment(file_name=env_file,
                                    seed=seed,
                                    side_channels=[self.channel],
                                    worker_id=worker_id)
        self.env.reset()  # 所有操作之前一定要reset
        self.behavior_name = list(self.env.behavior_specs)[0]  # 假设只有一种agent类型
        self.decision_steps, self.terminal_steps = self.env.get_steps(
            self.behavior_name)  # list，存每个agent的信息，用于获取所有agent在step后的环境信息(reward，obs等)
        self.agent_id_list = self.decision_steps.agent_id

        self.done = False  # denote whether to use decision_steps or terminal_steps

        # spec_list
        self.spec = self.env.behavior_specs[self.behavior_name]  # 保存该name种类的agent的相关信息

        self.episode_limit = 5000

        self.n_agents = len(self.decision_steps)
        self.action_shape = self.spec.action_spec.continuous_size  # 只取连续动作
        self.obs_shape = self.spec.observation_specs[0].shape[0]  # 只取vector类型的观测数据

        # gym info
        self.action_space = spaces.Box(-1., 1., shape=(self.action_shape,), dtype='float32')
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.obs_shape, ), dtype='float32')

    def _action_adapter(self, ac):
        if isinstance(ac, list):
            ac = np.array(ac)
        ac = ac.reshape(1, -1)
        actions = ActionTuple()
        actions.add_continuous(ac)
        return actions

    def reset(self):
        self.env.reset()

        # update env info since the reset() call
        self.behavior_name = list(self.env.behavior_specs)[0]
        self.decision_steps, self.terminal_steps = self.env.get_steps(
            self.behavior_name)
        self.agent_id_list = self.decision_steps.agent_id
        self.agent_id = self.agent_id_list[0]
        self.done = False
        self.spec = self.env.behavior_specs[self.behavior_name]

        return self.get_obs()[0]

    def step(self, action):
        actions = [action]
        if isinstance(actions, list):
            actions = actions
        else:
            actions = action.cpu().numpy()

        for (agent_id, act) in zip(self.agent_id_list, actions):
            act = self._action_adapter(act)
            self.env.set_action_for_agent(self.behavior_name, agent_id, act)


        self.env.step()  # step()

        # update env info
        self.decision_steps, self.terminal_steps = self.env.get_steps(
            self.behavior_name)  # list，存每个agent的信息，用于获取所有agent在step后的环境信息(reward，obs等)

        for agent_id in self.agent_id_list:
            if agent_id in self.terminal_steps:
                self.done = True

        rewards = []
        if self.done:
            for agent_id in self.agent_id_list:
                reward = self.terminal_steps[agent_id].reward
                group_reward = self.terminal_steps[agent_id].group_reward
                rewards.append(reward + group_reward)
        else:
            for agent_id in self.agent_id_list:
                reward = self.decision_steps[agent_id].reward
                group_reward = self.decision_steps[agent_id].group_reward
                rewards.append(reward + group_reward)

        env_info = {}
        return self.get_obs()[0], rewards[0], self.done, env_info

    def close(self):
        self.env.close()

    def get_obs_agent(self, agent_id):
        if self.done:
            return self.terminal_steps[agent_id].obs[0]  # ndarray
        else:
            return self.decision_steps[agent_id].obs[0]  # ndarray

    def get_obs(self):
        obs = []
        for agent_id in self.agent_id_list:
            obs.append(self.get_obs_agent(agent_id))
        return obs

class JuggleBallEnv(gym.Env):
    def __init__(self, env_file, seed=1, worker_id=0, time_scale=10):
        super().__init__()
        self.channel = EngineConfigurationChannel()
        self.channel.set_configuration_parameters(height=256, width=512, time_scale=time_scale)
        self.env = UnityEnvironment(file_name=env_file,
                                    seed=seed,
                                    side_channels=[self.channel],
                                    worker_id=worker_id)
        self.env.reset()  # 所有操作之前一定要reset
        self.behavior_name = list(self.env.behavior_specs)[0]  # 假设只有一种agent类型
        self.decision_steps, self.terminal_steps = self.env.get_steps(
            self.behavior_name)  # list，存每个agent的信息，用于获取所有agent在step后的环境信息(reward，obs等)
        self.agent_id_list = self.decision_steps.agent_id

        self.done = False  # denote whether to use decision_steps or terminal_steps

        # spec_list
        self.spec = self.env.behavior_specs[self.behavior_name]  # 保存该name种类的agent的相关信息

        self.episode_limit = 5000

        self.n_agents = len(self.decision_steps)
        self.action_shape = self.spec.action_spec.continuous_size  # 只取连续动作
        self.obs_shape = self.spec.observation_specs[0].shape[0]  # 只取vector类型的观测数据

        # gym info
        self.action_space = spaces.Box(-1., 1., shape=(self.action_shape,), dtype='float32')
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.obs_shape, ), dtype='float32')

    def _action_adapter(self, ac):
        if isinstance(ac, list):
            ac = np.array(ac)
        ac = ac.reshape(1, -1)
        actions = ActionTuple()
        actions.add_continuous(ac)
        return actions

    def reset(self):
        self.env.reset()

        # update env info since the reset() call
        self.behavior_name = list(self.env.behavior_specs)[0]
        self.decision_steps, self.terminal_steps = self.env.get_steps(
            self.behavior_name)
        self.agent_id_list = self.decision_steps.agent_id
        self.agent_id = self.agent_id_list[0]
        self.done = False
        self.spec = self.env.behavior_specs[self.behavior_name]

        return self.get_obs()[0]

    def step(self, action):
        actions = [action]
        if isinstance(actions, list):
            actions = actions
        else:
            actions = action.cpu().numpy()

        for (agent_id, act) in zip(self.agent_id_list, actions):
            act = self._action_adapter(act)
            self.env.set_action_for_agent(self.behavior_name, agent_id, act)


        self.env.step()  # step()

        # update env info
        self.decision_steps, self.terminal_steps = self.env.get_steps(
            self.behavior_name)  # list，存每个agent的信息，用于获取所有agent在step后的环境信息(reward，obs等)

        for agent_id in self.agent_id_list:
            if agent_id in self.terminal_steps:
                self.done = True

        rewards = []
        if self.done:
            for agent_id in self.agent_id_list:
                reward = self.terminal_steps[agent_id].reward
                group_reward = self.terminal_steps[agent_id].group_reward
                rewards.append(reward + group_reward)
        else:
            for agent_id in self.agent_id_list:
                reward = self.decision_steps[agent_id].reward
                group_reward = self.decision_steps[agent_id].group_reward
                rewards.append(reward + group_reward)

        env_info = {}
        return self.get_obs()[0], rewards[0], self.done, env_info

    def close(self):
        self.env.close()

    def get_obs_agent(self, agent_id):
        if self.done:
            return self.terminal_steps[agent_id].obs[0]  # ndarray
        else:
            return self.decision_steps[agent_id].obs[0]  # ndarray

    def get_obs(self):
        obs = []
        for agent_id in self.agent_id_list:
            obs.append(self.get_obs_agent(agent_id))
        return obs
