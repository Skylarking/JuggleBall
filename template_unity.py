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
from mlagents_envs.registry import default_registry


''' 设置旁落通信，可设置环境信息，并且直接操纵环境 '''
channel = EngineConfigurationChannel()

##################### 两种创建环境方式，第一种是unity自带环境，第二种是自己定义环境 ###################
# env_default = ['Basic', '3DBall', '3DBallHard', 'GridWorld', 'Hallway', 'VisualHallway',
#                'CrawlerDynamicTarget', 'CrawlerStaticTarget', 'Bouncer', 'SoccerTwos', 'PushBlock',
#                'VisualPushBlock', 'WallJump', 'Tennis', 'Reacher', 'Pyramids', 'VisualPyramids', 'Walker',
#                'FoodCollector', 'VisualFoodCollector', 'StrikersVsGoalie', 'WormStaticTarget',
#                'WormDynamicTarget']
# env_id = "StrikersVsGoalie"
# env = default_registry[env_id].make(side_channels = [channel])


env = UnityEnvironment(file_name='JuggleBall', seed=1, side_channels=[channel], no_graphics=False)

############################################################################################
channel.set_configuration_parameters(time_scale=1.0, height=1024, width=1920)  # 10倍速渲染
env.reset() # 所有操作之前一定要reset

''' 打印behavior信息 '''
# We will only consider the first Behavior
print(len(list(env.behavior_specs)))
behavior_name = list(env.behavior_specs)[0] # TODO 多agent时list中有多个behavior
print(f"Name of the behavior : {behavior_name}")
spec = env.behavior_specs[behavior_name]


''' 打印每个agent观测种类个数 '''
# Examine the number of observations per Agent
print("Number of observations : ", len(spec.observation_specs))

''' 是否有图像视觉观察 '''
# Is there a visual observation ?
# Visual observation have 3 dimensions: Height, Width and number of channels
vis_obs = any(len(spec.shape) == 3 for spec in spec.observation_specs)
print("Is there a visual observation ?", vis_obs)

''' 动作空间，包括连续和离散 '''
# Is the Action continuous or multi-discrete ?
if spec.action_spec.continuous_size > 0:
  print(f"There are {spec.action_spec.continuous_size} continuous actions")
if spec.action_spec.is_discrete():
  print(f"There are {spec.action_spec.discrete_size} discrete actions")

''' 动作空间大小 '''
# How many actions are possible ?
#print(f"There are {spec.action_size} action(s)")

''' 离散动作空间大小 '''
# For discrete actions only : How many different options does each action has ?
if spec.action_spec.discrete_size > 0:
  for action, branch_size in enumerate(spec.action_spec.discrete_branches):
    print(f"Action number {action} has {branch_size} different options")

''' 运行流程 '''
for episode in range(100):
    env.reset()
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    tracked_agent = -1 # -1 indicates not yet tracking
    done = False # For the tracked_agent
    episode_rewards = 0 # For the tracked_agent
    while not done:
        # Track the first agent we see if not tracking
        # Note : len(decision_steps) = [number of agents that requested a decision]
        if tracked_agent == -1 and len(decision_steps) >= 1:
            tracked_agent = decision_steps.agent_id[0]  # TODO 将tracked_agent改成多个，可以获得多agent观察与动作

        # Get the current obs
        s = decision_steps[tracked_agent].obs[0]

        # Generate an action for all agents
        action = spec.action_spec.random_action(len(decision_steps))

        # Set the actions
        # env.set_action_for_agent(behavior_name,tracked_agent,action)
        env.set_actions(behavior_name, action)

        # Move the simulation forward
        env.step()

        # Get the new simulation results: next_state,reward,done
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        if tracked_agent in decision_steps:  # The agent requested a decision
            reward = decision_steps[tracked_agent].reward
            episode_rewards += reward
            s_next = decision_steps[tracked_agent].obs[0]
        if tracked_agent in terminal_steps:  # The agent terminated its episode
            reward = terminal_steps[tracked_agent].reward
            episode_rewards += reward
            s_next = terminal_steps[tracked_agent].obs[0]
            done = True
        print('n')
    print(f"Total rewards for episode {episode} is {episode_rewards}")

env.close()
print("Closed environment")