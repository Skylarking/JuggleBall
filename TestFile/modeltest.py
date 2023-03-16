import torch.nn as nn
import torch.nn.functional as F
from utils import make_env
from ReplayBuffer import ReplayBuffer
import model
import torch
import gym

env = make_env('../RollerAgent', seed=1)

obs_shape = env.observation_space.shape
action_shape = env.action_space.shape
action_range = [env.action_space.low, env.action_space.high]    # action 范围
print(action_range)


max_buffer_len = 50000
Buffer = ReplayBuffer(max_buffer_len)
agent = model.PPO(obs_shape[0], action_shape[0], Buffer, actor=model.actor(obs_shape[0], action_shape[0]), critic=model.critic(obs_shape[0]))
obs = env.reset()

for step in range(100):
    action, log_old_prob = agent.take_action(obs)
    obs_next, reward, done, info = env.step(action)
    data = (obs, action, reward, log_old_prob, obs_next, done)
    agent.buffer.push(data)
    env.render()

    if agent.buffer.buffer_len() % 5 == 0:
        training_step = agent.update(batch_size=5, type='clip')
        agent.buffer.clear()

    obs = obs_next
    if done:
        break