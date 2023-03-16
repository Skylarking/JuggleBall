import model
import torch
import gym
import torch.nn as nn
import torch.nn.functional as F
from utils import make_env, load_actor, load_critic
from ReplayBuffer import ReplayBuffer

env = make_env('../RollerAgent', seed=1)

obs_shape = env.observation_space.shape
action_shape = env.action_space.shape
action_range = [env.action_space.low, env.action_space.high]    # action 范围
print(action_range)


max_buffer_len = 50000
Buffer = ReplayBuffer(max_buffer_len)
actor = load_actor(obs_shape[0], action_shape[0], file='../RollerAgent/saved_model/episode5600_1649406856_actor.pkl')
critic = load_critic(obs_shape[0], file='../RollerAgent/saved_model/episode5600_1649406856_critic.pkl')
agent = model.PPO(obs_shape[0], action_shape[0], actor=actor, critic=critic)

g_score = 0

for i in range(1000):
    obs = env.reset()
    score = 0
    for step in range(30):
        env.render()
        action = agent.evaluate(obs)
        obs_next, reward, done, info = env.step(action)

        score += reward

        if done :
            break

    g_score += score
    if (i + 1) % 100 == 0:
        print('averge score : {}'.format(g_score / (100)))
        g_score = 0
