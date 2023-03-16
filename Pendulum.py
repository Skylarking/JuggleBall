import model
import torch
import gym
import torch.nn as nn
import torch.nn.functional as F
from utils import make_env, action_adapter, evaluate_policy, display_data, load_actor, load_critic
from ReplayBuffer import ReplayBuffer
import pickle
from ppo import PPO

env = gym.make('Pendulum-v1')

obs_shape = env.observation_space.shape
action_shape = env.action_space.shape
action_range = [env.action_space.low, env.action_space.high]    # action 范围
print(action_range)


max_buffer_len = 50000
Buffer = ReplayBuffer(max_buffer_len)

# 从之前模型训练
# actor = load_actor(obs_shape[0], action_shape[0], file='./RollerAgent/saved_model/episode2200_1649327979_actor.pkl')
# critic = load_critic(obs_shape[0], file='./RollerAgent/saved_model/episode2200_1649327979_critic.pkl')
agent = model.PPO(obs_shape[0], action_shape[0], Buffer, actor=model.BetaActor(obs_shape[0], action_shape[0]), critic=model.critic(obs_shape[0]), actor_lr=0.0004, critic_lr=0.0004)
# agent = PPO(obs_shape[0],action_shape[0],False)
T = 2048

# 打印图表信息
score_episode = []
score_list = []
episode_list = []
v_loss_list = []
pi_loss_list = []
score = 0

length = 0
num_steps = 0
traj_lenth = 0
for i in range(60000):


    obs = env.reset()
    done = False
    steps = 0
    while not done:
        traj_lenth += 1

        action, log_old_prob = agent.take_action(obs)
        obs_next, reward, done, info = env.step(action)


        data = (obs, action, (reward + 8) / 8, obs_next, log_old_prob, done)
        agent.put_data(data)

        # env.render()

        obs = obs_next
        score += reward
        num_steps += 1
        steps += 1
        length += 1

        if length % T == 0:
            agent.update()
            length = 0

    if num_steps % 5000 == 0:
        score = evaluate_policy(env, agent, env._max_episode_steps, render=False)
        print("{}k steps eval score : {}".format(int(num_steps/1000), score))

        score = 0


    # 保存一次模型


