import model
import torch
import gym
import torch.nn as nn
import torch.nn.functional as F
from utils import make_env, action_adapter, evaluate_policy, display_data, load_actor, load_critic
from ReplayBuffer import ReplayBuffer
import pickle

env = make_env('../RollerAgent', seed=1)

obs_shape = env.observation_space.shape
action_shape = env.action_space.shape
action_range = [env.action_space.low, env.action_space.high]    # action 范围
print(action_range)


max_buffer_len = 50000
Buffer = ReplayBuffer(max_buffer_len)

# 从之前模型训练
# actor = load_actor(obs_shape[0], action_shape[0], file='./RollerAgent/saved_model/episode2200_1649327979_actor.pkl')
# critic = load_critic(obs_shape[0], file='./RollerAgent/saved_model/episode2200_1649327979_critic.pkl')
agent = model.PPO(obs_shape[0], action_shape[0], Buffer, actor=model.actor(obs_shape[0], action_shape[0]), critic=model.critic(obs_shape[0]), actor_lr=0.0004, critic_lr=0.0004)
T = 2018

# 打印图表信息
score_episode = []
score_list = []
episode_list = []
v_loss_list = []
pi_loss_list = []
score = 0

for i in range(60000):

    num_steps = 0
    done = False
    steps = 0
    obs = env.reset()

    for step in range(30):
        action, log_old_prob = agent.take_action(obs)
        obs_next, reward, done, info = env.step(action)
        data = (obs, action, reward, log_old_prob, obs_next, done)
        agent.buffer.push(data)
        env.render()
        if agent.buffer.buffer_len() % T == 0:
            training_step, v_loss, pi_loss = agent.update(batch_size=T, type='clip')
            print("###################################################################################")
            print("更新参数,episode:{}".format(i))
            print("训练次数:{}".format(training_step))
            agent.buffer.clear()    # 清空数据


            print("v_loss : {}".format(v_loss))
            print("pi_loss : {}".format(pi_loss))
            print("###################################################################################")



            # 保存打印信息
            episode_list.append(i + 1)
            v_loss_list.append(v_loss)
            pi_loss_list.append(pi_loss)

            # 绘制图像
            #display_data(episode_list, v_loss_list)
            #display_data(episode_list, pi_loss_list)




        obs = obs_next
        score += reward
        num_steps += 1
        if done:
            break

    if (i+1) % 100 == 0:
        score_episode.append(i+1)
        score_list.append(score)  # 保存episode和评估得分
        print("100 epsode total score : {}".format(score))

        score = 0


    # 保存一次模型
    if (i+1) % 200 == 0:
        agent.save_model(path='../RollerAgent/saved_model/', para='episode{}'.format(i + 1))
        print("episode:{}：模型保存".format(i + 1))



display_data(score_episode, score_list)
display_data(episode_list, v_loss_list)
display_data(episode_list, pi_loss_list)

data_dict = {
    'episode':episode_list,
    'score_episode':score_episode,
    'score':score_list,
    'v_loss':v_loss_list,
    'pi_loss':pi_loss_list
}

with open('data222.pickle', 'wb') as f:
    pickle.dump(data_dict,f)