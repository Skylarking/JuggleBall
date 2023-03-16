import torch
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
import matplotlib.pyplot as plt
import model


def make_env(file_name=None,seed=1):
    '''
    返回gym环境

    :param file_name:
    :param seed:
    :return: gym环境
    '''
    unity_env = UnityEnvironment(file_name, seed=seed)
    env = UnityToGymWrapper(unity_env)
    return env

def load_actor(obs_shape, action_shape, file = None):
    actor = model.actor(obs_shape, action_shape)
    actor.load_state_dict(torch.load(file))
    actor.eval()
    return actor

def load_critic(obs_shape, file = None):
    critic = model.critic(obs_shape)
    critic.load_state_dict(torch.load(file))
    critic.eval()
    return critic

def action_adapter(action, action_range):
    '''
    将高斯policy所采样的动作弄到环境适合的范围

    :param action:
    :param low:
    :param high:
    :return:
    '''
    return action.clamp(action, action_range[0], action_range[1])

def evaluate_policy(env, model, steps_per_epoch, render = True):
    '''
    评估该策略
    用该策略玩一局游戏，计算平均得分

    :param env:
    :param model:
    :param render: 是否可视化
    :param steps_per_epoch: 一局游戏最大步数
    :return: 一局游戏平均得分
    '''
    scores = 0
    turns = 3
    for j in range(turns):
        obs, done, ep_r, steps = env.reset(), False, 0, 0
        while not (done or (steps >= steps_per_epoch)):

            action = model.evaluate(obs)
            # act = action_adapter(a, env.action_range)  # [0,1] to [-max,max]
            obs_next, reward, done, info = env.step(action)

            ep_r += reward
            steps += 1
            obs = obs_next
            if render:
                env.render()
        scores += ep_r
    return scores/turns


class Accumulator:
    """在`n`个变量上累加。
        首先要初始化 n

    """
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]#将arges中的所有参数都累加

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def display_data(x_list, y_list):
    x,y = x_list, y_list
    plt.plot(x, y, color='r', marker='o')


    # plt.axis([0, 6, 0, 20]) # axis[xmin, xmax, ymin, ymax]，坐标轴范围
    plt.show()


def cal_gpu(module):
    '''
    查看模型在哪个设备

    :param module:
    :return:
    '''
    if isinstance(module, torch.nn.DataParallel):
        module = module.module
    for submodule in module.children():
        if hasattr(submodule, "_parameters"):
            parameters = submodule._parameters
            if "weight" in parameters:
                return parameters["weight"].device



