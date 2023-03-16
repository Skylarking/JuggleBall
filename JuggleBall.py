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
import unity_gym
#################################### HyperParameters ##########################################
opt = RollerParms.parms()
##########################################################################################
def action_adapter(ac):
    action = ac
    if isinstance(ac, list):
        action = np.array(ac)
    action = (action - 0.5) * 2
    return action
def main():
    write = opt.write
    Max_train_steps = opt.Max_train_steps
    save_interval = opt.save_interval  # in steps
    eval_interval = opt.eval_interval  # in steps
    T_horizon = opt.T_horizon
    random_seed = opt.seed
    EnvName = 'JuggleBall'
    isTest = False #TODO opt.isTest


    ################################### env set & env info####################################
    env = gym.make('JuggleBall-v0', env_file = EnvName, time_scale = 1000)
    env.reset()



    ######################## 保存环境信息，创建保存文件夹 #########################################
    timenow = str(datetime.now())[0:-10]
    timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
    if write:
        writepath = 'runs/{}'.format(EnvName) + timenow
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)

    ##########################################################################################

    Dist = ['Beta', 'GS_ms', 'GS_m']  # type of probility distribution




    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    kwargs = {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "gamma": opt.gamma,
            "lambd": opt.lambd,     #For GAE
            "clip_rate": opt.clip_rate,  #0.2
            "env_with_Dead" : False,
            "K_epochs": opt.K_epochs,
            "net_width": opt.net_width,
            "a_lr": opt.a_lr,
            "c_lr": opt.c_lr,
            "dist": Dist[opt.distnum],
            "l2_reg": opt.l2_reg,   #L2 regulization for Critic
            "a_optim_batch_size":opt.a_optim_batch_size,
            "c_optim_batch_size": opt.c_optim_batch_size,
            "entropy_coef":opt.entropy_coef, #Entropy Loss for Actor: Large entropy_coef for large exploration, but is harm for convergence.
            "entropy_coef_decay":opt.entropy_coef_decay
        }

    ################################## 测试时加载之前的模型 ########################################
    file_path = "./model/" + EnvName + timenow + "/"
    if not os.path.exists(file_path): os.mkdir(file_path)


    agent = PPO(**kwargs)
    if isTest: agent.load(episode=55000, file_path=file_path)    # 测试加载模型
    ################################## ################ ########################################






    total_steps = 0
    traj_lenth = 0
    total_score = 0 # 计算5k steps内的总score
    episode_num = 0  # 5k steps中总的episode个数

    while total_steps < Max_train_steps:

        episode_num += 1

        # 初始化信息
        s = env.reset()
        done, steps = False, 0
        episode_rewards = 0 # For the tracked_agent


        while not done:

            steps += 1  # for the steps per episode
            traj_lenth += 1 # for the trajectory length
            total_steps += 1

            # Generate an action for all agents
            if isTest:
                action, log_prob = agent.evaluate(s)
            else:
                action, log_prob = agent.select_action(s)

            # step
            s_next, reward, done, info = env.step(action_adapter(action))# 将action弄到合适的范围之内

            agent.put_data((s, action, reward, s_next, log_prob, done))


            # 训练 or 测试
            if not isTest:
                if traj_lenth % T_horizon == 0:
                    agent.train()
                    traj_lenth = 0

            # 打印训练信息
            total_score += reward
            if total_steps % eval_interval == 0:
                if write:
                    writer.add_scalar('score in 5k steps', total_score, global_step=total_steps)
                print('EnvName:',EnvName,'seed:',random_seed,'steps: {}k'.format(int(total_steps/1000)),'average score:', total_score)
                total_score = 0
                episode_num = 0


            '''save model'''
            if not isTest:
                if total_steps % save_interval == 0:
                    agent.save(episode=total_steps, file_path=file_path)

if __name__ == '__main__':
    main()