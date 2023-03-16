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



def Action_adapter(ac):
    ac = ac.reshape(1,-1)
    ac = 2*(ac-0.5)
    actions = ActionTuple()
    actions.add_continuous(ac)
    return actions

#################################### HyperParameters ##########################################
opt = RollerParms.parms()
##########################################################################################

def main():
    write = opt.write
    Max_train_steps = opt.Max_train_steps
    save_interval = opt.save_interval  # in steps
    eval_interval = opt.eval_interval  # in steps
    T_horizon = opt.T_horizon
    random_seed = opt.seed
    EnvName = 'Roller'
    isTest = opt.isTest


    ################################### env set & env info####################################
    channel = EngineConfigurationChannel()
    env = UnityEnvironment(file_name=EnvName, seed=random_seed, side_channels=[channel])
    channel.set_configuration_parameters(time_scale=opt.time_scale, width=256, height=256)  # 10倍速渲染
    env.reset()


    # We will only consider the first Behavior
    behavior_name = list(env.behavior_specs)[0]
    print(f"Name of the behavior : {behavior_name}")
    spec = env.behavior_specs[behavior_name]

    # Examine the number of observations per Agent
    print("Number of observations : ", len(spec.observation_specs))
    state_dim = spec.observation_specs[0].shape[0]

    # Is there a visual observation ?
    # Visual observation have 3 dimensions: Height, Width and number of channels
    vis_obs = any(len(spec.shape) == 3 for spec in spec.observation_specs)
    print("Is there a visual observation ?", vis_obs)

    # Is the Action continuous or multi-discrete ?
    action_dim = spec.action_spec.continuous_size
    if spec.action_spec.continuous_size > 0:
        print(f"There are {spec.action_spec.continuous_size} continuous actions")
    if spec.action_spec.is_discrete():
        print(f"There are {spec.action_spec.discrete_size} discrete actions")


    # How many actions are possible ?
    #print(f"There are {spec.action_size} action(s)")

    # For discrete actions only : How many different options does each action has ?
    if spec.action_spec.discrete_size > 0:
        for action, branch_size in enumerate(spec.action_spec.discrete_branches):
            print(f"Action number {action} has {branch_size} different options")
    ##########################################################################################


    ######################## 保存环境信息，创建保存文件夹 #########################################
    if write:
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
        writepath = 'runs/{}'.format(EnvName) + timenow
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)

    ##########################################################################################

    Dist = ['Beta', 'GS_ms', 'GS_m']  # type of probility distribution





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
    if not os.path.exists('model'): os.mkdir('model')
    file_path = "./model/"

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
        env.reset()
        done, steps = False, 0
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        tracked_agent = -1 # -1 indicates not yet tracking
        done = False # For the tracked_agent
        episode_rewards = 0 # For the tracked_agent


        while not done:

            steps += 1  # for the steps per episode
            traj_lenth += 1 # for the trajectory length
            total_steps += 1

            # Track the first agent we see if not tracking
            # Note : len(decision_steps) = [number of agents that requested a decision]
            if tracked_agent == -1 and len(decision_steps) >= 1:
                tracked_agent = decision_steps.agent_id[0]  # 单agent，跟踪第一个agent

            # Get the current obs
            current_decision_steps, current_terminal_steps = env.get_steps(behavior_name)
            s = current_decision_steps[tracked_agent].obs[0]


            # Generate an action for all agents
            if isTest:
                action, log_prob = agent.evaluate(s)
            else:
                action, log_prob = agent.select_action(s)
            actions = Action_adapter(action)    # 将action弄到合适的范围之内

            # Set the actions
            env.set_actions(behavior_name, actions)

            # Move the simulation forward
            env.step()

            # Get the new simulation results
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            if tracked_agent in decision_steps: # The agent requested a decision
                reward = decision_steps[tracked_agent].reward
                episode_rewards += reward
                s_next = decision_steps[tracked_agent].obs[0]
            if tracked_agent in terminal_steps: # The agent terminated its episode
                reward = terminal_steps[tracked_agent].reward
                episode_rewards += reward
                s_next = terminal_steps[tracked_agent].obs[0]
                done = True

            agent.put_data((s, action, reward, s_next, log_prob, done))


            # 训练 or 测试
            if not isTest:  # 是否是测试
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