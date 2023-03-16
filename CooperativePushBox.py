import numpy
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import numpy as np
from MAPPO import MAPPO
import RollerParms
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import os, shutil
import torch
from mlagents_envs.registry import default_registry

def action_adapter(action):
    actions = ActionTuple()
    actions.add_continuous(np.zeros([0]).reshape(1,-1))
    actions.add_discrete(np.array(action).reshape(1,-1))
    return actions


EnvName = 'CooperativePushBox'
n_agents = 3



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

env = UnityEnvironment(file_name=EnvName, seed=1, side_channels=[channel])

############################################################################################
channel.set_configuration_parameters(time_scale=10.0, height=512, width=768)  # 10倍速渲染
env.reset() # 所有操作之前一定要reset

''' 打印behavior信息 '''
# We will only consider the first Behavior
print(len(list(env.behavior_specs)))
behavior_name = list(env.behavior_specs)
print(f"Name of the behavior : {behavior_name}")
spec = env.behavior_specs[behavior_name[0]]    # agent




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
action_dim_ = []
if spec.action_spec.discrete_size > 0:
    for action, branch_size in enumerate(spec.action_spec.discrete_branches):
        print(f"Action number {action} has {branch_size} different options")
        action_dim_.append(branch_size)
action_dim = [action_dim_ for n in range(n_agents)]

write = True
if write:
    timenow = str(datetime.now())[0:-10]
    timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
    writer = []
    for n in range(n_agents):
        writepath = 'runs/{}'.format(EnvName) + timenow + '/agent{}'.format(n)
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer.append(SummaryWriter(log_dir=writepath))
    writer2 = SummaryWriter(log_dir='runs/{}'.format(EnvName) + timenow + '/reward')


if not os.path.exists('model_ma_linear_c'): os.mkdir('model_ma_linear_c')
file_path = "./model_ma_linear_c/"



state_dim = [spec.observation_specs[0].shape[0], spec.observation_specs[0].shape[0], spec.observation_specs[0].shape[0]]
model = MAPPO(n_agents=n_agents, state_dim=state_dim, action_dim=action_dim)

behavior_list = [behavior_name[0], behavior_name[0], behavior_name[0]]
''' 运行流程 '''







T_horizon = 2048
total_steps = 0
Max_train_steps = 5e7
eval_interval = 200
random_seed = 1
total_reward = [0 for n in range(n_agents)]
save_interval = 5000
traj_lenth = 0


while total_steps < Max_train_steps:
    env.reset()

    episode_rewards = [0 for n in range(n_agents)]
    episode_steps = 0


    decision_steps_list = []
    terminal_steps_list = []
    for n in range(n_agents):
        decision_steps, terminal_steps = env.get_steps(behavior_list[n])
        decision_steps_list.append(decision_steps)
        terminal_steps_list.append(terminal_steps)

    done = False # For the tracked_agent

    while not done:
        episode_steps += 1
        traj_lenth += 1
        total_steps += 1
        # Track the first agent we see if not tracking
        # Note : len(decision_steps) = [number of agents that requested a decision]

        # agent id
        agent_goalie_id_1 = decision_steps_list[0].agent_id[0]
        agent_soccer_id_1 = decision_steps_list[0].agent_id[1]
        agent_soccer_id_2 = decision_steps_list[0].agent_id[2]
        agent_id = [agent_goalie_id_1, agent_soccer_id_1, agent_soccer_id_2]



        # Get the current obs
        s = [[] for i in range(n_agents)]
        for n in range(n_agents):
            s[n].append(decision_steps_list[n][agent_id[n]].obs[0])


        # Generate an action for all agents
        a = []
        prob_n = []
        for n in range(n_agents):
            action, prob = model.agent[n].select_action(s[n])
            a.append(action)
            prob_n.append(prob)
            env.set_action_for_agent(behavior_list[n], agent_id[n], action_adapter(action))



        # Move the simulation forward
        env.step()

        # Get the new simulation results: next_state,reward,done
        s_next = [[] for i in range(n_agents)]
        reward = []
        for n in range(n_agents):
            decision_steps, terminal_steps = env.get_steps(behavior_list[n])
            if agent_id[n] in terminal_steps:  # The agent terminated its episode
                r = terminal_steps[agent_id[n]].reward
                s_next[n].append(terminal_steps[agent_id[n]].obs[0])
                done = True
            else:
                r = decision_steps[agent_id[n]].reward
                s_next[n].append(decision_steps[agent_id[n]].obs[0])

            reward.append(r)
            episode_rewards[n] += r
            total_reward[n] += r


        # train
        data = (s, a, reward, s_next, prob_n, done)
        model.put_data(data)

        if traj_lenth % T_horizon == 0:
            pi_loss_n, v_loss_n = model.train_all()
            print("loss:", pi_loss_n, v_loss_n)
            for n in range(n_agents):
                writer[n].add_scalar('agent{} pi_loss'.format(n), pi_loss_n[n],
                                     global_step=total_steps)
                writer[n].add_scalar('agent{} v_loss'.format(n), v_loss_n[n],
                                     global_step=total_steps)
            traj_lenth = 0


        if total_steps % eval_interval == 0:
            if write:
                writer2.add_scalars('total reward in 200', {"agent0": total_reward[0],
                                                          "agent1": total_reward[1],
                                                          "agent2": total_reward[2]}, total_steps)
                print('EnvName:', EnvName, 'seed:', random_seed, 'steps: {}'.format(int(total_steps)),
                    'total score:', [total_reward[i] for i in range(n_agents)])
            total_reward = [0 for n in range(n_agents)]

        '''save model'''
        if total_steps % save_interval == 0:
            model.save(episode=total_steps, file_path=file_path)



env.close()
print("Closed environment")
