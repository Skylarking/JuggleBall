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
def state_adapter(s, n_agents=3):
    '''
    s shape: (n_agents,n_obs,n_feature)
    将s[n] cat起来组成整个观测
    :param s:
    :return:  s shape: (n_agents,n_obs * n_feature)
    '''
    for n in range(n_agents):
        s[n] = np.concatenate(s[n])
    return s


########################### step1.创建环境和打印环境信息 ########################
EnvName = './linux_env/StrikerVsGoalie/SvsG_Linux.x86_64'
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
channel.set_configuration_parameters(time_scale=100.0, height=512, width=768)  # 10倍速渲染
env.reset() # 所有操作之前一定要reset



''' behavior信息,每种behavior表示一个agent的种类，同种agent观测动作一致，可以用同一个网络 '''
behavior_name = list(env.behavior_specs)
spec_lst = [env.behavior_specs[name] for name in behavior_name]
agent_group = [1, 2]# TODO 每种behavior一共有多少个agent，维度和behavior_name一致

''' 打印环境信息 '''
# Is the Action continuous or multi-discrete ?
for i, spec in enumerate(spec_lst):
    print('##################################################')
    print(behavior_name[i])
    print('    action:')
    if spec.action_spec.continuous_size > 0:
        print(f"        There are {spec.action_spec.continuous_size} continuous actions")
    if spec.action_spec.is_discrete():
        print(f"        There are {spec.action_spec.discrete_size} discrete actions")
        for action, branch_size in enumerate(spec.action_spec.discrete_branches):
            print(f"        Action number {action} has {branch_size} different options")


    print('    observation:')
    vis_obs = any(len(spec.shape) == 3 for spec in spec.observation_specs)
    print("        Visual observation: ", vis_obs)
    print("        Number of observations : ", len(spec.observation_specs))
    print('##################################################')

n_agents = 3


############################## step2.tensorboard打印信息 #################################
write = False
if write:
    timenow = str(datetime.now())[0:-10]
    timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
    writer = []
    for n in range(n_agents):
        writepath = 'runs/{}'.format(EnvName) + timenow + '/agent{}'.format(n)
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer.append(SummaryWriter(log_dir=writepath))
    writer2 = SummaryWriter(log_dir='runs/{}'.format(EnvName) + timenow + '/reward')

############################ step3.加载模型 ###############################
if not os.path.exists('model_ma_linear_c'): os.mkdir('model_ma_linear_c')
file_path = "./model_ma_linear_c/"


# TODO 若环境修改，根据环境获得每种agent的state和action维度
state_dim_lst = []
action_dim_lst = []
for i, spec in enumerate(spec_lst):
    for j in range(agent_group[i]):
        state_dim_lst.append( [state_dim.shape for state_dim in spec.observation_specs] )
        action_dim_lst.append( [branch_size for action_id, branch_size in enumerate(spec.action_spec.discrete_branches)] )


state_dim_lst = [738, 294, 294]


# TODO 加载模型
model = MAPPO(n_agents=n_agents, state_dim=state_dim_lst, action_dim=action_dim_lst)
model.load(episode=100000, file_path=file_path)






''' 运行流程 '''



################# step4.训练过程 #############################################



T_horizon = 2048
total_steps = 0
Max_train_steps = 5e7
eval_interval = 1000
random_seed = 1
total_reward = [0 for n in range(n_agents)]
save_interval = 5000
traj_lenth = 0

def get_steps(behavior_name, env):
    '''
    获取每个behavior_name的decision_steps, terminal_steps，形状和behavior_name一致

    :param behavior_name:
    :return:
    '''
    decision_steps_list = []
    terminal_steps_list = []
    for name in behavior_name:
        decision_steps, terminal_steps = env.get_steps(name)
        decision_steps_list.append(decision_steps)
        terminal_steps_list.append(terminal_steps)
    return decision_steps_list, terminal_steps_list

while total_steps < Max_train_steps:
    env.reset()

    episode_rewards = [0 for n in range(n_agents)]
    episode_steps = 0


    decision_steps_list, terminal_steps_list = get_steps(behavior_name, env)


    done = False # For the tracked_agent

    while not done:
        episode_steps += 1
        traj_lenth += 1
        total_steps += 1
        # Track the first agent we see if not tracking
        # Note : len(decision_steps) = [number of agents that requested a decision]

        # agent id
        agent_goalie_id_1 = decision_steps_list[0].agent_id[0]
        agent_soccer_id_1 = decision_steps_list[1].agent_id[0]
        agent_soccer_id_2 = decision_steps_list[1].agent_id[1]
        agent_id = [agent_goalie_id_1, agent_soccer_id_1, agent_soccer_id_2]



        # Get the current obs
        s = []
        for i, decision_steps in enumerate(decision_steps_list):
            for j in range(agent_group[i]):
                AgentId = i + j
                s.append( decision_steps[AgentId].obs )
        s = state_adapter(s,n_agents)

        # Generate an action for all agents
        a = []
        prob_n = []
        for i, decision_steps in enumerate(decision_steps_list):
            for j in range(agent_group[i]):
                AgentId = i + j
                n = AgentId
                action, prob = model.agent[n].evaluate(s[n])
                a.append(action)
                prob_n.append(prob)
                env.set_action_for_agent(behavior_name[i], AgentId, action_adapter(action))



        # Move the simulation forward
        env.step()

        # Get the new simulation results: next_state,reward,done
        decision_steps_list, terminal_steps_list = get_steps(behavior_name, env)
        s_next = []
        reward = []
        for i, (decision_steps, terminal_steps) in enumerate(zip(decision_steps_list, terminal_steps_list)):
            for j in range(agent_group[i]):
                AgentId = i + j
                n = AgentId
                if AgentId in terminal_steps:
                    r = terminal_steps[AgentId].reward
                    s_next.append(terminal_steps[AgentId].obs)
                    done = True
                else:
                    r = decision_steps[AgentId].reward
                    s_next.append(decision_steps[AgentId].obs)
                reward.append(r)
                episode_rewards[n] += r
                total_reward[n] += r
        # for i, name in enumerate(behavior_name):
        #     for j in range(agent_group[i]):
        #         decision_steps, terminal_steps = decision_steps_list[i], terminal_steps_list[i]
        #
        #         AgentId = i + j
        #         if AgentId in terminal_steps:
        #             r = terminal_steps[AgentId].reward
        #             s_next.append(terminal_steps[AgentId].obs)
        #             done = True
        #         else:
        #             r = decision_steps[AgentId].reward
        #             s_next.append(decision_steps[AgentId].obs)
        #         reward.append(r)
        #         episode_rewards[n] += r
        #         total_reward[n] += r

        s_next = state_adapter(s_next, n_agents)




        # train
        data = (s, a, reward, s_next, prob_n, done)
        model.put_data(data)

        '''打印信息'''
        if total_steps % eval_interval == 0:
            if write:
                writer2.add_scalars('total reward in 1000', {"agent0": total_reward[0],
                                                          "agent1": total_reward[1],
                                                          "agent2": total_reward[2]}, total_steps)
            print('EnvName:', EnvName, 'seed:', random_seed, 'steps: {}'.format(int(total_steps)),
                'total score:', [total_reward[i] for i in range(n_agents)])
            total_reward = [0 for n in range(n_agents)]




env.close()
print("Closed environment")





