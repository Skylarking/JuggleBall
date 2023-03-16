from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment

max_step = 1000  # 整个训练的最大总步数
max_step_per_episode = 200  # 每个episode最大探索步数

if __name__ == '__main__':
    total_step = 0
    env_path = None
    # env_path = "你的unity场景编译后的exe文件路径"
    unity_env = UnityEnvironment(file_name="RollerAgent_render", seed=1)


    """
    参数allow_multiple_obs=False代表不启用多种类型的观察值，因为我的状态空间没有包括视觉，
    所以设为False，如果你的用到了视觉，根据官方文档，必须设为True才可正常接收到unity的视觉
    状态信息
    """
    env = UnityToGymWrapper(unity_env, allow_multiple_obs=False)

    state_dim = env.observation_space.shape[0]  # 获取状态空间的维度
    action_dim = env.action_space.shape[0]  # 获取动作空间的维度
    print(state_dim,action_dim)

    while total_step < max_step:  # 这里是按照最大总步数结束训练的，当然你也可以改成指定最大episode次数
        state = env.reset()
        current_ep_reward = 0

        for t in range(1, max_step_per_episode + 1):
            env.render()
            # action = agent.get_action(state)      # 对于外接算法，这里的action应该是你的强化学习算法给出
            action = [0.1, 0.05]  # 时间原因，只测试Python与unity Editor对接功能，这里就固定了action
            state_, reward, done, _ = env.step(action)
            print(reward)
            if done:
                break
