from mlagents_envs.environment import UnityEnvironment,ActionTuple
import numpy as np



# This is a non-blocking call that only loads the environment.
env = UnityEnvironment(file_name="Roller", seed=1, side_channels=[])
env.reset()
# Start interacting with the environment.
"""
file_name is the name of the environment binary (located in the root directory of the python project).
worker_id indicates which port to use for communication with the environment.
    For use in parallel training regimes such as A3C.
seed indicates the seed to use when generating random numbers during the training process.
    In environments which are deterministic, setting the seed enables reproducible experimentation by ensuring
    that the environment and trainers utilize the same random seed.
side_channels provides a way to exchange data with the Unity simulation
    that is not related to the reinforcementlearning loop.
    For example: configurations or properties.More on them in the Modifying the environment from Python section.
If you want to directly interact with the Editor, you need to use file_name=None,
    then press the Play button in the Editor when the message
    "Start training by pressing the Play button in the Unity Editor" is displayed on the screen
"""
#env = UnityEnvironment(file_name=None, seed=1, side_channels=[UnityStaticLogChannel()])
# set time scale
#config_channel.set_configuration_parameters(time_scale=1.0)
# Start interacting with the environment.


"""
Returns a Mapping of BehaviorName to BehaviorSpec objects (read only).
    A BehaviorSpec contains the observation shapes and the ActionSpec (which defines the action shape).
    Note that the BehaviorSpec for a specific group is fixed throughout the simulation.
    The number of entries in the Mapping can change over time in the simulation if new Agent behaviors
        are created in the simulation.
An Agent "Behavior" is a group of Agents identified by a BehaviorName that share the same observations
    and action types (described in their BehaviorSpec).
"""

behavior_names = env.behavior_specs.keys()

for i in behavior_names:
    print("[Info] Behavior Name: ", i.title())


count = 0

for i in range(100):
    env.reset()
    done = False
    while not done:
        for name in behavior_names:
            # 获得当前状态
            DecisionSteps, TerminalSteps = env.get_steps(name)
            obs = DecisionSteps.obs


            # 获得动作，此添加算法
            actions = ActionTuple()
            ac = 2 * (np.random.random(size=2) - 0.5).reshape(1, 2)  # 第一个维度为agent个数，第二个维度为action shape
            actions.add_continuous(ac)
            env.set_actions(name, actions)

            # 虚拟环境执行动作，并且获得下一个状态
            env.step()
            DecisionSteps_, TerminalSteps_ = env.get_steps(name)
            obs_, reward = DecisionSteps_.obs, DecisionSteps_.reward

            # 一局游戏是否结束
            done = bool(TerminalSteps_.interrupted)
            if done:
                obs_, reward = TerminalSteps_.obs, TerminalSteps_.reward

env.close()




