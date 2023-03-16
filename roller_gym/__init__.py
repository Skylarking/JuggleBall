from gym.envs.registration import register
from roller_gym.roller_env import RollerEnv
register(
    id='Roller-v0',
    entry_point='roller_gym.roller_env:RollerEnv',
)