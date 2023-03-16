from gym.envs.registration import register
from unity_gym.unity_env import RollerEnv, JuggleBallEnv
register(
    id='Roller-v0',
    entry_point='unity_gym.unity_env:RollerEnv',
)

register(
    id='JuggleBall-v0',
    entry_point='unity_gym.unity_env:JuggleBallEnv',
)