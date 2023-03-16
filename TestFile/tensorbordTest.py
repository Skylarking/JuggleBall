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
EnvName = 'test'
a = torch.arange(0,36)

timenow = str(datetime.now())[0:-10]
timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
writepath = 'runs/{}'.format(EnvName) + timenow
if os.path.exists(writepath): shutil.rmtree(writepath)
writer = SummaryWriter(log_dir=writepath)

for i in range(36):
    writer.add_scalar('s1',i,i)
    writer.add_scalar('s2',i*2,i)