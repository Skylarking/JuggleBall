import argparse

def str2bool(v):
    '''transfer str to bool for argparse'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


#################################### HyperParameters ##########################################
def parms():
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', type=str2bool, default=True, help='Use SummaryWriter to record the training')
    parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
    parser.add_argument('--ModelIdex', type=int, default=400, help='which model to load')
    parser.add_argument('--EnvName', type=str, default='Roller', help='Environment name')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--T_horizon', type=int, default=2048, help='lenth of long trajectory')
    parser.add_argument('--Max_train_steps', type=int, default=5e7, help='Max training steps')
    parser.add_argument('--save_interval', type=int, default=5e3, help='Model saving interval, in steps.')
    parser.add_argument('--eval_interval', type=int, default=5e3, help='Model evaluating interval, in steps.')
    parser.add_argument('--isTest', type=str2bool, default=True, help='is test or train mode.')
    parser.add_argument('--time_scale', type=float, default=100.0, help='物理环境加速倍速')


    parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
    parser.add_argument('--lambd', type=float, default=0.95, help='GAE Factor')
    parser.add_argument('--clip_rate', type=float, default=0.4, help='PPO Clip rate')
    parser.add_argument('--K_epochs', type=int, default=10, help='PPO update times')
    parser.add_argument('--net_width', type=int, default=256, help='Hidden net width')
    parser.add_argument('--a_lr', type=float, default=3e-4, help='Learning rate of actor')
    parser.add_argument('--c_lr', type=float, default=2e-6, help='Learning rate of critic')
    parser.add_argument('--l2_reg', type=float, default=1e-3, help='L2 regulization coefficient for Critic')
    parser.add_argument('--optim_batch_size', type=int, default=512, help='lenth of sliced trajectory of actor')
    parser.add_argument('--entropy_coef', type=float, default=1e-3, help='Entropy coefficient of Actor')
    parser.add_argument('--entropy_coef_decay', type=float, default=0.999, help='Decay rate of entropy_coef')
    return parser.parse_args()