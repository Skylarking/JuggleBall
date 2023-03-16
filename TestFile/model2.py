import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import copy
from torch.distributions import Normal,MultivariateNormal
from ReplayBuffer import ReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class actor(nn.Module):
    def __init__(self, obs_shape, action_shape, hidden_num=128):
        super(actor, self).__init__()
        # parameters
        self.obs_shape = obs_shape
        self.action_shape = action_shape

        # network
        self.linear_1_mu = nn.Linear(obs_shape, hidden_num)
        self.linear_1_sigma = nn.Linear(obs_shape, hidden_num)
        self.linear_2_mu = nn.Linear(hidden_num, action_shape)
        self.linear_2_sigma = nn.Linear(hidden_num, action_shape)


    def forward(self, input):
        mu = F.leaky_relu(self.linear_2_mu(F.leaky_relu(self.linear_1_mu(input))))
        sigma = F.leaky_relu(self.linear_2_sigma(F.leaky_relu(self.linear_1_sigma(input))))

        return mu, torch.abs(sigma)





class critic(nn.Module):
    def __init__(self, obs_shape, hidden_num=128):
        super(critic, self).__init__()
        # parameters
        self.obs_shape = obs_shape

        # network
        self.linear_1 = nn.Linear(obs_shape , hidden_num)
        self.linear_2 = nn.Linear(hidden_num, 1)



    def forward(self, input):
        value = F.leaky_relu(self.linear_2(F.leaky_relu(self.linear_1(input))))

        return value


class PPO(object):
    '''
    agent类
    '''
    def __init__(self, obs_shape, action_shape, buffer = ReplayBuffer(100000), gamma = 0.99, lambd = 0.95, clip_param = 0.2, actor_lr = 0.0001, critic_lr = 0.0001, actor = None, critic = None):
        super(PPO, self).__init__()
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.buffer = buffer

        self.training_step = 0
        self.max_grad_norm = 20

        self.gamma = gamma
        self.lambd = lambd
        self.clip_param = clip_param


        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actor = actor.to(device)
        # self.old_actor = old_actor
        self.critic = critic.to(device)

        # optimizer
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), self.actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), self.critic_lr)

        # loss function
        self.critic_loss = nn.MSELoss()

    def take_action(self, obs):
        '''
        产生一个动作，并返回动作和其概率值

        :param obs: agent的observation
        :return: action和action的概率
        '''

        # 将numpy格式转为tensor格式，并且增加batch维度
        obs = torch.from_numpy(obs).float().unsqueeze(0).to(device)

        with torch.no_grad():
            mu, sigma = self.actor(obs)
        dist = Normal(mu.view(-1), sigma.view(-1))
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        # 最好将均值mu弄到动作空间的均值
        action = action.clamp(-1, 1)    # TODO(将此处改为参数),将action限制在[-1,1]，超过的按2算，不够的按-2算
        return action.cpu().numpy(), action_log_prob.sum().cpu().item()    # 返回np格式数据，而不是tensor格式,注意要把log概率之和加起来（log内相乘）

    def get_value(self, obs):
        '''
        获得critic值

        :param obs:
        :return:
        '''
        with torch.no_grad():
            value = self.critic(obs)
        return value


    def save_model(self, path = '../param', para = None):
        '''
        保存模型参数

        :param path:
        :return:
        '''
        torch.save(self.actor.state_dict(), path + para + '_' + str(time.time())[:10] +'_actor.pkl')
        torch.save(self.critic.state_dict(), path + para + '_' + str(time.time())[:10] +'_critic.pkl')

    def update(self, batch_size =256, type = 'clip'):
        '''
        更新actor和critic

        batch_size: batch_size为horizon T,一个T中可能包含多个episode，也可能多个T才时一个episode
        :return:
        '''
        self.training_step += 1

        # batch数据
        obs, action, reward, log_old_prob, next_obs, done = self.buffer.sample(batch_size=batch_size)

        # 计算优势估计
        done = done.cpu().faltten().numpy()
        next_v = self.get_value(next_obs)
        current_v = self.get_value(obs)
        target_v = reward + self.gamma * next_v * (1 - done)

        delta = target_v - current_v    # TD error
        delta = delta.cpu().flatten().numpy()

        '''计算GAE'''
        adv = [0]
        for dlt, done in zip(delta[::-1], done[::-1]):  # batch内反向计算
            advantage = dlt + self.gamma * self.lambd * adv[-1] * (1 - done)
            adv.apend(advantage)
        adv.reverse()   # 再反向回来，就是batch内每个数据对应的advantage
        adv = copy.deepcopy(adv[0:-1])  # 把初始化的adv = [0]去掉
        adv = torch.tensor(adv).unsqueeze(1).float().to(device) # 增加batch维度
        td_target = adv + current_v
        adv = (adv - adv.mean()) / ((adv.std() + 1e-4)) # 归一化后，训练更稳定
        '''结束'''

        # 更新critic
        v_loss = self.critic_loss(current_v, td_target)
        v_loss.requires_grad_()
        self.critic_optim.zero_grad()
        v_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optim.step()
        print("v更新")

        # 更新actor
        if type == 'clip':
            # 计算概率动作之比
            mu, sigma = self.actor(obs)
            n = Normal(mu, sigma)
            action_log_prob = n.log_prob(action).sum(dim=-1, keepdim=True)  # 概率相乘（相当于log相加）
            ratio = torch.exp(action_log_prob - log_old_prob)

            # 计算两个loss，之后选取最小值
            L1 = ratio * adv
            L2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * adv
            action_loss = -torch.min(L1, L2).mean()
            self.actor_optim.zero_grad()
            action_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optim.step()
            print("pi更新")
        else :
            pass    # TODO PPO1，利用KL divergence计算


        return self.training_step



    def evaluate(self, obs):
        '''
        测试模型，训练不调用，只在测试时使用

        :return:一个动作
        '''
        with torch.no_grad():
            obs = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            mu, sigma = self.actor(obs)
            dist = Normal(mu.view(-1), sigma.view(-1))
            action = dist.sample()
            action = action.clamp(-1, 1)  # TODO(将此处改为参数),将action限制在[-1,1]，超过的按1算，不够的按-1算
            return action.cpu().flatten().numpy()






