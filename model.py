import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import time
import copy
from torch.distributions import Normal, MultivariateNormal, Beta
from ReplayBuffer import ReplayBuffer
from utils import action_adapter
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BetaActor(nn.Module):
    def __init__(self, obs_shape, action_shape, hidden_num=128):
        super(BetaActor, self).__init__()

        self.l1 = nn.Linear(obs_shape, hidden_num)
        self.l2 = nn.Linear(hidden_num, hidden_num)
        self.alpha_head = nn.Linear(hidden_num, action_shape)
        self.beta_head = nn.Linear(hidden_num, action_shape)

    def forward(self, input):
        a = torch.tanh(self.l1(input))
        a = torch.tanh(self.l2(a))

        alpha = F.softplus(self.alpha_head(a)) + 1.0
        beta = F.softplus(self.beta_head(a)) + 1.0

        return alpha, beta

    def get_dist(self, state):
        alpha, beta = self.forward(state)
        dist = Beta(alpha, beta)
        return dist

    def dist_mode(self, state):
        alpha, beta = self.forward(state)
        mode = (alpha) / (alpha + beta)
        return mode


class actor(nn.Module):
    def __init__(self, obs_shape, action_shape, hidden_num=128):
        super(actor, self).__init__()
        # parameters
        self.obs_shape = obs_shape
        self.action_shape = action_shape

        # network
        self.net = nn.Sequential(
            nn.Linear(obs_shape, hidden_num),
            nn.Tanh(),
            nn.Linear(hidden_num, hidden_num),
            nn.Tanh()
        )

        self.mu_head = nn.Linear(hidden_num, action_shape)
        self.sigma_head = nn.Linear(hidden_num, action_shape) + 0.01

    def forward(self, input):
        x = self.net(input)
        mu = torch.sigmoid(self.mu_head(x))
        sigma = F.softplus(self.sigma_head(x)) + 0.01

        return mu, sigma

    def get_dist(self, input):
        mu, sigma = self.forward(input)
        dist = Normal(mu, sigma)
        return dist


class critic(nn.Module):
    def __init__(self, obs_shape, hidden_num=128):
        super(critic, self).__init__()
        # parameters
        self.obs_shape = obs_shape

        # network
        self.net = nn.Sequential(
            nn.Linear(obs_shape, hidden_num),
            nn.Tanh(),
            nn.Linear(hidden_num, hidden_num),
            nn.Tanh(),
            nn.Linear(hidden_num, 1)
        )

    def forward(self, input):
        value = self.net(input)
        return value


class PPO(object):
    '''
    agent类
    '''

    def __init__(
            self,
             obs_shape,
             action_shape,
             buffer=ReplayBuffer(10000),
             max_grad_norm=40,
             gamma=0.99,
             lambd=0.95,
             K_epochs=10,
             a_optim_batch_size=64,
             c_optim_batch_size=64,
             clip_param=0.2,
             l2_reg=1e-3,
             actor_lr=0.0001,
             critic_lr=0.0001,
             entropy_coef = 0.001,
             entropy_coef_decay=0.9998,
             actor=None,
             critic=None
             ):
        super(PPO, self).__init__()
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.buffer = buffer

        self.training_step = 0
        self.max_grad_norm = max_grad_norm

        self.gamma = gamma
        self.lambd = lambd
        self.K_epochs = K_epochs
        self.a_optim_batch_size = a_optim_batch_size
        self.c_optim_batch_size = c_optim_batch_size

        self.clip_param = clip_param
        self.l2_reg = l2_reg

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actor = actor.to(device)
        self.critic = critic.to(device)

        self.entropy_coef = entropy_coef
        self.entropy_coef_decay = entropy_coef_decay

        self.data = []

        # optimizer
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), self.actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), self.critic_lr)

    def take_action(self, obs):
        '''
        产生一个动作，并返回动作和其概率值

        :param obs: agent的observation
        :return: action和action的概率
        '''

        # 将numpy格式转为tensor格式，并且增加batch维度
        with torch.no_grad():
            obs = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            dist = self.actor.get_dist(obs)
            action = dist.sample()
            action_log_prob = dist.log_prob(action)
            # 最好将均值mu弄到动作空间的均值
            action = action.clamp(-2, 2)  # TODO(将此处改为参数),将action限制在[-1,1]，超过的按2算，不够的按-2算
            return action.cpu().flatten().numpy(), action_log_prob.sum().cpu().item()  # 返回np格式数据，而不是tensor格式,注意要把log概率之和加起来（log内相乘）

    def get_value(self, obs):
        '''
        获得critic值

        :param obs:
        :return:
        '''
        with torch.no_grad():
            value = self.critic(obs)
            return value

    def save_model(self, path='../param', para=None):
        '''
        保存模型参数

        :param path:
        :return:
        '''
        torch.save(self.actor.state_dict(), path + para + '_' + str(time.time())[:10] + '_actor.pkl')
        torch.save(self.critic.state_dict(), path + para + '_' + str(time.time())[:10] + '_critic.pkl')

    # def update(self, batch_size =256, type = 'clip'):
    #
    #     obs, action, reward, log_old_prob, next_obs, done = self.buffer.sample(batch_size=batch_size)
    #     for i in range(self.K_epoch):
    #         next_v = self.get_value(next_obs)
    #         current_v = self.get_value(obs)
    #         target_v = reward + self.gamma * next_v * (1 - done.float())
    #         delta = target_v - current_v  # TD error
    #         delta = delta.cpu().flatten().numpy()
    #         done = done.cpu().flatten().numpy()
    #
    #         '''计算GAE'''
    #         adv = [0]
    #         for dlt, done in zip(delta[::-1], done[::-1]):  # batch内反向计算
    #             advantage = dlt + self.gamma * self.lambd * adv[-1] * (1 - done)
    #             adv.append(advantage)
    #         adv.reverse()  # 再反向回来，就是batch内每个数据对应的advantage
    #         adv = copy.deepcopy(adv[0:-1])  # 把初始化的adv = [0]去掉
    #         adv = torch.tensor(adv).unsqueeze(1).float().to(device)  # 增加batch维度
    #         td_target = adv + current_v
    #         adv = (adv - adv.mean()) / ((adv.std() + 1e-4))  # 归一化后，训练更稳定
    #         '''结束'''
    #
    #         if type == 'clip':
    #             # 计算概率动作之比
    #             mu, sigma = self.actor(obs)
    #             n = Normal(mu, sigma)
    #             action_log_prob = n.log_prob(action).sum(dim=-1, keepdim=True)  # 概率相乘（相当于log相加）
    #             ratio = torch.exp(action_log_prob - log_old_prob)  # TODO 问题：这里两个概率值一直一样，比率为1
    #
    #             # 计算两个loss，之后选取最小值
    #             L1 = ratio * adv
    #             L2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * adv
    #             action_loss = -torch.min(L1, L2).mean()
    #             self.actor_optim.zero_grad()
    #             action_loss.backward()
    #             nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
    #             self.actor_optim.step()
    #             print('pi_loss : {}'.format(action_loss))
    #
    #         else:
    #             pass  # TODO PPO1，利用KL divergence计算
    #
    #         # 更新critic
    #         v_loss = self.critic_loss(self.critic(obs), td_target) + self.l2_reg  # TODO 这里要重新使用self.critic(obs)，这里需要梯度
    #         self.critic_optim.zero_grad()
    #         v_loss.backward()
    #         nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
    #         self.critic_optim.step()
    #         print('v_loss : {}'.format(v_loss))
    #
    #     return

    # def update(self, type='clip'):
    #     '''
    #     更新actor和critic
    #
    #     batch_size: batch_size为horizon T,一个T中可能包含多个episode，也可能多个T才时一个episode
    #     :return:
    #     '''
    #
    #     self.training_step += 1
    #
    #     # batch数据
    #     obs, action, reward, next_obs, log_old_prob, done = self.make_batch()
    #
    #     # TODO 计算优势估计,这里不需要梯度
    #     with torch.no_grad():
    #         current_v = self.critic(obs)
    #         next_v = self.critic(next_obs)
    #         target_v = reward + self.gamma * next_v * (1 - done.float())
    #
    #         delta = target_v - current_v  # TD error
    #         delta = delta.cpu().flatten().numpy()
    #         done = done.cpu().flatten().numpy()
    #
    #         '''计算GAE'''
    #         adv = [0]
    #         for dlt, d in zip(delta[::-1], done[::-1]):  # batch内反向计算
    #             advantage = dlt + self.gamma * self.lambd * adv[-1] * (1 - d)
    #             adv.append(advantage)
    #         adv.reverse()  # 再反向回来，就是batch内每个数据对应的advantage
    #         adv = copy.deepcopy(adv[0:-1])  # 把初始化的adv = [0]去掉
    #         adv = torch.tensor(adv).unsqueeze(1).float().to(device)  # 增加batch维度
    #         td_target = adv + current_v
    #         adv = (adv - adv.mean()) / ((adv.std() + 1e-4))  # 归一化后，训练更稳定
    #         '''结束'''
    #
    #     a_optim_iter_num = int(math.ceil(obs.shape[0] / self.a_optim_batch_size))
    #     c_optim_iter_num = int(math.ceil(obs.shape[0] / self.c_optim_batch_size))
    #     # 更新actor
    #     for j in range(self.K_epoch):
    #
    #         # 用batch更新，先Shuffle数据，取得一个batch数据
    #         perm = np.arange(obs.shape[0])  # 整个数据的长度
    #         np.random.shuffle(perm)
    #         perm = torch.LongTensor(perm).to(device)
    #         obs, action, td_target, adv, log_old_prob = \
    #             obs[perm].clone(), action[perm].clone(), td_target[perm].clone(), adv[perm].clone(), log_old_prob[
    #                 perm].clone()
    #
    #         for i in range(a_optim_iter_num):
    #             index = slice(i * self.a_optim_batch_size,
    #                           min((i + 1) * self.a_optim_batch_size, obs.shape[0]))  # minibatch的切片
    #             old = log_old_prob[index]
    #
    #             # if type == 'clip':
    #             # 计算概率动作之比
    #             dist = self.actor.get_dist(obs[index])
    #             action_log_prob = dist.log_prob(action[index]).sum(dim=-1, keepdim=True)  # 概率相乘（相当于log相加）
    #             dist_entropy = dist.entropy().sum(1, keepdim=True)
    #             ratio = torch.exp(action_log_prob - old)
    #
    #             # 计算两个loss，之后选取最小值
    #             L1 = ratio * adv[index]
    #             L2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * adv[index]
    #             action_loss = -torch.min(L1, L2) - self.entropy_coef * dist_entropy  # TODO action_loss修改
    #             self.actor_optim.zero_grad()
    #             action_loss.mean().backward()
    #             ####################################
    #             # for name, parms in self.actor.named_parameters():
    #             #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
    #             #           ' -->grad_value:', parms.grad.mean(),'-->parms', parms)
    #
    #             ####################################
    #             nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
    #             self.actor_optim.step()
    #
    #         # 更新critic
    #         for i in range(c_optim_iter_num):
    #             index = slice(i * self.c_optim_batch_size, min((i + 1) * self.c_optim_batch_size, obs.shape[0]))
    #
    #             v_loss = F.mse_loss(self.critic(obs[index]), td_target[index])
    #             for name, param in self.critic.named_parameters():
    #                 if 'weight' in name:
    #                     v_loss += param.pow(2).sum() * self.l2_reg
    #             self.critic_optim.zero_grad()
    #             v_loss.backward()
    #
    #             #####################################
    #             # for name, parms in self.critic.named_parameters():
    #             #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
    #             #           ' -->grad_value:', parms.grad)
    #             #####################################
    #
    #             nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
    #             self.critic_optim.step()
    #
    #     return self.training_step, v_loss, action_loss

    def update(self):
        self.entropy_coef *= self.entropy_coef_decay
        s, a, r, s_prime, logprob_a, done_mask = self.make_batch()

        ''' Use TD+GAE+LongTrajectory to compute Advantage and TD target'''
        with torch.no_grad():
            vs = self.critic(s)
            vs_ = self.critic(s_prime)

            '''dw for TD_target and Adv'''
            deltas = r + self.gamma * vs_ * (1 - done_mask) - vs

            deltas = deltas.cpu().flatten().numpy()
            adv = [0]

            '''done for GAE'''
            for dlt, mask in zip(deltas[::-1], done_mask.cpu().flatten().numpy()[::-1]):  # 这里reverse一下
                advantage = dlt + self.gamma * self.lambd * adv[-1] * (1 - mask)
                adv.append(advantage)
            adv.reverse()
            adv = copy.deepcopy(adv[0:-1])
            adv = torch.tensor(adv).unsqueeze(1).float().to(device)
            td_target = adv + vs
            adv = (adv - adv.mean()) / ((adv.std() + 1e-4))  # sometimes helps

        """Slice long trajectopy into short trajectory and perform mini-batch PPO update"""
        a_optim_iter_num = int(math.ceil(s.shape[0] / self.a_optim_batch_size))
        c_optim_iter_num = int(math.ceil(s.shape[0] / self.c_optim_batch_size))
        for i in range(self.K_epochs):

            # Shuffle the trajectory, Good for training
            perm = np.arange(s.shape[0])
            np.random.shuffle(perm)
            perm = torch.LongTensor(perm).to(device)
            s, a, td_target, adv, logprob_a = \
                s[perm].clone(), a[perm].clone(), td_target[perm].clone(), adv[perm].clone(), logprob_a[perm].clone()

            '''update the actor，用minibatch更新，i是minibatch得index'''
            for i in range(a_optim_iter_num):
                index = slice(i * self.a_optim_batch_size,
                              min((i + 1) * self.a_optim_batch_size, s.shape[0]))  # minibatch的切片
                distribution = self.actor.get_dist(s[index])
                dist_entropy = distribution.entropy().sum(1, keepdim=True)
                logprob_a_now = distribution.log_prob(a[index])
                old = logprob_a[index]
                ratio = torch.exp(
                    logprob_a_now.sum(1, keepdim=True) - old)  # a/b == exp(log(a)-log(b))

                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * adv[index]
                a_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy

                self.actor_optim.zero_grad()
                a_loss.mean().backward()

                ############################
                # for name, parms in self.actor.named_parameters():
                # 	print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
                # 		  ' -->grad_value:', parms.grad.mean(), '-->parms', parms)
                #####################
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
                self.actor_optim.step()

            '''update the critic'''
            for i in range(c_optim_iter_num):
                index = slice(i * self.c_optim_batch_size, min((i + 1) * self.c_optim_batch_size, s.shape[0]))
                c_loss = (self.critic(s[index]) - td_target[index]).pow(2).mean()
                for name, param in self.critic.named_parameters():
                    if 'weight' in name:
                        c_loss += param.pow(2).sum() * self.l2_reg

                self.critic_optim.zero_grad()
                c_loss.backward()
                self.critic_optim.step()

    def evaluate(self, obs):
        '''
        测试模型，训练不调用，只在测试时使用

        :return:一个动作
        '''
        with torch.no_grad():
            obs = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            dist = self.actor.get_dist(obs)
            action = dist.sample()
            action = action.clamp(-2, 2)  # TODO(将此处改为参数),将action限制在[-1,1]，超过的按1算，不够的按-1算
            return action.cpu().flatten().numpy()

    def make_batch(self):
        '''
        将数据弄成batch

        :return:
        '''
        s_lst, a_lst, r_lst, s_prime_lst, logprob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, logprob_a, done = transition

            s_lst.append(s)
            a_lst.append(a)
            logprob_a_lst.append([logprob_a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_lst.append([done])

        self.data = []  # Clean history trajectory

        '''list to tensor'''
        with torch.no_grad():
            s, a, r, s_prime, logprob_a, done = \
                torch.tensor(s_lst, dtype=torch.float).to(device), \
                torch.tensor(a_lst, dtype=torch.float).to(device), \
                torch.tensor(r_lst, dtype=torch.float).to(device), \
                torch.tensor(s_prime_lst, dtype=torch.float).to(device), \
                torch.tensor(logprob_a_lst, dtype=torch.float).to(device), \
                torch.tensor(done_lst, dtype=torch.float).to(device)

        return s, a, r, s_prime, logprob_a, done

    def put_data(self, transition):
        self.data.append(transition)
