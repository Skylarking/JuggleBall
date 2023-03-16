import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta,Normal,Categorical
import math


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CentralizedCritic(nn.Module):
	def __init__(self, state_dim, net_width, num_kernel, kernel_size):
		super(CentralizedCritic, self).__init__()
		self.net = nn.Sequential(
			nn.Linear(state_dim, net_width),
			nn.Tanh(),
			nn.Linear(net_width, net_width),
			nn.Tanh(),
			nn.Linear(net_width, 1)
		)



	def forward(self, state):
		if len(state.shape) == 3:
			state = torch.squeeze(state, dim=-1)

		v = self.net(state)
		return v

'''循环卷积

	输入：(B,N,F),N是多少个feature，F是feature维数，F=1
	

'''
class cicr_conv(nn.Module):
	def __init__(self, feature=1, num_kernel=9, kernel_size=3, stride=1, padding=0):
		super(cicr_conv, self).__init__()
		self.net = nn.Conv1d(in_channels=feature, out_channels=num_kernel, kernel_size=kernel_size, stride=stride, padding=padding)
		self.pool = nn.MaxPool1d(kernel_size=2)
	def forward(self, input):
		i = input.permute(0,2,1)	#(b,f,n) ,n是agent个数
		out = self.net(i)
		out = torch.tanh(out)
		out = self.pool(out)
		out = out.permute(0,2,1)	#(b,n,f)
		return out

''' 
	实体concate，确保他们feature长度一致
'''
class EntityConcat(nn.Module):
	def __init__(self, in1, in2):
		
		super(EntityConcat, self).__init__()


class Actor(nn.Module):
	def __init__(
		self,
		state_dim,
		action_dim,
		net_width,
		kernel_size=3,
		num_kernel=9
	):
		super(Actor, self).__init__()

		self.action_dim = action_dim
		self.action_num = len(action_dim)

		self.net = nn.Sequential(
			nn.Linear(state_dim, net_width),
			nn.Tanh(),
			nn.Linear(net_width, net_width),
			nn.Tanh(),
		)

		# action head 每个agent有n个动作
		self.net2 = [nn.Sequential(nn.Linear(net_width, num), nn.Tanh()).to(device) for num in self.action_dim]


	def forward(self, state):

		a = self.net(state)
		return a

	def pi(self, state, softmax_dim=0):
		if len(state.shape) == 3:
			state = torch.squeeze(state, dim=-1)
		x = self.forward(state)

		prob_list = []
		for n in range(self.action_num):
			head = self.net2[n]
			# print(cal_gpu(head))
			prob = F.softmax(head(x), dim=softmax_dim)
			prob_list.append(prob)

		return prob_list	# 返回n种动作的每个概率值









class PPO(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		all_state_dim,
		gamma=0.99,
		lambd=0.96,
		clip_rate=0.2,
		K_epochs=10,
		net_width=256,
		a_lr=3e-4,
		c_lr=3e-4,
		l2_reg = 1e-3,
		optim_batch_size = 64,
		entropy_coef = 0,
		entropy_coef_decay = 0.9998
	):

		self.actor = Actor(state_dim, action_dim, net_width).to(device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=a_lr)


		self.critic = CentralizedCritic(all_state_dim, net_width, 9,3).to(device)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=c_lr)

		self.action_dim = action_dim
		self.all_state_dim = all_state_dim
		self.action_num = len(action_dim)		# action种类
		self.clip_rate = clip_rate
		self.gamma = gamma
		self.lambd = lambd
		self.clip_rate = clip_rate
		self.K_epochs = K_epochs
		self.l2_reg = l2_reg
		self.optim_batch_size = optim_batch_size
		self.entropy_coef = entropy_coef
		self.entropy_coef_decay = entropy_coef_decay

		self.adv_normalization = True



	def select_action(self, state):#only used when interact with the env
		with torch.no_grad():
			state = torch.FloatTensor(state).view(1,-1).to(device)
			state = torch.unsqueeze(state, dim=-1)


			pi_list = self.actor.pi(state, softmax_dim=1)
			dist = [Categorical(d) for d in pi_list]
			a = [dist[n].sample().item() for n in range(self.action_num)]	# 随即策略，不是选择最大的概率
			pi_a = [pi_list[n][0][a[n]].item() for n in range(self.action_num)]


			return a, pi_a


	def evaluate(self, state):#only used when evaluate the policy.Making the performance more stable
		with torch.no_grad():
			state = torch.FloatTensor(state).view(1, -1).to(device)
			state = torch.unsqueeze(state, dim=-1)

			pi_list = self.actor.pi(state, softmax_dim=1)
			a = [torch.argmax(pi_list[n]).item() for n in range(self.action_num)]

			return a, 0.0



	def train(self, transition):
		self.entropy_coef *= self.entropy_coef_decay
		all_s, s, a, r, all_s_prime, s_prime, old_prob_a, done_mask, dw_mask = transition

		''' Use TD+GAE+LongTrajectory to compute Advantage and TD target'''
		with torch.no_grad():
			vs = self.critic(all_s)
			vs_ = self.critic(all_s_prime)

			'''dw(dead and win) for TD_target and Adv'''
			deltas = r + self.gamma * vs_ * (1 - dw_mask) - vs
			deltas = deltas.cpu().flatten().numpy()
			adv = [0]

			'''done for GAE'''
			for dlt, mask in zip(deltas[::-1], done_mask.cpu().flatten().numpy()[::-1]):
				advantage = dlt + self.gamma * self.lambd * adv[-1] * (1 - mask)
				adv.append(advantage)
			adv.reverse()
			adv = copy.deepcopy(adv[0:-1])
			adv = torch.tensor(adv).unsqueeze(1).float().to(device)
			td_target = adv + vs
			if self.adv_normalization:
				adv = (adv - adv.mean()) / ((adv.std() + 1e-4))  # useful in some envs

		"""PPO update"""
		# Slice long trajectopy into short trajectory and perform mini-batch PPO update
		optim_iter_num = int(math.ceil(s.shape[0] / self.optim_batch_size))

		for _ in range(self.K_epochs):
			# Shuffle the trajectory, Good for training
			perm = np.arange(s.shape[0])
			np.random.shuffle(perm)
			perm = torch.LongTensor(perm).to(device)
			s, a, td_target, adv, old_prob_a = \
				s[perm].clone(), a[perm].clone(), td_target[perm].clone(), adv[perm].clone(), old_prob_a[perm].clone()

			'''mini-batch PPO update'''
			for i in range(optim_iter_num):
				index = slice(i * self.optim_batch_size, min((i + 1) * self.optim_batch_size, s.shape[0]))

				'''actor update'''
				prob_list = self.actor.pi(s[index], softmax_dim=1)		# 每种动作的概率
				prob_a = [prob_list[n].gather(1, a[index][:, n].view(-1, 1)) for n in range(self.action_num)] 	# 所取动作a下的概率




				entropy_list = [Categorical(prob_list[n]).entropy().sum(0, keepdim=True) for n in range(self.action_num)]	# 熵
				entropy = sum(entropy_list)
				log_prob = torch.zeros_like(prob_a[0]).to(device)
				for p in prob_a:
					log_prob += torch.log(p)

				old_log_prob = torch.log(old_prob_a[index]).sum(dim=-1)

				ratio = torch.exp(log_prob - old_log_prob.view(-1,1))  # a/b == exp(log(a)-log(b))

				surr1 = ratio * adv[index]
				surr2 = torch.clamp(ratio, 1 - self.clip_rate, 1 + self.clip_rate) * adv[index]
				a_loss = -torch.min(surr1, surr2) - self.entropy_coef * entropy		# TODO 损失函数加上熵做惩罚，不然学出来的策略很平均很随机，导致什么也没学到

				self.actor_optimizer.zero_grad()
				a_loss.mean().backward()
				torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
				self.actor_optimizer.step()

				'''critic update'''

				c_loss = (self.critic(all_s[index]) - td_target[index].detach()).pow(2).mean()
				for name, param in self.critic.named_parameters():
					if 'weight' in name:
						c_loss += param.pow(2).sum() * self.l2_reg

				self.critic_optimizer.zero_grad()
				c_loss.backward()
				torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
				self.critic_optimizer.step()
		return a_loss.mean().detach().cpu().item(), c_loss.detach().cpu().item()


	def train_without_minibatch(self, transition):
		self.entropy_coef *= self.entropy_coef_decay
		all_s, s, a, r, all_s_prime, s_prime, old_prob_a, done_mask, dw_mask = transition

		''' Use TD+GAE+LongTrajectory to compute Advantage and TD target'''
		with torch.no_grad():
			vs = self.critic(all_s)
			vs_ = self.critic(all_s_prime)

			'''dw(dead and win) for TD_target and Adv'''
			deltas = r + self.gamma * vs_ * (1 - dw_mask) - vs
			deltas = deltas.cpu().flatten().numpy()
			adv = [0]

			'''done for GAE'''
			for dlt, mask in zip(deltas[::-1], done_mask.cpu().flatten().numpy()[::-1]):
				advantage = dlt + self.gamma * self.lambd * adv[-1] * (1 - mask)
				adv.append(advantage)
			adv.reverse()
			adv = copy.deepcopy(adv[0:-1])
			adv = torch.tensor(adv).unsqueeze(1).float().to(device)
			td_target = adv + vs
			if self.adv_normalization:
				adv = (adv - adv.mean()) / ((adv.std() + 1e-4))  # useful in some envs

		"""PPO update"""

		for _ in range(self.K_epochs):

			'''actor update'''
			prob_list = self.actor.pi(s, softmax_dim=1)		# 每种动作的概率
			prob_a = [prob_list[n].gather(1, a[:, n].view(-1, 1)) for n in range(self.action_num)] 	# 所取动作a下的概率




			entropy_list = [Categorical(prob_list[n]).entropy().sum(0, keepdim=True) for n in range(self.action_num)]	# 熵
			entropy = sum(entropy_list)
			log_prob = torch.zeros_like(prob_a[0]).to(device)
			for p in prob_a:
				log_prob += torch.log(p)

			old_log_prob = torch.log(old_prob_a).sum(dim=-1)

			ratio = torch.exp(log_prob - old_log_prob.view(-1,1))  # a/b == exp(log(a)-log(b))

			surr1 = ratio * adv
			surr2 = torch.clamp(ratio, 1 - self.clip_rate, 1 + self.clip_rate) * adv
			a_loss = -torch.min(surr1, surr2) - self.entropy_coef * entropy		# TODO 损失函数加上熵做惩罚，不然学出来的策略很平均很随机，导致什么也没学到

			self.actor_optimizer.zero_grad()
			a_loss.mean().backward()
			torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
			self.actor_optimizer.step()

			'''critic update'''

			c_loss = (self.critic(all_s) - td_target.detach()).pow(2).mean()
			for name, param in self.critic.named_parameters():
				if 'weight' in name:
					c_loss += param.pow(2).sum() * self.l2_reg

			self.critic_optimizer.zero_grad()
			c_loss.backward()
			torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
			self.critic_optimizer.step()

		return a_loss.mean().detach().cpu().item(), c_loss.detach().cpu().item()






class MAPPO():
	def __init__(
		self,
		n_agents,
		state_dim,
		action_dim,
		gamma=0.99,
		lambd=0.95,
		clip_rate=0.2,
		env_with_Dead = False,
		K_epochs=10,
		net_width=256,
		a_lr=3e-4,
		c_lr=3e-4,
		l2_reg=1e-3,
		optim_batch_size=64,
		entropy_coef=0,
		entropy_coef_decay=0.9998
	):
		super(MAPPO, self).__init__()

		self.n_agents = n_agents
		self.data = []
		self.all_state_dim = sum(state_dim)


		self.agent = [PPO(
			state_dim=state_dim[n],
			action_dim=action_dim[n],
			all_state_dim=self.all_state_dim,
			gamma=gamma,
			lambd=lambd,
			clip_rate=clip_rate,
			K_epochs=K_epochs,
			net_width=net_width,
			a_lr=a_lr,
			c_lr=c_lr,
			l2_reg=l2_reg,
			optim_batch_size=optim_batch_size,
			entropy_coef=entropy_coef,
			entropy_coef_decay=entropy_coef_decay
		) for n in range(self.n_agents)]


	def train_all(self):
		all_s, all_a, all_r, all_s_prime, all_prob_a, done_mask, dw_mask = self.make_batch()

		with torch.no_grad():
			all_a, all_r, all_prob_a, done_mask, dw_mask = \
				torch.tensor(all_a, dtype=torch.int64).to(device), \
				torch.tensor(all_r, dtype=torch.float).to(device), \
				torch.tensor(all_prob_a, dtype=torch.float).to(device), \
				torch.tensor(done_mask, dtype=torch.float).to(device), \
				torch.tensor(dw_mask, dtype=torch.float).to(device)

			s_n, s_prime_n = [], []
			for i in range(self.n_agents):
				s = torch.squeeze(torch.tensor(all_s[i], dtype=torch.float).to(device), dim=1)
				s = (s - s.mean()) / (s.std() + 1e-4)
				s_n.append(s)

				s_prime = torch.squeeze(torch.tensor(all_s_prime[i], dtype=torch.float).to(device), dim=1)
				s_prime = (s_prime - s_prime.mean()) / (s_prime.std() + 1e-4)
				s_prime_n.append(s_prime)



			all_s = torch.concat(s_n, dim=-1)
			all_s_prime = torch.concat(s_prime_n, dim=-1)
			all_s = (all_s - all_s.mean()) /  ((all_s.std() + 1e-4))
			all_s_prime = (all_s_prime - all_s_prime.mean()) / ((all_s_prime.std() + 1e-4))

		pi_loss_n, v_loss_n = [], []
		for n in range(self.n_agents):
			s, a, r, s_prime, prob_a = s_n[n], all_a[::, n], all_r[::, n], s_prime_n[n], all_prob_a[::, n]
			transition = (all_s, s, a, r.view(-1,1), all_s_prime, s_prime, prob_a, done_mask, dw_mask)
			pi_loss, v_loss = self.agent[n].train(transition)
			pi_loss_n.append(pi_loss)
			v_loss_n.append(v_loss)

		return pi_loss_n, v_loss_n


	def make_batch(self):
		s_lst, a_lst, r_lst, s_prime_lst, logprob_a_lst, done_lst, dw_lst = [[] for n in range(self.n_agents)], [], [], [[] for n in range(self.n_agents)], [], [], []
		for transition in self.data:
			s, a, r, s_prime, logprob_a, done, dw = transition


			for n in range(self.n_agents):
				s_lst[n].append(s[n])
				s_prime_lst[n].append(s_prime[n])
			a_lst.append(a)
			logprob_a_lst.append(logprob_a)
			r_lst.append(r)
			done_lst.append([done])
			dw_lst.append([dw])

		self.data = []  # Clean history trajectory

		return s_lst, a_lst, r_lst, s_prime_lst, logprob_a_lst, done_lst, dw_lst

	def put_data(self, transition):
		self.data.append(transition)

	def save(self, episode, file_path="./model_ma/"):
		for n in range(self.n_agents):
			torch.save(self.agent[n].critic.state_dict(), file_path + "ppo_critic{}_{}.pth".format(n, episode))
			torch.save(self.agent[n].actor.state_dict(), file_path + "ppo_actor{}_{}.pth".format(n, episode))

	def load(self, episode, file_path="./model_ma/"):
		for n in range(self.n_agents):
			self.agent[n].critic.load_state_dict(torch.load(file_path + "ppo_critic{}_{}.pth".format(n, episode)))
			self.agent[n].actor.load_state_dict(torch.load(file_path + "ppo_actor{}_{}.pth".format(n, episode)))

