import numpy as np
import random
import collections
import torch

'''
    创建ReplayBuffer和生成batch数据
'''

#两个函数：push存数据，sample采样一个batch
class ReplayBuffer():
    def __init__(self, buffer_maxlen):
        self.buffer = collections.deque(maxlen=buffer_maxlen)#双端队列

    def push(self, data):#data是一个tuple，格式为(s,a,r,log_old_prob,next_s,is_done)
        self.buffer.append(data)

    def sample(self, batch_size):#从buffer中随机取出一个batch
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        state_list = []
        action_list = []
        reward_list = []
        log_old_prob_list = []
        next_state_list = []
        done_list = []

        batch = random.sample(self.buffer, batch_size)
        for experience in batch:    # experience和data格式一样
            s, a, r, log_old_prob, n_s, d = experience
            # state, action, reward, log_old_prob, next_state, done

            state_list.append(s)
            action_list.append(a)
            reward_list.append(r)
            log_old_prob_list.append(log_old_prob)
            next_state_list.append(n_s)
            done_list.append(d)

        #返回一个batch
        return torch.tensor(state_list, dtype=torch.float).to(device), \
               torch.tensor(action_list, dtype=torch.float).to(device), \
               torch.tensor(reward_list, dtype=torch.float).view(-1,1).to(device), \
               torch.tensor(log_old_prob_list, dtype=torch.float).view(-1,1).to(device), \
               torch.tensor(next_state_list, dtype=torch.float).to(device), \
               torch.tensor(done_list, dtype=torch.float).view(-1,1).to(device)



    def buffer_len(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()

