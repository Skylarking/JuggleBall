import pickle
from utils import display_data

with open('data.pickle', 'rb') as f:
    data_dict = pickle.load(f)


episode_list = data_dict['episode']
score_episode = data_dict['score_episode']
score_list = data_dict['score']
v_loss_list = data_dict['v_loss']
pi_loss_list = data_dict['pi_loss']

display_data(score_episode,score_list)
display_data(episode_list, v_loss_list)
display_data(episode_list, pi_loss_list)