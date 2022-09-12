from copy import deepcopy
from icecream import ic
import numpy as np
import torch
from torch.optim import Adam
import gym
import time 

env = gym.make('HalfCheetah-v2')
ic(env)

ic(env.action_space)
a = env.action_space.sample()
ic(a)
