import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'osmesa'
# os.environ['MUJOCO_GL'] = 'egl'
# os.environ['MUJOCO_GL'] = 'glfw'

import sys
sys.path.append("..") 

import time
import hydra
import numpy as np
from icecream import ic
from dm_env import specs
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
torch.backends.cudnn.benchmark = True

import dmc
import utils
import vpg_core as core 
from logger import Logger
from replay_buffer import VPGBuffer 
from video import TrainVideoRecorder, VideoRecorder


def vpg(env, actor_critic, replay_buffer, logger,
        episodes, steps_per_episode,
        lr_actor, lr_critic, train_v_iters, 
        save_freq, train_video_recorder):
    """
    Vanilla Policy Gradient 

    (with GAE-Lambda for advantage estimation)

    Args:
        env (object): an environment object

        replay_buffer (object): a buffer object

        logger (object): a logger object
        
        episodes (int): Number of episodes of interaction (equivalent to
                      number of policy updates) to perform.

        steps_per_episode (int): Number of steps of interaction (state-action pairs) 
                               for the agent and the environment in each episode.


        lr_actor (float): Learning rate for policy optimizer.

        lr_critic (float): Learning rate for value function optimizer.

        train_v_iters (int): Number of gradient descent steps to take on 
                             value function per episode.

        save_freq (int): How often (in terms of gap between episodes) to save
                         the current policy and value function.
        
        train_video_recorder(object): save frames to video
        
        actor_critic (object): The constructor method for a PyTorch Module with a 
                     ``step`` method, an ``act`` method, a ``pi`` module, and a ``v`` module. 
                      
                      The ``step`` method should accept a batch of observations and return:
                      ===========  ================  ======================================
                      Symbol       Shape             Description
                      ===========  ================  ======================================
                      ``a``        (batch, act_dim)  | Numpy array of actions for each 
                                                     | observation.
                      ``v``        (batch,)          | Numpy array of value estimates
                                                     | for the provided observations.
                      ``logp_a``   (batch,)          | Numpy array of log probs for the
                                                     | actions in ``a``.
                      ===========  ================  ======================================

                      The ``act`` method behaves the same as ``step`` but only returns ``a``.

                      The ``pi`` module's forward call should accept a batch of observations 
                      and optionally a batch of actions, and return:
                      ===========  ================  ======================================
                      Symbol       Shape             Description
                      ===========  ================  ======================================
                      ``pi``       N/A               | Torch Distribution object, containing
                                                     | a batch of distributions describing
                                                     | the policy for the provided observations.
                      ``logp_a``   (batch,)          | Optional (only returned if batch of
                                                     | actions is given). Tensor containing 
                                                     | the log probability, according to 
                                                     | the policy, of the provided actions.
                                                     | If actions not given, will contain
                                                     | ``None``.
                      ===========  ================  ======================================

                      The ``v`` module's forward call should accept a batch of observations
                      and return:
                      ===========  ================  ======================================
                      Symbol       Shape             Description
                      ===========  ================  ======================================
                      ``v``        (batch,)          | Tensor containing the value estimates
                                                     | for the provided observations. (Critical: 
                                                     | make sure to flatten this!)
                      ===========  ================  ======================================

    """

    # Set up function for computing VPG policy loss
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        pi, logp = actor_critic.pi(obs, act)
        loss_pi = -(logp * adv).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        info_pi = dict(kl=approx_kl, ent=ent)

        return loss_pi, info_pi

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        pre_v = actor_critic.v(obs)
        return ((pre_v - ret)**2).mean()

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(actor_critic.pi.parameters(), lr=lr_actor)
    vf_optimizer = Adam(actor_critic.v.parameters(), lr=lr_critic)

    def update():
        data = replay_buffer.get()

        # Get loss and info values before update
        loss_pi_old, info_pi_old = compute_loss_pi(data)
        loss_v_old = compute_loss_v(data)

        # Train policy network with a single step of gradient descent
        pi_optimizer.zero_grad()
        loss_pi, info_pi = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Train value network with multi-step like regression
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            vf_optimizer.step()

        # Log changes from update
        kl, ent = info_pi['kl'], info_pi_old['ent']

        DeltaLossPi = loss_pi.item() - loss_pi_old.item()
        DeltaLossV = loss_v.item() - loss_v_old.item()

        return loss_pi_old.item(), loss_v_old.item(), kl, ent, DeltaLossPi, DeltaLossV
    

    start_time = time.time()
    global_frame = 0

    # Prepare for interaction with environment
    time_step = env.reset()
    (step_type, r, discount, o, a) = time_step 
    ep_ret, ep_len = 0, 0

    # init train video
    train_video_recorder.init(o)

    print('\nTraining....')
    # Main loop: collect experience in env and update/log each episode
    for episode in range(episodes):
        print(f'========================================')
        for t in range(steps_per_episode):
            print('\nepisode:%d, step:%d'%(episode, t))
            flat_o = torch.flatten(torch.as_tensor(o, dtype=torch.float32))
            a, v, logp = actor_critic.step(flat_o)
            time_step = env.step(a)
            (step_type, r, discount, next_o, _) = time_step 
            ep_ret += r
            ep_len += 1
            global_frame += 1

            # save and log
            replay_buffer.store(flat_o, a, r, v, logp)
            vvals = v
            logger.store(VVals=vvals)
            
            # video record 
            train_video_recorder.record(o)

            # Update obs 
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = time_step.last() or timeout
            episode_ended = t==steps_per_episode-1

            if terminal or episode_ended:

                if episode_ended and not(terminal):
                    print('\nWarning: trajectory cut off by episode at %d steps.'%ep_len, flush=True)

                if episode_ended:
                    print('\nHave not reach terminal state, bootstrap value target')
                    flat_o = torch.flatten(torch.as_tensor(o, dtype=torch.float32))
                    _, v, _ = actor_critic.step(flat_o)
                if terminal:
                    print('\nReach terminal state, set target value to 0')
                    v = 0
                replay_buffer.finish_path(last_val=v)

                logger.store(episode_reward=ep_ret, episode_length=ep_len)
                with logger.log_and_dump_ctx(global_frame, ty='train') as log:
                    log('episode', episode)
                    log('episode_reward', ep_ret)
                    log('episode_length', ep_len)
                    log('Time', time.time()-start_time)
                
                print('\nSave video....')
                train_video_recorder.save(f'{global_frame}.mp4') 

                print('\nReset Env....')
                (step_type, r, discount, o, a) = env.reset()
                ep_ret, ep_len = 0, 0
                train_video_recorder.init(o)

        # Perform VPG update
        print('\nUpdate model.....')
        LossPi, LossV, KL, Entropy, DeltaLossPi, DeltaLossV = update()
        # Log changes from update
        logger.store(LossPi=LossPi, LossV=LossV,
                     KL=KL, Entropy=Entropy,
                     DeltaLossPi=DeltaLossPi, DeltaLossV = DeltaLossV)

        # Save model
        if (episode % save_freq == 0) or (episode == episodes-1):
            print('\nSaving model....')
            logger.save_model(actor_critic.pi, actor_critic.v, episode)


        # Log info about episode
        logger.log_tabular('episode', episode)
        logger.log_tabular('episode_reward', with_min_and_max=True)
        logger.log_tabular('episode_length', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (episode+1)*steps_per_episode)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()



@hydra.main(config_path='cfgs', config_name='config')
def main(cfg):
    ic(cfg)

    work_dir = Path.cwd()
    print(f'workspace: {work_dir}')

    # create logger
    logger = Logger(work_dir, use_tb=cfg.use_tb)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
    
    print(f'seed: {cfg.seed}')
    utils.set_seed_everywhere(cfg.seed)

    device = torch.device(cfg.gpu_id)
    print(f'device: {device}')

    print(f'\nInit environment: {cfg.task_name}')
    env = dmc.make(name = cfg.task_name, 
                   frame_stack = 1,
                   action_repeat = 1,     
                   seed = cfg.seed)

    print(f'\nMake an agent.....')
    obs_dim = env.observation_spec().shape
    flat_obs_dim = obs_dim[0] * obs_dim[1] * obs_dim[2]
    act_dim = env.action_spec().shape[0]
    # Create actor-critic module
    ac = core.ActorCritic(flat_obs_dim, act_dim, (cfg.hidden_dim_1, cfg.hidden_dim_2, cfg.hidden_dim_3), activation=nn.Tanh)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    print('Number of parameters: \t pi: %d, \t v: %d'%var_counts)
    ic(ac.pi, ac.v)

    print('\nSet up experience buffer...')
    buf = VPGBuffer(flat_obs_dim, act_dim, cfg.steps_per_episode, cfg.gamma, cfg.lam)

    print("\nCreate video recorder.....")
    video_recorder = TrainVideoRecorder(work_dir)

    vpg(env=env, 
        actor_critic = ac,
        logger = logger,
        replay_buffer = buf,
        episodes = cfg.episodes,
        steps_per_episode = cfg.steps_per_episode, 
        lr_actor = cfg.lr_actor,
        lr_critic = cfg.lr_critic,
        train_v_iters = cfg.train_v_iters,
        save_freq = cfg.save_freq,
        train_video_recorder = video_recorder,
        )
    
if __name__ == '__main__':
    main()
