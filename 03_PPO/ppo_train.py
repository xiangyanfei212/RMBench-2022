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
import core
import utils
from logger import Logger
from replay_buffer import PPOBuffer 
from video import TrainVideoRecorder, VideoRecorder

def ppo(algo:str,
        env,
        actor_critic,
        replay_buffer,
        logger,
        steps_per_episode,
        episodes,
        max_ep_len,
        lr_actor, lr_critic,
        clip_ratio,
        train_v_iters,
        train_pi_iters,
        gamma, lam,
        target_kl,
        train_video_recorder,
        save_freq,
        device):
    """
    Proximal Policy Optimization (by clipping)
    with early stopping based on approximate KL

    Args:
        env
        actor_critic
        steps_per_episode (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        episodes (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while 
            still profiting (improving the objective function)? The new policy 
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
            denoted by :math:`\epsilon`. 

        lr_actor (float): Learning rate for policy optimizer.

        lr_critic (float): Learning rate for value function optimizer.

        train_pi_iters (int): Maximum number of gradient descent steps to take 
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used 
            for early stopping. (Usually small, 0.01 or 0.05.)

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """
        
    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        obs = torch.as_tensor(obs, device=device)
        act = torch.as_tensor(act, device=device)
        adv = torch.as_tensor(adv, device=device)
        logp_old = torch.as_tensor(logp_old, device=device)

        # Policy loss
        pi, logp = actor_critic.pi(obs, act)

        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        obs = torch.as_tensor(obs, device=device)
        ret = torch.as_tensor(ret, device=device)
        pre_v = actor_critic.v(obs)
        return ((pre_v - ret)**2).mean()

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(actor_critic.pi.parameters(), lr=lr_actor)
    vf_optimizer = Adam(actor_critic.v.parameters(), lr=lr_critic)

    def update():
        data = replay_buffer.get()

        pi_loss_old, pi_info_old = compute_loss_pi(data)
        pi_loss_old = pi_loss_old.item()
        v_loss_old = compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            pi_loss, pi_info = compute_loss_pi(data)
            kl = pi_info['kl']
            if kl > 1.5 * target_kl:
                print('Early stopping at step %d due to reaching max kl.'%i)
                break
            pi_loss.backward()
            pi_optimizer.step()
        logger.store(StopIter=i)

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            v_loss = compute_loss_v(data)
            v_loss.backward()
            vf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']

        logger.store(LossPi=pi_loss_old, LossV=v_loss_old,
                     KL=kl, Entropy=ent, 
                     ClipFrac=cf,
                     DeltaLossPi=(pi_loss.item() - pi_loss_old),
                     DeltaLossV=(v_loss.item() - v_loss_old))

    # Prepare for interaction with environment
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
        print(f'\n================== Episode: {episode} ===================')
        for t in range(steps_per_episode):
            print('\nepisode:%d, step:%d'%(episode, t))
            flat_o = torch.flatten(torch.as_tensor(o, dtype=torch.float32, device=device))
            a, v, logp = actor_critic.step(flat_o)
            time_step = env.step(a)
            (step_type, r, discount, next_o, _) = time_step 
            ep_ret += r
            ep_len += 1
            global_frame += 1

            # save and log
            replay_buffer.store(flat_o.cpu().numpy(), a, r, v, logp)
            logger.store(VVals=v)

            # video record 
            train_video_recorder.record(o)

            # Update obs (critical!)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = time_step.last() or timeout
            episode_ended = t==steps_per_episode-1

            if terminal or episode_ended:
                if episode_ended and not(terminal):
                    print('Warning: trajectory cut off by episode at %d steps.'%ep_len, flush=True)

                if timeout or episode_ended:
                    print('\nHave not reach terminal state, bootstrap value target')
                    flat_o = torch.flatten(torch.as_tensor(o, dtype=torch.float32))
                    flat_o = torch.as_tensor(flat_o, device=device)
                    _, v, _ = actor_critic.step(flat_o)
                else:
                    print('\nReach terminal state, set target value to 0')
                    v = 0
                replay_buffer.finish_path(last_val=v)

                logger.store(episode_reward=ep_ret, episode_length=ep_len)
                with logger.log_and_dump_ctx(global_frame, ty='train') as log:
                    log('episode', episode)
                    log('episode_reward', ep_ret)
                    log('episode_length', ep_len)
                    log('global_frame', global_frame)
                    log('Time', time.time()-start_time)

                print('\nSaving video and inti a new recorder')
                train_video_recorder.save(f'{global_frame}.mp4') 

                print('\nReset Env....')
                (step_type, r, discount, o, a) = env.reset()
                ep_ret, ep_len = 0, 0
                train_video_recorder.init(o)

        # Perform PPO update!
        print('\nUpdate model.....')
        update()

        # Save model
        if (episode % save_freq == 0) or (episode == episodes - 1):
            print('\nSaving model.....')
            logger.save_model(actor_critic.pi, actor_critic.v, None, episode)

        # Log info about episode
        logger.log_tabular('episode', episode)
        logger.log_tabular('episode_reward', with_min_and_max=True)
        logger.log_tabular('episode_length', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (episode + 1) * steps_per_episode)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()

@hydra.main(config_path='cfgs', config_name='config')
def main(cfg):
    ic(cfg)

    work_dir = Path.cwd()
    print(f'workspace: {work_dir}')

    # create logger
    logger = Logger(work_dir, use_tb=cfg.use_tb)

    # os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
    
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
    ac = core.MLPActorCritic(flat_obs_dim, act_dim, (cfg.hidden_dim_1, cfg.hidden_dim_2, cfg.hidden_dim_3), activation=nn.Tanh)
    ac.pi.to(device)
    ac.v.to(device)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    print('Number of parameters: \t pi: %d, \t v: %d'%var_counts)
    ic(ac.pi, ac.v)

    print('\nSet up experience buffer...')
    buf = PPOBuffer(obs_dim = flat_obs_dim, 
                    act_dim = act_dim, 
                    size = cfg.steps_per_episode, 
                    gamma = cfg.gamma, lam = cfg.lam)

    print("\nCreate video recorder.....")
    video_recorder = TrainVideoRecorder(work_dir)

    ppo(algo = 'trpo',
        env = env,
        actor_critic = ac,
        logger = logger,
        replay_buffer = buf,
        episodes = cfg.episodes,
        steps_per_episode = cfg.steps_per_episode,
        max_ep_len = cfg.max_ep_len,
        lr_actor = cfg.lr_actor,
        lr_critic = cfg.lr_critic,
        clip_ratio = cfg.clip_ratio,
        train_v_iters = cfg.train_v_iters,
        train_pi_iters = cfg.train_pi_iters,
        gamma = cfg.gamma,
        lam = cfg.lam,
        target_kl = cfg.target_kl,
        train_video_recorder = video_recorder,
        save_freq = cfg.save_freq,
        device = device,
        )

if __name__ == '__main__':
    main()
