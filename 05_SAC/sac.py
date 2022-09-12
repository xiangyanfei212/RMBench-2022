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
import itertools
import numpy as np
from icecream import ic
from dm_env import specs
from pathlib import Path
from copy import deepcopy

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
from video import TrainVideoRecorder, VideoRecorder

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}

def sac(env, actor_critic, actor_critic_targ, logger, steps_per_episode, episodes, max_ep_len, replay_buffer, act_dim, act_limit, gamma, polyak, lr, alpha, batch_size, start_steps, update_after, update_every, num_test_episodes, train_video_recorder, test_video_recorder, save_freq, device, pretrain_episodes):
    """
    Args:
        env: an environment object
        actor_critic: a MLP network
        actor_critic_targ: 
        steps_per_episode: Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each episodes.) 
        episodes (int): Number of episodes to run and train agent.
        gamma (float): Discount factor. (Always between 0 and 1.)
        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:
                \theta_targ <- \rho \theta_targ + (1-\rho) \theta
            where :math:`\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)
        lr (float): Learning rate (used for both policy and value learning).
        alpha (float): Entropy regularization coefficient. (Equivalent to 
            inverse of reward scale in the original SAC paper.)
        batch_size (int): Minibatch size for SGD.
        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.
        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.
        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to 1.
        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.
        max_ep_len (int): Maximum length of trajectory / episode / rollout.
    """

    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(actor_critic.q1.parameters(), actor_critic.q2.parameters())

    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        o = torch.as_tensor(o, device=device)
        a = torch.as_tensor(a, device=device)
        r = torch.as_tensor(r, device=device)
        o2 = torch.as_tensor(o2, device=device)
        d = torch.as_tensor(d, device=device)

        q1 = actor_critic.q1(o,a)
        q2 = actor_critic.q2(o,a)

        # Bellman bachup for Q functions 
        with torch.no_grad():
            # Target actions come from current policy
            a2, logp_a2 = actor_critic.pi(o2)
            
            # Target Q-values
            q1_pi_targ = actor_critic_targ.q1(o2, a2) 
            q2_pi_targ = actor_critic_targ.q2(o2, a2) 
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)

            backup = r + gamma * (1-d) * (q_pi_targ - alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().cpu().numpy(),
                      Q2Vals=q2.detach().cpu().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(data):
        o = data['obs']
        o = torch.as_tensor(o, device=device)

        pi, logp_pi = actor_critic.pi(o)
        q_pi = torch.min(actor_critic.q1(o, pi), actor_critic.q2(o, pi))

        # Entropy-regularized policy loss
        loss_pi = (alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().cpu().numpy())

        return loss_pi, pi_info

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(actor_critic.pi.parameters(), lr=lr)
    q_optimizer = Adam(q_params, lr=lr)

    def update(data):

        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, q_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Record things
        logger.store(LossQ=loss_q.item(), **q_info)

        # Freeze Q-network so you don't waste computational effort
        # computing gradients for them during the policy learning step 
        for p in q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next step.
        for p in q_params:
            p.requires_grad = True

        # Record things
        logger.store(LossPi=loss_pi.item(), **pi_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(actor_critic.parameters(), actor_critic_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, deterministic=False):
        return actor_critic.act(o, deterministic)

    # prepare for interaction with environment 
    start_steps = steps_per_episode * pretrain_episodes
    total_steps = steps_per_episode * episodes
    start_time = time.time()
    global_frame = start_steps 

    time_step = env.reset()
    (_, _, _, o, _) = time_step 
    ep_ret, ep_len = 0, 0

    # init train video
    train_video_recorder.init(o)

    # Main loop: collect experience in env and update/log each episode
    print('\nTraining....')
    for t in range(start_steps, total_steps):

        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy
        flat_o = torch.flatten(torch.as_tensor(o, dtype=torch.float32, device=device))
        if t > start_steps:
            a = get_action(flat_o).cpu().numpy()
        else:
            a = np.random.uniform(env.action_spec().minimum, env.action_spec().maximum, env.action_spec().shape)
        
        # Step the env
        time_step = env.step(a)
        (_, r, _, o2, _) = time_step
        d = time_step.last()
        ep_ret += r
        ep_len += 1
        global_frame += 1

        # video record 
        train_video_recorder.record(o)

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d
        
        # Store experience to replay buffer
        flat_o2 = torch.flatten(torch.as_tensor(o2, dtype=torch.float32))
        replay_buffer.store(flat_o.cpu().numpy(), a, r, flat_o2.cpu().numpy(), d)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len) or ((t+1) % steps_per_episode == 0):
            print(f'End of trajectory handling, step:{t}/{total_steps}')
            logger.store(episode_reward=ep_ret, episode_length=ep_len)
            episode = (t+1) // steps_per_episode

            with logger.log_and_dump_ctx(global_frame, ty='train') as log:
                log('episode', episode)
                log('episode_reward', ep_ret)
                log('episode_length', ep_len)
                log('global_frame', global_frame)
                log('Time', time.time()-start_time)

            try:
                print('\nSaving video and init a new recorder')
                train_video_recorder.save(f'{global_frame}.mp4') 
            except:
                print('Something wrong when save video. continue')

            time_step = env.reset()
            (_, _, _, o, _) = time_step 
            ep_ret, ep_len = 0, 0
            train_video_recorder.init(o)

        # Update handling
        if t >= update_after and t % update_every == 0:
            print(f'\nUpdate model....., step:{t}/{total_steps}')
            for _ in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch)

        # End of episode handling
        if (t+1) % steps_per_episode == 0:
            print(f'End of episode {episode}, step:{t}/{total_steps}')
            episode = (t+1) // steps_per_episode

            # Save model
            if (episode % save_freq == 0) or (episode == episodes):
                print('\nSaving model.....')
                logger.save_model(actor_critic.pi, actor_critic.q1, actor_critic.q2, episode)

            # print('\nTest the performance of the deterministic version of the agent.')
            # test_agent()

            # Log info about episode
            logger.log_tabular('episode', episode)
            logger.log_tabular('global_frame', global_frame)
            logger.log_tabular('episode_reward', with_min_and_max=True)
            logger.log_tabular('episode_length', average_only=True)
            # logger.log_tabular('test_episode_reward', with_min_and_max=True)
            # logger.log_tabular('test_episode_length', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            # logger.log_tabular('Q1Vals', with_min_and_max=True)
            # logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()
    

@hydra.main(config_path='cfgs', config_name='config')
def main(cfg):
    ic(cfg)

    if cfg.load_pretrain_model:
        work_dir = cfg.work_dir
        seed = cfg.pretrain_seed 
    else:
        work_dir = Path.cwd()
        seed = cfg.seed
    print(f'\nworkspace: {work_dir}')

    # create logger
    logger = Logger(work_dir, use_tb=cfg.use_tb)

    print(f'seed: {seed}')
    utils.set_seed_everywhere(seed)

    device = torch.device(cfg.gpu_id)
    print(f'device: {device}')

    print(f'\nInit an environment: {cfg.task_name}')
    env = dmc.make(name = cfg.task_name, 
                   frame_stack = 1,
                   action_repeat = 1,     
                   seed = seed)
    
    obs_dim = env.observation_spec().shape
    flat_obs_dim = obs_dim[0] * obs_dim[1] * obs_dim[2]
    act_dim = env.action_spec().shape[0]
    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_spec().maximum
    act_limit = torch.as_tensor(act_limit, device=device)

    # Create actor-critic module and target networks
    ac = core.MLPActorCritic(flat_obs_dim, act_dim, act_limit, (cfg.hidden_dim_1, cfg.hidden_dim_2, cfg.hidden_dim_3))
    if cfg.load_pretrain_model:
        pretrain_episodes = cfg.pretrain_episodes

        model_pi_path = os.path.join(work_dir, 'model_save', f'actor_model_Ep{int(pretrain_episodes)}.pt')
        print(f'\nLoad actor model from:\n{model_pi_path}')
        ac.pi = torch.load(model_pi_path)

        model_q1_path = os.path.join(work_dir, 'model_save', f'critic_model_1_Ep{int(pretrain_episodes)}.pt')
        print(f'Load critic model 1 from:\n{model_q1_path}')
        ac.q1 = torch.load(model_q1_path)

        model_q2_path = os.path.join(work_dir, 'model_save', f'critic_model_2_Ep{int(pretrain_episodes)}.pt')
        print(f'Load critic model 2 from:\n{model_q2_path}')
        ac.q2 = torch.load(model_q2_path)
    else:
        pretrain_episodes = 0

    ac_targ = deepcopy(ac)
    ac.pi.to(device)
    ac.q1.to(device)
    ac.q2.to(device)
    ac_targ.pi.to(device)
    ac_targ.q1.to(device)
    ac_targ.q2.to(device)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    print('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)
    
    # Experience buffer
    ic(flat_obs_dim, act_dim, cfg.replay_size)
    replay_buffer = ReplayBuffer(obs_dim=flat_obs_dim, act_dim=act_dim, size=int(cfg.replay_size))

    print("\nCreate video recorder.....")
    train_video_recorder = TrainVideoRecorder(work_dir)
    test_video_recorder = VideoRecorder(work_dir)
    
    sac(env=env, actor_critic=ac, actor_critic_targ=ac_targ, logger=logger, 
        steps_per_episode=cfg.steps_per_episode, 
        episodes=cfg.episodes, 
        max_ep_len=cfg.max_ep_len, 
        replay_buffer=replay_buffer, 
        act_dim = act_dim, act_limit=act_limit, 
        gamma=cfg.gamma, polyak=cfg.polyak, lr=cfg.lr, alpha=cfg.alpha, 
        batch_size=cfg.batch_size, start_steps=cfg.start_steps, 
        update_after=cfg.update_after, update_every=cfg.update_every, 
        num_test_episodes=cfg.num_test_episodes, 
        train_video_recorder=train_video_recorder, 
        test_video_recorder=test_video_recorder, 
        pretrain_episodes = pretrain_episodes,
        save_freq=cfg.save_freq, 
        device=device)


if __name__ == '__main__':
    main()
