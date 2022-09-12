import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'osmesa'
# os.environ['MUJOCO_GL'] = 'egl'
# os.environ['MUJOCO_GL'] = 'glfw'
# os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"

import sys
sys.path.append("..") 

import time
import hydra
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
    A simple FIFO experience replay buffer for DDPG agents.
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



def ddpg(env, env_test, actor_critic, actor_critic_target, logger, steps_per_episode, episodes, start_steps, max_ep_len, replay_buffer, act_dim, act_limit,
         gamma, polyak, lr_actor, lr_critic, batch_size, update_after, update_every,
         act_noise, num_test_episodes, train_video_recorder, test_video_recorder, save_freq, device, pretrain_episodes):
    """
    Deep Deterministic Policy Gradient (DDPG)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act`` 
            method, a ``pi`` module, and a ``q`` module. The ``act`` method and
            ``pi`` module should accept batches of observations as inputs,
            and ``q`` should accept a batch of observations and a batch of 
            actions as inputs. When called, these should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``pi``       (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``q``        (batch,)          | Tensor containing the current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to DDPG.

        seed (int): Seed for random number generators.

        steps_per_episode (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each episode.

        episodes (int): Number of episodes to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        pi_lr (float): Learning rate for policy.

        q_lr (float): Learning rate for Q-networks.

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

        act_noise (float): Stddev for Gaussian exploration noise added to 
            policy at training time. (At test time, no noise is added.)

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each episode.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between episodes) to save
            the current policy and value function.

    """
    # Set up function for computing DDPG Q-loss
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        o = torch.as_tensor(o, device=device)
        a = torch.as_tensor(a, device=device)
        r = torch.as_tensor(r, device=device)
        o2 = torch.as_tensor(o2, device=device)
        d = torch.as_tensor(d, device=device)

        q = actor_critic.q(o,a)

        # Bellman backup for Q function
        with torch.no_grad():
            q_pi_targ = actor_critic_target.q(o2, actor_critic_target.pi(o2))
            backup = r + gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q = ((q - backup)**2).mean()

        # Useful info for logging
        loss_info = dict(QVals=q.detach().cpu().numpy())

        return loss_q, loss_info

    # Set up function for computing DDPG pi loss
    def compute_loss_pi(data):
        o = data['obs']
        o = torch.as_tensor(o, device=device)
        q_pi = actor_critic_target.q(o, actor_critic.pi(o))
        return -q_pi.mean()

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(actor_critic_target.pi.parameters(), lr=lr_actor)
    q_optimizer = Adam(actor_critic_target.q.parameters(), lr=lr_critic)

    def update(data):
        # First run one gradient descent step for Q.
        q_optimizer.zero_grad()
        loss_q, loss_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Freeze Q-network so you don't waste computational effort 
        # computing gradients for it during the policy learning step.
        for p in actor_critic.q.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in actor_critic.q.parameters():
            p.requires_grad = True

        # Record things
        logger.store(LossQ=loss_q.item(), LossPi=loss_pi.item(), **loss_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(actor_critic.parameters(), actor_critic_target.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, noise_scale):
        flat_o = torch.flatten(torch.as_tensor(o, dtype=torch.float32, device=device))
        a = actor_critic.act(flat_o)
        add = noise_scale * np.random.randn(act_dim)
        a += torch.tensor(add, device=device)
        return torch.clip(a, -act_limit, act_limit)

    def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = env_test.reset(), False, 0, 0
            (_, _, _, o, _) = env_test.reset()
            d = False
            ep_ret, ep_len = 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                time_step = env_test.step(get_action(o, 0))
                (_, r, _, o, _) = time_step
                d = time_step.last()
                ep_ret += r
                ep_len += 1
            logger.store(test_episode_reward=ep_ret, test_episode_length=ep_len)

    # Prepare for interaction with environment
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
        
        # print(f'\n================== step:{t}/{total_steps} ===================')
        
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy (with some noise, via act_noise). 
        flat_o = torch.flatten(torch.as_tensor(o, dtype=torch.float32, device=device))
        if t > start_steps:
            a = get_action(flat_o, act_noise).cpu().numpy()
        else:
            a = np.random.uniform(env.action_spec().minimum, env.action_spec().maximum, env.action_spec().shape)
            # a = env.action_spec().sample()

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
            print('End of trajectory handling')
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

        if t >= update_after and t % update_every == 0:
            print(f'\nUpdate model....., step:{t}/{total_steps}')
            for _ in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch)

        # End of episode handling
        if (t+1) % steps_per_episode == 0:
            print('End of episode handling')
            episode = (t+1) // steps_per_episode

            # Save model
            if (episode % save_freq == 0) or (episode == episodes):
                print('\nSaving model.....')
                logger.save_model(actor_critic.pi, actor_critic.q, None, episode)

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
            # logger.log_tabular('QVals', with_min_and_max=True)
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

    # os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
    
    print(f'seed: {seed}')
    utils.set_seed_everywhere(seed)

    device = torch.device(cfg.gpu_id)
    print(f'device: {device}')

    print(f'\nInit an environment: {cfg.task_name}')
    env = dmc.make(name = cfg.task_name, 
                   frame_stack = 1,
                   action_repeat = 1,     
                   seed = seed)

    print(f'\nInit an environment for testing: {cfg.task_name}')
    env_test = dmc.make(name = cfg.task_name, 
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

        model_q_path = os.path.join(work_dir, 'model_save', f'critic_model_1_Ep{int(pretrain_episodes)}.pt')
        print(f'Load critic model from:\n{model_q_path}')
        ac.q = torch.load(model_q_path)
    else:
        pretrain_episodes = 0
    ac_targ = deepcopy(ac)
    ac.pi.to(device)
    ac.q.to(device)
    ac_targ.pi.to(device)
    ac_targ.q.to(device)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q])
    print('\nNumber of parameters: \t pi: %d, \t q: %d\n'%var_counts)

    # Experience buffer
    buf = ReplayBuffer(obs_dim=flat_obs_dim, act_dim=act_dim, size=cfg.replay_size)

    print("\nCreate video recorder.....")
    train_video_recorder = TrainVideoRecorder(work_dir)
    test_video_recorder = VideoRecorder(work_dir)

    ddpg(env=env, env_test=env_test, 
         actor_critic=ac, actor_critic_target=ac_targ, 
         act_dim = act_dim, act_limit = act_limit,
         logger=logger, 
         steps_per_episode=cfg.steps_per_episode, 
         episodes=cfg.episodes, 
         max_ep_len = cfg.max_ep_len,
         start_steps=cfg.start_steps,
         replay_buffer = buf, 
         gamma=cfg.gamma, polyak=cfg.polyak, 
         lr_actor=cfg.lr_actor, lr_critic=cfg.lr_critic, 
         batch_size=cfg.batch_size, 
         update_after=cfg.update_after, 
         update_every=cfg.update_every,
         act_noise=cfg.act_noise, 
         num_test_episodes=cfg.num_test_episodes, 
         train_video_recorder = train_video_recorder, 
         test_video_recorder = test_video_recorder, 
         pretrain_episodes = pretrain_episodes,
         save_freq = cfg.save_freq, 
         device = device)


if __name__ == '__main__':
    main()
