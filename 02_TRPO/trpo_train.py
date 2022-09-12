"""
Trust Region Policy Optimization
TRPO is almost the same as PPO. The only difference is the update rule that
1) computes the search direction via conjugate
2) compute step by backtracking
"""
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
import trpo_core as core 
from logger import Logger
from replay_buffer import GAEBuffer 
from video import TrainVideoRecorder, VideoRecorder

EPS = 1e-8

def trpo(algo:str,
         env, 
         actor_critic, 
         replay_buffer,
         logger,
         episodes,
         steps_per_episode,
         max_ep_len,
         lr_actor, lr_critic,
         train_v_iters,
         train_video_recorder,
         gamma, lam,
         delta, 
         damping_coeff, 
         cg_iters, 
         backtrack_iters, backtrack_coeff,
         save_freq,
         device):

    """
    Trust Region Policy Optimization
    (with support for Natural Policy Gradient)
    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.
        actor_critic: The constructor method for a PyTorch Module with a
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v``
            module. The ``step`` method should accept a batch of observations
            and return:

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

            The ``pi`` module's forward call should accept a batch of
            observations and optionally a batch of actions, and return:

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
        ac_kwargs (dict): Any kwargs appropriate for the actor_critic
            function you provided to TRPO.
        seed (int): Seed for random number generators.
        steps_per_episode (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each episode.
        episodes (int): Number of episodes of interaction (equivalent to
            number of policy updates) to perform.
        gamma (float): Discount factor. (Always between 0 and 1.)
        delta (float): KL-divergence limit for TRPO / NPG update.
            (Should be small for stability. Values like 0.01, 0.05.)
        vf_lr (float): Learning rate for value function optimizer.
        train_v_iters (int): Number of gradient descent steps to take on
            value function per episode.
        damping_coeff (float): Artifact for numerical stability, should be
            smallish. Adjusts Hessian-vector product calculation:

            .. math:: Hv \\rightarrow (\\alpha I + H)v
            where :math:`\\alpha` is the damping coefficient.
            Probably don't play with this hyperparameter.
        cg_iters (int): Number of iterations of conjugate gradient to perform.
            Increasing this will lead to a more accurate approximation
            to :math:`H^{-1} g`, and possibly slightly-improved performance,
            but at the cost of slowing things down.
            Also probably don't play with this hyperparameter.
        backtrack_iters (int): Maximum number of steps allowed in the
            backtracking line search. Since the line search usually doesn't
            backtrack, and usually only steps back once when it does, this
            hyperparameter doesn't often matter.
        backtrack_coeff (float): How far back to step during backtracking line
            search. (Always between 0 and 1, usually above 0.5.)
        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)
        max_ep_len (int): Maximum length of trajectory / episode / rollout.
        logger_kwargs (dict): Keyword args for EpisodeLogger.
        save_freq (int): How often (in terms of gap between episodes) to save
            the current policy and value function.
        algo: Either 'trpo' or 'npg': this code supports both, since they are
            almost the same.
    """

    # Set up function for computing TRPO policy loss
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        obs = torch.as_tensor(obs, device=device)
        act = torch.as_tensor(act, device=device)
        adv = torch.as_tensor(adv, device=device)
        logp_old = torch.as_tensor(logp_old, device=device)

        # Policy loss
        _, logp = actor_critic.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        loss_pi = -(ratio * adv).mean()
        return loss_pi

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        obs = torch.as_tensor(obs, device=device)
        ret = torch.as_tensor(ret, device=device)
        return ((actor_critic.v(obs) - ret) ** 2).mean()

    def compute_kl(data, old_pi):
        obs, act = data['obs'], data['act']
        obs = torch.as_tensor(obs, device=device)
        act = torch.as_tensor(act, device=device)
        pi, _ = actor_critic.pi(obs, act)
        kl_loss = torch.distributions.kl_divergence(pi, old_pi).mean()
        return kl_loss

    @torch.no_grad()
    def compute_kl_loss_pi(data, old_pi):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        obs = torch.as_tensor(obs, device=device)
        act = torch.as_tensor(act, device=device)
        adv = torch.as_tensor(adv, device=device)
        logp_old = torch.as_tensor(logp_old, device=device)

        # Policy loss
        pi, logp = actor_critic.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        loss_pi = -(ratio * adv).mean()
        kl_loss = torch.distributions.kl_divergence(pi, old_pi).mean()
        return loss_pi, kl_loss

    def hessian_vector_product(data, old_pi, v):
        kl = compute_kl(data, old_pi)

        grads = torch.autograd.grad(kl, actor_critic.pi.parameters(), create_graph=True)
        flat_grad_kl = core.flat_grads(grads)

        kl_v = (flat_grad_kl * v).sum()
        grads = torch.autograd.grad(kl_v, actor_critic.pi.parameters())
        flat_grad_grad_kl = core.flat_grads(grads)

        return flat_grad_grad_kl + v * damping_coeff


    # Set up optimizers for policy and value function
    pi_optimizer = Adam(actor_critic.pi.parameters(), lr=lr_actor)
    vf_optimizer = Adam(actor_critic.v.parameters(), lr=lr_critic)

    def update():
        data = replay_buffer.get()

        # compute old pi distribution
        obs, act = data['obs'], data['act']
        obs = torch.as_tensor(obs, device=device)
        act = torch.as_tensor(act, device=device)
        with torch.no_grad():
            old_pi, _ = actor_critic.pi(obs, act)

        pi_loss = compute_loss_pi(data)
        pi_l_old = pi_loss.item()
        v_l_old = compute_loss_v(data).item()

        grads = core.flat_grads(torch.autograd.grad(pi_loss, actor_critic.pi.parameters()))

        # Core calculations for TRPO or NPG
        Hx = lambda v: hessian_vector_product(data, old_pi, v)
        x = core.conjugate_gradients(Hx, grads, cg_iters)

        alpha = torch.sqrt(2 * delta / (torch.matmul(x, Hx(x)) + EPS))

        old_params = core.get_flat_params_from(actor_critic.pi)

        def set_and_eval(step):
            new_params = old_params - alpha * x * step
            core.set_flat_params_to(actor_critic.pi, new_params)
            loss_pi, kl_loss = compute_kl_loss_pi(data, old_pi)
            return kl_loss.item(), loss_pi.item()

        if algo == 'npg':
            # npg has no backtracking or hard kl constraint enforcement
            kl, pi_loss_new = set_and_eval(step=1.)

        elif algo == 'trpo':
            # trpo augments npg with backtracking line search, hard kl
            for j in range(backtrack_iters):
                kl, pi_l_new = set_and_eval(step=backtrack_coeff ** j)
                if kl <= delta and pi_l_new <= pi_l_old:
                    # logger.log('Accepting new params at step %d of line search.' % j)
                    print('Accepting new params at step %d of line search.' % j)
                    logger.store(BacktrackIters=j)
                    break
                if j == backtrack_iters - 1:
                    print('Line search failed! Keeping old params.')
                    kl, pi_l_new = set_and_eval(step=0.)
                    logger.store(BacktrackIters=j)

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            v_loss = compute_loss_v(data)
            v_loss.backward()
            vf_optimizer.step()

        # Log changes from update
        logger.store(LossPi=pi_l_old, LossV=v_l_old, KL=kl,
                     DeltaLossPi=(pi_l_new - pi_l_old),
                     DeltaLossV=(v_loss.item() - v_l_old))

    # Prepare for interaction with environment
    start_time = time.time()
    global_frame = 0

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
            flat_o = torch.flatten(torch.as_tensor(o, dtype=torch.float32))
            flat_o = torch.as_tensor(flat_o, device=device)
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
                if episode_ended and not (terminal):
                    print('Warning: trajectory cut off by episode at %d steps.' % ep_len, flush=True)

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

        # Perform TRPO update!
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
        if algo == 'trpo':
            logger.log_tabular('BacktrackIters', average_only=True)
        logger.log_tabular('Time', time.time() - start_time)
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
    buf = GAEBuffer(flat_obs_dim, act_dim, cfg.steps_per_episode, cfg.gamma, cfg.lam)

    print("\nCreate video recorder.....")
    video_recorder = TrainVideoRecorder(work_dir)

    trpo(
         algo = 'trpo',
         env = env,
         actor_critic = ac,
         logger = logger,
         replay_buffer = buf,
         episodes = cfg.episodes,
         steps_per_episode = cfg.steps_per_episode,
         max_ep_len = cfg.max_ep_len, 
         lr_actor = cfg.lr_actor, 
         lr_critic = cfg.lr_critic, 
         train_v_iters = cfg.train_v_iters, 
         train_video_recorder = video_recorder, 
         gamma = cfg.gamma, 
         lam = cfg.lam,
         delta = cfg.delta, 
         damping_coeff = cfg.damping_coeff, 
         cg_iters = cfg.cg_iters, 
         backtrack_iters = cfg.backtrack_iters, 
         backtrack_coeff = cfg.backtrack_coeff,
         save_freq = cfg.save_freq,
         device = device,
    )


if __name__ == '__main__':
    main()
