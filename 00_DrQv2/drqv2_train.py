# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'osmesa'
# os.environ['MUJOCO_GL'] = 'egl'
# os.environ['MUJOCO_GL'] = 'glfw'

import sys
sys.path.append("..")

import torch
import hydra
import numpy as np
from dm_env import specs
from pathlib import Path

import dmc
import utils
from logger import Logger
from video import TrainVideoRecorder, VideoRecorder
from replay_buffer import ReplayBufferStorage, make_replay_loader

from icecream import ic
torch.backends.cudnn.benchmark = True

def make_agent(obs_spec, action_spec, cfg):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)

class Workspace:
    def __init__(self, cfg):

        self.work_dir = Path.cwd()
        print(f'\nworkspace: {self.work_dir}')

        # Step 2.1 : Some config
        self.cfg = cfg
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        print(f'\ndevice: {self.device}')

        # Step 2.2 : setup logger, train & eval environment, replay buffer & loader, video recorder 
        print('\nSetup......')
        self.setup()

        # Step 2.3 : Make an agent 
        print('\nMake an agent......')
        print(f'train_env.observation: {self.train_env.observation_spec()}')
        print(f'train_env.action_spec: {self.train_env.action_spec()}')
        print(f'config of agent: {self.cfg.agent}')
        self.agent = make_agent(self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                self.cfg.agent)

        # Step 2.4 : recorder
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)

        # create envs
        print(f'\nInit environment: {self.cfg.task_name}')
        self.train_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                  self.cfg.action_repeat, self.cfg.seed)
        print(f'self.cfg.task_name: {self.cfg.task_name}')
        self.eval_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                 self.cfg.action_repeat, self.cfg.seed)

        # create video recorder
        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None)

        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if self.cfg.save_train_video else None)

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            self.global_step,
                                            eval_mode=True)
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            self.video_recorder.save(f'{self.global_frame}.mp4')

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)

    def train(self):

        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)

        # create replay buffer
        print('\nCreate replay buffer.....')
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        self.replay_storage = ReplayBufferStorage(data_specs,
                                                  self.work_dir / 'buffer')

        self.replay_loader = make_replay_loader(
            self.work_dir / 'buffer', self.cfg.replay_buffer_size,
            self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot, self.cfg.nstep, self.cfg.discount)

        self._replay_iter = None

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset() # wrapper to an experience tuple (state/obs, action, reward, discount) 
        self.replay_storage.add(time_step) # add experience to replay storage
        self.train_video_recorder.init(time_step.observation) # init train video
        metrics = None

        print('\nBegin Train....')
        while train_until_step(self.global_step): # if current step < until step
            # print(f'time_step: {time_step}')

            if time_step.last(): # if an episode ends
                print('\nAn episode ends....')
                self._global_episode += 1
                self.train_video_recorder.save(f'{self.global_frame}.mp4')

                elapsed_time, total_time = self.timer.reset()
                episode_frame = episode_step * self.cfg.action_repeat
                with self.logger.log_and_dump_ctx(self.global_frame, ty='train') as log:
                    log('fps', episode_frame / elapsed_time)
                    log('total_time', total_time)
                    log('episode', self.global_episode)
                    log('episode_reward', episode_reward)
                    log('episode_frame', episode_frame)
                    log('episode_step', episode_step)
                    log('buffer_size', len(self.replay_storage))
                    log('step', self.global_step)
                    log('frame', self.global_frame)

                # %% Reset for next episode 
                print('\nReset for next episode......')
                # reset env
                time_step = self.train_env.reset()
                # add an experience to replay storage
                self.replay_storage.add(time_step)
                # init train video 
                self.train_video_recorder.init(time_step.observation)
                # try to save snapshot
                if self.cfg.save_snapshot:
                    self.save_snapshot()
                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                print(f'\nEvaluate ......')
                self.logger.log('eval_total_time', 
                                self.timer.total_time(),
                                self.global_frame)
                self.eval()

            # sample action
            print(f'\nSample an action......')
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation,
                                        self.global_step,
                                        eval_mode=False)

            # try to update the agent
            if not seed_until_step(self.global_step):
                print(f'\nUpdate the agent(network)......')
                metrics = self.agent.update(self.replay_iter, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            print(f'\nTake env step.......')
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            self.replay_storage.add(time_step)
            self.train_video_recorder.record(time_step.observation)
            episode_step += 1
            self._global_step += 1

    def save_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        print(f'Saving snapshot to {snapshot}')

        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}

        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        print(f'Loading snapshot from {snapshot}')

        with snapshot.open('rb') as f:
            payload = torch.load(f)

        for k, v in payload.items():
            self.__dict__[k] = v


@hydra.main(config_path='cfgs', config_name='config')
def main(cfg):
    print(',,,,,,,,,,,,,,,')
    ic(cfg)
    
    workspace = Workspace(cfg)

    root_dir = Path.cwd()
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()

    workspace.train()


if __name__ == '__main__':
    main()
