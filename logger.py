# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import csv
import datetime
from icecream import ic
from collections import defaultdict

import numpy as np
import torch
import torchvision
from termcolor import colored
from torch.utils.tensorboard import SummaryWriter
from utils import * 

COMMON_TRAIN_FORMAT = [('frame', 'F', 'int'), ('step', 'S', 'int'),
                       ('episode', 'E', 'int'), ('episode_length', 'L', 'int'),
                       ('episode_reward', 'R', 'float'),
                       ('buffer_size', 'BS', 'int'), ('fps', 'FPS', 'float'),
                       ('total_time', 'T', 'time')]

COMMON_EVAL_FORMAT = [('frame', 'F', 'int'), ('step', 'S', 'int'),
                      ('episode', 'E', 'int'), ('episode_length', 'L', 'int'),
                      ('episode_reward', 'R', 'float'),
                      ('total_time', 'T', 'time')]


class AverageMeter(object):
    def __init__(self):
        self._sum = 0
        self._count = 0

    def update(self, value, n=1):
        self._sum += value
        self._count += n

    def value(self):
        return self._sum / max(1, self._count)


class MetersGroup(object):
    def __init__(self, csv_file_name, formating):
        self._csv_file_name = csv_file_name
        self._formating = formating

        self._meters = defaultdict(AverageMeter)
        self._csv_file = None
        self._csv_writer = None

    def log(self, key, value, n=1):
        self._meters[key].update(value, n)

    def _prime_meters(self):
        data = dict()
        for key, meter in self._meters.items():
            if key.startswith('train'):
                key = key[len('train') + 1:]
            else:
                key = key[len('eval') + 1:]
            key = key.replace('/', '_')
            data[key] = meter.value()
        return data

    def _remove_old_entries(self, data):
        rows = []
        with open(self._csv_file_name, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if float(row['episode']) >= data['episode']:
                    break
                rows.append(row)
        with open(self._csv_file_name, 'w') as f:
            writer = csv.DictWriter(f,
                                    fieldnames=sorted(data.keys()),
                                    restval=0.0)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _dump_to_csv(self, data):
        if self._csv_writer is None:
            should_write_header = True
            if os.path.exists(self._csv_file_name):
                self._remove_old_entries(data)
                should_write_header = False

            self._csv_file = open(self._csv_file_name, 'a')
            self._csv_writer = csv.DictWriter(self._csv_file,
                                              fieldnames=sorted(data.keys()),
                                              restval=0.0)
            if should_write_header:
                self._csv_writer.writeheader()

        self._csv_writer.writerow(data)
        self._csv_file.flush()

    def _format(self, key, value, ty):
        if ty == 'int':
            value = int(value)
            return f'{key}: {value}'
        elif ty == 'float':
            return f'{key}: {value:.04f}'
        elif ty == 'time':
            value = str(datetime.timedelta(seconds=int(value)))
            return f'{key}: {value}'
        else:
            raise f'invalid format type: {ty}'

    def _dump_to_console(self, data, prefix):
        prefix = colored(prefix, 'yellow' if prefix == 'train' else 'green')
        pieces = [f'| {prefix: <14}']
        for key, disp_key, ty in self._formating:
            value = data.get(key, 0)
            pieces.append(self._format(disp_key, value, ty))
        print(' | '.join(pieces))

    def dump(self, step, prefix):
        if len(self._meters) == 0:
            return
        data = self._prime_meters()
        data['frame'] = step
        self._dump_to_csv(data)
        self._dump_to_console(data, prefix)
        self._meters.clear()


class Logger(object):
    def __init__(self, log_dir, use_tb):
        self._log_dir = log_dir
        self._train_mg = MetersGroup(os.path.join(log_dir, 'train.csv'), formating=COMMON_TRAIN_FORMAT)
        self._eval_mg = MetersGroup(os.path.join(log_dir, 'eval.csv'), formating=COMMON_EVAL_FORMAT)
        self.output_file = open(os.path.join(self._log_dir, 'progress.txt'), 'a')

        if use_tb:
            self._sw = SummaryWriter(str(log_dir / 'tb'))
        else:
            self._sw = None

        print(f'Init a log at {self._log_dir}')
        print(f'output_file: {self.output_file}')

        self.epoch_dict = dict()
        self.first_row=True
        self.log_headers = []
        self.log_current_row = {}

    def _try_sw_log(self, key, value, step):
        if self._sw is not None:
            self._sw.add_scalar(key, value, step)

    def store(self, **kwargs):
        """
        Save something into the epoch_logger's current state.
        Provide an arbitrary number of keyword arguments with numerical 
        values.
        """
        for k,v in kwargs.items():
            if not(k in self.epoch_dict.keys()):
                self.epoch_dict[k] = []
            self.epoch_dict[k].append(v)


    def log(self, key, value, step):
        assert key.startswith('train') or key.startswith('eval')
        if type(value) == torch.Tensor:
            value = value.item()
        self._try_sw_log(key, value, step)
        mg = self._train_mg if key.startswith('train') else self._eval_mg
        mg.log(key, value)
        print(f'{key}:{value}')

    def log_metrics(self, metrics, step, ty):
        for key, value in metrics.items():
            self.log(f'{ty}/{key}', value, step)

    def dump(self, step, ty=None):
        if ty is None or ty == 'eval':
            self._eval_mg.dump(step, 'eval')
        if ty is None or ty == 'train':
            self._train_mg.dump(step, 'train')

    def log_and_dump_ctx(self, step, ty):
        return LogAndDumpCtx(self, step, ty)


    def log_format(self, key, val):
        """
        Log a value of some diagnostic.
        Call this only once for each diagnostic quantity, each iteration.
        After using ``log_tabular`` to store values for each diagnostic,
        make sure to call ``dump_tabular`` to write them out to file and
        stdout (otherwise they will not get saved anywhere).
        """
        if self.first_row:
            self.log_headers.append(key)
        else:
            assert key in self.log_headers, "Trying to introduce a new key %s that you didn't include in the first iteration"%key
        assert key not in self.log_current_row, "You already set %s this iteration. Maybe you forgot to call dump_tabular()"%key
        self.log_current_row[key] = val


    def log_tabular(self, key, val=None, with_min_and_max=False, average_only=False):
        """
        Log a value or possibly the mean/std/min/max values of a diagnostic.
        Args:
            key (string): The name of the diagnostic. If you are logging a
                diagnostic whose state has previously been saved with ``store``, 
                the key here has to match the key you used there.
            val: A value for the diagnostic. If you have previously saved
                values for this key via ``store``, do *not* provide a ``val``
                here.
            with_min_and_max (bool): If true, log min and max values of the 
                diagnostic over the epoch.
            average_only (bool): If true, do not log the standard deviation
                of the diagnostic over the epoch.
        """


        if val is not None:
            self.log_format(key, val)
        else:
            v = self.epoch_dict[key]
            vals = np.concatenate(v) if isinstance(v[0], np.ndarray) and len(v[0].shape)>0 else v
            stats = statistics_scalar(vals, with_min_and_max=with_min_and_max)

            self.log_format(key if average_only else 'Average' + key, stats[0])
            if not(average_only):
                self.log_format('Std'+key, stats[1])
            if with_min_and_max:
                self.log_format('Max'+key, stats[3])
                self.log_format('Min'+key, stats[2])
        self.epoch_dict[key] = []

    def dump_tabular(self):
        """
        Write all of the diagnostics from the current iteration.
        Writes both to stdout, and to the output file.
        """
        vals = []
        key_lens = [len(key) for key in self.log_headers]
        max_key_len = max(15, max(key_lens))
        keystr = '%'+'%d'%max_key_len
        fmt = "| " + keystr + "s | %15s |"
        n_slashes = 22 + max_key_len
        print("-"*n_slashes)

        for key in self.log_headers:
            val = self.log_current_row.get(key, "")
            valstr = "%8.3g"%val if hasattr(val, "__float__") else val
            print(fmt%(key, valstr))
            vals.append(val)
        print("-"*n_slashes, flush=True)
        
        if self.output_file is not None:
            if self.first_row:
                self.output_file.write("\t".join(self.log_headers)+"\n")
            self.output_file.write("\t".join(map(str,vals))+"\n")
            self.output_file.flush()
        self.log_current_row.clear()
        self.first_row=False

    def save_model(self, actor_model, critic_model_1, critic_model_2=None, episode=0):

        fpath = os.path.join(self._log_dir, 'model_save')
        os.makedirs(fpath, exist_ok=True)

        act_fname = f'actor_model_Ep{int(episode)}.pt' 
        cri_fname_1 = f'critic_model_1_Ep{int(episode)}.pt' 

        act_save_path = os.path.join(fpath, act_fname)
        cri_save_path_1 = os.path.join(fpath, cri_fname_1)

        print(f'Saving actor model to {act_save_path}')
        torch.save(actor_model, act_save_path)

        print(f'Saving critic model to {cri_save_path_1}')
        torch.save(critic_model_1, cri_save_path_1)

        if critic_model_2 != None:
            cri_fname_2 = f'critic_model_2_Ep{int(episode)}.pt' 
            cri_save_path_2 = os.path.join(fpath, cri_fname_2)
            print(f'Saving critic model to {cri_save_path_2}')
            torch.save(critic_model_2, cri_save_path_2)


class LogAndDumpCtx:
    def __init__(self, logger, step, ty):
        self._logger = logger
        self._step = step
        self._ty = ty

    def __enter__(self):
        return self

    def __call__(self, key, value):
        self._logger.log(f'{self._ty}/{key}', value, self._step)

    def __exit__(self, *args):
        self._logger.dump(self._step, self._ty)
