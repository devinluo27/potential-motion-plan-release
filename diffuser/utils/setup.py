import os
import importlib
import random
import numpy as np
import torch
from tap import Tap
import pdb

import os.path as osp
import datetime
import sys, argparse
from termcolor import colored
from .serialization import mkdir
from .git_utils import (
    get_git_rev,
    save_git_diff,
)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def watch(args_to_watch):
    """construct exp_name out of the value of 'args_to_watch' in args"""
    def _fn(args):
        exp_name = []
        for key, label in args_to_watch:
            if not hasattr(args, key):
                continue
            val = getattr(args, key)
            if type(val) == dict:
                val = '_'.join(f'{k}-{v}' for k, v in val.items())
            exp_name.append(f'{label}{val}')
            print(key, label, val)
        exp_name = '_'.join(exp_name)
        exp_name = exp_name.replace('/_', '/')
        exp_name = exp_name.replace('(', '').replace(')', '')
        exp_name = exp_name.replace(', ', '-')
        print('exp_name', exp_name)
        return exp_name
    return _fn

def lazy_fstring(template, args):
    ## https://stackoverflow.com/a/53671539
    return eval(f"f'{template}'")

class Parser(Tap):

    def save(self):
        fullpath = os.path.join(self.savepath, 'args.json')
        print(f'[ utils/setup ] Saved args to {fullpath}')
        super().save(fullpath, skip_unpicklable=True)

    def parse_args(self, experiment=None, from_jupyter=False, use_config_2=False, 
                   not_parse=False, input_args=None):
        if not_parse:
            args = super().parse_args(known_only=True) # True
            args = input_args
        else:
            args = super().parse_args(known_only=True) # True
        if use_config_2:
            ## tmp = args.config
            args.config = args.config_2
        ## if not loading from a config script, skip the result of the setup
        if not hasattr(args, 'config'): return args
        args = self.read_config(args, experiment)
        ## [NOTE] should always add_extras
        if not from_jupyter:
            self.add_extras(args)
        self.eval_fstrings(args)
        self.set_seed(args)
        self.get_commit(args)
        self.generate_exp_name(args)
        self.mkdir(args)
        self.save_diff(args)
        self.set_wandb(args)
        self.check_sinPosEmb(args)
        
        return args

    def read_config(self, args, experiment):
        '''
            Step 1: Read Config
            Load parameters from config file
        '''
        # dataset = args.dataset.replace('-', '_')
        # print(f'[ utils/setup ] Reading config: {args.config}:{dataset}')
        ## enable loading a .py relative path directly 
        if args.config.endswith('.py'):
            args.config = args.config[:-3]
            sys.path.append(os.path.dirname(args.config))
            module = importlib.import_module(os.path.basename(args.config))
            print('module', module)
        else:
            module = importlib.import_module(args.config)

        ## [NOTE] have bugs before but are fixed now maybe
        if 'dataset' in getattr(module, 'base').keys():
            dataset = getattr(module, 'base')['dataset'].replace('-', '_')
            args.dataset = dataset.replace('_', '-')
        else:
            dataset = args.dataset.replace('-', '_')
        if experiment == 'plan':
            args.dataset_eval = args.dataset[:-3] + '-eval' + args.dataset[-3:]

        print(f'[ utils/setup ] Reading config: {args.config}:{dataset}')

        params = getattr(module, 'base')[experiment]

        if hasattr(module, dataset) and experiment in getattr(module, dataset):
            print(f'[ utils/setup ] Using overrides | config: {args.config} | dataset: {dataset}')
            overrides = getattr(module, dataset)[experiment]
            params.update(overrides)
        else:
            print(f'[ utils/setup ] Not using overrides | config: {args.config} | dataset: {dataset}')

        self._dict = {}
        for key, val in params.items():
            setattr(args, key, val)
            self._dict[key] = val

        return args

    def add_extras(self, args):
        '''
            Override config parameters with command-line arguments
        '''
        extras = args.extra_args
        if not len(extras):
            return

        print(f'[ utils/setup ] Found extras: {extras}')
        assert len(extras) % 2 == 0, f'Found odd number ({len(extras)}) of extras: {extras}'
        for i in range(0, len(extras), 2):
            key = extras[i].replace('--', '')
            val = extras[i+1]
            extra_args_list = ['plan_n_maze', 'diffusion_epoch', 'config_2'] # used in commandline only
            if key in extra_args_list:
                # assert not hasattr(args, key)
                setattr(args, key, val)
                print(colored(f'args: set {key} to {val}','cyan'))
            else:
                assert hasattr(args, key), f'[ utils/setup ] {key} not found in config: {args.config}'

            old_val = getattr(args, key)
            old_type = type(old_val)
            print(f'[ utils/setup ] Overriding config | {key} : {old_val} --> {val}')
            if val == 'None':
                val = None
            elif val == 'latest':
                val = 'latest'
            elif old_type in [bool, type(None)]:
                try:
                    val = eval(val)
                except:
                    print(f'[ utils/setup ] Warning: could not parse {val} (old: {old_val}, {old_type}), using str')
            else:
                val = old_type(val)
            setattr(args, key, val)
            self._dict[key] = val

    def eval_fstrings(self, args):
        for key, old in self._dict.items():
            if type(old) is str and old[:2] == 'f:':
                val = old.replace('{', '{args.').replace('f:', '')
                new = lazy_fstring(val, args)
                print(f'[ utils/setup ] Lazy fstring | {key} : {old} --> {new}')
                setattr(self, key, new)
                self._dict[key] = new

    def set_seed(self, args):
        if not 'seed' in dir(args):
            return
        print(f'[ utils/setup ] Setting seed: {args.seed}')
        set_seed(args.seed)

    def generate_exp_name(self, args):
        if not 'exp_name' in dir(args):
            return
        exp_name = getattr(args, 'exp_name')
        if callable(exp_name):
            exp_name_string = exp_name(args)
            print(f'[ utils/setup ] Setting exp_name to: {exp_name_string}')
            setattr(args, 'exp_name', exp_name_string)
            self._dict['exp_name'] = exp_name_string

    def mkdir(self, args):
        if 'logbase' in dir(args) and 'dataset' in dir(args) and 'exp_name' in dir(args):
            args.savepath = os.path.join(args.logbase, args.dataset, args.exp_name)
            self.savepath = args.savepath # Luo added, Sep 5-10
            self._dict['savepath'] = args.savepath
            if 'suffix' in dir(args):
                args.savepath = os.path.join(args.savepath, args.suffix)
            if mkdir(args.savepath):
                print(f'[ utils/setup ] Made savepath: {args.savepath}')
            self.save()

    def get_commit(self, args):
        args.commit = get_git_rev()

    def save_diff(self, args):
        try:
            # save_git_diff(os.path.join(args.savepath, 'diff.txt'))
            pass # no need to save, slow down
        except:
            print('[ utils/setup ] WARNING: did not save git diff')

    def set_wandb(self, args):
        'logs/maze2d-large-v1/diffusion/H384_T256_maze2d_test',
        exp_name = osp.split(args.exp_name)[1]
        exp_date = datetime.datetime.now().strftime("%m%d")
        SLURM_JOB_ID = os.environ.get('SLURM_JOB_ID')
        args.logger_name = f"{exp_date}-{exp_name}-{SLURM_JOB_ID}"
        args.logger_id = args.logger_name

    def check_sinPosEmb(self, args):
        wall_sinPosEmb = getattr(args, 'wall_sinPosEmb', None)
        if wall_sinPosEmb:
            assert args.dataset_config['use_normed_wallLoc']  == True
