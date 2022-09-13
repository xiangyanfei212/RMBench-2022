# RMBench-2022


## Brief Intro

n this work, we present RMBench, the first benchmark for robotic manipulations, which have high-dimensional continuous action and state spaces. We implement and evaluate reinforcement learning algorithms that directly use observed pixels as inputs.

### RL algirithms

- VPG Sutton et al., 2000:  https://papers.nips.cc/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html
- TRPO Schulman et al., 2015 : https://proceedings.mlr.press/v37/schulman15.html
- PPO Schulman et al., 2017: https://arxiv.org/abs/1707.06347
- DDPG Silver et al., 2014: https://arxiv.org/abs/1509.02971
- TD3 Fujimoto et al., 2018 : https://arxiv.org/abs/1802.09477
- SAC Haarnoja et al., 2018: https://arxiv.org/abs/1801.01290
- DrQ-v2 Yarats et al., 2021: http://arxiv.org/abs/2107.09645

### Tasks

<table>
    <tr>
        <td>\label{table:task</td>
        <td>description}</td>
    </tr>
    <tr>
        <td>\bf{Category}</td>
        <td>\bf{Task}</td>
        <td>\bf{Description}</td>
    </tr>
    <tr>
        <td>\multirow{2}*{Lifting}</td>
        <td>Lift brick</td>
        <td>Elevate a brick above a threshold height.</td>
    </tr>
    <tr>
        <td>~</td>
        <td>Lift large box</td>
        <td>Elevate a large box above a threshold height. The box is too large to be grasped by the gripper, requiring non-prehensile manipulation.</td>
    </tr>
    <tr>
        <td>\multirow{2}*{Reaching}</td>
        <td>Reach site</td>
        <td>Move the end effector to a target location in 3D space.</td>
    </tr>
    <tr>
        <td>~</td>
        <td>Reach brick</td>
        <td>Move the end effector to a brick resting on the ground.</td>
    </tr>
    <tr>
        <td>\multirow{2}*{Placing}</td>
        <td>Place cradle</td>
        <td>Place a brick inside a concave `cradle' situated on a pedestal.</td>
    </tr>
    <tr>
        <td>~</td>
        <td>Place brick</td>
        <td>Place a brick on top of another brick that is attached to the top of a pedestal. Unlike the stacking tasks below, the two bricks are not required to be snapped together in order to obtain maximum reward.</td>
    </tr>
    <tr>
        <td>\multirow{2}*{Stacking}</td>
        <td>Stack 2 bricks</td>
        <td>Snap together two bricks, one of which is attached to the floor.</td>
    </tr>
    <tr>
        <td>~</td>
        <td>Stack 2 bricks movable base</td>
        <td>Same as `stack 2 bricks', except both bricks are movable.</td>
    </tr>
    <tr>
        <td>Reassembling</td>
        <td>Reassemble 5 bricks random order</td>
        <td>The episode begins with all five bricks already assembled in a stack, with the bottom brick being attached to the floor. The agent must disassemble the top four bricks in the stack, and reassemble them in the opposite order.</td>
    </tr>
</table>





## Installation

1. Install MuJoCo 
- Obtain a license on the MuJoCo website:https://www.roboti.us/license.html.
- Download MuJoCo binaries here:https://www.roboti.us/download.html. such as 'mujoco210\_linux.zip'
- Unzip the downloaded archive into ~/.mujoco/mujoco210 

$ mkdir ~/.mujoco/mujoco210
$ cp mujoco210\_linux.zip ~/.mujoco/mujoco210 
$ cd ~/.mujoco/mujoco210 
$ unzip mujoco210\_linux.zip

- Place your license key file mjkey.txt at ~/.mujoco/mujoco210.

$ cp mjkey.txt ~/.mujoco/mujoco210 
$ cp mjkey.txt ~/.mujoco/mujoco210/mujoco210_linux/bin

- Add environment variables: Use the env variables MUJOCO\_PY\_MJKEY\_PATH and MUJOCO\_PY\_MUJOCO\_PATH to specify the MuJoCo license key path and the MuJoCo directory path. 
$ export MUJOCO\_PY\_MJKEY\_PATH=$MUJOCO\_PY\_MJKEY\_PATH:~/.mujoco/mujoco210/mjkey.txt
$ export MUJOCO\_PY\_MUJOCO\_PATH=$MUJOCO\_PY\_MUJOCO\_PATH:~/.mujoco/mujoco210/mujoco210\_linux

- Append the MuJoCo subdirectory bin path into the env variable LD\_LIBRARY\_PATH.
$ export LD\_LIBRARY\_PATH=$LD\_LIBRARY\_PATH:~/.mujoco/mujoco210/bin 

2. Install the required python library
pip install -r requirements.txt

## How to run？

For example, we want to train agents using DrQ-v2 algorithms for 'reaching site' tasks:
$ cd 00\_DrQv2
$ python drqv2_train.py task=reach_site

## Acknowledgements

Part of this code is inspired by SpinningUp2018:https://spinningup.openai.com/en/latest/spinningup/rl_intro.html and DrQ-v2:https://github.com/facebookresearch/drqv2


