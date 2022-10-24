import os 
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from icecream import ic

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

plt.style.use(['science', 'no-latex'])
plt.rcParams["font.sans-serif"]=["SimHei"]
plt.rcParams["axes.unicode_minus"]=False

place_brick_exp_dirs = {
    'DrQ-v2': ['./00_DrQv2/exp_local/2022.05.09/112328_task=place_brick',
                 './00_DrQv2/exp_local/2022.05.09/144229_task=place_brick',
                 './00_DrQv2/exp_local/2022.05.09/144133_task=place_brick',
                 './00_DrQv2/exp_local/2022.05.09/144012_task=place_brick',
                 './00_DrQv2/exp_local/2022.05.09/143825_task=place_brick'],

    'VPG': ['./01_VPG/exp_local/2022.05.10/113333_place_brick',
            './01_VPG/exp_local/2022.05.10/113357_place_brick',
            './01_VPG/exp_local/2022.05.10/113419_place_brick',
            './01_VPG/exp_local/2022.05.10/113437_place_brick',
            './01_VPG/exp_local/2022.05.10/113458_place_brick'],

    'TRPO': ['./02_TRPO/exp_local/2022.05.09/145704_place_brick',
             './02_TRPO/exp_local/2022.05.09/145615_place_brick',
             './02_TRPO/exp_local/2022.05.09/145541_place_brick',
             './02_TRPO/exp_local/2022.05.09/145412_place_brick',
             './02_TRPO/exp_local/place_brick_seed_3/2022.07.31_111232'],

    'PPO': ['./03_PPO/exp_local/2022.05.10/111429_place_brick',
            './03_PPO/exp_local/2022.05.10/111552_place_brick',
            './03_PPO/exp_local/2022.05.10/111613_place_brick',
            './03_PPO/exp_local/2022.05.10/111632_place_brick',
            './03_PPO/exp_local/2022.05.10/111648_place_brick'],

    'DDPG': ['./04_DDPG/exp_local/2022.05.12/150217_place_brick',
             './04_DDPG/exp_local/2022.05.12/195154_place_brick',
             './04_DDPG/exp_local/2022.05.12/195112_place_brick',
             './04_DDPG/exp_local/2022.05.12/185915_place_brick',
             './04_DDPG/exp_local/2022.05.12/185926_place_brick'],

    'TD3': ['./06_TD3/exp_local/2022.06.05/100739_place_brick',
            './06_TD3/exp_local/2022.06.05/100802_place_brick',
            './06_TD3/exp_local/2022.06.05/100820_place_brick',
            './06_TD3/exp_local/2022.06.05/100903_place_brick',
            './06_TD3/exp_local/2022.06.05/100941_place_brick'],

    'SAC': ['./05_SAC/exp_local/2022.06.05/101313_place_brick',
            './05_SAC/exp_local/2022.06.05/101250_place_brick',
            './05_SAC/exp_local/2022.06.05/101219_place_brick',
            './05_SAC/exp_local/2022.06.05/101126_place_brick',
            './05_SAC/exp_local/2022.06.05/101108_place_brick'],
}
place_cradle_exp_dirs = {
    'DrQ-v2': ['./00_DrQv2/exp_local/place_cradle_seed_0/2022.06.23_094706',
               './00_DrQv2/exp_local/place_cradle_seed_1/2022.06.23_094747',
               './00_DrQv2/exp_local/place_cradle_seed_2/2022.06.23_094852',
               './00_DrQv2/exp_local/place_cradle_seed_3/2022.06.23_223450',
               './00_DrQv2/exp_local/place_cradle_seed_4/2022.06.23_095224'],
    'VPG':    ['./01_VPG/exp_local/place_cradle_seed_0/2022.06.23_095528',
               './01_VPG/exp_local/place_cradle_seed_1/2022.06.23_095559',
               './01_VPG/exp_local/place_cradle_seed_2/2022.06.23_223546',
               './01_VPG/exp_local/place_cradle_seed_3/2022.06.23_095634',
               './01_VPG/exp_local/place_cradle_seed_4/2022.06.23_223610',],
    'TRPO':   ['./02_TRPO/exp_local/place_cradle_seed_0/2022.07.01_163105',
               './02_TRPO/exp_local/place_cradle_seed_1/2022.06.23_223137',
               './02_TRPO/exp_local/place_cradle_seed_2/2022.06.23_223137',
               './02_TRPO/exp_local/place_cradle_seed_3/2022.07.01_163105',
               './02_TRPO/exp_local/place_cradle_seed_4/2022.07.01_163105'],
    'PPO':    ['./03_PPO/exp_local/place_cradle_seed_0/2022.06.23_102308',
               './03_PPO/exp_local/place_cradle_seed_1/2022.06.23_102308',
               './03_PPO/exp_local/place_cradle_seed_2/2022.06.23_102308',
               './03_PPO/exp_local/place_cradle_seed_3/2022.06.23_102308',
               './03_PPO/exp_local/place_cradle_seed_4/2022.06.23_102308'],
    'DDPG':   ['./04_DDPG/exp_local/place_cradle_seed_0/2022.07.06_200052',
               './04_DDPG/exp_local/place_cradle_seed_1/2022.07.01_100029',
               './04_DDPG/exp_local/place_cradle_seed_2/2022.07.06_200052',
               './04_DDPG/exp_local/place_cradle_seed_3/2022.07.01_100029',
               './04_DDPG/exp_local/place_cradle_seed_4/2022.07.01_100029'],
    'SAC':    ['./05_SAC/exp_local/place_cradle_seed_0/2022.07.01_095214',
               './05_SAC/exp_local/place_cradle_seed_1/2022.07.01_095214',
               './05_SAC/exp_local/place_cradle_seed_2/2022.07.01_095214',
               './05_SAC/exp_local/place_cradle_seed_3/2022.07.01_095214',
               './05_SAC/exp_local/place_cradle_seed_4/2022.07.01_095214'],
    'TD3':    ['./06_TD3/exp_local/place_cradle_seed_0/2022.07.06_200217',
               './06_TD3/exp_local/place_cradle_seed_1/2022.07.06_200729',
               './06_TD3/exp_local/place_cradle_seed_2/2022.07.08_133551',
               './06_TD3/exp_local/place_cradle_seed_3/2022.07.14_090751',
               './06_TD3/exp_local/place_cradle_seed_4/2022.07.14_090751'], 
}

reach_duplo_exp_dirs = {
    'DrQ-v2': ['./00_DrQv2/exp_local/reach_duplo_seed_0/2022.07.18_141211',
               './00_DrQv2/exp_local/reach_duplo_seed_1/2022.07.14_091012',
               './00_DrQv2/exp_local/reach_duplo_seed_2/2022.07.08_142949',
               './00_DrQv2/exp_local/reach_duplo_seed_3/2022.07.08_143214',
               './00_DrQv2/exp_local/reach_duplo_seed_4/2022.07.08_142715'],
    'VPG':    ['./01_VPG/exp_local/reach_duplo_seed_0/2022.07.08_143402',
               './01_VPG/exp_local/reach_duplo_seed_1/2022.07.08_143402',
               './01_VPG/exp_local/reach_duplo_seed_2/2022.07.08_143402',
               './01_VPG/exp_local/reach_duplo_seed_3/2022.07.08_143402',
               './01_VPG/exp_local/reach_duplo_seed_4/2022.07.08_143402'],
    'TRPO':   ['./02_TRPO/exp_local/reach_duplo_seed_0/2022.07.21_084221',
               './02_TRPO/exp_local/reach_duplo_seed_1/2022.07.08_143612',
               './02_TRPO/exp_local/reach_duplo_seed_2/2022.07.18_141325',
               './02_TRPO/exp_local/reach_duplo_seed_3/2022.07.14_091314',
               './02_TRPO/exp_local/reach_duplo_seed_4/2022.07.08_143612'],
    'PPO':    ['./03_PPO/exp_local/reach_duplo_seed_0/2022.07.21_084012',
               './03_PPO/exp_local/reach_duplo_seed_1/2022.07.14_091457',
               './03_PPO/exp_local/reach_duplo_seed_2/2022.07.14_091457',
               './03_PPO/exp_local/reach_duplo_seed_3/2022.07.18_141458',
               './03_PPO/exp_local/reach_duplo_seed_4/2022.07.14_091457'],
    'DDPG':   ['./04_DDPG/exp_local/reach_duplo_seed_0/2022.07.14_091729',
               './04_DDPG/exp_local/reach_duplo_seed_1/2022.07.14_091729',
               './04_DDPG/exp_local/reach_duplo_seed_2/2022.07.14_091729',
               './04_DDPG/exp_local/reach_duplo_seed_3/2022.07.14_091729',
               './04_DDPG/exp_local/reach_duplo_seed_4/2022.07.14_091729'],
    'SAC':    ['./05_SAC/exp_local/reach_duplo_seed_0/2022.07.18_141628',
               './05_SAC/exp_local/reach_duplo_seed_1/2022.07.18_141628',
               './05_SAC/exp_local/reach_duplo_seed_2/2022.07.18_141628',
               './05_SAC/exp_local/reach_duplo_seed_3/2022.07.18_141628',
               './05_SAC/exp_local/reach_duplo_seed_4/2022.07.18_141628'],
    'TD3':    ['./06_TD3/exp_local/reach_duplo_seed_0/2022.07.18_141815',
               './06_TD3/exp_local/reach_duplo_seed_1/2022.07.21_084402',
               './06_TD3/exp_local/reach_duplo_seed_2/2022.07.18_141815',
               './06_TD3/exp_local/reach_duplo_seed_3/2022.07.20_170150',
               './06_TD3/exp_local/reach_duplo_seed_4/2022.07.20_170150'],
}

reach_site_exp_dirs = {
    'DrQ-v2': ['./00_DrQv2/exp_local/reach_site_seed_0/2022.07.23_133735', 
               './00_DrQv2/exp_local/reach_site_seed_1/2022.07.22_192024', 
               './00_DrQv2/exp_local/reach_site_seed_2/2022.07.22_192024', 
               './00_DrQv2/exp_local/reach_site_seed_3/2022.07.22_192025', 
               './00_DrQv2/exp_local/reach_site_seed_4/2022.07.23_133735',],
    'VPG':    ['./01_VPG/exp_local/reach_site_seed_0/2022.07.22_210938',
               './01_VPG/exp_local/reach_site_seed_1/2022.07.22_210938', 
               './01_VPG/exp_local/reach_site_seed_2/2022.07.24_090540', 
               './01_VPG/exp_local/reach_site_seed_3/2022.07.25_195247', 
               './01_VPG/exp_local/reach_site_seed_4/2022.07.24_090540',],
    'TRPO':   ['./02_TRPO/exp_local/reach_site_seed_0/2022.07.23_133904', 
               './02_TRPO/exp_local/reach_site_seed_1/2022.07.23_133904', 
               './02_TRPO/exp_local/reach_site_seed_2/2022.07.24_090651', 
               './02_TRPO/exp_local/reach_site_seed_3/2022.07.23_133904', 
               './02_TRPO/exp_local/reach_site_seed_4/2022.07.23_133904',],
    'PPO':    ['./03_PPO/exp_local/reach_site_seed_0/2022.07.24_090912', 
               './03_PPO/exp_local/reach_site_seed_1/2022.07.24_090913', 
               './03_PPO/exp_local/reach_site_seed_2/2022.07.24_090913', 
               './03_PPO/exp_local/reach_site_seed_3/2022.07.24_090913', 
               './03_PPO/exp_local/reach_site_seed_4/2022.07.24_090912',] ,
    'DDPG':   ['./04_DDPG/exp_local/reach_site_seed_0/2022.07.25_195619', 
               './04_DDPG/exp_local/reach_site_seed_1/2022.08.01_084619', 
               './04_DDPG/exp_local/reach_site_seed_2/2022.07.25_195619', 
               './04_DDPG/exp_local/reach_site_seed_3/2022.08.01_084619', 
               './04_DDPG/exp_local/reach_site_seed_4/2022.08.01_084619',],
    'SAC':    ['./05_SAC/exp_local/reach_site_seed_0/2022.07.26_141402', 
               './05_SAC/exp_local/reach_site_seed_1/2022.07.26_141402', 
               './05_SAC/exp_local/reach_site_seed_2/2022.07.26_141402', 
               './05_SAC/exp_local/reach_site_seed_3/2022.07.28_135758', 
               './05_SAC/exp_local/reach_site_seed_4/2022.07.26_141713',],
    'TD3':    ['./06_TD3/exp_local/reach_site_seed_0/2022.07.26_141635', 
               './06_TD3/exp_local/reach_site_seed_1/2022.07.26_141635', 
               './06_TD3/exp_local/reach_site_seed_2/2022.07.26_141635', 
               './06_TD3/exp_local/reach_site_seed_3/2022.07.26_141635', 
               './06_TD3/exp_local/reach_site_seed_4/2022.07.26_141635',]
}

lift_large_box_exp_dirs = {
    'DrQ-v2': ['./00_DrQv2/exp_local/lift_large_box_seed_0/2022.07.27_145509', 
               './00_DrQv2/exp_local/lift_large_box_seed_1/2022.07.27_145509', 
               './00_DrQv2/exp_local/lift_large_box_seed_2/2022.07.28_140051', 
               './00_DrQv2/exp_local/lift_large_box_seed_3/2022.07.27_145509', 
               './00_DrQv2/exp_local/lift_large_box_seed_4/2022.07.28_140051',],
    'VPG':    ['./01_VPG/exp_local/lift_large_box_seed_0/2022.07.28_140342', 
               './01_VPG/exp_local/lift_large_box_seed_1/2022.07.30_094030', 
               './01_VPG/exp_local/lift_large_box_seed_2/2022.07.28_140342', 
               './01_VPG/exp_local/lift_large_box_seed_3/2022.07.31_100841', 
               './01_VPG/exp_local/lift_large_box_seed_4/2022.07.31_100841',],
    'TRPO':   ['./02_TRPO/exp_local/lift_large_box_seed_0/2022.07.28_140625', 
               './02_TRPO/exp_local/lift_large_box_seed_1/2022.07.31_111328', 
               './02_TRPO/exp_local/lift_large_box_seed_2/2022.07.30_094013', 
               './02_TRPO/exp_local/lift_large_box_seed_3/2022.07.28_140625', 
               './02_TRPO/exp_local/lift_large_box_seed_4/2022.07.30_094013',],
    'PPO':    ['./03_PPO/exp_local/lift_large_box_seed_0/2022.07.30_094604', 
               './03_PPO/exp_local/lift_large_box_seed_1/2022.07.30_094604', 
               './03_PPO/exp_local/lift_large_box_seed_2/2022.07.30_094604', 
               './03_PPO/exp_local/lift_large_box_seed_3/2022.07.31_101044', 
               './03_PPO/exp_local/lift_large_box_seed_4/2022.07.30_094604',] ,
    'DDPG':   ['./04_DDPG/exp_local/lift_large_box_seed_0/2022.07.31_102312', 
               './04_DDPG/exp_local/lift_large_box_seed_1/2022.07.31_102312', 
               './04_DDPG/exp_local/lift_large_box_seed_2/2022.07.31_102312', 
               './04_DDPG/exp_local/lift_large_box_seed_3/2022.07.31_102312', 
               './04_DDPG/exp_local/lift_large_box_seed_4/2022.07.31_102312',],
    'SAC':    ['./05_SAC/exp_local/lift_large_box_seed_0/2022.07.31_103147', 
               './05_SAC/exp_local/lift_large_box_seed_1/2022.07.31_103147', 
               './05_SAC/exp_local/lift_large_box_seed_2/2022.07.31_103147', 
               './05_SAC/exp_local/lift_large_box_seed_3/2022.07.31_103147', 
               './05_SAC/exp_local/lift_large_box_seed_4/2022.07.31_103147',],
    'TD3':    ['./06_TD3/exp_local/lift_large_box_seed_0/2022.07.31_103147', 
               './06_TD3/exp_local/lift_large_box_seed_1/2022.07.31_103147', 
               './06_TD3/exp_local/lift_large_box_seed_2/2022.07.31_103147', 
               './06_TD3/exp_local/lift_large_box_seed_3/2022.07.31_103147', 
               './06_TD3/exp_local/lift_large_box_seed_4/2022.07.31_103147',]
}

lift_brick_exp_dirs = {
    'DrQ-v2': ['./00_DrQv2/exp_local/lift_brick_seed_0/2022.08.02_203252', 
               './00_DrQv2/exp_local/lift_brick_seed_1/2022.08.02_203252', 
               './00_DrQv2/exp_local/lift_brick_seed_2/2022.08.03_143028', 
               './00_DrQv2/exp_local/lift_brick_seed_3/2022.08.02_203252', 
               './00_DrQv2/exp_local/lift_brick_seed_4/2022.08.02_203252',],
    'VPG':    ['./01_VPG/exp_local/lift_brick_seed_0/2022.08.01_084427', 
               './01_VPG/exp_local/lift_brick_seed_1/2022.08.01_084427', 
               './01_VPG/exp_local/lift_brick_seed_2/2022.08.01_084427', 
               './01_VPG/exp_local/lift_brick_seed_3/2022.08.01_084427', 
               './01_VPG/exp_local/lift_brick_seed_4/2022.08.01_084427',],
    'TRPO':   ['./02_TRPO/exp_local/lift_brick_seed_0/2022.08.04_145615', 
               './02_TRPO/exp_local/lift_brick_seed_1/2022.08.04_145615', 
               './02_TRPO/exp_local/lift_brick_seed_2/2022.08.04_145615', 
               './02_TRPO/exp_local/lift_brick_seed_3/2022.08.04_145615', 
               './02_TRPO/exp_local/lift_brick_seed_4/2022.08.06_204306',],
    'PPO':    ['./03_PPO/exp_local/lift_brick_seed_0/2022.08.04_145718', 
               './03_PPO/exp_local/lift_brick_seed_1/2022.08.04_145718', 
               './03_PPO/exp_local/lift_brick_seed_2/2022.08.04_084540', 
               './03_PPO/exp_local/lift_brick_seed_3/2022.08.04_145718', 
               './03_PPO/exp_local/lift_brick_seed_4/2022.08.04_084540',] ,
    'DDPG':   ['./04_DDPG/exp_local/lift_brick_seed_0/2022.08.03_143126', 
               './04_DDPG/exp_local/lift_brick_seed_1/2022.08.03_143126', 
               './04_DDPG/exp_local/lift_brick_seed_2/2022.08.03_143126', 
               './04_DDPG/exp_local/lift_brick_seed_3/2022.08.03_143126', 
               './04_DDPG/exp_local/lift_brick_seed_4/2022.08.03_143126',],
    'SAC':    ['./05_SAC/exp_local/lift_brick_seed_0/2022.08.04_151904', 
               './05_SAC/exp_local/lift_brick_seed_1/2022.08.04_151903', 
               './05_SAC/exp_local/lift_brick_seed_2/2022.08.04_151904', 
               './05_SAC/exp_local/lift_brick_seed_3/2022.08.04_151903', 
               './05_SAC/exp_local/lift_brick_seed_4/2022.08.04_151904',],
    'TD3':    ['./06_TD3/exp_local/lift_brick_seed_0/2022.08.03_143329',
               './06_TD3/exp_local/lift_brick_seed_1/2022.08.03_143329', 
               './06_TD3/exp_local/lift_brick_seed_2/2022.08.03_143329', 
               './06_TD3/exp_local/lift_brick_seed_3/2022.08.03_143329', 
               './06_TD3/exp_local/lift_brick_seed_4/2022.08.03_143329',]
}

reassemble_5_bricks_random_order_exp_dirs = {
    'DrQ-v2': ['./00_DrQv2/exp_local/reassemble_5_bricks_random_order_seed_0/2022.08.03_144304', 
               './00_DrQv2/exp_local/reassemble_5_bricks_random_order_seed_1/2022.08.03_144304', 
               './00_DrQv2/exp_local/reassemble_5_bricks_random_order_seed_2/2022.08.05_134137', 
               './00_DrQv2/exp_local/reassemble_5_bricks_random_order_seed_3/2022.08.05_134137', 
               './00_DrQv2/exp_local/reassemble_5_bricks_random_order_seed_4/2022.08.03_144304',],
    'VPG':    ['./01_VPG/exp_local/reassemble_5_bricks_random_order_seed_0/2022.08.03_144543',
               './01_VPG/exp_local/reassemble_5_bricks_random_order_seed_1/2022.08.05_134229', 
               './01_VPG/exp_local/reassemble_5_bricks_random_order_seed_2/2022.08.05_134229', 
               './01_VPG/exp_local/reassemble_5_bricks_random_order_seed_3/2022.08.03_144543', 
               './01_VPG/exp_local/reassemble_5_bricks_random_order_seed_4/2022.08.05_134229',],
    'TRPO':   ['./02_TRPO/exp_local/reassemble_5_bricks_random_order_seed_0/2022.08.03_144815',
               './02_TRPO/exp_local/reassemble_5_bricks_random_order_seed_1/2022.08.03_144815',
               './02_TRPO/exp_local/reassemble_5_bricks_random_order_seed_2/2022.08.07_194940',
               './02_TRPO/exp_local/reassemble_5_bricks_random_order_seed_3/2022.08.06_204250',
               './02_TRPO/exp_local/reassemble_5_bricks_random_order_seed_4/2022.08.03_144815',],
    'PPO':    ['./03_PPO/exp_local/reassemble_5_bricks_random_order_seed_0/2022.08.06_204421',
               './03_PPO/exp_local/reassemble_5_bricks_random_order_seed_1/2022.08.04_152103',
               './03_PPO/exp_local/reassemble_5_bricks_random_order_seed_2/2022.08.04_152103', 
               './03_PPO/exp_local/reassemble_5_bricks_random_order_seed_3/2022.08.04_152103', 
               './03_PPO/exp_local/reassemble_5_bricks_random_order_seed_4/2022.08.07_194940',] ,
    'DDPG':   ['./04_DDPG/exp_local/reassemble_5_bricks_random_order_seed_0/2022.08.04_152205', 
               './04_DDPG/exp_local/reassemble_5_bricks_random_order_seed_1/2022.08.04_152205', 
               './04_DDPG/exp_local/reassemble_5_bricks_random_order_seed_2/2022.08.04_152205', 
               './04_DDPG/exp_local/reassemble_5_bricks_random_order_seed_3/2022.08.06_204745', 
               './04_DDPG/exp_local/reassemble_5_bricks_random_order_seed_4/2022.08.04_152205',],
    'SAC':    ['./05_SAC/exp_local/reassemble_5_bricks_random_order_seed_0/2022.08.07_195106', 
               './05_SAC/exp_local/reassemble_5_bricks_random_order_seed_1/2022.08.07_195106', 
               './05_SAC/exp_local/reassemble_5_bricks_random_order_seed_2/2022.08.07_195106', 
               './05_SAC/exp_local/reassemble_5_bricks_random_order_seed_3/2022.08.07_195106', 
               './05_SAC/exp_local/reassemble_5_bricks_random_order_seed_4/2022.08.07_195106',],
    'TD3':    ['./06_TD3/exp_local/reassemble_5_bricks_random_order_seed_0/2022.08.08_075523', 
               './06_TD3/exp_local/reassemble_5_bricks_random_order_seed_1/2022.08.08_220212', 
               './06_TD3/exp_local/reassemble_5_bricks_random_order_seed_2/2022.08.08_220212', 
               './06_TD3/exp_local/reassemble_5_bricks_random_order_seed_3/2022.08.08_220212', 
               './06_TD3/exp_local/reassemble_5_bricks_random_order_seed_4/2022.08.08_075523',]
}

stack_2_bricks_exp_dirs = {
    'DrQ-v2': ['./00_DrQv2/exp_local/stack_2_bricks_seed_0/2022.08.09_140145', 
               './00_DrQv2/exp_local/stack_2_bricks_seed_1/2022.08.09_140145', 
               './00_DrQv2/exp_local/stack_2_bricks_seed_2/2022.08.09_140145', 
               './00_DrQv2/exp_local/stack_2_bricks_seed_3/2022.08.09_140145', 
               './00_DrQv2/exp_local/stack_2_bricks_seed_4/2022.08.09_140145',],
    'VPG':    ['./01_VPG/exp_local/stack_2_bricks_seed_0/2022.08.08_173351', 
               './01_VPG/exp_local/stack_2_bricks_seed_1/2022.08.23_232127', 
               './01_VPG/exp_local/stack_2_bricks_seed_2/2022.08.23_205246', 
               './01_VPG/exp_local/stack_2_bricks_seed_3/2022.08.23_205246', 
               './01_VPG/exp_local/stack_2_bricks_seed_4/2022.08.23_205246',],
    'TRPO':   ['./02_TRPO/exp_local/stack_2_bricks_seed_0/2022.08.16_091956', 
               './02_TRPO/exp_local/stack_2_bricks_seed_1/2022.08.16_091956', 
               './02_TRPO/exp_local/stack_2_bricks_seed_2/2022.08.16_091956', 
               './02_TRPO/exp_local/stack_2_bricks_seed_3/2022.08.16_091956', 
               './02_TRPO/exp_local/stack_2_bricks_seed_4/2022.08.16_091956',],
    'PPO':    ['./03_PPO/exp_local/stack_2_bricks_seed_0/2022.08.23_233014', 
               './03_PPO/exp_local/stack_2_bricks_seed_1/2022.08.23_233014', 
               './03_PPO/exp_local/stack_2_bricks_seed_2/2022.08.23_233014', 
               './03_PPO/exp_local/stack_2_bricks_seed_3/2022.08.23_233014', 
               './03_PPO/exp_local/stack_2_bricks_seed_4/2022.08.23_233014',] ,
    'DDPG':   ['./04_DDPG/exp_local/stack_2_bricks_seed_0/2022.08.23_233215', 
               './04_DDPG/exp_local/stack_2_bricks_seed_1/2022.08.23_233215', 
               './04_DDPG/exp_local/stack_2_bricks_seed_2/2022.08.23_233215', 
               './04_DDPG/exp_local/stack_2_bricks_seed_3/2022.08.23_233215', 
               './04_DDPG/exp_local/stack_2_bricks_seed_4/2022.08.23_233215',],
    'SAC':    ['./05_SAC/exp_local/stack_2_bricks_seed_0/2022.08.18_074559', 
               './05_SAC/exp_local/stack_2_bricks_seed_1/2022.08.17_213609', 
               './05_SAC/exp_local/stack_2_bricks_seed_2/2022.08.18_074559', 
               './05_SAC/exp_local/stack_2_bricks_seed_3/2022.08.17_213609', 
               './05_SAC/exp_local/stack_2_bricks_seed_4/2022.08.17_213609',],
    'TD3':    ['./06_TD3/exp_local/stack_2_bricks_seed_0/2022.08.17_210942', 
               './06_TD3/exp_local/stack_2_bricks_seed_1/2022.08.17_210942', 
               './06_TD3/exp_local/stack_2_bricks_seed_2/2022.08.17_210942', 
               './06_TD3/exp_local/stack_2_bricks_seed_3/2022.08.17_210942',     
               './06_TD3/exp_local/stack_2_bricks_seed_4/2022.08.17_210942',]
}

stack_2_bricks_moveable_base_exp_dirs = {
    'DrQ-v2': ['./00_DrQv2/exp_local/stack_2_bricks_moveable_base_seed_0/2022.08.23_085520',
               './00_DrQv2/exp_local/stack_2_bricks_moveable_base_seed_1/2022.08.23_085520',
               './00_DrQv2/exp_local/stack_2_bricks_moveable_base_seed_2/2022.08.23_085520',
               './00_DrQv2/exp_local/stack_2_bricks_moveable_base_seed_3/2022.08.22_223332',
               './00_DrQv2/exp_local/stack_2_bricks_moveable_base_seed_4/2022.08.22_223332'],
    'VPG':    ['./01_VPG/exp_local/stack_2_bricks_moveable_base_seed_0/2022.08.23_232438',
               './01_VPG/exp_local/stack_2_bricks_moveable_base_seed_1/2022.08.23_232437',
               './01_VPG/exp_local/stack_2_bricks_moveable_base_seed_2/2022.08.23_232437',
               './01_VPG/exp_local/stack_2_bricks_moveable_base_seed_3/2022.08.23_232437',
               './01_VPG/exp_local/stack_2_bricks_moveable_base_seed_4/2022.08.23_232437'],
    'TRPO':   ['./02_TRPO/exp_local/stack_2_bricks_moveable_base_seed_0/2022.08.23_232709',
               './02_TRPO/exp_local/stack_2_bricks_moveable_base_seed_1/2022.08.25_091421',
               './02_TRPO/exp_local/stack_2_bricks_moveable_base_seed_2/2022.08.24_105832',
               './02_TRPO/exp_local/stack_2_bricks_moveable_base_seed_3/2022.08.23_232709',
               './02_TRPO/exp_local/stack_2_bricks_moveable_base_seed_4/2022.08.25_091421'],
    'PPO':    ['./03_PPO/exp_local/stack_2_bricks_moveable_base_seed_0/2022.08.24_105409',
               './03_PPO/exp_local/stack_2_bricks_moveable_base_seed_1/2022.08.24_105408',
               './03_PPO/exp_local/stack_2_bricks_moveable_base_seed_2/2022.08.24_105408',
               './03_PPO/exp_local/stack_2_bricks_moveable_base_seed_3/2022.08.24_105408',
               './03_PPO/exp_local/stack_2_bricks_moveable_base_seed_4/2022.08.25_091101'],
    'DDPG':   ['./04_DDPG/exp_local/stack_2_bricks_moveable_base_seed_0/2022.08.24_110309',
               './04_DDPG/exp_local/stack_2_bricks_moveable_base_seed_1/2022.08.24_110309',
               './04_DDPG/exp_local/stack_2_bricks_moveable_base_seed_2/2022.08.24_110309',
               './04_DDPG/exp_local/stack_2_bricks_moveable_base_seed_3/2022.08.24_110309',
               './04_DDPG/exp_local/stack_2_bricks_moveable_base_seed_4/2022.08.24_110309'],
    'SAC':    ['./05_SAC/exp_local/stack_2_bricks_moveable_base_seed_0/2022.08.17_214013',
               './05_SAC/exp_local/stack_2_bricks_moveable_base_seed_1/2022.08.17_214013',
               './05_SAC/exp_local/stack_2_bricks_moveable_base_seed_2/2022.08.17_214013',
               './05_SAC/exp_local/stack_2_bricks_moveable_base_seed_3/2022.08.17_214013',
               './05_SAC/exp_local/stack_2_bricks_moveable_base_seed_4/2022.08.17_214013'],
    'TD3':    ['./06_TD3/exp_local/stack_2_bricks_moveable_base_seed_0/2022.08.17_213933',
               './06_TD3/exp_local/stack_2_bricks_moveable_base_seed_1/2022.08.17_213933',
               './06_TD3/exp_local/stack_2_bricks_moveable_base_seed_2/2022.08.17_213933',
               './06_TD3/exp_local/stack_2_bricks_moveable_base_seed_3/2022.08.17_213933',
               './06_TD3/exp_local/stack_2_bricks_moveable_base_seed_4/2022.08.17_213933'],
}

def plot_curve_for_single_task(mode:str, task_dir:str, task_name:str, algo:str, epoch_len:int):

    in_df = pd.read_csv(os.path.join(task_dir, 'train.csv'))
    if 'Epoch' in in_df.columns:
        in_df.rename(columns={'Epoch': 'episode', 'EpRet': 'episode_reward'})
    in_df = in_df[(in_df['episode']>=1) & (in_df['episode']<=epoch_len)]
    if algo != 'drqv2':
        # in_df.drop_duplicates(subset='episode', inplace=True)
        in_df2 = in_df.groupby('episode')['episode_reward'].mean()
        in_df2 = pd.DataFrame(in_df2)
        in_df2.reset_index(inplace=True)
        in_df = in_df2
    ic(in_df)

    x = in_df['episode'].values
    y = in_df['episode_reward'].values

    cum_mean_y = cumsum_avg(x, y)
    conv_mean_y = move_avg(y, 5)
    ic(conv_mean_y)

    fig = plt.figure(figsize=(20, 5))

    ax1 = fig.add_subplot(131)
    ax1.grid(linestyle="--")
    lns1 = ax1.plot(x, y, label=algo, marker='o', color="blue", linewidth=1.5, linestyle='--')
    ax1.set_xlabel(f'{mode} episode', fontsize=14)
    ax1.set_ylabel(f'{mode} Episode Reward', fontsize=14)
    # ax1.set_ylim([0, np.max(y)+1])

    # ax2 = ax1.twinx()
    ax2 = fig.add_subplot(132)
    ax2.grid(linestyle="--")
    lns2 = ax2.plot(x, cum_mean_y, label=algo, marker='o', color="red", linewidth=1.5, linestyle='--')
    ax2.set_ylabel(f'{mode} episode mean reward', fontsize=14)
    ax2.set_xlabel(f'{mode} episode', fontsize=14)
    # ax2.set_ylim([0, np.max(y)+1])

    # ax3 = ax1.twinx()
    ax3 = fig.add_subplot(133)
    ax3.grid(linestyle="--")
    lns3 = ax3.plot(x[:-3], conv_mean_y[:-3], label=algo, marker='o', color="green", linewidth=1.5, linestyle='--')
    ax3.set_ylabel(f'{mode} episode smooth reward', fontsize=14)
    ax3.set_xlabel(f'{mode} episode', fontsize=14)
    # ax3.set_ylim([0, np.max(y)+1])

    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax3.legend(lns, labs, loc='best')

    fig.suptitle(f'{task_name}', fontsize=16)
    out_img = os.path.join(task_dir, f'{mode}.png')
    print(f'Saving to {out_img}')
    plt.savefig(out_img, dpi=400)
    plt.show()

def move_avg(a,n,mode="same"):
    return (np.convolve(a, np.repeat(1/n, n), mode=mode))

def cumsum_avg(x, y):
    return (np.cumsum(y)/x) 

def get_data_1(task_name:str, algo_name:str):

    all_data = pd.DataFrame()

    if task_name == 'place_brick':
        exp_dirs = place_brick_exp_dirs[algo_name]
    if task_name == 'place_cradle':
        exp_dirs = place_cradle_exp_dirs[algo_name]
    if task_name == 'reach_duplo':
        exp_dirs = reach_duplo_exp_dirs[algo_name]
    if task_name == 'reach_site':
        exp_dirs = reach_site_exp_dirs[algo_name]
    if task_name == 'lift_large_box':
        exp_dirs = lift_large_box_exp_dirs[algo_name]
    if task_name == 'lift_brick':
        exp_dirs = lift_brick_exp_dirs[algo_name]
    if task_name == 'reassemble_5_bricks_random_order':
        exp_dirs = reassemble_5_bricks_random_order_exp_dirs[algo_name]
    if task_name == 'stack_2_bricks':
        exp_dirs = stack_2_bricks_exp_dirs[algo_name]
    if task_name == 'stack_2_bricks_moveable_base':
        exp_dirs = stack_2_bricks_moveable_base_exp_dirs[algo_name]
        
    for seed, exp_dir in enumerate(exp_dirs):
        in_f = os.path.join(exp_dir, 'train.csv')
        print(f'Reading {in_f}')
        in_df = pd.read_csv(in_f)
        if 'episode_frame' in in_df.columns:
            in_df.rename(columns = {'episode_frame':'episode_length', 'total_time':'Time'}, inplace = True)

        in_df = in_df[in_df['episode'] <= 200]
        in_df = in_df[in_df['episode_length'] >= 249]

        in_df1 = in_df.groupby('episode')['episode_reward'].mean()
        in_df1 = pd.DataFrame(in_df1)
        in_df1.reset_index(inplace=True)
        x = in_df1['episode'].values
        y = in_df1['episode_reward'].values
        in_df1['conv_mean_reward'] = move_avg(y, 20)
        in_df1['mean_reward'] = np.mean(y) 
        in_df1['seed'] = seed
        in_df1['frame'] = (in_df1['episode']+1)*2000
        ic(in_df1)
        
        all_data = pd.concat([all_data, in_df1])

    all_data.reset_index(drop=True, inplace=True)

    return all_data


def get_data(task_name:str, algo_name:str):

    all_data = pd.DataFrame()

    if task_name == 'place_brick':
        exp_dirs = place_brick_exp_dirs[algo_name]
    if task_name == 'place_cradle':
        exp_dirs = place_cradle_exp_dirs[algo_name]
    if task_name == 'reach_duplo':
        exp_dirs = reach_duplo_exp_dirs[algo_name]
    if task_name == 'reach_site':
        exp_dirs = reach_site_exp_dirs[algo_name]
    if task_name == 'lift_large_box':
        exp_dirs = lift_large_box_exp_dirs[algo_name]
    if task_name == 'lift_brick':
        exp_dirs = lift_brick_exp_dirs[algo_name]
    if task_name == 'reassemble_5_bricks_random_order':
        exp_dirs = reassemble_5_bricks_random_order_exp_dirs[algo_name]
    if task_name == 'stack_2_bricks':
        exp_dirs = stack_2_bricks_exp_dirs[algo_name]
    if task_name == 'stack_2_bricks_moveable_base':
        exp_dirs = stack_2_bricks_moveable_base_exp_dirs[algo_name]
        
    for seed, exp_dir in enumerate(exp_dirs):
        in_f = os.path.join(exp_dir, 'train.csv')
        # print(f'Reading {in_f}')
        in_df = pd.read_csv(in_f)

        if 'episode_frame' in in_df.columns:
            in_df.rename(columns = {'episode_frame':'episode_length', 'total_time':'Time'}, inplace = True)

        in_df = in_df[in_df['episode'] <= 250]
        in_df = in_df[in_df['episode_length'] >= 249]

        in_df1 = in_df.groupby('episode')['episode_reward', 'Time'].mean()
        in_df1 = pd.DataFrame(in_df1)
        in_df1.reset_index(inplace=True)
        in_df1['Time'] = in_df1['Time']/(60*60) # hours

        x = in_df1['episode'].values
        y = in_df1['episode_reward'].values

        # cum_mean_y = cumsum_avg(x, y)
        # in_df['cum_mean_reward'] = cum_mean_y

        in_df1['conv_mean_reward'] = move_avg(y, 10)
        in_df1['mean_reward'] = np.mean(y) 
        in_df1['seed'] = seed
        
        all_data = pd.concat([all_data, in_df1])

    all_data.reset_index(drop=True, inplace=True)
    # ic(all_data[['mean_reward', 'seed']].drop_duplicates())

    all_data1 = all_data.groupby('episode')['Time'].mean()
    all_data = all_data.merge(all_data1, left_on='episode', right_on='episode')
    all_data.drop(columns=['Time_x'], inplace=True)
    all_data.rename(columns={'Time_y':'Time'}, inplace=True)
    # 均值
    mean_reward_all_seed = np.mean(all_data[['mean_reward', 'seed']].drop_duplicates()['mean_reward'].values) 
    # 标准差
    std_reward_all_seed = np.std(all_data[['mean_reward', 'seed']].drop_duplicates()['mean_reward'].values) 

    return all_data, mean_reward_all_seed, std_reward_all_seed


def compare_hour_episode_reward(task_name:str, algo_list:list, max_episode:int):
    plt.style.use(['science', 'no-latex'])
    plt.rcParams["font.sans-serif"]=["SimHei"] # 设置字体
    plt.rcParams["axes.unicode_minus"]=False # 解决图像中的“-”负号的乱码问题

    fig = plt.figure(figsize=(8,5), dpi=150)
    ax1 = fig.add_subplot(111)
    ax1.grid(linestyle="--")
    max_y = 0
    for algo in algo_list:
        data, mean_reward_all_seed, std_reward_all_seed = get_data(task_name, algo)
        # data = data[(data['episode']>=1) & (data['episode']<=max_episode)]
        data = data[data['episode']<=max_episode-5]
        # ic(data)

        x = data["Time"]
        y = data["conv_mean_reward"]

        if np.max(y) > max_y:
            max_y = np.max(y)
        
        sns.lineplot(x, y, ci=95, label=algo, ax=ax1)
        print(f'algo: {algo}, mean_reward_all_seed: {round(mean_reward_all_seed,3)}, std_reward_all_seed:  {round(std_reward_all_seed,3)}')

    ax1.set_ylabel(f'Training Episode Reward', fontsize=18)
    # ax3.set_ylabel(f'Training smooth episode reward', fontsize=14)
    ax1.set_xlabel(f'Training Hours', fontsize=18)

    if task_name == 'place_brick':
        ax1.set_ylim([2, 6])
    if task_name == 'place_cradle':
        ax1.set_ylim([2, 6])

    # fig.suptitle(f'task: {task_name}', fontsize=16)
    plt.legend(loc='lower right', ncol=3, fontsize=14)

    now_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    plt.savefig(f'./img/{task_name}_time_{len(algo_list)}.png')
    plt.show()

def compare_frame_episode_reward(task_name:str, algo_list:list, max_episode:int):
    
    plt.style.use(['science', 'no-latex'])
    plt.rcParams["font.sans-serif"]=["SimHei"] # 设置字体
    plt.rcParams["axes.unicode_minus"]=False # 解决图像中的“-”负号的乱码问题

    fig = plt.figure(figsize=(8,5), dpi=150)
    ax1 = fig.add_subplot(111)
    ax1.grid(linestyle="--")
    max_y = 0
    for algo in algo_list:
        data = get_data_1(task_name, algo)
        x = data["frame"]/1000
        y = data["conv_mean_reward"]

        if np.max(y) > max_y:
            max_y = np.max(y)
        
        sns.lineplot(x, y, ci=95, label=algo, ax=ax1)

    ax1.set_ylabel(f'Training Episode Reward', fontsize=18)
    ax1.set_xlabel(f'Training Frame ($10^3$)', fontsize=18)

    ax1.set_ylim([0, 0.6])

    # fig.suptitle(f'task: {task_name}', fontsize=16)
    plt.legend(loc='upper right', fontsize=18)

    now_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    plt.savefig(f'./img/{task_name}_frame_{len(algo_list)}.png')
    plt.show()
    
def bak_compare_episode_episode_reward(task_name:str, algo_list:list, max_episode:int):

    plt.style.use(['science', 'no-latex'])
    plt.rcParams["font.sans-serif"]=["SimHei"] # 设置字体
    plt.rcParams["axes.unicode_minus"]=False # 解决图像中的“-”负号的乱码问题

    # fig = plt.figure(figsize=(8, 5), dpi=100)
    # ax1 = fig.add_subplot(111)
    # ax1.grid(linestyle="--")
    # for algo in algo_list:
    #     data, mean_reward_all_seed, std_reward_all_seed = get_data(task_name, algo)
    #     data = data[(data['episode']>=1) & (data['episode']<=max_episode)]
    #     x = data["episode"]
    #     y = data["episode_reward"]
    #     sns.lineplot(x, y, ci=95, label=algo, ax=ax1)
    #     
    # ax1.set_ylabel(f'Training episode reward', fontsize=14)
    # ax1.set_xlabel(f'Training episode', fontsize=14)
    # ax1.set_ylim([0, np.max(y)-1])

    # ax2 = fig.add_subplot(132)
    # ax2.grid(linestyle="--")
    # for algo in algo_list:
    #    data = get_data(task_name, algo)
    #    data = data[(data['episode']>=1) & (data['episode']<=max_episode)]
    #    x = data["episode"]
    #    y = data["cum_mean_reward"]
    #    sns.lineplot(x, y, ci=95, label=algo, ax=ax2)
    # ax2.set_ylabel(f'{mode} average episode reward', fontsize=14)
    # ax2.set_xlabel(f'{mode} episode', fontsize=14)
    # ax2.set_ylim([0, np.max(y)+1])

    fig = plt.figure(figsize=(8,5), dpi=200)
    ax3 = fig.add_subplot(111)
    ax3.grid(linestyle="--")
    max_y = 0
    for algo in algo_list:
        data, mean_reward_all_seed, std_reward_all_seed = get_data(task_name, algo)
        # data = data[(data['episode']>=1) & (data['episode']<=max_episode-3)]
        data = data[data['episode']<=(max_episode-5)]
        # ic(data)
        
        x = data["episode"]
        y = data["conv_mean_reward"]

        if np.max(y) > max_y:
            max_y = np.max(y)
        
        sns.lineplot(x, y, ci=95, label=algo, ax=ax3)
        print(f'algo: {algo}, mean_reward_all_seed: {round(mean_reward_all_seed,3)}, std_reward_all_seed:  {round(std_reward_all_seed,3)}')
    ax3.set_ylabel(f'Training Episode Reward', fontsize=18)
    ax3.set_xlabel(f'Training Episode', fontsize=18)
    if task_name == 'lift_brick':
        ax3.set_ylim([0, 0.008])

    # fig.suptitle(f'task: {task_name}', fontsize=16)
    plt.legend(loc='upper right', fontsize=18)

    now_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    plt.savefig(f'./img/{task_name}_{len(algo_list)}.png')
    plt.show()

def compare_episode_episode_reward(task_name:str, algo_list:list, max_episode:int):

    plt.style.use(['science', 'no-latex'])
    plt.rcParams["font.sans-serif"]=["SimHei"] # 设置字体
    plt.rcParams["axes.unicode_minus"]=False # 解决图像中的“-”负号的乱码问题

    ic(task_name)

    fig = plt.figure(figsize=(8,5), dpi=200)
    ax1 = fig.add_subplot(111)
    ax1.grid(linestyle="--")
    for algo in algo_list:
        data, mean_reward_all_seed, std_reward_all_seed = get_data(task_name, algo)
        data = data[(data['episode']>=0) & (data['episode']<=max_episode-5)]
        # ic(data)
        
        x = data["episode"]
        y = data["conv_mean_reward"]
        # y = y*10000

        sns.lineplot(x, y, ci=95, label=algo, ax=ax1, linewidth = 2)
        print(f'algo: {algo}, mean_reward_all_seed: {round(mean_reward_all_seed,3)}, std_reward_all_seed:  {round(std_reward_all_seed,3)}')
    ax1.set_ylabel(f'Training Episode Reward', fontsize=18)
    ax1.set_xlabel(f'Training Episode', fontsize=18)
    ax1.set_xlim([0, 160])
    if task_name == 'place_brick': 
        ax1.set_ylim([2, 6])
    if task_name == 'place_cradle': 
        ax1.set_ylim([2, 6])
    if task_name == 'lift_brick': 
        ax1.set_ylim([0, 0.005])
    if task_name == 'lift_large_box': 
        ax1.set_ylim([0, 0.014])
    if task_name == 'reach_site': 
        ax1.set_ylim([0, 10])
    if task_name == 'reach_duplo': 
        ax1.set_ylim([0, 16])
    if task_name == 'stack_2_bricks': 
        ax1.set_ylim([0, 0.4])
    if task_name == 'stack_2_bricks_moveable_base': 
        ax1.set_ylim([0, 0.4])
    if task_name == 'reassemble_5_bricks_random_order': 
        ax1.set_ylim([0, 90])

    # fig.suptitle(f'task: {task_name}', fontsize=16)
    if task_name == 'place_brick':
        plt.legend(loc='lower right', fontsize=16, ncol=3)
    else:
        plt.legend('')

    now_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    plt.savefig(f'./img/{task_name}_{len(algo_list)}.png', dpi=200)
    plt.show()

if __name__ == '__main__':

    # plot_curve_for_single_task('train', place_brick_exp_dirs['DrQv2'][1], 'place_brick', 'DrQv2', 200)

    # compare_episode_episode_reward(task_name='place_brick',      algo_list=['PPO', 'TRPO'], max_episode=200)
    # compare_episode_episode_reward(task_name='place_cradle',     algo_list=['PPO', 'TRPO'], max_episode=200)
    # compare_episode_episode_reward(task_name='reach_duplo',      algo_list=['PPO', 'TRPO'], max_episode=200)
    # compare_episode_episode_reward(task_name='reach_site',       algo_list=['PPO', 'TRPO'], max_episode=200)
    # compare_episode_episode_reward(task_name='lift_large_box',   algo_list=['PPO', 'TRPO'], max_episode=200)
    # compare_episode_episode_reward(task_name='lift_brick',       algo_list=['PPO', 'TRPO'], max_episode=200)
    # compare_episode_episode_reward(task_name='stack_2_bricks',   algo_list=['PPO', 'TRPO'], max_episode=200)
    # compare_episode_episode_reward(task_name='reassemble_5_bricks_random_order',   algo_list=['PPO', 'TRPO'], max_episode=200)
    # 
    # compare_episode_episode_reward(task_name='place_brick',      algo_list=['DDPG', 'TD3', 'DrQ-v2'], max_episode=200)
    # compare_episode_episode_reward(task_name='place_cradle',     algo_list=['DDPG', 'TD3', 'DrQ-v2'], max_episode=200)
    # compare_episode_episode_reward(task_name='reach_duplo',      algo_list=['DDPG', 'TD3', 'DrQ-v2'], max_episode=200)
    # compare_episode_episode_reward(task_name='reach_site',       algo_list=['DDPG', 'TD3', 'DrQ-v2'], max_episode=200)
    # compare_episode_episode_reward(task_name='lift_large_box',   algo_list=['DDPG', 'TD3', 'DrQ-v2'], max_episode=200)
    # compare_episode_episode_reward(task_name='lift_brick',       algo_list=['DDPG', 'TD3', 'DrQ-v2'], max_episode=200)
    # compare_episode_episode_reward(task_name='stack_2_bricks',   algo_list=['DDPG', 'TD3', 'DrQ-v2'], max_episode=200)
    # compare_episode_episode_reward(task_name='reassemble_5_bricks_random_order',   algo_list=['DDPG', 'TD3', 'DrQ-v2'], max_episode=200)

    # compare_episode_episode_reward(task_name='place_brick',    algo_list=['VPG', 'TRPO', 'PPO', 'DDPG', 'TD3', 'SAC', 'DrQ-v2'], max_episode=200)
    # compare_episode_episode_reward(task_name='place_cradle',   algo_list=['VPG', 'TRPO', 'PPO', 'DDPG', 'TD3', 'SAC', 'DrQ-v2'], max_episode=200)
    # compare_episode_episode_reward(task_name='reach_duplo',    algo_list=['VPG', 'TRPO', 'PPO', 'DDPG', 'TD3', 'SAC', 'DrQ-v2'], max_episode=200)
    # compare_episode_episode_reward(task_name='reach_site',     algo_list=['VPG', 'TRPO', 'PPO', 'DDPG', 'TD3', 'SAC', 'DrQ-v2'], max_episode=200)
    # compare_episode_episode_reward(task_name='lift_large_box', algo_list=['VPG', 'TRPO', 'PPO', 'DDPG', 'TD3', 'SAC', 'DrQ-v2'], max_episode=200)
    # compare_episode_episode_reward(task_name='lift_brick',     algo_list=['VPG', 'TRPO', 'PPO', 'DDPG', 'TD3', 'SAC', 'DrQ-v2'], max_episode=200)
    # compare_episode_episode_reward(task_name='reassemble_5_bricks_random_order', algo_list=['VPG', 'TRPO', 'PPO', 'DDPG', 'TD3', 'SAC', 'DrQ-v2'], max_episode=200)
    # compare_episode_episode_reward(task_name='stack_2_bricks', algo_list=['VPG', 'TRPO', 'PPO', 'DDPG', 'TD3', 'SAC', 'DrQ-v2'], max_episode=200)
    compare_episode_episode_reward(task_name='stack_2_bricks_moveable_base', algo_list=['VPG', 'TRPO', 'PPO', 'DDPG', 'TD3', 'SAC', 'DrQ-v2'], max_episode=200)

    # compare_hour_episode_reward(task_name='place_brick',    algo_list=['VPG', 'TRPO', 'PPO', 'DDPG', 'TD3', 'SAC', 'DrQ-v2'], max_episode=200)
    # compare_hour_episode_reward(task_name='place_cradle',   algo_list=['VPG', 'TRPO', 'PPO', 'DDPG', 'TD3', 'SAC', 'DrQ-v2'], max_episode=200)
    # compare_hour_episode_reward(task_name='reach_duplo',    algo_list=['VPG', 'TRPO', 'PPO', 'DDPG', 'TD3', 'SAC', 'DrQ-v2'], max_episode=200)
    # compare_hour_episode_reward(task_name='reach_site',     algo_list=['VPG', 'TRPO', 'PPO', 'DDPG', 'TD3', 'SAC', 'DrQ-v2'], max_episode=200)
    # compare_hour_episode_reward(task_name='lift_large_box', algo_list=['VPG', 'TRPO', 'PPO', 'DDPG', 'TD3', 'SAC', 'DrQ-v2'], max_episode=200)
    # compare_hour_episode_reward(task_name='lift_brick',     algo_list=['VPG', 'TRPO', 'PPO', 'DDPG', 'TD3', 'SAC', 'DrQ-v2'], max_episode=200)
    # compare_hour_episode_reward(task_name='reassemble_5_bricks_random_order', algo_list=['VPG', 'TRPO', 'PPO', 'DDPG', 'TD3', 'SAC', 'DrQ-v2'], max_episode=200)
    # compare_hour_episode_reward(task_name='stack_2_bricks', algo_list=['VPG', 'TRPO', 'PPO', 'DDPG', 'TD3', 'SAC', 'DrQ-v2'], max_episode=200)

    # compare_frame_episode_reward(task_name='stack_2_bricks', algo_list=['VPG', 'TRPO', 'PPO', 'DDPG', 'TD3', 'SAC', 'DrQ-v2'], max_episode=200)
