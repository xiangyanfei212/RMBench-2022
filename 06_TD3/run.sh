# nohup python -u td3.py task=place_brick seed=0 > logs/place_brick_seed_0.log 2>&1 &
# nohup python -u td3.py task=place_brick seed=1 > logs/place_brick_seed_1.log 2>&1 &
# nohup python -u td3.py task=place_brick seed=2 > logs/place_brick_seed_2.log 2>&1 &
# nohup python -u td3.py task=place_brick seed=3 > logs/place_brick_seed_3.log 2>&1 &
# nohup python -u td3.py task=place_brick seed=4 > logs/place_brick_seed_4.log 2>&1 &

# nohup python -u td3.py task=place_cradle seed=0 gpu_id=4 > logs/place_cradle_seed_0.log 2>&1 &
# nohup python -u td3.py task=place_cradle seed=1 gpu_id=4 > logs/place_cradle_seed_1.log 2>&1 &
# nohup python -u td3.py task=place_cradle seed=2 gpu_id=0 > logs/place_cradle_seed_2.log 2>&1 &
# nohup python -u td3.py task=place_cradle seed=3 gpu_id=0 > logs/place_cradle_seed_3.log 2>&1 &
# nohup python -u td3.py task=place_cradle seed=4 gpu_id=0 > logs/place_cradle_seed_4.log 2>&1 &

# nohup python -u td3.py task=reach_duplo seed=0 gpu_id=0 > logs/reach_duplo_seed_0.log 2>&1 &
# nohup python -u td3.py task=reach_duplo seed=1 gpu_id=2 > logs/reach_duplo_seed_1.log 2>&1 &
# nohup python -u td3.py task=reach_duplo seed=2 gpu_id=0 > logs/reach_duplo_seed_2.log 2>&1 &
# nohup python -u td3.py task=reach_duplo seed=3 gpu_id=2 > logs/reach_duplo_seed_3.log 2>&1 &
# nohup python -u td3.py task=reach_duplo seed=4 gpu_id=2 > logs/reach_duplo_seed_4.log 2>&1 &

# nohup python -u td3.py task=reach_site seed=0 gpu_id=1 > logs/reach_site_seed_0.log 2>&1 &
# nohup python -u td3.py task=reach_site seed=1 gpu_id=1 > logs/reach_site_seed_1.log 2>&1 &
# nohup python -u td3.py task=reach_site seed=2 gpu_id=2 > logs/reach_site_seed_2.log 2>&1 &
# nohup python -u td3.py task=reach_site seed=3 gpu_id=2 > logs/reach_site_seed_3.log 2>&1 &
# nohup python -u td3.py task=reach_site seed=4 gpu_id=2 > logs/reach_site_seed_4.log 2>&1 &

# nohup python -u td3.py task=lift_large_box seed=0 gpu_id=1 > logs/lift_large_box_seed_0.log 2>&1 &
# nohup python -u td3.py task=lift_large_box seed=1 gpu_id=1 > logs/lift_large_box_seed_1.log 2>&1 &
# nohup python -u td3.py task=lift_large_box seed=2 gpu_id=2 > logs/lift_large_box_seed_2.log 2>&1 &
# nohup python -u td3.py task=lift_large_box seed=3 gpu_id=2 > logs/lift_large_box_seed_3.log 2>&1 &
# nohup python -u td3.py task=lift_large_box seed=4 gpu_id=2 > logs/lift_large_box_seed_4.log 2>&1 &

# nohup python -u td3.py task=lift_brick seed=0 gpu_id=0 > logs/lift_brick_seed_0.log 2>&1 &
# nohup python -u td3.py task=lift_brick seed=1 gpu_id=1 > logs/lift_brick_seed_1.log 2>&1 &
# nohup python -u td3.py task=lift_brick seed=2 gpu_id=1 > logs/lift_brick_seed_2.log 2>&1 &
# nohup python -u td3.py task=lift_brick seed=3 gpu_id=2 > logs/lift_brick_seed_3.log 2>&1 &
# nohup python -u td3.py task=lift_brick seed=4 gpu_id=2 > logs/lift_brick_seed_4.log 2>&1 &

# nohup python -u td3.py task=reassemble_5_bricks_random_order seed=0 gpu_id=1 > logs/reassemble_5_bricks_random_order_seed_0.log 2>&1 &
# nohup python -u td3.py task=reassemble_5_bricks_random_order seed=1 gpu_id=0 > logs/reassemble_5_bricks_random_order_seed_1.log 2>&1 &
# nohup python -u td3.py task=reassemble_5_bricks_random_order seed=2 gpu_id=0 > logs/reassemble_5_bricks_random_order_seed_2.log 2>&1 &
# nohup python -u td3.py task=reassemble_5_bricks_random_order seed=3 gpu_id=0 > logs/reassemble_5_bricks_random_order_seed_3.log 2>&1 &
# nohup python -u td3.py task=reassemble_5_bricks_random_order seed=4 gpu_id=1 > logs/reassemble_5_bricks_random_order_seed_4.log 2>&1 &

# nohup python -u td3.py task=reassemble_5_bricks_random_order pretrain_seed=0 load_pretrain_model=true work_dir=/data1/xiangyf/24_DRL_manipulation_benchmark/06_TD3/exp_local/reassemble_5_bricks_random_order_seed_0/2022.08.08_075523 pretrain_episodes=140 gpu_id=1 > logs/reassemble_5_bricks_random_order_seed_0_pretrained.log 2>&1 &
# nohup python -u td3.py task=reassemble_5_bricks_random_order pretrain_seed=1 load_pretrain_model=true work_dir=/data1/xiangyf/24_DRL_manipulation_benchmark/06_TD3/exp_local/reassemble_5_bricks_random_order_seed_1/2022.08.08_220212 pretrain_episodes=130 gpu_id=0 > logs/reassemble_5_bricks_random_order_seed_1_pretrained.log 2>&1 &
# nohup python -u td3.py task=reassemble_5_bricks_random_order pretrain_seed=2 load_pretrain_model=true work_dir=/data1/xiangyf/24_DRL_manipulation_benchmark/06_TD3/exp_local/reassemble_5_bricks_random_order_seed_2/2022.08.08_220212 pretrain_episodes=80 gpu_id=0 > logs/reassemble_5_bricks_random_order_seed_2_pretrained.log 2>&1 &
# nohup python -u td3.py task=reassemble_5_bricks_random_order pretrain_seed=3 load_pretrain_model=true work_dir=/data1/xiangyf/24_DRL_manipulation_benchmark/06_TD3/exp_local/reassemble_5_bricks_random_order_seed_3/2022.08.08_220212 pretrain_episodes=130 gpu_id=0 > logs/reassemble_5_bricks_random_order_seed_3_pretrained.log 2>&1 &
# nohup python -u td3.py task=reassemble_5_bricks_random_order pretrain_seed=4 load_pretrain_model=true work_dir=/data1/xiangyf/24_DRL_manipulation_benchmark/06_TD3/exp_local/reassemble_5_bricks_random_order_seed_4/2022.08.08_075523 pretrain_episodes=140 gpu_id=1 > logs/reassemble_5_bricks_random_order_seed_4_pretrained.log 2>&1 &

# nohup python -u td3.py task=stack_2_bricks seed=0 gpu_id=0 load_pretrain_model=false > logs/stack_2_bricks_seed_0.log 2>&1 &
# nohup python -u td3.py task=stack_2_bricks seed=1 gpu_id=0 load_pretrain_model=false > logs/stack_2_bricks_seed_1.log 2>&1 &
# nohup python -u td3.py task=stack_2_bricks seed=2 gpu_id=0 load_pretrain_model=false > logs/stack_2_bricks_seed_2.log 2>&1 &
# nohup python -u td3.py task=stack_2_bricks seed=3 gpu_id=0 load_pretrain_model=false > logs/stack_2_bricks_seed_3.log 2>&1 &
# nohup python -u td3.py task=stack_2_bricks seed=4 gpu_id=0 load_pretrain_model=false > logs/stack_2_bricks_seed_4.log 2>&1 &

nohup python -u td3.py task=stack_3_bricks seed=0 gpu_id=0 load_pretrain_model=false > logs/stack_3_bricks_seed_0.log 2>&1 &
nohup python -u td3.py task=stack_3_bricks seed=1 gpu_id=0 load_pretrain_model=false > logs/stack_3_bricks_seed_1.log 2>&1 &
nohup python -u td3.py task=stack_3_bricks seed=2 gpu_id=0 load_pretrain_model=false > logs/stack_3_bricks_seed_2.log 2>&1 &
nohup python -u td3.py task=stack_3_bricks seed=3 gpu_id=0 load_pretrain_model=false > logs/stack_3_bricks_seed_3.log 2>&1 &
nohup python -u td3.py task=stack_3_bricks seed=4 gpu_id=0 load_pretrain_model=false > logs/stack_3_bricks_seed_4.log 2>&1 &

# nohup python -u td3.py task=stack_2_bricks_moveable_base seed=0 gpu_id=0 load_pretrain_model=false > logs/stack_2_bricks_moveable_base_seed_0.log 2>&1 &
# nohup python -u td3.py task=stack_2_bricks_moveable_base seed=1 gpu_id=0 load_pretrain_model=false > logs/stack_2_bricks_moveable_base_seed_1.log 2>&1 &
# nohup python -u td3.py task=stack_2_bricks_moveable_base seed=2 gpu_id=0 load_pretrain_model=false > logs/stack_2_bricks_moveable_base_seed_2.log 2>&1 &
# nohup python -u td3.py task=stack_2_bricks_moveable_base seed=3 gpu_id=0 load_pretrain_model=false > logs/stack_2_bricks_moveable_base_seed_3.log 2>&1 &
# nohup python -u td3.py task=stack_2_bricks_moveable_base seed=4 gpu_id=0 load_pretrain_model=false > logs/stack_2_bricks_moveable_base_seed_4.log 2>&1 &

