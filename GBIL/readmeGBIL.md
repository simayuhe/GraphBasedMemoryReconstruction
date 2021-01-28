这是一个将GQ 和 IL 放在一起，用较好的轨迹中的点来替代原本方案中的聚类中心
多了一个参数，--expert_memory_size=3
我们要存多少条获得奖励最高的轨迹

CUDA_VISIBLE_DEVICES=7 python mainGBIL0119.py --env="Alien-v4" --training_iters=10000000 --memory_size=100000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=20 --dist_th=2 --riqi="0119-2-20-3" --expert_memory_size=3 --save_path='/home/kpl/'

GBIL 每个两组参数 2-40-5 2-40-3 2-20-5 2-20-3

tmux a -t 30
CUDA_VISIBLE_DEVICES=2 python mainGBIL0119.py --env="Alien-v4" --training_iters=10000000 --memory_size=100000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=20 --dist_th=2 --riqi="0122-2-20-32" --expert_memory_size=3 --save_path='/home/kpl/'
2021年1月26日 重来

tmux a -t 31
CUDA_VISIBLE_DEVICES=2 python mainGBIL0119.py --env="Alien-v4" --training_iters=10000000 --memory_size=100000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=20 --dist_th=2 --riqi="0122-2-20-52" --expert_memory_size=5 --save_path='/home/kpl/'
2021年1月26日 重来

tmux a -t 32
CUDA_VISIBLE_DEVICES=2 python mainGBIL0119.py --env="Alien-v4" --training_iters=10000000 --memory_size=100000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=40 --dist_th=2 --riqi="0122-2-40-32" --expert_memory_size=3 --save_path='/home/kpl/'

tmux a -t 33
CUDA_VISIBLE_DEVICES=2 python mainGBIL0119.py --env="Alien-v4" --training_iters=10000000 --memory_size=100000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=40 --dist_th=2 --riqi="0122-2-40-52" --expert_memory_size=5 --save_path='/home/kpl/'


MsPacmanNoFrameskip-v4

tmux a -t 34
CUDA_VISIBLE_DEVICES=4 python mainGBIL0119.py --env="MsPacmanNoFrameskip-v4" --training_iters=10000000 --memory_size=100000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=20 --dist_th=2 --riqi="0122-2-20-32" --expert_memory_size=3 --save_path='/home/kpl/'

2021年1月26日 挂掉了，重开

tmux a -t 35
CUDA_VISIBLE_DEVICES=4 python mainGBIL0119.py --env="MsPacmanNoFrameskip-v4" --training_iters=10000000 --memory_size=100000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=20 --dist_th=2 --riqi="0122-2-20-52" --expert_memory_size=5 --save_path='/home/kpl/'

tmux a -t 36
CUDA_VISIBLE_DEVICES=4 python mainGBIL0119.py --env="MsPacmanNoFrameskip-v4" --training_iters=10000000 --memory_size=100000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=40 --dist_th=2 --riqi="0122-2-40-32" --expert_memory_size=3 --save_path='/home/kpl/'

tmux a -t 37
CUDA_VISIBLE_DEVICES=4 python mainGBIL0119.py --env="MsPacmanNoFrameskip-v4" --training_iters=10000000 --memory_size=100000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=40 --dist_th=2 --riqi="0122-2-40-52" --expert_memory_size=5 --save_path='/home/kpl/'

--------------------------------------------------------------------------------
2021年1月28日

tmux a -t 6
CUDA_VISIBLE_DEVICES=3 python mainGBIL0119.py --env="MsPacmanNoFrameskip-v4" --training_iters=10000000 --memory_size=100000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=20 --dist_th=2 --riqi="0128-2-20-3" --expert_memory_size=3 --save_path='/home/kpl/'

tmux a -t 7
CUDA_VISIBLE_DEVICES=3 python mainGBIL0119.py --env="MsPacmanNoFrameskip-v4" --training_iters=10000000 --memory_size=100000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=20 --dist_th=2 --riqi="0128-2-20-5" --expert_memory_size=5 --save_path='/home/kpl/'

tmux a -t 8
CUDA_VISIBLE_DEVICES=4 python mainGBIL0119.py --env="Alien-v4" --training_iters=10000000 --memory_size=100000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=20 --dist_th=2 --riqi="0128-2-20-3" --expert_memory_size=3 --save_path='/home/kpl/'

tmux a -t 9
CUDA_VISIBLE_DEVICES=4 python mainGBIL0119.py --env="Alien-v4" --training_iters=10000000 --memory_size=100000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=20 --dist_th=2 --riqi="0128-2-20-5" --expert_memory_size=5 --save_path='/home/kpl/'