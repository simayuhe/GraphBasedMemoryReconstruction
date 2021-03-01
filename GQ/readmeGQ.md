CUDA_VISIBLE_DEVICES=3 python mainGQ.py --env="Alien-v4" --training_iters=10000000 --memory_size=100000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=40 --dist_th=2 --riqi="0118-2-40"

CUDA_VISIBLE_DEVICES=7 python mainGQ.py --env="MsPacmanNoFrameskip-v4" --training_iters=10000000 --memory_size=100000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=40 --dist_th=2 --riqi="0120-2-40"


MsPacmanNoFrameskip-v4 中比较好的参数 2-20 2-40 但是到后来4million之后都趋于平稳的状态了。可能跟记忆的数量有关，分数也与原文差了很多，所以要设计一个机制，一边训练，一边保存成功轨迹，看看到底学到了什么。

Alien-v4 中比较好的是 2-20 ，3million 的得分700 左右，2-40 还在努力中，后来的分数也会趋于平稳



GQ 每个游戏两个参数 2-40 2-20 每个做两组

在 0112种有的：

tmux a -t 3
CUDA_VISIBLE_DEVICES=0 python mainGQ.py --env="MsPacmanNoFrameskip-v4" --training_iters=10000000 --memory_size=100000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=20 --dist_th=2 --riqi="0114-2-20" 

结束了 20210126

tmux a -t 4
CUDA_VISIBLE_DEVICES=0 python mainGQ.py --env="MsPacmanNoFrameskip-v4" --training_iters=10000000 --memory_size=100000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=20 --dist_th=4 --riqi="0112-4-20"

结束了 20210126


tmux a -t 5
CUDA_VISIBLE_DEVICES=0 python mainGQ.py --env="MsPacmanNoFrameskip-v4" --training_iters=10000000 --memory_size=100000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=40 --dist_th=2 --riqi="0118-2-40"

还在跑 20210126

alien :

tmux a -t 8
CUDA_VISIBLE_DEVICES=1 python mainGQ.py --env="Alien-v4" --training_iters=10000000 --memory_size=100000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=20 --dist_th=4 --riqi="0112-4-20"ti
结束了 2021年1月26日

tmux a -t 15

CUDA_VISIBLE_DEVICES=3 python mainGQ.py --env="Alien-v4" --training_iters=10000000 --memory_size=100000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=20 --dist_th=2 --riqi="0114-2-20"
结束了 2021年1月26日

tmux a -t 17

CUDA_VISIBLE_DEVICES=3 python mainGQ.py --env="Alien-v4" --training_iters=10000000 --memory_size=100000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=40 --dist_th=2 --riqi="0118-2-40"

还在跑 4/10 2021年1月26日

应该可以再加两组

ms ：

tmux a -t 6
CUDA_VISIBLE_DEVICES=0 python mainGQ.py --env="MsPacmanNoFrameskip-v4" --training_iters=10000000 --memory_size=100000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=20 --dist_th=2 --riqi="0122-2-20-2" --save_path='/home/kpl/'

tmux a -t 9
CUDA_VISIBLE_DEVICES=0 python mainGQ.py --env="MsPacmanNoFrameskip-v4" --training_iters=10000000 --memory_size=100000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=40 --dist_th=2 --riqi="0118-2-40-2" --save_path='/home/kpl/'  

这个名字有笔误 22日开始的


alien:
tmux a -t 38
CUDA_VISIBLE_DEVICES=1 python mainGQ.py --env="Alien-v4" --training_iters=10000000 --memory_size=100000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=20 --dist_th=2 --riqi="0122-2-20-2"  --save_path='/home/kpl/'
挂掉了
2021年1月26日 继续 感觉2-20 还是有效果的


tmux a -t 11
CUDA_VISIBLE_DEVICES=1 python mainGQ.py --env="Alien-v4" --training_iters=10000000 --memory_size=100000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=40 --dist_th=2 --riqi="0122-2-40-2" --save_path='/home/kpl/'
挂掉l

2021年1月26日 0112文件夹中的还没有结果，这里再保存一组,仍然用原先的tmux 编号

_______________________________________________________________________
2021年1月28日
tmux a -t 2
CUDA_VISIBLE_DEVICES=1 python mainGQ.py --env="MsPacmanNoFrameskip-v4" --training_iters=10000000 --memory_size=100000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=20 --dist_th=2 --riqi="0128-2-20" 

tmux a -t 3
CUDA_VISIBLE_DEVICES=1 python mainGQ.py --env="MsPacmanNoFrameskip-v4" --training_iters=10000000 --memory_size=100000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=2 --riqi="0128-2-50"

tmux a -t 4
CUDA_VISIBLE_DEVICES=2 python mainGQ.py --env="Alien-v4" --training_iters=10000000 --memory_size=100000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=20 --dist_th=2 --riqi="0128-2-20"  --save_path='/home/kpl/'

tmux a -t 5
CUDA_VISIBLE_DEVICES=2 python mainGQ.py --env="Alien-v4" --training_iters=10000000 --memory_size=100000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=2 --riqi="0128-2-50" --save_path='/home/kpl/'

______________________________________________________________________________

2021年1月29日
tmux a -t 2
CUDA_VISIBLE_DEVICES=1 python mainGQ.py --env="MsPacmanNoFrameskip-v4" --training_iters=10000000 --memory_size=100000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=20 --dist_th=2 --riqi="0128-2-20b" 

tmux a -t 3
CUDA_VISIBLE_DEVICES=1 python mainGQ.py --env="MsPacmanNoFrameskip-v4" --training_iters=10000000 --memory_size=100000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=2 --riqi="0128-2-50b"

tmux a -t 4
CUDA_VISIBLE_DEVICES=2 python mainGQ.py --env="Alien-v4" --training_iters=10000000 --memory_size=100000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=20 --dist_th=2 --riqi="0128-2-20b"  --save_path='/home/kpl/'

tmux a -t 5
CUDA_VISIBLE_DEVICES=2 python mainGQ.py --env="Alien-v4" --training_iters=10000000 --memory_size=100000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=2 --riqi="0128-2-50b" --save_path='/home/kpl/'


2021年2月27日
tmux a -t 2
CUDA_VISIBLE_DEVICES=1 python mainGQ.py --env="MsPacman-v4" --training_iters=10000000 --memory_size=100000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=20 --dist_th=2 --riqi="0227-2-20b" 

tmux a -t 3
CUDA_VISIBLE_DEVICES=1 python mainGQ.py --env="MsPacman-v4" --training_iters=10000000 --memory_size=100000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=2 --riqi="0227-2-50b"

tmux a -t 4
CUDA_VISIBLE_DEVICES=2 python mainGQ.py --env="Frostbite-v4" --training_iters=10000000 --memory_size=100000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=20 --dist_th=2 --riqi="0227-2-20b"  --save_path='/home/kpl/'

tmux a -t 5
CUDA_VISIBLE_DEVICES=2 python mainGQ.py --env="Frostbite-v4" --training_iters=10000000 --memory_size=100000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=2 --riqi="0227-2-50b" --save_path='/home/kpl/'


日期 2021年3月1日
gbil

tmux a -t 2
CUDA_VISIBLE_DEVICES=1 python mainGQ.py --env="MsPacman-v4" --training_iters=10000000 --memory_size=100000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=20 --dist_th=2 --riqi="0227-2-20c" 
tmux a -t 3
CUDA_VISIBLE_DEVICES=1 python mainGQ.py --env="MsPacman-v4" --training_iters=10000000 --memory_size=100000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=2 --riqi="0227-2-50c"

tmux a -t 4
CUDA_VISIBLE_DEVICES=2 python mainGQ.py --env="Frostbite-v4" --training_iters=10000000 --memory_size=100000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=20 --dist_th=2 --riqi="0227-2-20c"  --save_path='/home/kpl/'

tmux a -t 5
CUDA_VISIBLE_DEVICES=2 python mainGQ.py --env="Frostbite-v4" --training_iters=10000000 --memory_size=100000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=2 --riqi="0227-2-50c" --save_path='/home/kpl/'

tmux a -t 22
CUDA_VISIBLE_DEVICES=2 python mainGQ.py --env="Hero-v4" --training_iters=10000000 --memory_size=100000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=2 --riqi="0227-2-50c" --save_path='/home/kpl/'


gbmr

tmux a -t 23
CUDA_VISIBLE_DEVICES=3 python mainGQ.py --env="Hero-v4" --training_iters=10000000 --memory_size=100000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=2 --riqi="0227-2-50b" --save_path='/home/kpl/'