CUDA_VISIBLE_DEVICES=3 python mainGQ.py --env="Alien-v4" --training_iters=10000000 --memory_size=100000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=40 --dist_th=2 --riqi="0118-2-40"

CUDA_VISIBLE_DEVICES=7 python mainGQ.py --env="MsPacmanNoFrameskip-v4" --training_iters=10000000 --memory_size=100000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=40 --dist_th=2 --riqi="0120-2-40"


MsPacmanNoFrameskip-v4 中比较好的参数 2-20 2-40 但是到后来4million之后都趋于平稳的状态了。可能跟记忆的数量有关，分数也与原文差了很多，所以要设计一个机制，一边训练，一边保存成功轨迹，看看到底学到了什么。

Alien-v4 中比较好的是 2-20 ，3million 的得分700 左右，2-40 还在努力中，后来的分数也会趋于平稳



GQ 每个游戏两个参数 2-40 2-20 每个做两组

在 0112种有的：

tmux a -t 3
CUDA_VISIBLE_DEVICES=0 python mainGQ.py --env="MsPacmanNoFrameskip-v4" --training_iters=10000000 --memory_size=100000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=20 --dist_th=2 --riqi="0114-2-20" 

tmux a -t 4
CUDA_VISIBLE_DEVICES=0 python mainGQ.py --env="MsPacmanNoFrameskip-v4" --training_iters=10000000 --memory_size=100000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=20 --dist_th=4 --riqi="0112-4-20"

tmux a -t 5
CUDA_VISIBLE_DEVICES=0 python mainGQ.py --env="MsPacmanNoFrameskip-v4" --training_iters=10000000 --memory_size=100000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=40 --dist_th=2 --riqi="0118-2-40"

alien :

tmux a -t 8
CUDA_VISIBLE_DEVICES=1 python mainGQ.py --env="Alien-v4" --training_iters=10000000 --memory_size=100000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=20 --dist_th=4 --riqi="0112-4-20"ti

tmux a -t 15

CUDA_VISIBLE_DEVICES=3 python mainGQ.py --env="Alien-v4" --training_iters=10000000 --memory_size=100000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=20 --dist_th=2 --riqi="0114-2-20"


tmux a -t 17

CUDA_VISIBLE_DEVICES=3 python mainGQ.py --env="Alien-v4" --training_iters=10000000 --memory_size=100000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=40 --dist_th=2 --riqi="0118-2-40"

应该可以再加两组

ms ：

tmux a -t 6
CUDA_VISIBLE_DEVICES=0 python mainGQ.py --env="MsPacmanNoFrameskip-v4" --training_iters=10000000 --memory_size=100000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=20 --dist_th=2 --riqi="0122-2-20-2" --save_path='/home/kpl/'

tmux a -t 9
CUDA_VISIBLE_DEVICES=0 python mainGQ.py --env="MsPacmanNoFrameskip-v4" --training_iters=10000000 --memory_size=100000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=40 --dist_th=2 --riqi="0118-2-40-2" --save_path='/home/kpl/'  

这个名字有笔误


alien:
tmux a -t 10
CUDA_VISIBLE_DEVICES=3 python mainGQ.py --env="Alien-v4" --training_iters=10000000 --memory_size=100000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=20 --dist_th=2 --riqi="0122-2-20-2"  --save_path='/home/kpl/'
挂掉了


tmux a -t 11
CUDA_VISIBLE_DEVICES=3 python mainGQ.py --env="Alien-v4" --training_iters=10000000 --memory_size=100000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=40 --dist_th=2 --riqi="0122-2-40-2" --save_path='/home/kpl/'
挂掉l