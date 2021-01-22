代码执行的基本命令和执行窗口

CUDA_VISIBLE_DEVICES=7 python main.py --env="Frostbite-v4" --training_iters=50000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --riqi="0120test" 

CUDA_VISIBLE_DEVICES=7 python main.py --env="MsPacmanNoFrameskip-v4" --training_iters=50000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --riqi="0120test" --save_path='/home/kpl/'


MsPacmanNoFrameskip-v4 Alien-v4

NEC 每个两组 做参考

tmux a -t 0
CUDA_VISIBLE_DEVICES=7 python main.py --env="MsPacmanNoFrameskip-v4" --training_iters=50000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --riqi="0122-1" 

tmux a -t 1
CUDA_VISIBLE_DEVICES=7 python main.py --env="MsPacmanNoFrameskip-v4" --training_iters=50000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --riqi="0122-2" 

完全一样的参数跑两组

tmux a -t 2
CUDA_VISIBLE_DEVICES=7 python main.py --env="Alien-v4" --training_iters=50000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --riqi="0122-1" 

tmux a -t 29
CUDA_VISIBLE_DEVICES=7 python main.py --env="Alien-v4" --training_iters=50000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --riqi="0122-2" 