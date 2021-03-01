代码执行的基本命令和执行窗口

CUDA_VISIBLE_DEVICES=7 python main.py --env="Frostbite-v4" --training_iters=50000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --riqi="0120test" 

CUDA_VISIBLE_DEVICES=7 python main.py --env="MsPacmanNoFrameskip-v4" --training_iters=50000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --riqi="0120test" --save_path='/home/kpl/'


MsPacmanNoFrameskip-v4 Alien-v4

NEC 每个两组 做参考

tmux a -t 0
CUDA_VISIBLE_DEVICES=7 python main.py --env="MsPacmanNoFrameskip-v4" --training_iters=50000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --riqi="0122-1" 

tmux a -t 39
CUDA_VISIBLE_DEVICES=7 python main.py --env="MsPacmanNoFrameskip-v4" --training_iters=50000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --riqi="0122-2" 

日期:2021年1月25日 挂掉了
问题：
22:04:16,   40000/50000000it |   4 episodes,q: 31.054, avr_ep_r: 527.5, max_ep_r: 840.0, epsilon: 0.100, entries: 0
  0%|                              | 40480/50000000 [33:24<998:48:21, 13.89it/s]terminate called after throwing an instance of 'std::system_error'
  what():  Resource temporarily unavailable

为啥？
2021年1月26日 重开，tmux 39


完全一样的参数跑两组

tmux a -t 2
CUDA_VISIBLE_DEVICES=7 python main.py --env="Alien-v4" --training_iters=50000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --riqi="0122-1" 

tmux a -t 40
CUDA_VISIBLE_DEVICES=7 python main.py --env="Alien-v4" --training_iters=50000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --riqi="0122-2" 

日期:2021年1月25日 挂掉了
同样的问题挂掉了
21:58:54,   30000/50000000it |  10 episodes,q: 105.726, avr_ep_r: 242.0, max_ep_r: 830.0, epsilon: 0.100, entries: 0
  0%|                             | 30103/50000000 [21:27<1219:39:59, 11.38it/s]tot nodes in dnd 30104
  0%|                             | 30875/50000000 [22:34<1187:57:10, 11.68it/s]tot nodes in dnd 30877
  0%|                             | 31622/50000000 [23:40<1215:45:00, 11.42it/s]tot nodes in dnd 31623
  0%|                             | 32484/50000000 [24:57<1041:18:43, 13.33it/s]tot nodes in dnd 32486
  0%|                             | 33379/50000000 [26:16<1128:53:50, 12.29it/s]tot nodes in dnd 33381
  0%|                             | 34322/50000000 [27:41<1252:55:44, 11.08it/s]tot nodes in dnd 34323
  0%|                             | 35313/50000000 [29:07<1142:50:49, 12.14it/s]tot nodes in dnd 35314
  0%|                             | 36382/50000000 [30:41<1276:30:10, 10.87it/s]tot nodes in dnd 36383
  0%|                             | 37681/50000000 [32:36<1094:27:23, 12.68it/s]terminate called after throwing an instance of 'std::system_error'
  what():  Resource temporarily unavailable
Aborted (core dumped)
2021年1月26日 重新开启

2021年1月28日 又挂掉了，这次挂的比较彻底，所有代码重启，所以，进行重新部署：每台机器上只部署一组实验

tmux a -t 0
CUDA_VISIBLE_DEVICES=0 python main.py --env="MsPacmanNoFrameskip-v4" --training_iters=50000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --riqi="0128-1" 

tmux a -t 1
CUDA_VISIBLE_DEVICES=0 python main.py --env="Alien-v4" --training_iters=50000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --riqi="0128-1"

日期 2021年1月29日 多加一组备份 在gbil 节点上,明明的结尾加了一个b,重新复制显示文件

tmux a -t 0
CUDA_VISIBLE_DEVICES=0 python main.py --env="MsPacmanNoFrameskip-v4" --training_iters=50000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --riqi="0128-1b" 

tmux a -t 1
CUDA_VISIBLE_DEVICES=0 python main.py --env="Alien-v4" --training_iters=50000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --riqi="0128-1b"


2021年2月27日
tmux a -t 0
CUDA_VISIBLE_DEVICES=0 python main.py --env="MsPacman-v4" --training_iters=50000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --riqi="0227-1b" 

tmux a -t 1
CUDA_VISIBLE_DEVICES=0 python main.py --env="Frostbite-v4" --training_iters=50000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --riqi="0227-1b"


日期： 2021年3月1日

gbil 备份， 为了区分，名称中以c 标出

tmux a -t 0
CUDA_VISIBLE_DEVICES=0 python main.py --env="MsPacman-v4" --training_iters=50000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --riqi="0227-1c" 


tmux a -t 1
CUDA_VISIBLE_DEVICES=0 python main.py --env="Frostbite-v4" --training_iters=50000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --riqi="0227-1c"


tmux a -t 21
CUDA_VISIBLE_DEVICES=2 python main.py --env="Hero-v4" --training_iters=50000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --riqi="0227-1c"
