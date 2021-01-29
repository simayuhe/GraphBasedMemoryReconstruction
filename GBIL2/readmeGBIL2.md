# 目的

1. 中间结果提取并显示
2. 换一种关键点的寻找方式

执行 

tmux a -t 0
CUDA_VISIBLE_DEVICES=1 python mainGBIL0127.py --env="MsPacmanNoFrameskip-v4" --training_iters=10000000 --memory_size=100000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=20 --dist_th=2 --riqi="0128-2-40-52" --expert_memory_size=10 --save_path='/data/kyx_data/GraphBasedMemoryReconstruction/GBIL2/'

obs 到 embedding的的时候少了一个观测，搞清楚少的是第一个还是最后一个

在getaction的时候把当前状态的embedding 存到trj中

在reset 中 obs 比embedding 多了一个
        self.trajectory_observations = [self.preproc(obs)]
        self.trajectory_embeddings = []

        所以他们的序号是相对应的

向expertmemory 中写的时候是每条轨迹写一次
但是我们进行重构是隔10000步重构的


这里的临时方案是对累积奖励排序，得分较高的轨迹中找到相同的点，以他们为关键点，然后进行探索
这样的结果就会让大多数轨迹在起点附近徘徊，只关注到很少的奖励
改进策略：先对探索进行鼓励，也就是得到足够多的轨迹，然后再进行评价，再进行重构；也就是先通过随机探索完成两件事情，第一是训练编码器，第二是得到覆盖空间的轨迹。

可以有两种方式鼓励探索，第一种是纯手工，就是只有探索，不利用任何信息；第二种是逐渐减小epsilon


日期：2021年1月28日
tmux a -t 10
CUDA_VISIBLE_DEVICES=5 python mainGBIL0127.py --env="MsPacmanNoFrameskip-v4" --training_iters=10000000 --memory_size=100000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=20 --dist_th=2 --riqi="0128-2-20" --expert_memory_size=10 --save_path='/home/kpl/'

tmux a -t 11
CUDA_VISIBLE_DEVICES=5 python mainGBIL0127.py --env="Alien-v4" --training_iters=10000000 --memory_size=100000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=20 --dist_th=2 --riqi="0128-2-20" --expert_memory_size=10 --save_path='/home/kpl/'

日期：2021年1月29日
tmux a -t 10
CUDA_VISIBLE_DEVICES=5 python mainGBIL0127.py --env="MsPacmanNoFrameskip-v4" --training_iters=10000000 --memory_size=100000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=20 --dist_th=2 --riqi="0128-2-20b" --expert_memory_size=10 --save_path='/home/kpl/'

tmux a -t 11
CUDA_VISIBLE_DEVICES=5 python mainGBIL0127.py --env="Alien-v4" --training_iters=10000000 --memory_size=100000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=20 --dist_th=2 --riqi="0128-2-20b" --expert_memory_size=10 --save_path='/home/kpl/'