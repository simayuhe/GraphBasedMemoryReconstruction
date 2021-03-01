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
————————————————————————————————————————————————————————————————————————————————————————————
——————————————————————————————————————————————————————————————————————————————————————

2021年2月25日

目标：得到一个可行的方案，以nec已有数据为基础进行比较，并部署到更多的雅达利游戏上，

原有方案：先对探索进行鼓励，也就是得到足够多的轨迹，然后再进行评价，再进行重构；也就是先通过随机探索完成两件事情，第一是训练编码器，第二是得到覆盖空间的轨迹。

可以有两种方式鼓励探索，第一种是纯手工，就是只有探索，不利用任何信息；第二种是逐渐减小epsilon

存在的问题：
+ 刚开始的时候随机得到的可能有累计奖励较高的，但是后来随着学习的进步，随机得到的奖励就不会被排到关键状态的索取中了，所以就没啥用了
        + 可以尝试记录当前最好成绩的步数，在到达这些步数的0.9的时候开始进行随机探索
        + 记录每条轨迹所用step, 把它写到expertmemory 中，排序的时候作为一个附加量，进行更新
+ 我们每次都走那几条路线，导致刚开始的起点附近都是重叠的点，这件事情会被不断地加强，然后形成（思维定势）。
        + 关键点做成堆栈，每次都只用最后的几个
        + 或者可以给它附加步数，只用步数较远的那些
+ 越往后，我们提取到的关键状态反而越少，而且就在开始的两个点，这是为啥
        + ？？？
+ 还有个疑问，那个nofreamskip 和普通的有啥区别，要存几张实验图看看
        + https://blog.csdn.net/qq_27008079/article/details/100126060
        + 表示每4帧执行一个动作，所以这里不应该用noframeskip, 

改进思路：

1. 将随机探索建立在当前已有的策略之上，但是这件事情要如何完成呢？
+ 现在的做法是探索步骤超过95%平均步骤的时候就进行随机探索，然后把总得分较高的轨迹拿出来，对他们的交叉点进行重构
+ 这么做的原因是：针对迷宫探索且时刻都有奖励的游戏，接近尾声的时候通常是接近死亡的时候，所以与其让他按照剧本进行探索，不如随机走，有可能发现新的机会。对蒙特祖玛那种稀疏奖励的游戏可能并不好用。

把出度入度加和比较高的状态作为关键状态
+ 还没实现


CUDA_VISIBLE_DEVICES=0 python mainGBIL0225.py --env="MsPacman-v4" --training_iters=10000000 --memory_size=100000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=20 --dist_th=2 --riqi="0225-2-20" --expert_memory_size=10 --save_path='/data/kyx_data/'

CUDA_VISIBLE_DEVICES=0 python mainGBIL0225.py --env="MsPacman-v4" --training_iters=30000 --memory_size=10000 --epsilon=0.1 --display_step=3000 --learn_step=4 --num_neighbours=20 --dist_th=2 --riqi="0225-2-20" --expert_memory_size=10 --save_path='/data/kyx_data/'

bowling
frostbite
hero

2021年2月27日
tmux a -t 14
CUDA_VISIBLE_DEVICES=6 python mainGBIL0225.py --env="MsPacman-v4" --training_iters=10000000 --memory_size=100000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=20 --dist_th=2 --riqi="0227-2-20b" --expert_memory_size=10 --save_path='/home/kpl/'

tmux a -t 17
CUDA_VISIBLE_DEVICES=6 python mainGBIL0225.py --env="MsPacman-v4" --training_iters=10000000 --memory_size=100000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=2 --riqi="0227-2-50b" --expert_memory_size=10 --save_path='/home/kpl/'

tmux a -t 18
CUDA_VISIBLE_DEVICES=7 python mainGBIL0225.py --env="Alien-v4" --training_iters=10000000 --memory_size=100000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=20 --dist_th=2 --riqi="0227-2-20b" --expert_memory_size=10 --save_path='/home/kpl/'

tmux a -t 19
CUDA_VISIBLE_DEVICES=7 python mainGBIL0225.py --env="Alien-v4" --training_iters=10000000 --memory_size=100000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=2 --riqi="0227-2-50b" --expert_memory_size=10 --save_path='/home/kpl/'

tmux a -t 20
CUDA_VISIBLE_DEVICES=7 python mainGBIL0225.py --env="Frostbite-v4" --training_iters=10000000 --memory_size=100000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=20 --dist_th=2 --riqi="0227-2-20b" --expert_memory_size=10 --save_path='/home/kpl/'

tmux a -t 21
CUDA_VISIBLE_DEVICES=7 python mainGBIL0225.py --env="Frostbite-v4" --training_iters=10000000 --memory_size=100000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=2 --riqi="0227-2-50b" --expert_memory_size=10 --save_path='/home/kpl/'\

2021年3月1日
gbil

tmux a -t 13
CUDA_VISIBLE_DEVICES=6 python mainGBIL0225.py --env="MsPacman-v4" --training_iters=10000000 --memory_size=100000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=20 --dist_th=2 --riqi="0227-2-20c" --expert_memory_size=10 --save_path='/home/kpl/' 

tmux a -t 14
CUDA_VISIBLE_DEVICES=6 python mainGBIL0225.py --env="MsPacman-v4" --training_iters=10000000 --memory_size=100000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=2 --riqi="0227-2-50c" --expert_memory_size=10 --save_path='/home/kpl/'

tmux a -t 15
CUDA_VISIBLE_DEVICES=7 python mainGBIL0225.py --env="Alien-v4" --training_iters=10000000 --memory_size=100000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=20 --dist_th=2 --riqi="0227-2-20c" --expert_memory_size=10 --save_path='/home/kpl/'

tmux a -t 16
CUDA_VISIBLE_DEVICES=7 python mainGBIL0225.py --env="Alien-v4" --training_iters=10000000 --memory_size=100000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=2 --riqi="0227-2-50c" --expert_memory_size=10 --save_path='/home/kpl/'

tmux a -t 17
CUDA_VISIBLE_DEVICES=7 python mainGBIL0225.py --env="Frostbite-v4" --training_iters=10000000 --memory_size=100000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=20 --dist_th=2 --riqi="0227-2-20c" --expert_memory_size=10 --save_path='/home/kpl/'

tmux a -t 18
CUDA_VISIBLE_DEVICES=7 python mainGBIL0225.py --env="Frostbite-v4" --training_iters=10000000 --memory_size=100000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=2 --riqi="0227-2-50c" --expert_memory_size=10 --save_path='/home/kpl/'

tmux a -t 23
CUDA_VISIBLE_DEVICES=3 python mainGBIL0225.py --env="Hero-v4" --training_iters=10000000 --memory_size=100000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=2 --riqi="0227-2-50c" --expert_memory_size=10 --save_path='/home/kpl/'


gbmr


tmux a -t 24
CUDA_VISIBLE_DEVICES=4 python mainGBIL0225.py --env="Hero-v4" --training_iters=10000000 --memory_size=100000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=2 --riqi="0227-2-50b" --expert_memory_size=10 --save_path='/home/kpl/'