这是一个将GQ 和 IL 放在一起，用较好的轨迹中的点来替代原本方案中的聚类中心
多了一个参数，--expert_memory_size=3
我们要存多少条获得奖励最高的轨迹

CUDA_VISIBLE_DEVICES=7 python mainGBIL0119.py --env="Alien-v4" --training_iters=10000000 --memory_size=100000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=20 --dist_th=2 --riqi="0119-2-20-3" --expert_memory_size=3