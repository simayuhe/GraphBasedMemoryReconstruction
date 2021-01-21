代码执行的基本命令和执行窗口

CUDA_VISIBLE_DEVICES=7 python main.py --env="Frostbite-v4" --training_iters=50000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --riqi="0120test" 

CUDA_VISIBLE_DEVICES=7 python main.py --env="MsPacmanNoFrameskip-v4" --training_iters=50000000 --memory_size=500000  --epsilon=0.1  --display_step=10000 --learn_step=4 --num_neighbours=50 --riqi="0120test" --save_path='/home/kpl/'