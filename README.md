# GBMR
Graph Based Memory Reconstruction
2021年1月20日

TODO:
+ 先理顺代码的更新逻辑，确保没有问题
+ 重新回顾所有代码笔记，把记录的问题整理出来
+ 对几个游戏的特点进行分析总结，看看到底问题出现在哪儿
+ 部署ENVTEST,NEC，GQ, GQIL, GQSI
+ 建立结果分析比较平台


# 一.值迭代的基本逻辑检查

1. NEC的更新方法回顾
    + 如何计算Q值：n_steps 是一个超参数 100； 算完这个累积奖励之后就作为q-value值，都不进行改动的。但是不进行更新是如何学到东西的呢？
    + 如何更新Q值: 代码中没有更新，但是论文里有，核对它的更新公式？？
    原文用的是n-step Q,而且 其中最后一步的Q 是用最大Q做的估计
    + 如何使用Q值：从表中把return值查出来，做了一个加权，然后从各个动作给出的候选中选出一个。表示在当前状态上如果采取某个动作有可能得到的奖励有多少。given by normalised kernels between the lookup key and the corresponding key in memory.
    注意：加权的权重不是训练出来的，是根据编码的相似程度计算出来的，其实这里也是一种聚类。


2. GQ 的更新方法
    + 计算Q： 第一次见到的状态还是以n-step returns 作为基础来做，但是第二次见到的时候就是按照Q值更新的方式进行的
    + 更新Q: new_weight = old_weight + self.eta * (values[i-1]-old_weight)
    这里的weight 是n 步return 并不是Q ，这是否会影响最终结果
    这里更新Q值的同时还会更新对应的编码向量（这一点是否能够实现论文中所说缓慢更新编码，快速更新Q值，是否有必要调节两个参数的更新比例）
    + 使用Q：使用的时候还是查近邻，然后加权，做决策

3. 重构的更新方法:
    + Q的计算：仍然用的是n步return，
    + Q的更新：分为两部分
        + 交互过程中，和上面GQ是一样的更新公式，根据实际当中算出的n步return 和记录的n-步return，进行迭代。
        + 反思过程，我们自己构造了一条轨迹，这个轨迹的每个边上都有R值，那么我们该如何更新呢？
        现在是new_weight = old_weight + self.eta*(q_next-old_weight) 
        q_next 是下一个节点对应边权中最大的那个，合理吗？
    + q的使用：从表中读近邻，然后得到权值，这里的权值是经过重构更新的权值，决策


# 二.代码笔记整理

主要梳理各个版本中的遗留问题

### 20200522

+ 在maze环境跑通的是上一次的IJCAI，但是只有聚类没有图传播，也没有训练
这个代码最开始是用tvt的框架进行整理的，包括控制，读写，记忆等模块#2020年5月22日
+ 主要关注最初设计的权重更新方式，看看和现在有什么区别，是否有纰漏：
在这个版本中，直接用位置对Mspacman 进行编码。这里甚至把monster 拿出来进行编码
+ 这里其实遗留了一个问题，如何使用恰当的GNN模型训练得到关键点信息。在早期的版本中是直接用固定权值的aggregater + 聚类算法 完成的 
+ tvt 的构思逻辑还是很精巧的，有空要重读它的代码
+ 如果真想用聚类来完成图中的信息发掘，我们要多尝试一些特征图聚类的方法，不能只用简单的canopy

### 20200908

+ 这个版本主要是对maze环境评测，有一定参考价值，但不大
+ 可以用现在的方法替换原先的算法，看看是否有变化，能不能得到相似的将结果
+ networkx 这个库有更好的可以尝试换掉，主要是太慢，在求最短路，和有权值路径的时候很慢，写权重的时候又不能并行
+ 这个版本中Q的更新逻辑：因为这里存的是奖励，所以用的是一步的Q值

        def update_edge(self,state, action, w, state_):
            old_w= self.Gmemory.edges[state,state_]['weight']
            old_visits = self.Gmemory.edges[state,state_]['visits']
            if state_ != 'terminal':
                edge_list = self.Gmemory.edges(state_)
                if len(edge_list)==0:
                    #下一个状态既不是终点
                    tar = w
                else:
                    weight_list = []
                    for edge_i in edge_list:
                        weight_list.append(self.Gmemory.edges[edge_i]['weight'])
                    weight_max = np.max(weight_list)
                    tar =  w + self.gamma*weight_max
            else:
                tar = w
            delta_w = tar - old_w
            new_w = old_w + self.lr*delta_w
            self.Gmemory.add_edge(state,state_,weight=new_w,labels=action,reward=w,visits=old_visits+1)

+ 0908是maze环境中的新版本，这个版本**需要重新运行**

### 202012

最初的计划：

+ 几个需要实现的baseline算法：A3C DQN PER NEC 

+ 我们自己的算法：GBMR(所有模块都有的)，GCMR(只有聚类，没有邻居信息传播的)， GQ(基于图的Q-learning),Q-learning（NEC）。

+  最终的指标是得到10million frames的时候一个比分表格，和表现比较好的几个游戏的40millions曲线图, 和一些比较好的中间结果

中间曾经用过一个fassi的库，速度没有太大提升，弃之

+ 最开始的版本没有区分动作，所以每个节点存储的是v 不是Q 
+ pyfunction 的使用
+ 存储达到极限会怎么样，是否会影响性能，这个要根据实验结果看一下，要先知道什么时候到达极限，然后要找到对应步骤之后的效果 
+ 压缩边，动作约简，这个思路还没有尝试，我们每个节点的输出边通常都很多
+ kd-tree的实现 

1214基本能够理顺NEC的逻辑

+ 重构代码，找到会影响结果的变量，进行筛选调节
+ 训练的时候：NEC 中，target q 是环境交互过程中算出来的累积回报 ， DQN 中，直接用的是即时奖励， 这一点会不会导致结果的根本不同呢?
+ 区分一个问题，再GetAction中得到的value是查表查出来的，是个预测值，在train中用的value 是通过轨迹算出来的，是真实值。也就是往表里写的是真的，但是由于编码有误差，我们查出来的并不一定是真的，我们就是利用这个误差对编码网络进行训练。

可能存在的问题：
1. 计算逻辑： 对边权的更新方式：
    原先有一个update edge的模块，好像被舍弃了，现在实际上是有这一部分的，放在addby indices中了
2. 近邻数量：现在查的是5个，原文是50（和kd-tree的数量相同）
3. 记忆存储的多少： 原文的default 是500，000， 现在用的是100，000

### 202101

开始的思路还是要进行聚类得到keypoint，由于实现的不好临时放弃了这个想法
+ pong 一直没有效果是怎么回事儿呢？是不是截取的区域不一样呢
+ 这个pong 的参数设置有问题，他没有结束，但也没有任何效果，要尝试观察一下输出的距离，或者是不是输入就不一样。
+ 老赵写过一个并行的版本，没会用，有空问问看是咋弄的
+ slowly changing state representations and rapidly updated estimates of the value function.

+ 要求我们在更新特征表示的时候不能进行完全替代，要把相近的进行加权，这个过程只发生在写入的过程，而与交互过程无关，
 
+ Replay memory 似乎是有上限的，虽然现在没有遇到这个问题
+ 在NEC 的参考代码中有几个参考编码方案，是否会有不同
+ 我们的版本如果不求近替换会怎么样？（这样可以减少一个参数调节的过程 dist 就可以省略掉了）
+ 编码网络的训练间隔是怎么调整的呢？为啥这里选4
+ 要等到每个记忆槽中都有了一定量的样本之后再进行训练

+ history_len 是用来控制每次编码多少帧观测，这里默认4帧

+ 在NEC 算法中，所有的决策和查询都是多点同时完成的，而我们后面的思路局限于单点完成，并没有进行多点并发。比如我们在重构的时候都是根据某一个点进行的重构，而不是根据多个点来估计重构的结果
+ GQ是否能够实现论文中所说缓慢更新编码，快速更新Q值，是否有必要调节两个参数的更新比例
+ 有很多的奖励点不如我们只关注一个奖励点来得快（pvnd中的做法）
+ 关键点可以是多条轨迹上共同的且有奖励的点（或者只是共同的点，从后往前更新会导致前面的Q值增大，但是这点没关系，因为我们用的不是内部奖励，而是一个学习得到的Q值）

+ 从现有的结果看，距离指标越小，得到的曲线越稳定，虽然没有太多分数提升

+ 做重构的时候是任意起止点的，我们是否有必要把起点固定呢？就算起点到所有关键点的轨迹？？？？

+ clock问题，环境中有智能体不能控制但是经常变化的问题，我们可以通过智能体定位自身影响范围问题

发现可行的思路和改进方案

+ TVT 是一个更宏观一点儿的框架，可以考虑把记忆读写的训练部分加入到终版的模块中

+ 与环境相关的联想记忆是可以和位置相关的记忆相结合的一个点

主要参数： dist_th 1 2 3  
num_neighbours = 20 40 50
env  MsPacmanNoFrameskip-v4 Alien-v4 Pong-v4 Frostbite-v4 "HeroNoFrameskip-v4"

# 三. Atari 游戏特点分析

1. Mspacman

2:86

2. Alien

6:90

3. PingPong
18 :102 合着这个截距就是给它设计的，它的效果还不好


4. "HeroNoFrameskip-v4"
8：92


# 四. 新的实验部署方案

1. 整理代码方案

MsPacmanNoFrameskip-v4 Alien-v4

NEC 每个两组 做参考
GQ 每个种参数 2-40 2-20 每个做两组
GBIL 每个两组参数 2-40-5 2-40-3 2-20-5 2-20-3


2. 备份部署


0119
0114
都有代码在运行
要注意监控结果
还要尝试把中间结果保存成图片拿出来用

2021年1月28日 又挂掉了，这次挂的比较彻底，所有代码重启，所以，进行重新部署：

# 五. 结果比较方案

1. 单个游戏的调参对比

2. 单个游戏的多算法对比

3. 多个游戏的整体筛选

4. 相关工作算法的实现
    + DQN (不同版本的)
    + 几个需要实现的baseline算法：A3C DQN PER NEC 
    + NEC
        https://github.com/hiwonjoon/NEC/blob/master/fast_dictionary.py
        https://github.com/mjacar/pytorch-nec


