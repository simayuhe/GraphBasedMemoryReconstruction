
'''
# 0106 



一旦我维护的是多幅图，那么我们在后来的路径重构中要怎么找到各个图之间的路径呢？
或者有没有其它的路径重构的方式呢

尝试让一个图带19个表，表中的index 和图中的node要有对应关系，表有容量要求的


'''

import numpy as np
from annoy import AnnoyIndex
import networkx as nx


class Graph_dict():
    def __init__(self,capacity,key_dimension,act_dimension,dist_th,batch_size=8):
        # 一个图带19个表，每个表是一种动作的特征集合
        self.capacity = capacity # 每个表的容量
        self.key_dimension = key_dimension
        self.act_dimension = act_dimension
        self.G = nx.DiGraph()
        #self.node_features = np.zeros((capacity,key_dimension))
        self.node_features_list = [np.zeros((capacity,key_dimension)) for i in range(self.act_dimension)]
        self.indices_list = [AnnoyIndex(key_dimension, metric='euclidean') for i in range(self.act_dimension)] # 用来给节点特征快速查近邻的
        #self.index.set_seed(123)
        # 暂时不知道该不该重新写
        self.initial_update_size = batch_size
        self.min_update_size = self.initial_update_size
        self.cached_keys = [[] for i in range(self.act_dimension)]
        self.cached_indices =  [[] for i in range(self.act_dimension)]
        # 下面两个用来给节点的使用频率计数的
        self.lru = [np.zeros(self.capacity) for i in range(self.act_dimension)]
        self.tm = 0.0
        self.curr_capacity = [0 for i in range(self.act_dimension)] # 是图的容量
        self.built_capacity = [0 for i in range(self.act_dimension)]
        #self.dist_th =4
        self.eta = 0.1 # 控制边权的增长速度
        self.dist_th = dist_th


    def queryable(self,act,k):
        # 返回 动作act 对应的表中是否有k 个值供查询
        if self.curr_capacity[act]>k and self.built_capacity[act]>k:
            return True
        else:
            return False    

    def _insert(self, a, keys, indices):
        self.cached_keys[a] = self.cached_keys[a] + keys
        # self.cached_values = self.cached_values + values
        self.cached_indices[a] = self.cached_indices[a] + indices

        if len(self.cached_indices[a]) >= self.min_update_size:# 为啥设置这个阈值
            self.min_update_size = max(self.initial_update_size, self.curr_capacity[a]*0.02)
            self._update_index(a)

    def _update_index(self,a):
        #self.index.unbuild()
        self.indices_list[a].unbuild()
        for i, ind in enumerate(self.cached_indices[a]):
            new_key = self.cached_keys[a][i]
            # new_value = self.cached_values[i]
            # self.embeddings[ind] = new_key
            # self.values[ind] = new_value
            self.node_features_list[a][ind,:] = new_key
            self.indices_list[a].add_item(ind,new_key)
            #self.index.add_item(ind, new_key)

        self.cached_keys[a] = []
        #self.cached_values = []
        self.cached_indices[a] = []

        self.indices_list[a].build(50)
        #self.index.build(50)
        self.built_capacity[a] = self.curr_capacity[a]

    def add_by_features(self,features,actions,values):
        len_features =len(features)
        # 每个动作要对应一个即将新加的表格
        temp_indices = [[] for i in range(self.act_dimension)]
        temp_keys   = [[] for i in range(self.act_dimension)]
        for i in range(len_features):
            f = features[i]
            a = actions[i]
            if self.queryable(a,1): # 每个表里有值
                ind,dist = self.indices_list[a].get_nns_by_vector(f,1,include_distances=True)
                d= dist[0]
                #print(d)
                # print("self.indices_list[a].get_n_items()",self.indices_list[a].get_n_items())
                # print("self.cur_capacity[a]",self.curr_capacity[a])
                #print(d)
                if d < self.dist_th:
                    temp_index = ind[0]
                    key_ = (self.indices_list[a].get_item_vector(temp_index) + f*d)/(1+d)
                #elif self.indices_list[a].get_n_items() > self.capacity: #问题可能出现在等于的时候
                elif self.curr_capacity[a] >= self.capacity: 
                    temp_index = np.argmin(self.lru[a])# + a*self.capacity # 序号的编码要和图一致 
                    self.G.remove_node(temp_index + a*self.capacity)# 序号的编码要和图一致 
                    key_ = f
                else:
                    temp_index = self.curr_capacity[a]
                    self.curr_capacity[a] += 1
                    key_ = f
            else: #原先是个空表
                temp_index = self.curr_capacity[a]
                key_ = f
                self.curr_capacity[a] += 1
            # print("temp_index",temp_index)
            self.lru[a][temp_index] = self.tm
            temp_indices[a].append(temp_index)
            temp_keys[a].append(key_)

            index_in_Graph = temp_index + a*self.capacity  
            # 至此，我们完成了一个点的编码
            # 接下来把这个点写入图中     
            if i==0:# 第一个点还没有看到下一个节点
                start_node = index_in_Graph
                continue
            elif i==len_features:# 最后一个点米有下一个状态，不写了
                continue
            else: 
                # 在轨迹中间的时候，开始的点在上一轮写过，
                # 这一轮得到的是结束的点和下一个动作
                end_node = index_in_Graph
                # 加入边和权重
                # self.adj_matrix[start_node][end_node][actions[i-1]] = 1 
                # self.weight_matrix[start_node][end_node][actions[i-1]] = values[i-1]
                # 如果a点到b点，既可以用动作1 到达，也可以用动作2 到达，如何处理呢
                if self.G.has_edge(start_node,end_node):
                    old_weight = self.G[start_node][end_node]["weight"]
                    new_weight = old_weight + self.eta * (values[i-1]-old_weight)
                    self.G.add_edge(start_node,end_node,label=actions[i-1],weight=new_weight)
                else:
                    self.G.add_edge(start_node,end_node,label=actions[i-1],weight=values[i-1])
                start_node = end_node   

        # 把这条轨迹中属于各个动作的状态分别送到各自的索引表中存起来 
        for a in range(self.act_dimension):
            self._insert(a,temp_keys[a],temp_indices[a])
        self.tm += 0.01

    def query_by_features(self,features,actions,k):
        Q_s = np.zeros((len(features),k))
        Embs = np.zeros((len(features),k,self.key_dimension))
        for i,f in enumerate(features):
            a = actions[i]
            inds,dists = self.indices_list[a].get_nns_by_vector(f,k,include_distances= True)
            for j,ind in enumerate(inds):
                index = ind + a*self.capacity
                for edge_i in self.G.out_edges(index):
                    if len(edge_i)==0:
                        continue
                    if self.G[edge_i[0]][edge_i[1]]["label"] == a: # 如果边的类型是a ，我们读取相应的Q值
                        Q_s[i,j] = self.G[edge_i[0]][edge_i[1]]["weight"] 
            Embs[i,:,:] = self.node_features_list[a][inds,:]
            self.lru[a][inds] = self.tm 
        self.tm += 0.01

        return Q_s,Embs   