
'''
# 0106 

一旦我维护的是多幅图，那么我们在后来的路径重构中要怎么找到各个图之间的路径呢？
或者有没有其它的路径重构的方式呢

尝试让一个图带19个表，表中的index 和图中的node要有对应关系，表有容量要求的


# 0114 尝试让搜索最短路的过程并行


'''

import numpy as np
from annoy import AnnoyIndex
import networkx as nx
#from CanopyForNodeAttributes import Canopy as Cluster
import time
import multiprocessing

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

    
    def Graphisfull(self):
        for a in range(self.act_dimension):
            if self.built_capacity[a]<self.capacity:
                return False
        return True

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
        #print("len(features)",len(features))
        len_features =len(features)
        # 每个动作要对应一个即将新加的表格
        temp_indices = [[] for i in range(self.act_dimension)]
        temp_keys   = [[] for i in range(self.act_dimension)]
        encoded_trj =  []
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
            encoded_trj.append(index_in_Graph) 
            # 至此，我们完成了一个点的编码
            # 接下来把这个点写入图中     
            if i==0:# 第一个点还没有看到下一个节点
                start_node = index_in_Graph
                if start_node in self.G:
                    #print("start node re added",start_node)
                    old_visit = self.G.nodes[start_node]["visit"]
                    new_visit = old_visit + 1
                    self.G.add_node(start_node,visit=new_visit)
                else:
                    #print("start node added",start_node)
                    self.G.add_node(start_node,visit=1)
                continue
            # elif i==len_features:# 最后一个点米有下一个状态，不写了
            #     end_node = index_in_Graph
            #     if end_node in self.G:
            #         old_visit = self.G.node[end_node]["visit"]
            #         new_visit = old_visit+1
            #         self.G.add_node(end_node,visit=new_visit)
            #     else:
            #         self.G.add_node(end_node,visit=1)
            #     continue
            else: 
                # 在轨迹中间的时候，开始的点在上一轮写过，
                # 这一轮得到的是结束的点和下一个动作
                end_node = index_in_Graph
                if end_node in self.G:
                    #print("end node readded",end_node)
                    old_visit = self.G.nodes[end_node]["visit"]
                    new_visit = old_visit+1
                    self.G.add_node(end_node,visit=new_visit)
                else:
                    #print("endnode added",end_node)
                    self.G.add_node(end_node,visit=1)
                # 加入边和权重
                # self.adj_matrix[start_node][end_node][actions[i-1]] = 1 
                # self.weight_matrix[start_node][end_node][actions[i-1]] = values[i-1]
                # 如果a点到b点，既可以用动作1 到达，也可以用动作2 到达，如何处理呢
                if self.G.has_edge(start_node,end_node):
                    old_weight = self.G[start_node][end_node]["weight"]
                    new_weight = old_weight + self.eta * (values[i-1]-old_weight)# 这里用了现实交互的值更新已有的值
                    self.G.add_edge(start_node,end_node,label=actions[i-1],weight=new_weight)
                else:
                    self.G.add_edge(start_node,end_node,label=actions[i-1],weight=values[i-1])  
                    # 加入新边的时候加入了节点，但是我们后面要用到的节点属性没有加入

                start_node = end_node
                   

        # 把这条轨迹中属于各个动作的状态分别送到各自的索引表中存起来 
        for a in range(self.act_dimension):
            self._insert(a,temp_keys[a],temp_indices[a])
        self.tm += 0.01
        #print("len(encoded_trj)",len(encoded_trj))
        return  encoded_trj
        
    def add_by_indices(self,start_nodes,end_nodes,actions,weights):
        # 因为已经知道点的标号了，也就是我们并不准备在这个函数中改变图的拓扑结构，只更新边权
        #print("guiji",start_nodes,end_nodes,actions,weights)
        for i,start_node in enumerate(start_nodes):
            old_weight = weights[i]
            q_next = 0
            for edge_i in self.G.out_edges(end_nodes[i]):
                # 下一个节点上不一定有动作，也不论是什么动作，都取过来
                if len(edge_i) ==0:
                    continue
                if q_next< self.G[edge_i[0]][edge_i[1]]["weight"] :
                    q_next =self.G[edge_i[0]][edge_i[1]]["weight"] 
            new_weight = old_weight + self.eta*(q_next-old_weight) # 话说这不是自己加自己吗有啥用
            #print("qizhidian ",start_node,end_nodes[i])
            self.G[start_node][end_nodes[i]]['weight'] =new_weight
            
            #同时还要更新相应点的使用度
            a =actions[i]
            temp_index = start_node - a*self.capacity
            self.lru[a][temp_index] = self.tm  # 这里其实可以把lru 拉长 
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

    def GetKeyPointByDegree(self,num_center):
        # # 每个节点的出度，入度 加和，并除以 访问次数开根号
        # # 这只是众多可以找到关键节点的方法中的一种
        # # 先看看大家的出度入度都是什么量级的
        # print(nx.degree_histogram(self.G)) # [0, 2, 587, 31, 55, 18, 29, 15, 13, 12, 7, 3, 3, 2, 1, 1, 2, 0, 1] 是个统计图，表示出度为18的点有1个
        # # 我们可以对这个分布截尾10%，然后最低的那个作为阈值，得到相应的点，这些点的个数应该不会多
        # degree_th = len(nx.degree_histogram(self.G))*0.7
        # print(degree_th) 
        # # 把这些点的频次拿出来，除，再取前50% 然后得到的就是出现次数不是很多，但是节点连接丰富的地方
        # # 重构的时候想办法按奖励找路径，同时还要考量一下中间节点个数不能太多
        N = np.sum(self.built_capacity)
        self.SI = np.zeros(self.capacity*self.act_dimension)
        #self.SU = np.zeros(self.capacity*self.act_dimension)
        for node in self.G.nodes():
            fb = len(self.G.in_edges(node)) # before
            fa = len(self.G.out_edges(node)) # after
            ns = self.G.nodes[node]["visit"]
            si = (fa+fb)/(np.sqrt(ns/N+1e-6))
            #print("fb",fb,"fa",fa,"ns",ns,"ns/N",ns/N,"fa+fb",fa+fb,"Si",si)
            self.SI[node]= si
            #self.SU[node]= fb+fa
        Sorted_si_ind = np.argsort(self.SI,axis=0)[-num_center:-1]
        # Sorted_si = self.SI[Sorted_si_ind]
        # Sorted_su_ind = np.argsort(self.SU,axis=0)[-20:-1]
        # Sorted_su = self.SU[Sorted_su_ind]
        # print("S si ind",Sorted_si_ind,"s si",Sorted_si,"\n","s su ind",Sorted_su_ind,"s su",Sorted_su)
        center_list=list(Sorted_si_ind)
        return center_list
            # if + >degree_th:
            #     # for IN_EDGE in self.G.in_edges(node):
            #     #     print(IN_EDGE,self.G[IN_EDGE[0]][IN_EDGE[1]]["label"])
            #     # for OUT_EDGE in self.G.out_edges(node):  
            #     #     print(OUT_EDGE,self.G[OUT_EDGE[0]][OUT_EDGE[1]]["label"])
            #     print("visit",)
            #     # 入边的动作还有很大的多样性，出边的动作基本是一致的
            #     #print(node,len(self.G.in_edges(node))+len(self.G.out_edges(node)),"IN",self.G.in_edges(node),"OUT",self.G.out_edges(node))


    def GraphCluster(self,num_center):
        # 这个操作要在内存刚存满的时候，在接下来要决定谁留下来 

        cls_time_a =time.time()
        #center_list = []
        # for a in range(self.act_dimension):
        #     #print("len of a features",len(self.node_features_list[a]))
        #     gc = Cluster(self.node_features_list[a][0:self.built_capacity[a],:])#有太多的零不能挤进来，要么等满了再聚，要么就把零删了
        #     gc.setThreshold(t1,t2) 
        #     #当T1过大时，会使许多点属于多个Canopy，可能会造成各个簇的中心点间距离较近，各簇间区别不明显；
        #     #当T2过大时，增加强标记数据点的数量，会减少簇个个数；T2过小，会增加簇的个数，同时增加计算时间
        #     center_i = gc.clustering()
        #     for i in range(len(center_i)):
        #         #print("center",center_i[i][0])
        #         center_list.append(center_i[i][0] + a*self.capacity)
        center_list = self.GetKeyPointByDegree(num_center)
        print("center list",center_list)
        cls_time_b =time.time()
        print("cls time ",cls_time_b-cls_time_a)

        #print("G.NODES",self.G.nodes())
        reconstruct_time_a = time.time()

        # pool = multiprocessing.Pool(multiprocessing.cpu_count() - 2)
        
        for i in range(len(center_list)):
            for j in range(len(center_list)):
                if i==j:
                    pass
                else:
                    #print(len(self.G.nodes()),i,j)
                    self.search(center_list, i, j)
                    # pool.apply_async(self.search, (i, j))
        # pool.close()
        # pool.join()

        reconstruct_time_b =time.time()
        print("rec _time",reconstruct_time_b-reconstruct_time_a)


    def search(self, center_list, i, j):
        if nx.algorithms.shortest_paths.generic.has_path(self.G,center_list[i],center_list[j]):
            #print("reconstruct")
            path = nx.shortest_path(self.G,center_list[i],center_list[j])
            # print("path",path)
            temp_nodes=[]
            temp_actions = []
            temp_weights = []
            temp_nodes_ = []
            for idx in range(len(path)-1):
                pair_start = path[idx]
                pair_end = path[idx+1]
                temp_nodes.append(pair_start)
                temp_nodes_.append(pair_end)
                temp_actions.append(self.G.edges[pair_start,pair_end]['label'])
                temp_weights.append(self.G.edges[pair_start,pair_end]['weight'])
                #temp.append([pair_start,self.G.edges[pair_start,pair_end]['label'],self.G.edges[pair_start,pair_end]['weight'],pair_end])
                # print("temp",temp)
            self.add_by_indices(temp_nodes,temp_nodes_,temp_actions,temp_weights)
        else:
            #print("no path")
            pass

    def ReconstructGraph(self,center_list):
        print("keypoints",center_list)
        reconstruct_time_a = time.time()

        # pool = multiprocessing.Pool(multiprocessing.cpu_count() - 2)
        
        for i in range(len(center_list)):
            for j in range(len(center_list)):
                if i==j:
                    pass
                else:
                    #print(len(self.G.nodes()),i,j)
                    self.search(center_list, i, j)
                    # pool.apply_async(self.search, (i, j))
        # pool.close()
        # pool.join()

        reconstruct_time_b =time.time()
        print("rec _time",reconstruct_time_b-reconstruct_time_a)

# 用来测试这个类的
if __name__ == '__main__':
    # 用来对类内新增函数作简单测试
    G= Graph_dict(70,8,9) # 7个数据，每个数据的特征是8维，有9个动作
    # print("self.adj_matrix",G.adj_matrix.shape)
    # print("self.weight_matrix",G.weight_matrix.shape)
    # print("self.node_features ",G.node_features.shape)
    # print("self.node_attributes ",G.node_attributes.shape)

    embs = []
    acts = []
    rews = []
    for i in range(40):
        e = np.random.randn(8)
        a = np.random.randint(9)
        r = np.random.rand()
        embs.append(e)
        acts.append(a)
        rews.append(r)
    #print(embs,acts,rews)

    G.add_by_features(embs,acts,rews)
    # for i,a in enumerate(acts):
    #     adjm = G.adj_matrix
    #     weightm = G.weight_matrix
    #     print("action ",a)
    #     print(adjm[:,:,a])
    #     #print(weightm[:,:,a])
    embs = []
    acts = []
    rews = []
    for i in range(50):
        e = np.random.randn(8)
        a = np.random.randint(9)
        r = np.random.rand()
        embs.append(e)
        acts.append(a)
        rews.append(r)
    #print(embs,acts,rews)

    G.add_by_features(embs,acts,rews)
    # for i,a in enumerate(acts):
    #     adjm = G.adj_matrix
    #     weightm = G.weight_matrix
    #     print("action ",a)
    #     print(adjm[:,:,a])
    #     #print(weightm[:,:,a])
    embs = []
    acts = []
    rews = []
    for i in range(30):
        e = np.random.randn(8)
        a = np.random.randint(9)
        r = np.random.rand()
        embs.append(e)
        acts.append(a)
        rews.append(r)
    #print(embs,acts,rews)

    G.add_by_features(embs,acts,rews)
    # for i,a in enumerate(acts):
    #     adjm = G.adj_matrix
    #     weightm = G.weight_matrix
    #     print("action ",a)
    #     print(adjm[:,:,a])
    #     #print(weightm[:,:,a])

    G.GraphCluster(5,4)