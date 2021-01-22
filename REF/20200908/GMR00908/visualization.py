#https://blog.csdn.net/monotonomo/article/details/83342768

import re

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.cm as cm

import shutil

import os

sns.set_style('whitegrid')

import numpy as np


# paths = []
# paths.append('./RESULT/Q/temp_step_r00')
# paths.append('./RESULT/GQ/temp_step_r00')
# paths.append('./RESULT/GCQ/temp_step_r00')

# data = []
# for path in paths:
#     namelist=[]
#     data =[]
#     for i in range(3):
#         name=path+str(i+1)+'.npy'
#         data.append(np.load(name))
#     datas.append(data)


Qpath = './RESULT/Q/temp_step_r00'
Q1name= Qpath + '1.npy'
Q2name= Qpath + '2.npy'
Q3name= Qpath + '3.npy'
Q4name= Qpath + '4.npy'
Q5name= Qpath + '5.npy'
Q1=np.load(Q1name)
Q2=np.load(Q2name)
Q3=np.load(Q3name)
Q4=np.load(Q4name)
Q5=np.load(Q5name)
Q1=Q1.tolist()
Q2=Q2.tolist()
Q3=Q3.tolist()
Q4=Q4.tolist()
Q5=Q5.tolist()
GQpath = './RESULT/GQ/temp_step_r00'
GQ1name= GQpath + '1.npy'
GQ2name= GQpath + '2.npy'
GQ3name= GQpath + '3.npy'
GQ4name= GQpath + '4.npy'
GQ5name= GQpath + '5.npy'
GQ1=np.load(GQ1name)
GQ2=np.load(GQ2name)
GQ3=np.load(GQ3name)
GQ4=np.load(GQ4name)
GQ5=np.load(GQ5name)
GQ1=GQ1.tolist()
GQ2=GQ2.tolist()
GQ3=GQ3.tolist()
GQ4=GQ4.tolist()
GQ5=GQ5.tolist()
GCQpath = './RESULT/GCQ/temp_step_r00'
GCQ1name= GCQpath + '1.npy'
GCQ2name= GCQpath + '2.npy'
GCQ3name= GCQpath + '3.npy'
GCQ4name= GCQpath + '4.npy'
GCQ5name= GCQpath + '5.npy'
GCQ1=np.load(GCQ1name)
GCQ2=np.load(GCQ2name)
GCQ3=np.load(GCQ3name)
GCQ4=np.load(GCQ4name)
GCQ5=np.load(GCQ5name)
GCQ1=GCQ1.tolist()
GCQ2=GCQ2.tolist()
GCQ3=GCQ3.tolist()
GCQ4=GCQ4.tolist()
GCQ5=GCQ5.tolist()
GNNpath = './RESULT/GNN/temp_step_r00'
GNN1name= GNNpath + '1.npy'
GNN2name= GNNpath + '2.npy'
GNN3name= GNNpath + '3.npy'
GNN4name= GNNpath + '4.npy'
GNN5name= GNNpath + '5.npy'
GNN1=np.load(GNN1name)
GNN2=np.load(GNN2name)
GNN3=np.load(GNN3name)
GNN4=np.load(GNN4name)
GNN5=np.load(GNN5name)
GNN1=GNN1.tolist()
GNN2=GNN2.tolist()
GNN3=GNN3.tolist()
GNN4=GNN4.tolist()
GNN5=GNN5.tolist()


meanQ=[(Q1[i]+Q2[i]+Q3[i]+Q4[i]+Q5[i])/5 for i in range(12000)]
maxQ=[max([Q1[i],Q2[i],Q3[i],Q4[i],Q5[i]]) for i in range(12000)]
minQ=[min([Q1[i],Q2[i],Q3[i],Q4[i],Q5[i]]) for i in range(12000)]

meanGQ=[(GQ1[i]+GQ2[i]+GQ3[i]+GQ4[i]+GQ5[i])/5 for i in range(12000)]
maxGQ=[max([GQ1[i],GQ2[i],GQ3[i],GQ4[i],GQ5[i]]) for i in range(12000)]
minGQ=[min([GQ1[i],GQ2[i],GQ3[i],GQ4[i],GQ5[i]]) for i in range(12000)]

meanGCQ=[(GCQ1[i]+GCQ2[i]+GCQ3[i]+GCQ4[i]+GCQ5[i])/5 for i in range(12000)]
maxGCQ=[max([GCQ1[i],GCQ2[i],GCQ3[i],GCQ4[i],GCQ5[i]]) for i in range(12000)]
minGCQ=[min([GCQ1[i],GCQ2[i],GCQ3[i],GCQ4[i],GCQ5[i]]) for i in range(12000)]

meanGNN=[(GNN1[i]+GNN2[i]+GNN3[i]+GNN4[i]+GNN5[i])/5 for i in range(12000)]
maxGNN=[max([GNN1[i],GNN2[i],GNN3[i],GNN4[i],GNN5[i]]) for i in range(12000)]
minGNN=[min([GNN1[i],GNN2[i],GNN3[i],GNN4[i],GNN5[i]]) for i in range(12000)]


f, x = plt.subplots(1,1)
x.plot(range(0,len(meanQ)), meanQ, color='goldenrod',label='Q-Learning')
x.plot(range(0,len(meanGQ)),meanGQ,color='green',label='GBRL')
x.plot(range(0,len(meanGCQ)),meanGCQ,color='red',label='CBMR')
x.plot(range(0,len(meanGNN)),meanGNN,color='blue',label='GBMR')
# # r1 = list(map(lambda x: x[0]-x[1], zip(returnavg, returnstd)))
# # r2 = list(map(lambda x: x[0]+x[1], zip(returnavg, returnstd)))
x.fill_between(range(0,len(meanQ)), maxQ, minQ, color='goldenrod', alpha=0.2)
x.fill_between(range(0,len(meanGQ)),maxGQ,minGQ, color='green',alpha=0.2)
x.fill_between(range(0,len(meanGCQ)),maxGCQ,minGCQ, color='red',alpha=0.2)
x.fill_between(range(0,len(meanGNN)),maxGNN,minGNN, color='blue',alpha=0.2)
x.legend()
x.set_xlabel('step')
x.set_ylabel('rewards')
# # exp_dir = 'Plot/'
# # if not os.path.exists(exp_dir):
# #     os.makedirs(exp_dir, exist_ok=True)
#.savefig('./RESULT/c.png', dpi=1000)
f.savefig('./RESULT/C.pdf')