import numpy as np
import matplotlib.pyplot as plt
T=600
a=50

def average(M,a,T):
    t = M[0:T]
    for i in range(a):
        t=t+M[i:T+i]
    return t/a


name="GQ0114-08-50Pong-v4aver"
M0850=np.load(name+".npy")
m0850 = average(M0850,a,T)

name1="GQ0112-1-20Pong-v4aver"
M120=np.load(name1+".npy")
m120 = average(M120,a,T)

name2="GQ0112-06-20Pong-v4aver"
M0620=np.load(name2+".npy")
m0620 = average(M0620,a,T)

l1, = plt.plot(range(T),m0850,label="14-08-50",color="r")
l2, = plt.plot(range(T),m120,label="12-1-20",color="g")
l3, = plt.plot(range(T),m0620,label="12-06-20",color="b")

plt.legend([l1,l2,l3],["14-08-50","12-1-20","12-06-20"],loc = 'upper right')
plt.savefig("pongcompare"+".png")
plt.close()


# 标记符    颜色
# r          红
# g          绿
# b          蓝
# c          蓝绿
# m          紫红
# y          黄
# k          黑
# w          白

