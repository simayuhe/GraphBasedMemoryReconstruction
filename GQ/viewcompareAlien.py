import numpy as np
import matplotlib.pyplot as plt
T=600
a=50

def average(M,a,T):
    t = M[0:T]
    for i in range(a):
        t=t+M[i:T+i]
    return t/a


name="GQ0112-45-20Alien-v4aver"
M4520=np.load(name+".npy")
m4520 = average(M4520,a,T)

name1="GQ0112-4-20Alien-v4aver"
M420=np.load(name1+".npy")
m420 = average(M420,a,T)

name2="GQ0112-5-20Alien-v4aver"
M520=np.load(name2+".npy")
m520 = average(M520,a,T)

name3="GQ0114-2-20Alien-v4aver"
M220=np.load(name3+".npy")
m220 = average(M220,a,T)

name4="GQ0118-2-40Alien-v4aver"
M240=np.load(name4+".npy")
m240 = average(M240,a,T)

l1, = plt.plot(range(T),m4520,label="12-45-20",color="r")
l2, = plt.plot(range(T),m420,label="12-4-20",color="g")
l3, = plt.plot(range(T),m520,label="12-5-20",color="b")
l4, = plt.plot(range(T),m220,label="14-2-20",color="c")
l5, = plt.plot(range(T),m240,label="18-2-40",color="m")

plt.legend([l1,l2,l3,l4,l5],["12-45-20","12-4-20","12-5-20","14-2-20","18-2-40"],loc = 'upper right')
plt.savefig("aliencompare"+".png")
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

