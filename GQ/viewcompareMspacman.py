import numpy as np
import matplotlib.pyplot as plt
T=600
a=20

def average(M,a,T):
    t = M[0:T]
    for i in range(a):
        t=t+M[i:T+i]
    return t/a


name="GQ0114-45-50MsPacmanNoFrameskip-v4aver"
M45=np.load(name+".npy")
m45 = average(M45,a,T)
name1 ="GQ0114-5-40MsPacmanNoFrameskip-v4aver"
M5=np.load(name1+".npy")
m5 = average(M5,a,T)
name3 ="GQ0114-4-50MsPacmanNoFrameskip-v4aver"
M4=np.load(name3+".npy")
m4 = average(M4,a,T)
name4 = "GQ0114-2-20MsPacmanNoFrameskip-v4aver"
M2=np.load(name4+".npy")
m2 = average(M2,a,T)
name5 = "GQ0112-4-20MsPacmanNoFrameskip-v4aver"
M420=np.load(name5+".npy")
m420 = average(M420,a,T)
name6 = "GQ0112-5-20MsPacmanNoFrameskip-v4aver"
M520=np.load(name6+".npy")
m520 = average(M520,a,T)
name7 = "GQ0118-2-40MsPacmanNoFrameskip-v4aver"
M240=np.load(name7+".npy")
m240 = average(M240,a,T)

l1, = plt.plot(range(T),m45,label="14-45-50",color="r")
l2, = plt.plot(range(T),m5,label="14-5-40",color="g")
l3, = plt.plot(range(T),m4,label="14-4-50",color="b")
l4, = plt.plot(range(T),m2,label="14-2-20",color="c")
l5, = plt.plot(range(T),m420,label="12-4-20",color="m")
l6, = plt.plot(range(T),m520,label="12-5-20",color="y")
l7, = plt.plot(range(T),m240,label="18-2-40",color="k")

plt.legend([l1,l2,l3,l4,l5,l6,l7],["14-45-50","14-5-40","14-4-50","14-2-20","14-4-20","12-5-20","18-2-40"],loc = 'upper right')
plt.savefig("mspacmancompare"+".png")
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

