import numpy as np
import matplotlib.pyplot as plt
T=600

a=np.load("GQ0112-1-20Pong-v4aver.npy")
plt.plot(range(T),a[0:T])
plt.savefig("GQ0112-1-20Pong-v4aver.png")
plt.close()

b=np.load("GQ0112-2-20Frostbite-v4aver.npy")
plt.plot(range(T),b[0:T])
plt.savefig("GQ0112-2-20Frostbite-v4aver.png")
plt.close()

c=np.load("GQ0112-2-20Pong-v4aver.npy")
plt.plot(range(T),c[0:T])
plt.savefig("GQ0112-2-20Pong-v4aver.png")
plt.close()


d=np.load("GQ0112-3-20Frostbite-v4aver.npy")
plt.plot(range(T),d[0:T])
plt.savefig("GQ0112-3-20Frostbite-v4aver.png")
plt.close()


e=np.load("GQ0112-4-10MsPacmanNoFrameskip-v4aver.npy")
plt.plot(range(T),e[0:T])
plt.savefig("GQ0112-4-10MsPacmanNoFrameskip-v4aver.png")
plt.close()

name ="GQ0112-4-20Alien-v4aver"
f=np.load(name+".npy")
plt.plot(range(T),f[0:T])
plt.savefig(name+".png")
plt.close()

name ="GQ0112-4-20MsPacmanNoFrameskip-v4aver"
g=np.load(name+".npy")
plt.plot(range(T),g[0:T])
plt.savefig(name+".png")
plt.close()



name ="GQ0112-5-20Alien-v4aver"
h=np.load(name+".npy")
plt.plot(range(T),h[0:T])
plt.savefig(name+".png")
plt.close()

#tmux a -t 15
name ="GQ0114-2-20Alien-v4aver"
hi=np.load(name+".npy")
plt.plot(range(T),hi[0:T])
plt.savefig(name+".png")
plt.close()

name ="GQ0112-45-20Alien-v4aver"
i=np.load(name+".npy")
plt.plot(range(T),i[0:T])
plt.savefig(name+".png")
plt.close()

name ="GQ0112-06-20Pong-v4aver"
l=np.load(name+".npy")
plt.plot(range(T),l[0:T])
plt.savefig(name+".png")
plt.close()


#tmux a -t 2
name ="GQ0112-5-20MsPacmanNoFrameskip-v4aver"
m=np.load(name+".npy")
plt.plot(range(T),m[0:T])
plt.savefig(name+".png")
plt.close()

name ="GQ0114-5-40MsPacmanNoFrameskip-v4aver"
m=np.load(name+".npy")
plt.plot(range(T),m[0:T])
plt.savefig(name+".png")
plt.close()


# tmux a -t 3
name ="GQ0112-45-20MsPacmanNoFrameskip-v4aver"
n=np.load(name+".npy")
plt.plot(range(T),n[0:T])
plt.savefig(name+".png")
plt.close()

name ="GQ0114-2-20MsPacmanNoFrameskip-v4aver"
n=np.load(name+".npy")
plt.plot(range(T),n[0:T])
plt.savefig(name+".png")
plt.close()

name ="GQ0112-45-10MsPacmanNoFrameskip-v4aver"
p=np.load(name+".npy")
plt.plot(range(T),p[0:T])
plt.savefig(name+".png")
plt.close()


name="GQ0112-5-10MsPacmanNoFrameskip-v4aver"
q=np.load(name+".npy")
plt.plot(range(T),q[0:T])
plt.savefig(name+".png")
plt.close()

name="GQ0114-08-50Pong-v4aver"
r=np.load(name+".npy")
plt.plot(range(T),r[0:T])
plt.savefig(name+".png")
plt.close()

name="GQ0114-45-50MsPacmanNoFrameskip-v4aver"
o=np.load(name+".npy")
plt.plot(range(T),o[0:T])
plt.savefig(name+".png")
plt.close()

name="GQ0114-4-50MsPacmanNoFrameskip-v4aver"
o=np.load(name+".npy")
plt.plot(range(T),o[0:T])
plt.savefig(name+".png")
plt.close()

# tmux a -t 17
name ="GQ0118-2-40Alien-v4aver"
hiS=np.load(name+".npy")
plt.plot(range(T),hiS[0:T])
plt.savefig(name+".png")
plt.close()

name="GQ0118-2-40MsPacmanNoFrameskip-v4aver"
o=np.load(name+".npy")
plt.plot(range(T),o[0:T])
plt.savefig(name+".png")
plt.close()

# tmux a -t 18

name="GQ0118-02-30Pong-v4aver"
r=np.load(name+".npy")
plt.plot(range(T),r[0:T])
plt.savefig(name+".png")
plt.close()

# tmux a -t 19
Name = "GQ0118-1-40Frostbite-v4aver"
b=np.load(Name+".npy")
plt.plot(range(T),b[0:T])
plt.savefig(Name+".png")
plt.close()