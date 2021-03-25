import gym
import matplotlib.pyplot as plt
from skimage.transform import resize
import numpy as np

def greyscale_preprocessor(state):
    #state = cv2.cvtColor(state,cv2.COLOR_BGR2GRAY)/255.
    state = np.dot(state[...,:3], [0.299, 0.587, 0.114])
    return state

def deepmind_preprocessor1(state):
    state = greyscale_preprocessor(state)
    #state = np.array(cv2.resize(state, (84, 84)))
    #resized_screen = scipy.misc.imresize(state, (110,84))
    resized_screen = resize(state, (110,84))
    state = resized_screen[8:92, :]
    return state

def deepmind_preprocessor(state):
    state = greyscale_preprocessor(state)
    #state = np.array(cv2.resize(state, (84, 84)))
    #resized_screen = scipy.misc.imresize(state, (110,84))
    resized_screen = resize(state, (110,84))
    state = resized_screen[18:102, :]
    return state

def testenvs():
    #env = gym.make("CartPole-v1")
    #env = gym.make("MsPacman-v0")
    #name = "HeroNoFrameskip-v4"
    # name = "Amidar-v4" 
    # name = "Atlantis-v4" #射击类
    # name = "BankHeist-v4" # 追逐类，不一定好用，要识别物体之间的相对关系
    name ="DemonAttack-v4"
    env = gym.make(name)
    observation = env.reset()
    print("observation",observation.shape)
   
    for _ in range(10000):
        #env.render()
        action = env.action_space.sample() # your agent here (this takes random actions)
        observation, reward, done, info = env.step(action)
        print("reward",reward)
        if done:
            a = deepmind_preprocessor1(observation)
            plt.imshow(a)
            plt.savefig('./tmp/'+name+'re0.png')
            plt.close()
            observation = env.reset()
    env.close()

if __name__ == '__main__':
    print("first come here")
    from gym import envs
    print(envs.registry.all())
    testenvs()