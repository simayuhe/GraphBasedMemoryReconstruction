import gym

def testenvs():
    #env = gym.make("CartPole-v1")
    #env = gym.make("MsPacman-v0")
    env = gym.make("Alien-v0")
    observation = env.reset()
    print("observation",observation.shape)
    for _ in range(100):
        env.render()
        action = env.action_space.sample() # your agent here (this takes random actions)
        observation, reward, done, info = env.step(action)
        print("reward",reward)
        if done:
            observation = env.reset()
    env.close()

if __name__ == '__main__':
    print("first come here")
    from gym import envs
    print(envs.registry.all())
    testenvs()