import gym
from collections import defaultdict
import time
import copy
# _game_envs = defaultdict(set)

def obs_ram_Mspac(observation):
        #仅限于MsPacman
        #print("observation",observation)
        observation= observation.astype('int')
        mine = [observation[10],observation[16]]
        monsters =[]
        dists = []
        distmin = 32
        distmin_index = 8
        direction = "f" # free
        distance = "v" # very far
        for i in range(4):
            print(i)
            monster = [observation[6+i],observation[12+i]]
            monsters.append(monster)
            dist = abs(observation[6+i]-observation[10])+abs(observation[12+i]-observation[16])
            dists.append(dist)
            if dist <distmin:
                distmin_index = i
                distmin =dist
        print("distmin_index",distmin_index,"distmin",distmin)
        if distmin_index != 8:
            print("monster(s) are following you")
            distx = observation[6+distmin_index]-observation[10]
            disty = observation[12+distmin_index]-observation[16]
            if abs(distx)<10 or abs(disty)<10:
                distance = "n" # n
            else:
                if abs(distx)>20 and abs(disty)>20:
                    distance = "f"
                else:
                    distance = "m"

            if distx == 0 :
                if disty ==0:
                    pass
                if disty >0 :
                    direction = "u" # up
                if disty < 0:
                    direction = "d" # down
            else:
                if distx >0:
                    if disty ==0:
                        direction = "r" #right
                    if disty >0:
                        direction = "ru" # 右上方
                    if disty <0:
                        direction = "rd" # 右下方
                if distx<0:
                    if disty ==0:
                        direction = "l" # left
                    if disty >0 :
                        direction = "lu" # 左上方
                    if disty < 0:
                        direction = "ld" # 左下方
        
        print("monster at your ",direction," distance ",distmin,"named",distance )
        print("mine",mine)
        print("monsters",monsters,"dists",dists)
        state = [observation[10],observation[16]]
        #state = state.astype('float32')
        #state = state.astype('int')
        state = str(state[0])+str(state[1])+direction+distance
        print(state)
        return state

i = 0
for env_spec in gym.envs.registry.all():
    i =i+1
    if i !=640:
        continue
    else:
        break
print(i,env_spec,env_spec.id)
env = gym.make("MsPacman-ramDeterministic-v4")
# env = gym.make("Alien-ramDeterministic-v4")
# env = gym.make('Blackjack-v0')
observation = env.reset()

#print("observation",observation.shape)
step =0
for _ in range(10000):
    step = step +1
    env.render()
    state = obs_ram_Mspac(observation)
    action = env.action_space.sample() # your agent here (this takes random actions)
    observation_, reward, done, info = env.step(action)
    if step<80:
        time.sleep(0.01)
    else:
        time.sleep(1)
    delta_obs = observation_-observation   
    #print("step",step,"agent position",[observation[10],observation[16]],"obs 1 x y",[observation[7],observation[8]],"obs2 x y",[observation[9],observation[15]],"obs3 x y",[observation[6],observation[12]], "obs4 x  y", [ 0,observation[13]])
    print("step",step,"agent position",[observation[10],observation[16]],"obs 0 x y",[observation[6],observation[12]],"obs 1 x y",[observation[7],observation[13]],"obs2 x y",[observation[8],observation[14]],"obs3 x y",[observation[9],observation[15]])
    
    t=copy.deepcopy(observation)
    t[7]=0
    t[8]=0
    t[9]=0
    t[10]=0
    t[15]=0
    t[16]=0
    t[6]=0
    t[12]=0
    t[13]=0
    print("step",step,"observation",observation)
    #print("reward",reward) 
    observation = observation_
    if done or info['ale.lives']<3:
        #observation = env.reset()
        break
env.close()

#     # TODO: solve this with regexes
#     env_type = env.entry_point.split(':')[0].split('.')[-1]
#     _game_envs[env_type].add(env.id)

# # reading benchmark names directly from retro requires
# # importing retro here, and for some reason that crashes tensorflow
# # in ubuntu
# _game_envs['retro'] = {
#     'BubbleBobble-Nes',
#     'SuperMarioBros-Nes',
#     'TwinBee3PokoPokoDaimaou-Nes',
#     'SpaceHarrier-Nes',
#     'SonicTheHedgehog-Genesis',
#     'Vectorman-Genesis',
#     'FinalFight-Snes',
#     'SpaceInvaders-Snes',
# }



    #env = gym.make("CartPole-v1")
    #env = gym.make("MsPacman-v0")
# i = 0
# for env_spec in gym.envs.registry.all():
#     i+=1
#     print(i,env_spec,env_spec.id)
#     if env_spec.id =='Blackjack-v0' or i<355:
#         #print("unwork ! pass !")
#         continue
#     env = gym.make(env_spec.id)
#     #env = gym.make('Blackjack-v0')
#     observation = env.reset()

#     #print("observation",observation.shape)
#     for _ in range(100):
#         env.render()
#         action = env.action_space.sample() # your agent here (this takes random actions)
#         observation, reward, done, info = env.step(action)
#         # print("obs",[observation[10],observation[16]])
#         #print("reward",reward)
#         if done:
#             #observation = env.reset()
#             break
#     env.close()