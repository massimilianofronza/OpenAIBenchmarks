import gym
#import neat

# Classic control
#env = gym.make('CartPole-v1')
#env = gym.make('MountainCar-v0')
#env = gym.make('Pendulum-v1')
#env = gym.make('MountainCarContinuous-v0')
#env = gym.make('Acrobot-v1')

# Box2D
env = gym.make('LunarLander-v2')
#env = gym.make('LunarLanderContinuous-v2')
#env = gym.make('BipedalWalker-v3')
#env = gym.make('BipedalWalkerHardcore-v3')
#env = gym.make('CarRacing-v0')

config_files = ['./config_ffnn_cartpole']

for episode in range(5):
    observation = env.reset()

    for t in range(200):
        
        # Comment for faster learning
        env.render()

        #print("Obs[2, 3, 4, 5, 6, 7]:\t", observation[2], "\t", observation[3], "\t", observation[4],
#             "\t", observation[5], "\t\t", observation[6], " - ", observation[7])
        
        action = env.action_space.sample() # or given a custom model, action = policy(observation)
        observation, reward, done, info = env.step(action)

        if done:
            print("\nEpisode finished after {} timesteps".format(t+1))
            print("Final reward:", reward)
            break
    
    print("end of episode.\n")

# Useful information
print("Action space: ", env.action_space)
print("Observation space: ", env.observation_space)
#print("\thigh: ", env.observation_space.high)
#print("\tlow: ", env.observation_space.low)

env.close()
