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

for episode in range(5):
    observation = env.reset()

    for t in range(100):
        env.render()
        #print(observation)
        action = 3#env.action_space.sample() # or given a custom model, action = policy(observation)
        print(action)
        # action = 0 moves cart to the left
        # action = 1 moves cart to the right
        observation, reward, done, info = env.step(action)

        if done:
            print("Episode finished after {} timesteps".format(t+1), "\n")
            break

# Useful information
print("Action space: ", env.action_space)
print("Observation space: ", env.observation_space)
print("\thigh: ", env.observation_space.high)
print("\tlow: ", env.observation_space.low)

env.close()
