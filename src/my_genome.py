import gym
import neat
import numpy as np

# Classic control
env = gym.make('CartPole-v1')


### Custom genome class for the CartPole problem
class CartGenome(neat.DefaultGenome):

    # Initializes the first {pop_size} individuals
    def __init__(self, key):
        super().__init__(key)
        #print("CartGenome - __init__")
    
    def eval_genome(self, config):
        #print("CartGenome - eval_genome()")
        network = neat.nn.FeedForwardNetwork.create(genome=self, config=config)
        #print("network created")
        netScore = self.simulate(network)
        return netScore

    def simulate(self, network):
        #print("Start simulating")
        score = 0
        observation = env.reset()

        while True:
            output = network.activate(observation)
            action = np.argmax(output)
            observation, reward, done, info = env.step(action)
            score += reward

            if done:
                #print("Simulation ended with score: ", score)
                env.close()
                break

        return score
