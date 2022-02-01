import gym
import neat
import numpy as np


### Custom genome class for the CartPole problem
class CartGenome(neat.DefaultGenome):

    # Initializes the first {pop_size} individuals
    def __init__(self, key):
        super().__init__(key)
        self.fitness = 0
    
    def eval_genome(self, config):
        network = neat.nn.FeedForwardNetwork.create(genome=self, config=config)
        self.fitness = self.simulate(network, isRendered=False)
        return self.fitness

    def simulate(self, network, isRendered):
        env = gym.make('CartPole-v1')
        score = 0
        step = 0
        
        if isRendered:
            env = gym.wrappers.Monitor(env, './results', force=True)

        observation = env.reset()

        while True:
            if isRendered:
                env.render()

            step += 1
            output = network.activate(observation)
            action = np.argmax(output)
            observation, reward, done, info = env.step(action)
            score += reward

            if done:
                if isRendered:
                    print("Simulation ended with:\n\tScore: ", score, 
                    "\n\tSteps: ", step)
                env.close()
                break
        
        return score
