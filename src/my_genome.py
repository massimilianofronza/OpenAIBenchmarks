import gym
import neat
import numpy as np
import pickle


# Classic control
env = gym.make('CartPole-v1')


### Custom genome class for the CartPole problem
class CartGenome(neat.DefaultGenome):

    # Initializes the first {pop_size} individuals
    def __init__(self, key):
        super().__init__(key)
    
    def eval_genome(self, config):
        network = neat.nn.FeedForwardNetwork.create(genome=self, config=config)
        netScore = self.simulate(network, False)

        # Go to 501 if you want to disable the pickle.dump()
        if netScore == 500:
            try:
                # Open for writing the best individual in binary mode
                with open('./results/winner.pickle', 'xb') as file:
                    pickle.dump(self, file)
            except FileExistsError:
                print("\nAn optimal individual was already found and recorded!")
            except:
                print("Something else happened.")

        return netScore

    def simulate(self, network, isRendered):
        score = 0
        step = 0
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
                #print("Simulation ended with score: ", score)
                env.close()
                break
        
        #print("Steps: ", step, " score: ", score)
        return score
