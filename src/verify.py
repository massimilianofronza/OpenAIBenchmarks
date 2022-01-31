import os
import gym
import neat
import pickle

import main
import my_genome

CartGenome = my_genome.CartGenome

if __name__ == '__main__':
    
    winner = None

    with open('./results/winner.pickle', 'rb') as file:
        winner = pickle.load(file)
        print("Winner: ", winner, "\n")
    
    # Load the config file
    local_dir = os.path.dirname(__file__)
    config_file = os.path.join(local_dir, main.CONFIG_PATH)
    config = neat.Config(CartGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    
    # Simulate the 
    network = neat.nn.FeedForwardNetwork.create(genome=winner, config=config)
    score = winner.simulate(network, isRendered=True)
    

