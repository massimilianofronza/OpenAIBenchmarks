import os
import gym
import neat
import pickle

import main
import my_genome

CartGenome = my_genome.CartGenome

# Verification script, takes the winner of the previous main.py
# execution and shows it in action. It also records an .mp4 video
# in the "./results" folder
if __name__ == '__main__':
    
    winner = None

    with open('./results/winner.pickle', 'rb') as file:
        winner = pickle.load(file)
        print("\nWinner: ", winner, "\n")

    # Load the config file
    local_dir = os.path.dirname(__file__)
    config_file = os.path.join(local_dir, main.CONFIG_PATH)
    config = neat.Config(CartGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    
    # Simulate the winner genome
    network = neat.nn.FeedForwardNetwork.create(genome=winner, config=config)
    score = winner.simulate(network, isRendered=True)
    

