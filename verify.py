import os
import neat
import pickle

from CartPole import Genome_C
from LunarLander import Genome_L


ENVIRONMENTS = ["CartPole", "LunarLander"]
CONFIG_PATHS = [
    "./CartPole/config_ffnn_cartpole", 
    "./LunarLander/config_ffnn_lunarlander",
    "./LunarLander/singleGenomes/min_fitness/config_ffnn_lunarlander",
    "./LunarLander/singleGenomes/nice/config_ffnn_lunarlander"
]
RESULTS_FOLDER = [
    "./CartPole/results/", 
    "./LunarLander/results/",
    "./LunarLander/singleGenomes/min_fitness/",
    "./LunarLander/singleGenomes/nice/"
]

ENV_ID = 3      # Change index to switch between the environments


CustomGenome = None
Results = RESULTS_FOLDER[ENV_ID]


# Verification script, takes the winner of the previous main.py
# execution and shows it in action. It also records an .mp4 video
# in the "./results" folder of the respective problem
if __name__ == '__main__':
    
    winner = None

    if ENV_ID == 0:
        CustomGenome = Genome_C.CartGenome
        print("\n########## CartPole benchmark ##########\n")
    elif 1 <= ENV_ID <= 3:
        CustomGenome = Genome_L.LanderGenome
        print("\n########## LunarLander benchmark ##########\n")
    else:
        print("\nWrong environment selected! Exit...\n")
        exit(1)

    with open(Results + 'winner.pickle', 'rb') as file:
        winner = pickle.load(file)
        print("\nWinner: ", winner, "\n")

    # Load the config file
    local_dir = os.path.dirname(__file__)
    config_file = os.path.join(local_dir, CONFIG_PATHS[ENV_ID])
    config = neat.Config(CustomGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    
    # Simulate the winner genome
    network = neat.nn.FeedForwardNetwork.create(genome=winner, config=config)
    score = winner.simulate(network, isRendered=True)
    

