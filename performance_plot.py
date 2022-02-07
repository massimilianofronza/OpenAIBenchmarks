import multiprocessing
import neat
import os
import pickle
import pylab as pl

from CartPole import Genome_C
from LunarLander import Genome_L


ENVIRONMENTS = ["CartPole", "LunarLander"]
CONFIG_PATHS = [
    "./LunarLander/singleGenomes/min_fitness/config_ffnn_lunarlander",
    "./LunarLander/singleGenomes/nice/config_ffnn_lunarlander"
]
RESULTS_FOLDER = [
    "./LunarLander/singleGenomes/min_fitness/",
    "./LunarLander/singleGenomes/nice/"
]

ENV_IDS = [0, 1]        # Insert the genome ids that you want to plot
NUM_SIMULATIONS = 100
NUM_CORES = 8


# Performance script, takes a particular genome and test it for
# a given amount of times, reporting statistics about it.
if __name__ == '__main__':
    
    individual = None
    min_scores = []
    nice_scores = []
    CustomGenome = Genome_L.LanderGenome
    local_dir = os.path.dirname(__file__)

    # Start gathering scores from all the saved genomes
    for ID in ENV_IDS:
        
        # Obtain the winner
        with open(RESULTS_FOLDER[ID] + 'winner.pickle', 'rb') as file:
            individual = pickle.load(file)
            print("\nWinner: ", individual, "\n")

        config_file = os.path.join(local_dir, CONFIG_PATHS[ID])
        config = neat.Config(CustomGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            config_file)
        
        network = neat.nn.FeedForwardNetwork.create(genome=individual, config=config)

        # Define useful variables for the analysis
        idx = 0
        maxScore = 0
        scores = []
        pool = multiprocessing.Pool(NUM_CORES)
        jobs = []

        # Schedule the simulations in parallel
        for i in range(NUM_SIMULATIONS):
            jobs.append(pool.apply_async(individual.simulate, (network, False)))

        # Fetch the already scheduled jobs
        for job in jobs:
            newScore = job.get(timeout=None)
            if ID==0:
                min_scores.append(newScore)
            elif ID==1:
                nice_scores.append(newScore)
            else:
                print(">>>>>>>> You added an ENV_ID that's not associated <<<<<<<<")


    nice_scores.sort()
    min_scores.sort()

    # Boxplot creation
    fig = pl.figure('Genomes (best fitnesses)')
    ax = fig.gca()
    ax.boxplot([nice_scores, min_scores], notch=False)
    ax.set_xticklabels(['Nice genome', 'Min genome'])
    #ax.set_yscale('log')
    ax.set_xlabel('Best obtained genomes')
    ax.set_ylabel('Best fitness')
    pl.show()

    print("Values:", 
        "\n\tNice:\n\tHighest: ", nice_scores[-1], "\tLowest: ", nice_scores[0], len(nice_scores),
        "\n\tMin:\n\tHighest: ", min_scores[-1], "\tLowest: ", min_scores[0], len(min_scores))

