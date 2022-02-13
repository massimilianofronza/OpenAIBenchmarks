import multiprocessing
import neat
import os
import pickle
import pylab as pl

from LunarLander import Genome_L


RESULTS = [
    "./LunarLander/singleGenomes/min_fitness/",
    "./LunarLander/singleGenomes/no_hidden/",
    "./LunarLander/singleGenomes/mean_fitness/",
    "./LunarLander/singleGenomes/no_elitism/",
    "./LunarLander/singleGenomes/two_hidden/"
]

ENV_IDS = [0, 1, 2, 3, 4]        # Insert the genome ids that you want to plot
NUM_SIMULATIONS = 5000
NUM_CORES = 8


# Performance script, takes a particular genome and test it for
# a given amount of times, reporting statistics about it.
if __name__ == '__main__':
    
    individual = None
    
    min_scores = []
    no_hidden_scores = []
    mean_scores = []
    no_elitism_scores = []
    two_hidden_scores = []
    
    local_dir = os.path.dirname(__file__)

    # Start gathering scores from all the saved genomes
    for ID in ENV_IDS:
        
        # Obtain the winner
        with open(RESULTS[ID] + 'winner.pickle', 'rb') as file:
            individual = pickle.load(file)
            print("\nWinner: ", individual, "\n")

        config_file = os.path.join(local_dir, RESULTS[ID] + "config_ffnn_lunarlander")
        config = neat.Config(Genome_L.LanderGenome, neat.DefaultReproduction,
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
                no_hidden_scores.append(newScore)
            elif ID==2:
                mean_scores.append(newScore)
            elif ID==3:
                no_elitism_scores.append(newScore)
            elif ID==4:
                two_hidden_scores.append(newScore)
            else:
                print(">>>>>>>> You added an ENV_ID that's not associated <<<<<<<<")


    min_scores.sort()
    no_hidden_scores.sort()
    mean_scores.sort()
    no_elitism_scores.sort()
    two_hidden_scores.sort()

    # Boxplot creation
    fig = pl.figure('Genomes (best fitnesses)')
    ax = fig.gca()
    ax.boxplot([min_scores, no_hidden_scores, mean_scores, no_elitism_scores, two_hidden_scores], notch=False)
    ax.set_xticklabels(['Min fitness', 'No hidden nodes', 'Mean fitness', 'No elitism', 'Two hidden nodes'])
    #ax.set_yscale('log')
    ax.set_xlabel('Best obtained genomes')
    ax.set_ylabel('Best fitness')
    pl.show()

    print("Values:", 
        "\n\tMin:\n\tHighest: ", min_scores[-1], "\tLowest: ", min_scores[0], len(min_scores), min_scores,
        "\n\tNo hidden:\n\tHighest: ", no_hidden_scores[-1], "\tLowest: ", no_hidden_scores[0], len(no_hidden_scores), no_hidden_scores
    )

