import multiprocessing
import neat
import os
import pickle

from CartPole import Genome_C
from LunarLander import Genome_L
import visualize


# Program parameters
ENVIRONMENTS = ["CartPole", "LunarLander"]
CONFIG_PATHS = ["./CartPole/config_ffnn_cartpole", "./LunarLander/config_ffnn_lunarlander"]
RESULTS_FOLDER = ["./CartPole/results/", "./LunarLander/results/"]
ENV_ID = 0      # Change index to switch between the environments

NUM_GENERATIONS = 50

CustomGenome = None
Results = RESULTS_FOLDER[ENV_ID]


if __name__ == '__main__':

    if ENV_ID == 0:
        CustomGenome = Genome_C.CartGenome
        print("\n########## CartPole benchmark ##########\n")
    
    elif ENV_ID == 1:
        CustomGenome = Genome_L.LanderGenome
        print("\n########## LunarLander benchmark ##########\n")
    
    else:
        print("\nWrong environment selected! Exit...\n")
        exit(1)

    # Load the config file
    local_dir = os.path.dirname(__file__)
    config_file = os.path.join(local_dir, CONFIG_PATHS[ENV_ID])
    config = neat.Config(CustomGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    pop = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    stats = neat.StatisticsReporter()
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(stats)

    # Use the parallel module of neat to let the evaluations be computed together
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), CustomGenome.eval_genome)

    # run NEAT for num_generations
    winner = pop.run(pe.evaluate, NUM_GENERATIONS)

    # Save the winner
    try:
        # Open for writing the best individual in binary mode
        with open(Results + 'winner.pickle', 'xb') as file:
            pickle.dump(winner, file)
    except FileExistsError:
        print("##### An optimal individual was already found and recorded! #####")
        exit(1)
    except:
        print("##### Something else happened. Probably the /results/ folder doesn't exist #####")
        exit(1)

    visualize.plot_stats(stats, ylog=False, view=False, 
        filename="./" + ENVIRONMENTS[ENV_ID] + "/results/" + ENVIRONMENTS[ENV_ID] + "_fitness.png")
    
    visualize.plot_species(stats, view=False, 
        filename="./" + ENVIRONMENTS[ENV_ID] + "/results/" + ENVIRONMENTS[ENV_ID] + "_speciation.png")

    # Display the winning genome
    print('\nBest genome:\n{!s}'.format(winner))
