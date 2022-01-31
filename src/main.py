import multiprocessing
import neat
import os
import pickle

import my_genome

# Constant variables
CONFIG_PATH = "./config_ffnn_cartpole"
NUM_GENERATIONS = 100


CartGenome = my_genome.CartGenome

if __name__ == '__main__':
    
    # Load the config file
    local_dir = os.path.dirname(__file__)
    config_file = os.path.join(local_dir, CONFIG_PATH)
    config = neat.Config(CartGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    pop = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    stats = neat.StatisticsReporter()
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(stats)

    # Use the parallel module of neat to let the evaluations be computed together
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), CartGenome.eval_genome)

    # run NEAT for num_generations
    winner = pop.run(pe.evaluate, NUM_GENERATIONS)

    # Save the winner
    try:
        # Open for writing the best individual in binary mode
        with open('./results/winner.pickle', 'xb') as file:
            pickle.dump(winner, file)
    except FileExistsError:
        print("An optimal individual was already found and recorded!")
    except:
        print("Something else happened.")


    # Display the winning genome
    print('\nBest genome:\n{!s}'.format(winner))


    
