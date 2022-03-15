# OpenAIBenchmarks
University project for the "Bio-Inspired Artificial Intelligence" course at UniTN.
This work aims at applying the Neuroevolution of Augmenting Topologies algorithm(NEAT) to the CartPole and LunarLander benchmarks from the OpenAI Gym library to explore the NEAT's capabilities at evolving agents for sequential decision tasks.

--------------------------------

## Scripts list:
* gym_test.py           -> Base code for Gym
* main.py               -> Execution of NEAT
* verify.py             -> After main.py, you can verify the results by running this(check the END_ID)
* performance_plot.py   -> You can plot the 5 boxplots of the LunarLander singleGenomes by running this
* visualize.py          -> Used by main.py to plot speciation and fitness trends charts. This is the only script that's not supposed to be run individually

--------------------------------

## Installation instructions:

1. python -m venv env
2. source env/bin/activate
3. pip install -r requirements.txt
4. cd ./src
5. Execute!
