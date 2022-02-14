import gym
import neat
import numpy as np


### Custom genome class for the LunarLander problem
class LanderGenome(neat.DefaultGenome):

    # Initializes the first {pop_size} individuals
    def __init__(self, key):
        super().__init__(key)
        self.fitness = 0
    
    def eval_genome(self, config):
        network = neat.nn.FeedForwardNetwork.create(genome=self, config=config)
        self.fitness = self.simulate(network, isRendered=False)
        return self.fitness

    def simulate(self, network, isRendered):
        env = gym.make('LunarLander-v2')
        score = 0
        step = 0
        
        if isRendered:
            env = gym.wrappers.Monitor(env, './LunarLander/results', force=True)

        rewards = []
        observation = env.reset()

        while True:
            if isRendered:
                env.render()

            output = network.activate(observation)
            action = np.argmax(output)
            observation, reward, done, info = env.step(action)

            # Fitness evaluation data
            step += 1
            rewards.append(reward)
            # print("reward: ", reward)
            
            if done:
                if isRendered:
                    print("Simulation ended with:\n\tScore: ", score, 
                    "\n\tSteps: ", step, "\n\tLast reward: ", reward)
                    
                    for r in rewards:
                        print(r)

                env.close()
                break
        
        result = self.fitness_sum(rewards)
        return result

    # Cumulative fitness function, returns the sum of all the rewards
    # obtained during the simulation
    def fitness_sum(self, rewards):
        return sum(rewards)

    # Fitness here is calculated as the sum of the last n frames 
    # of the simulation
    def fitness_last(self, rewards):
        tmp = rewards[-25:]     # last half second of rewards
        return sum(tmp)
    