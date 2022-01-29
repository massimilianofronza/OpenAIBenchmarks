import neat


### Custom genome class for the CartPole problem
class CartGenome(neat.DefaultGenome):

    # Initializes the first {pop_size} individuals
    def __init__(self, key):
        super().__init__(key)
        #print("CartGenome -  __init__")
    
    def eval_genome(self, config):
        print("CartGenome - eval_genome()")
        return 1    # TODO dummy value
