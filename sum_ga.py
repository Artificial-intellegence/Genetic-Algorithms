from ga import *

class SumGA(GeneticAlgorithm):
    """
    An example of using the GeneticAlgorithm class to solve a particular
    problem, in this case finding strings with the maximum number of 1's.
    """
    def fitness(self, chromosome):
        """
        Fitness is the sum of the bits.
        """
        return sum(chromosome)

    def isDone(self):
        """
        Stop when the fitness of the the best member of the current
        population is equal to the maximum fitness.
        """
        return self.fitness(self.bestEver) == self.length


def main():
    # Chromosomes of length 20, population of size 50
    ga = SumGA(20, 50)
    # Evolve for 100 generations
    # High prob of crossover, low prob of mutation
    bestFound = ga.evolve(100, 0.6, 0.001)
    print(bestFound)
    ga.plotStats("Sum GA")

if __name__ == '__main__':
    main()
