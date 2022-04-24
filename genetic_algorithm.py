import random
import numpy as np

# MAX_SCORE = 28
CITIES_LIST = np.arange(0, 48)

def get_key(dictionary, val):
    for key, value in dictionary.items():
        if val == value:
            return key
    return "key doesn't exist"


def generate_chromosome(chromosome_size=None): #TODO: change chromosome representation
    chromosome = CITIES_LIST.copy()
    np.random.shuffle(chromosome)
    return chromosome


def generate_population(population_size=100, chromosome_size=8):
    population = []
    for _ in range(population_size):
        population.append(generate_chromosome(chromosome_size))
    return population


def population_fitness(population, cities): #TODO: change all this function to sum distances
    scores = []
    for chromosome in population:
        scores.append(chromosome_fitness(chromosome, cities))
    return scores

def euc_dist(a, b):
    return np.linalg.norm(a - b)

def chromosome_fitness(chromosome, cities):
    sum_dist = 0
    last_city = None # or numpy of two elements
    for city in chromosome:
        if last_city is None:
            last_city = cities[city]
        else:
            cur_dist = euc_dist(last_city, cities[city])
            sum_dist += cur_dist
    return 0 - sum_dist # negative distance for maximize the score



def selection(population_scores_dict):
    sum_of_scores = sum(population_scores_dict.values())
    pick = random.uniform(sum_of_scores, 0)
    current = sum_of_scores
    for chromosome, score in population_scores_dict.items():
        current -= score
        if current > pick:
            return chromosome

def singlepoint_crossover_one_chrom(parent1, parent2, point):
    offspring = parent1[:point]
    for i in range(point, len(parent2)):
        if parent2[i] in offspring:
            continue
        else:
            offspring.append(parent2[i])
    if len(offspring) < len(parent2):
        for city in CITIES_LIST:
            if city not in offspring:
                offspring.append(city)
    if len(offspring) < len(parent2):
        raise("Error singlepoint_crossover_one_chrom implementation. not a vaild solution")
    return offspring


def singlepoint_crossover(parent1, parent2, rate="random"):
    if rate == "random":
        single_point = random.randint(0, len(parent1))
    else:
        single_point = rate / len(parent1)

    parent1 = list(parent1)
    parent2 = list(parent2)
    singlepoint_crossover_one_chrom(parent1, parent2, single_point)
    offspring1 = parent1[:single_point] + parent2[single_point:]
    offspring2 = parent2[:single_point] + parent1[single_point:]
    return tuple(offspring1), tuple(offspring2)

def crossover(parent1, parent2, crossover_type=singlepoint_crossover, rate="random"):
    """

    :param parent1: first parent (tuple)
    :param parent2: second parent (tuple)
    :param crossove_type: function for crossover. get arguments: parent1, parent2, crossover rate
    :param rate: rate for crossover
    :return: two children
    """
    return crossover_type(parent1, parent2, rate=rate)


def mutation(offspring1, offspring2, rate=0.4):
    offspring1 = list(offspring1)
    offspring2 = list(offspring2)
    for i in range(len(offspring1)):
        if random.random() < rate: # to do mutation both children
            offspring1[i] = random.randint(0, 7)
            offspring2[i] = random.randint(0, 7)
    return tuple(offspring1), tuple(offspring2)


def create_population_scores_dict(population, scores):
    return {tuple(population[i]): scores[i] for i in range(len(scores))}


def alitism(popul_scores_dict, p=0.1):
    """

    :param popul_scores_dict: key: chromosom, value: score
    :param p: float (percentage) how many besh chromosoms to pass to the next generation.
    :return: list of best chromosoms
    """
    n_best = int(p * len(popul_scores_dict))
    sorted_popul = sorted(popul_scores_dict.items(), key=lambda item: item[1], reverse=True)
    best = sorted_popul[:n_best]
    best_rep = [s[0] for s in best] # save the representation of each solution from the best ones
    return best_rep




def run_algorithm(cities, population_size, crossover_type, crossover_rate, mutation_rate, max_iter):
    generation = 0
    population = generate_population(population_size)
    scores = population_fitness(population, cities)
    population_scores_dict = create_population_scores_dict(population, scores)
    # best_score = max(population_scores_dict.values())

    while generation <= max_iter:
        print(population)
        new_population = []
        new_population.extend(alitism(population_scores_dict))
        for _ in range(int(len(population) - len(new_population) / 2)): # run all population except the alitism we pass
            parent1 = selection(population_scores_dict)
            parent2 = selection(population_scores_dict)
            offspring1, offspring2 = crossover(parent1, parent2, crossover_type=crossover_type, rate=crossover_rate)
            offspring1, offspring2 = mutation(offspring1, offspring2, rate=mutation_rate)
            new_population.append(offspring1)
            new_population.append(offspring2)
            if len(new_population) == population_size:
                break

        population = new_population
        new_scores = population_fitness(new_population, cities)
        population_scores_dict = create_population_scores_dict(
            new_population, new_scores)
        best_score = max(population_scores_dict.values())
        best_chromosome = get_key(population_scores_dict, best_score)
        print(f"chromosome: {best_chromosome} score: {best_score}")
        generation += 1

    return population_scores_dict, generation


if __name__ == "__main__":
    config = {
        "population_size" : 5,
        "crossover_type": singlepoint_crossover,
        "crossover_rate": "random",
        "mutation_rate": 0.4,
        "max_iter": 100
    }
    cities = np.loadtxt("./tsp.txt")
    population, best_score = run_algorithm(cities, **config)
    counter = 0
    # for chromosome, score in population.items():
    #     if score == MAX_SCORE:
    #         counter += 1

    print("\n##################################")
    print(
        f"The best score is {best_score}")
    print("##################################\n")
