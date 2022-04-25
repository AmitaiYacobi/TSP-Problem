import random
import numpy as np
import matplotlib.pyplot as plt

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
            last_city = cities[city]
            sum_dist += cur_dist
    # return to for city
    sum_dist += euc_dist(last_city, cities[chromosome[0]])
    return sum_dist # negative distance for maximize the score



def selection(population_scores_dict):
    sum_of_scores = sum([1/s for s in population_scores_dict.values() ])
    pick = random.uniform(0, sum_of_scores)
    current = 0
    for chromosome, score in population_scores_dict.items():
        current += (1 / score)
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

def twopoint_crossover_one_chrom(parent1, parent2, point1, point2):
    offspring = [None] * len(parent1)
    # get parent1 gens
    offspring[:point1] = parent1[:point1]
    offspring[point2:] = parent1[point2:]
    # get parent2 gens
    for i in range(point1, point2):
        if parent2[i] in offspring:
            continue
        else:
            offspring[i] = parent2[i]
    # complete for a valid solution
    if None in offspring:
        none_idxes = [i for i in range(len(offspring)) if offspring[i] is None]
        j = 0
        for city in CITIES_LIST:
            if city not in offspring:
                offspring[none_idxes[j]] = city
                j += 1
    if sum(offspring) != sum(CITIES_LIST):
        raise("Error twopoint_crossover_one_chrom implementation. not a vaild solution")
    return offspring


def twopoints_crossover(parent1, parent2, rate="random"):
    """

    :param parent1: first parent
    :param parent2: second parent
    :param rate: where to cut the parents for crossover
    :return:
    """
    if rate == "random":
        point1 = random.randint(0, len(parent1)-1)
        point2 = random.randint(point1, len(parent1))
    else:
        point1 = int(rate[0] * len(parent1))
        point2 = int(rate[1] * len(parent1))

    parent1 = list(parent1)
    parent2 = list(parent2)
    offspring1 = twopoint_crossover_one_chrom(parent1, parent2, point1, point2)
    offspring2 = twopoint_crossover_one_chrom(parent2, parent1, point1, point2)
    return tuple(offspring1), tuple(offspring2)

def singlepoint_crossover(parent1, parent2, rate="random"):
    if rate == "random":
        single_point = random.randint(0, len(parent1))
    else:
        single_point = rate / len(parent1)

    parent1 = list(parent1)
    parent2 = list(parent2)
    offspring1 = singlepoint_crossover_one_chrom(parent1, parent2, single_point)
    offspring2 = singlepoint_crossover_one_chrom(parent2, parent1, single_point)
    return tuple(offspring1), tuple(offspring2)

def crossover(parent1, parent2, crossover_type="single_point", rate="random"):
    """

    :param parent1: first parent (tuple)
    :param parent2: second parent (tuple)
    :param crossove_type: function for crossover. get arguments: parent1, parent2, crossover rate
    :param rate: rate for crossover
    :return: two children
    """
    if crossover_type == "single_point":
        return singlepoint_crossover(parent1, parent2, rate=rate)
    if crossover_type == "two_points":
        return twopoints_crossover(parent1, parent2, rate=rate)


def mutation(offspring1, offspring2, rate=0.4):
    offspring1 = list(offspring1)
    offspring2 = list(offspring2)
    for i in range(len(offspring1)):
        if random.random() < rate: # to do mutation both children
            swip_index = random.randint(0, len(CITIES_LIST)-1) # choose index to swip
            offspring1[i], offspring1[swip_index] = offspring1[swip_index], offspring1[i]
            offspring2[i], offspring2[swip_index] = offspring2[swip_index], offspring2[i]
    return tuple(offspring1), tuple(offspring2)


def create_population_scores_dict(population, scores):
    return {tuple(population[i]): scores[i] for i in range(len(scores))}


def elitism(popul_scores_dict, p=0.1):
    """

    :param popul_scores_dict: key: chromosom, value: score
    :param p: float (percentage) how many besh chromosoms to pass to the next generation.
    :return: list of best chromosoms
    """
    n_best = int(p * len(popul_scores_dict)) + 1 # p << 1
    sorted_popul = sorted(popul_scores_dict.items(), key=lambda item: item[1], reverse=False)
    best = sorted_popul[:n_best]
    best_rep = [s[0] for s in best] # save the representation of each solution from the best ones
    return best_rep


def check_valid_child(offspring):
    return len(np.unique(offspring)) == len(CITIES_LIST)

def run_algorithm(cities, population_size, crossover_type, crossover_rate, mutation_rate, max_iter,
                  p_elitism):
    generation = 0
    population = generate_population(population_size)
    scores = population_fitness(population, cities)
    population_scores_dict = create_population_scores_dict(population, scores)
    best_score = max(population_scores_dict.values())
    avgs = []

    while generation <= max_iter:
        # print(population)
        new_population = []
        new_population.extend(elitism(population_scores_dict, p=p_elitism))
        remain_offstrings = len(population) - len(new_population)
        for i in range(remain_offstrings // 2): # run all population except the elitism we pass
            parent1 = selection(population_scores_dict)
            parent2 = selection(population_scores_dict)
            offspring1, offspring2 = crossover(parent1, parent2, crossover_type=crossover_type, rate=crossover_rate)
            offspring1, offspring2 = mutation(offspring1, offspring2, rate=mutation_rate)
            assert(check_valid_child(offspring1) and check_valid_child(offspring2))
            new_population.append(offspring1)
            new_population.append(offspring2)
            if len(new_population) == population_size:
                break

        population = new_population
        new_scores = population_fitness(new_population, cities)
        population_scores_dict = create_population_scores_dict(
            new_population, new_scores)
        best_score_new = min(population_scores_dict.values())
        if best_score_new > best_score:
            print("bug!")
        else:
            best_score = best_score_new
        best_chromosome = get_key(population_scores_dict, best_score)
        print(f"chromosome: {best_chromosome} score: {best_score}")
        generation += 1
        avg_score = sum([v for k,v in population_scores_dict.items()]) / len(population_scores_dict)
        # print(f"avg score is: {avg_score}")
        avgs.append(avg_score)

    plt.plot(avgs)
    plt.show()
    return population_scores_dict, best_score


if __name__ == "__main__":
    config = {
        "population_size" : 500,
        "crossover_type": "two_points",
        "crossover_rate": "random",
        "mutation_rate": 0.2,
        "max_iter": 10000,
        "p_elitism": 0.5
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
