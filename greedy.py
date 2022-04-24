import random
import numpy as np
from numpy import loadtxt
from collections import defaultdict

def dictionary():
    return defaultdict(dictionary)

def calculate_distance(city1, city2):
    return np.linalg.norm(city1 - city2)

def calculate_all_distances(cities):
    distances = dictionary()
    for source in cities:
        for dest in cities:
            if sum(source == dest) == 2: continue
            distance = calculate_distance(source, dest)
            source = tuple(source)
            dest = tuple(dest)
            distances[source][dest] = distance
    return distances

def greedy_algorithm(cities):
    distances = calculate_all_distances(cities)
    source_index = random.randint(0,47)
    source = cities[source_index]
    min_dist = min([distances[tuple(source)][tuple(dest)] for dest in distances[tuple(source)].keys()])
    

        

def main():
    cities = loadtxt("./tsp.txt")
    greedy_algorithm(cities)
    


if __name__ == "__main__":
    main()