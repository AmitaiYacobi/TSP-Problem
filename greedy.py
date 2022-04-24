from mimetypes import init
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

def calculate_min_distance(distances, source, visited_cities):
    distances = [
                    distances[tuple(source)][tuple(dest)] for dest in distances[tuple(source)].keys() 
                    if tuple(dest) not in visited_cities
                ] 
    # print(len(distances))
    return min(distances)

def find_cities_by_distance(distances, source, distance):
    for dest in distances[tuple(source)].keys():
        if distances[tuple(source)][tuple(dest)] == distance:
            return source ,dest

def generate_initial_source(cities):
    index = random.randint(0,47)
    return cities[index]

def greedy_algorithm(cities):
    visited_cities = []
    route_weight = []
    best_routes = []
    distances = calculate_all_distances(cities)
    # initial_source = generate_initial_source(cities)
    for initial_source in cities:
        visited_cities = []
        route_weight = []
        source = initial_source
        while len(visited_cities) < 47:
            min_distance = calculate_min_distance(distances, source, visited_cities)
            source, dest = find_cities_by_distance(distances, source, min_distance)
            visited_cities.append(tuple(source))
            route_weight.append(min_distance)
            source = dest 
        route_weight.append(calculate_distance(initial_source, dest))
        best_routes.append(sum(route_weight))
    
    print(min(best_routes))





def main():
    cities = loadtxt("./tsp.txt")
    greedy_algorithm(cities)
    


if __name__ == "__main__":
    main()