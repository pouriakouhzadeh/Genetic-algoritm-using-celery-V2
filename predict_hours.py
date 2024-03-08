import csv
import random
import numpy as np
import pandas as pd
from TR_MODEL import TrainModels

def generate_individual(min_allowed_hours=8, max_allowed_hours=18):  # تغییر داده شده
    hours = list(range(24))  # ساعات از 0 تا 23
    num_allowed_hours = random.randint(min_allowed_hours, max_allowed_hours)
    allowed_hours = random.sample(hours, num_allowed_hours)
    return allowed_hours

def crossover(parent1, parent2):
    crossover_idx = random.randint(1, min(len(parent1), len(parent2)) - 1)
    child = parent1[:crossover_idx] + parent2[crossover_idx:]
    child = list(set(child))  # حذف تکراری‌ها
    if len(child) > 18:
        child = random.sample(child, 18)  # اطمینان از حداکثر تعداد ساعات مجاز
    return child

def read_data(file_name, tail_size):
    chunk_size = 1000
    chunks = pd.read_csv(file_name, chunksize=chunk_size)
    tail_data = pd.concat([chunk for chunk in chunks]).tail(tail_size)
    if tail_data.empty:
        return None
    return tail_data

def mutate(individual, mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            possible_hours = list(set(range(24)) - set(individual))
            if possible_hours:  # اگر ساعت جایگزین وجود دارد
                individual[i] = random.choice(possible_hours)
    return individual

def model_prediction(allowed_hours, currency_files):
    total_accuracy = 0
    for file_name in currency_files:
        currency_data = read_data(file_name, 3000)  # باید داده‌های مربوط به file_name را بخوانید
        
        ACC, wins, loses = TrainModels().Train(currency_data, depth=2, page=2, feature=30, QTY=1000, iter=100, Thereshhold=60, primit_hours=allowed_hours)
        total_accuracy += ACC
    return total_accuracy / len(currency_files)  # میانگین دقت را برمی‌گرداند

def genetic_algorithm(currency_files, population_size, generations, mutation_rate):
    population = [generate_individual() for _ in range(population_size)]
    best_fitness = -1
    best_individual = None

    for generation in range(generations):
        fitness_scores = [model_prediction(individual, currency_files) for individual in population]
        population_fitness = list(zip(population, fitness_scores))
        population_fitness.sort(key=lambda x: x[1], reverse=True)

        best_individual_gen = population_fitness[0][0]
        best_fitness_gen = population_fitness[0][1]

        if best_fitness_gen > best_fitness:
            best_fitness = best_fitness_gen
            best_individual = best_individual_gen

        print(f"Generation {generation + 1}: Best Fitness = {best_fitness_gen}")

        selected = population_fitness[:population_size // 2]
        population = [ind for ind, _ in selected]

        while len(population) < population_size:
            parent1, parent2 = random.sample(population, 2)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            population.append(child)

    return best_individual, best_fitness

if __name__ == "__main__":
    currency_files = ["EURUSD60.csv", "AUDCAD60.csv"]
    population_size = 40
    generations = 150
    mutation_rate = 0.2

    best_allowed_hours, best_fitness = genetic_algorithm(currency_files, population_size, generations, mutation_rate)
    print(f"Best Allowed Hours: {best_allowed_hours}, Best Fitness: {best_fitness}")
