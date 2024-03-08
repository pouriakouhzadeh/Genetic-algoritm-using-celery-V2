import csv
import random
import pandas as pd
from TR_MODEL import TrainModels

def generate_individual(min_forbidden_hours=6, max_forbidden_hours=18):
    hours = list(range(24))
    num_forbidden_hours = random.randint(min_forbidden_hours, max_forbidden_hours)
    forbidden_hours = random.sample(hours, num_forbidden_hours)
    return forbidden_hours

def crossover(parent1, parent2):
    crossover_idx = random.randint(1, min(len(parent1), len(parent2)) - 1)
    child = parent1[:crossover_idx] + parent2[crossover_idx:]
    child = list(set(child))
    if len(child) > 18:
        child = random.sample(child, 18)
    return child

def mutate(individual, mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            possible_hours = list(set(range(24)) - set(individual))
            if possible_hours:
                individual[i] = random.choice(possible_hours)
    return individual

def read_data(file_name, tail_size):
    chunk_size = 1000
    chunks = pd.read_csv(file_name, chunksize=chunk_size)
    tail_data = pd.concat([chunk for chunk in chunks]).tail(tail_size)
    if tail_data.empty:
        return None
    return tail_data

def model_prediction(forbidden_hours, currency_files):
    total_accuracy = 0
    for file_name in currency_files:
        currency_data = read_data(file_name, 3000)
        if currency_data is not None:
            ACC, _, _ = TrainModels().Train(currency_data, depth=2, page=2, feature=30, QTY=1000, iter=100, Thereshhold=60, forbidden_hours = forbidden_hours)
            total_accuracy += ACC
    return total_accuracy / len(currency_files)

def genetic_algorithm(currency_files, population_size, generations, mutation_rate):
    population = [generate_individual() for _ in range(population_size)]
    best_fitness = -1
    best_individual = None

    for generation in range(generations):
        fitness_scores = [model_prediction(individual, currency_files) for individual in population]
        population_fitness = list(zip(population, fitness_scores))
        population_fitness.sort(key=lambda x: x[1], reverse=True)

        if population_fitness[0][1] > best_fitness:
            best_fitness = population_fitness[0][1]
            best_individual = population_fitness[0][0]

        print(f"Generation {generation + 1}: Best Fitness = {best_fitness}")

        # Selection
        selected = population_fitness[:population_size // 2]
        population = [ind for ind, _ in selected]

        # Crossover and Mutation
        while len(population) < population_size:
            parent1, parent2 = random.sample([ind for ind, _ in selected], 2)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            population.append(child)

    return best_individual, best_fitness

if __name__ == "__main__":
    currency_files = ["EURUSD60.csv", "AUDCAD60.csv", "AUDCHF60.csv", "AUDNZD60.csv", "AUDUSD60.csv",
                      "EURAUD60.csv", "EURCHF60.csv", "EURGBP60.csv", "GBPUSD60.csv", "USDCAD60.csv", "USDCHF60.csv"]
    population_size = 40
    generations = 50  # تعداد نسل‌ها را برای زمان‌بندی بهتر می‌توانید تنظیم کنید
    mutation_rate = 0.2

    best_hours_forbidden, best_fitness = genetic_algorithm(currency_files, population_size, generations, mutation_rate)
    print(f"Best Forbidden Hours: {best_hours_forbidden}, Best Fitness: {best_fitness}")

    # ذخیره‌سازی نتایج در فایل CSV
    with open('/mnt/data/best_forbidden_hours.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Best Forbidden Hours", "Best Fitness"])
        writer.writerow([best_hours_forbidden, best_fitness])
