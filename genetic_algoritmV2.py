import random
import pandas as pd
from celery import group
from tasks import train_model_task

currency_files = [
    "EURUSD60.csv", "AUDCAD60.csv", "AUDCHF60.csv",
    "AUDNZD60.csv", "AUDUSD60.csv", "EURAUD60.csv",
    "EURCHF60.csv", "EURGBP60.csv", "GBPUSD60.csv",
    "USDCAD60.csv", "USDCHF60.csv"
]

def read_data(file_name, tail_size=7000):
    try:
        data = pd.read_csv(file_name[-1])
        return data.tail(tail_size)
    except FileNotFoundError:
        print(f"File {file_name} not found.")
        return None



def generate_individual(param_space):
    # ایجاد قسمت‌های اصلی individual با استفاده از مقادیر تصادفی بر اساس param_space
    individual = [random.randint(param[0], param[1]) for param in param_space[:-1]]
    
    # ایجاد لیست allowed_hours به صورت تصادفی
    # توجه داشته باشید که این قسمت آخرین عنصر individual را تشکیل می‌دهد
    allowed_hours = random.sample(range(3, 24), random.randint(param_space[-1][0], param_space[-1][1]))
    
    # افزودن allowed_hours به عنوان آخرین عنصر individual
    individual.append(allowed_hours)
    
    return individual


def crossover(parent1, parent2):
    if len(parent1) < 3 or len(parent2) < 3:
        return random.choice([parent1, parent2])
    crossover_point = random.randint(1, len(parent1) - 2)
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return child


def mutate(individual, mutation_rate, param_space):
    # جهش برای تمام آرگومان‌های به جز 'allowed_hours'
    for i in range(len(individual) - 1):
        if random.random() < mutation_rate:
            individual[i] = random.randint(param_space[i][0], param_space[i][1])
    
    # بررسی و جهش برای 'allowed_hours'
    if random.random() < mutation_rate:
        # اطمینان حاصل کنید که individual[-1] به درستی یک لیست است
        if isinstance(individual[-1], list):
            allowed_hours = individual[-1]
            
            # تصمیم‌گیری برای اضافه کردن یا حذف یک ساعت تصادفی
            if random.random() < 0.5:
                # تلاش برای اضافه کردن ساعت جدید، اگر حداکثر طول رعایت نشده باشد
                if len(allowed_hours) < param_space[-1][1]:
                    new_hour = random.choice(list(set(range(3, 24)) - set(allowed_hours)))
                    allowed_hours.append(new_hour)
            else:
                # تلاش برای حذف یک ساعت، اگر بیش از یک ساعت وجود داشته باشد
                if len(allowed_hours) > 1:
                    allowed_hours.remove(random.choice(allowed_hours))
            
            individual[-1] = allowed_hours
        else:
            print("Error: individual[-1] is not a list. It is:", type(individual[-1]))
            # در صورت خطا، بازگرداندن individual بدون تغییر
            return individual
    
    return individual



def save_to_file(data, file_name="ga_results.txt"):
    with open(file_name, "a") as file:
        file.write(data + "\n")



def evaluate_population(population, currency_file):
    results = []  # نتایج برای هر فرد
    tasks = []
    fitness_scores = []  # اینجا یک لیست برای نگهداری امتیازات ایجاد می‌کنیم

    # حلقه بر روی جمعیت
    for individual in population:
        # حلقه بر روی فایل‌های ارز
        currency_data = read_data(currency_file, 7000)
        
        if currency_data is not None:
            temp = currency_data.to_json(orient='split')
            # اطمینان حاصل کنید که individual به صورت مناسبی آرگومان‌ها را برای train_model_task فراهم می‌کند
            # فرض بر این است که individual[-1] حاوی 'allowed_hours' است و بقیه عناصر آرگومان‌های دیگر را شامل می‌شوند
            task = train_model_task.s(temp, *individual[:-1], individual[-1])
            tasks.append(task)


    job = group(tasks)()
    results = job.get()  # results حاوی نتایج بازگشتی از هر کار است

    for result in results:
        try:
            # فرض می‌شود که هر نتیجه به صورت یک تاپل از (acc, wins, loses) بازگردانده شده است
            acc, wins, loses = result
            # این قسمت باید بر اساس ساختار دقیق نتایج بازگشتی از تابع train_model_task شما تنظیم شود
            if wins + loses >= 0.2 * (individual[3] * 0.033):  # این شرط باید بر اساس لاجیک مورد نظر شما تنظیم شود
                fitness_scores.append(acc) 
            else:
                 fitness_scores.append(0)    
        except Exception as e:
            print(f"Error processing task: {e}")


    print(f"Stage finish, fitness scores : {fitness_scores}")
    return fitness_scores  # برگرداندن لیست امتیازات دقت برای هر فرد در جمعیت



def genetic_algorithm_for_all_currencies(currency_files, population_size, generations, mutation_rate, param_space, results_file="ga_results.txt"):
    best_results = {}  # برای ذخیره بهترین نتایج برای هر جفت ارز

    for currency_file in currency_files:
        population = [generate_individual(param_space) for _ in range(population_size)]
        best_fitness = [-1]
        best_individual = None

        for generation in range(generations):
            fitness_scores = [evaluate_population(population, [currency_file])]
            # fitness_scores = [model_prediction(individual, [currency_file]) for individual in population]
            total_scores = 0
            for scores_list in fitness_scores:
                total_scores += sum(scores_list)
            print(f"Currency: {currency_file}, Generation {generation + 1}: Total Fitness = {total_scores}")

            population_fitness = list(zip(population, fitness_scores))
            population_fitness.sort(key=lambda x: x[1], reverse=True)
            best_individual_gen, best_fitness_gen = population_fitness[0]
            if int(best_fitness_gen[0]) > int(best_fitness[0]):
                best_fitness = best_fitness_gen
                best_individual = best_individual_gen
            print(f"Currency: {currency_file}, Generation {generation + 1}: Best Fitness = {max(best_fitness_gen)}")

            selected = population_fitness[:len(population) // 2]
            population = [ind for ind, _ in selected]
            while len(population) < population_size:
                if len(selected) >= 2:
                    parent1, parent2 = random.sample(selected, 2)
                else:
                    # این شرایط همیشه true خواهد بود، زیرا selected حداقل شامل نیمی از جمعیت است
                    parent1 = generate_individual(param_space)
                    parent2 = generate_individual(param_space)
                child = crossover(parent1, parent2)
                child = mutate(child, mutation_rate, param_space)
                population.append(child)

        best_results[currency_file] = (best_individual, best_fitness)

    save_to_file("Best Results for All Currencies:")
    for currency_file, (best_individual, best_fitness) in best_results.items():
        save_to_file(f"Currency: {currency_file}, Best Individual: {best_individual}, Fitness: {best_fitness}")



param_space = [(2, 10), (2, 20), (30, 500), (500, 6000), (100, 5000), (52, 75), (8, 18)]  # Example param_space including allowed_hours range

genetic_algorithm_for_all_currencies(currency_files=currency_files,
                                     population_size=150,
                                     generations=40,
                                     mutation_rate=0.02,
                                     param_space=param_space,
                                     results_file="ga_results.txt")
