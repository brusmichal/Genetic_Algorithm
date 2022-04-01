from genetic_algorithm import *
import numpy as np
import matplotlib as plt

parameters_number = 5
iterations_number = 25
total_budget = 10 ** 5

default_ps = 250
default_pm = 0.05
default_pc = 0.75

populations = np.array([25, 50, 100, 300, 500])
p_mutations = np.array([0.2, 0.1, 0.05, 0.01, 0.005])
p_crossovers = np.array([0.9, 0.75, 0.6, 0.5, 0.25])


def test(p_type, parameters, iterations):
    mean_best_values, mean_gen_means = compute_means(p_type, parameters, iterations)
    plot_means(mean_best_values, mean_gen_means, p_type, parameters)


def plot_means(mean_best_values, mean_gen_means, p_type, parameters):
    parameters_number = len(parameters)
    print("Średnia z najlepszych wyników:")
    for k in range(parameters_number):
        print(f"{p_type}: {parameters[k]}, optimum: {mean_best_values[k]}")
        y = np.array([mean_gen_means[k]])
        x = np.arange(len(mean_gen_means[k]))
        plt.figure(figsize=(20, 10))
        plt.scatter(x, y)
        plt.title("Wartość średnia funkcji w danej generacji")
        plt.xlabel("Generacje")
        plt.ylabel("E(q(x)")
        plt.grid(b=True)
        plt.show()


def compute_means(p_type, parameters, iterations):
    parameters_number = len(parameters)
    mean_best_values = np.empty(parameters_number)
    mean_gen_means = np.empty(parameters_number, dtype=object)
    for i in range(parameters_number):
        if p_type == "ps":
            number_of_gen = int(total_budget / parameters[i])
        else:
            number_of_gen = int(total_budget / default_ps)
        best_values = np.empty(iterations)
        gen_means = np.empty([iterations, number_of_gen])
        for j in range(iterations):
            if p_type == "ps":
                result = genetic_algorithm(profit_function, default_pm, default_pc, parameters[i], total_budget)
            elif p_type == "pm":
                result = genetic_algorithm(profit_function, parameters[i], default_pc, default_ps, total_budget)
            else:
                result = genetic_algorithm(profit_function, default_pm, parameters[i], default_ps, total_budget)
            best_values[j] = result[0][0]
            gen_means[j] = result[3]
        mean_best_values[i] = best_values.sum() / iterations
        mean_gen_means[i] = gen_means.sum(axis=0) / iterations
    return mean_best_values, mean_gen_means
