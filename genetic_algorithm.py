import numpy as np


def profit_function(engine_on_off_vector):
    height = 0
    velocity = 0
    fuel_units_mass = engine_on_off_vector.sum()  # vector consists of 1s and 0s so naturally the sum is a number of 1s
    total_mass = 20 + fuel_units_mass
    for t in range(len(engine_on_off_vector)):
        if engine_on_off_vector[t] == 1:
            total_mass = total_mass - 1
        engine_acceleration = 500 / total_mass
        friction_acceleration = -0.06 * velocity * abs(velocity) / total_mass
        gravity_acceleration = -0.9
        acceleration = engine_acceleration + friction_acceleration + gravity_acceleration
        velocity = velocity + acceleration  # v = v + a*t but t here is 1 time unit every time, so v = v + a
        height = height + velocity + acceleration / 2  # s = s + v*t + 0.5*a*t^2 but t here is 1 time unit every time,
        # so s = s + v + a/2
    if height >= 750:
        return 200 - fuel_units_mass
    else:
        return 0


def initialise_population(population_size):
    bits_number = 200
    population = np.empty(population_size, bits_number)
    rng = np.random.default_rng()
    for i in range(population_size):
        population[i] = rng.integers(2, size=bits_number)
    return population


def evaluate(population, q_function):
    evaluation = np.empty(population.shape()[0])
    for i in range(len(evaluation)):
        evaluation[i] = q_function(population[i])
    return evaluation


def roulette_selection(population, evaluation):
    population_size = population.shape()[0]
    selection_probability = np.empty(population_size)
    for i in range(len(selection_probability)):
        selection_probability[i] = evaluation[i] / evaluation.sum()
    rng = np.random.default_rng()
    selected = rng.choice(population, population_size, p=evaluation)   # should it be possible
    # for an individual to be selected more than once? Yes, otherwise everyone would be selected
    return selected


def reproduce_and_mutate(p_crossing, p_mutation, selected):
    return offspring


def genetic_algorithm(q_function, population_size, p_mutation, p_crossing, t_max):
    population = initialise_population(population_size)
    t = 0
    evaluation = evaluate(population, q_function)
    while t < t_max:
        selected = roulette_selection(population, evaluation)
        offspring = reproduce_and_mutate(p_crossing, p_mutation, selected)
