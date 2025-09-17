from random import random_float64, seed
from math import floor
from benchmark import keep

# 1. Representation: Integer Chromosome
alias Chromosome = SIMD[DType.int64, 1]
# type alias to assist readability
alias Fitness = Float64

# 2. Fitness function
fn fitness(chromosome: Chromosome) -> Fitness:
  var x = chromosome[0]
  return Float64(x * x) # We are looking for the maximum value of x^2

# 3. Initial population
fn generate_population(size: Int, lower_bound: Int, upper_bound: Int) -> List[Chromosome]:
  seed() # Seed the random number generator
  var population: List[Chromosome] = List[Chromosome]()
  for _ in range(size):
      var random_float = random_float64() # random number from 0 to 1
      var range_size = upper_bound - lower_bound
      var gene = lower_bound + floor(random_float * Float64(range_size))
      var chromosome: Chromosome = Chromosome() # Initialise chromosome to zero
      keep(chromosome)
      chromosome[0] = Int64(gene) # set the first (and only) element to gene
      population.append(chromosome)
  return population^

# 4. Selection: Tournament selection
fn tournament_selection(population: List[Chromosome], tournament_size: Int) -> Chromosome:
  var candidates: List[Chromosome] = List[Chromosome]()
  var pop_size = len(population)
  for _ in range(tournament_size):
    var random_float = random_float64()
    var index = floor(random_float * Float64(pop_size))
    keep(index)
    candidates.append(population[Int(index)])

  # find the fittest from the candidates
  var best_candidate: Chromosome = candidates[0]
  var best_fitness: Fitness = fitness(best_candidate)
  for i in range(1, len(candidates)):
    var current_candidate = candidates[i]
    var current_fitness: Fitness = fitness(current_candidate)
    if current_fitness > best_fitness:
      best_fitness = current_fitness
      best_candidate = current_candidate
  return best_candidate

# 5. Crossover: Single-point crossover
fn crossover(parent1: Chromosome, parent2: Chromosome) -> (Chromosome, Chromosome):
  var point: Int = 0 # There is only one gene, so no point to changing this
  var child1: Chromosome = parent1
  var child2: Chromosome = parent2
  child1[point] = parent2[point]
  child2[point] = parent1[point]
  return (child1, child2)

# 6. Mutation: Random change with probability
fn mutate(chromosome: Chromosome, mutation_rate: Float64, lower_bound: Int, upper_bound: Int) -> Chromosome:
  if random_float64() < mutation_rate:
    var mutated_chromosome: Chromosome = chromosome
    var random_float = random_float64() # random number from 0 to 1
    var range_size = upper_bound - lower_bound
    var gene = lower_bound + floor(random_float * Float64(range_size))
    mutated_chromosome[0] = Int(gene)
    return mutated_chromosome
  else:
    return chromosome

# Main GA function
fn genetic_algorithm(pop_size: Int, lower_bound: Int, upper_bound: Int, generations: Int, mutation_rate: Float64, tournament_size: Int) -> Chromosome:
  var population = generate_population(pop_size, lower_bound, upper_bound)
  var best_chromosome: Chromosome = population[0]  # Initialize the best chromosome
  var best_fitness: Fitness = fitness(best_chromosome)  # Initialize the best fitness
  print("Initial solution: " + String(best_chromosome) + " with fitness: " + String(best_fitness))   # Added initial solution

  for _gen in range(generations):
    var new_population: List[Chromosome] = List[Chromosome]()
    for _i in range(pop_size // 2):
      # Selection (Tournament)
      var parent1 = tournament_selection(population, tournament_size)
      var parent2 = tournament_selection(population, tournament_size)

      # Crossover
      var mut_tuple = crossover(parent1, parent2)
      child1 = mut_tuple[0]
      child2 = mut_tuple[1]

      # Mutation
      var mutated_child1 = mutate(child1, mutation_rate, lower_bound, upper_bound)
      var mutated_child2 = mutate(child2, mutation_rate, lower_bound, upper_bound)

      new_population.append(mutated_child1)
      new_population.append(mutated_child2)

    population = new_population^

  # Find the best chromosome in the final population

  for i in range(1,len(population)):
    var current_chromosome: Chromosome = population[i]
    var current_fitness: Fitness = fitness(current_chromosome)
    if current_fitness > best_fitness:
      print("Improved solution: " + String(current_chromosome) + " with fitness: " + String(current_fitness))
      best_fitness = current_fitness
      best_chromosome = current_chromosome

  return best_chromosome

fn main():
  var pop_size = 100
  var lower_bound = -10
  var upper_bound = 10
  var generations = 100
  var mutation_rate = 0.1
  var tournament_size = 4
  var result = genetic_algorithm(pop_size, lower_bound, upper_bound, generations, mutation_rate, tournament_size)
  var fitness = fitness(result)
  print("Best solution: " + String(result) + " with fitness: " + String(fitness))   # Expected solution is [-10] or [10], both of which result in 100 fitness
