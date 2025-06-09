import random
import math

# Definição da gramática em formato dicionário
grammar = {
    "<expr>": [
        ["<expr>", "<op>", "<expr>"],
        ["(", "<expr>", ")"],
        ["<var>"],
        ["<const>"]
    ],
    "<op>": [["+"], ["-"], ["*"], ["/"]],
    "<var>": [["x"]],
    "<const>": [["1"], ["2"], ["3"]]
}

# Dados de treino (x, y)
training_data = [(i+1, y) for i, y in enumerate([
    4.298439483869902,
    3.036754968514613,
    2.7903071013454395,
    2.5566124177995415,
    2.549163946064691,
    2.4543154705471175,
    2.4164705316308774,
    2.314514716147899,
    2.3401741189347853,
    2.3910344213724817])]

print("Dados de treino:", training_data)

def generate_individual(length=27):
    """Gera um indivíduo (genótipo) como lista de inteiros"""
    return [random.randint(0, 255) for _ in range(length)]

def map_genotype_to_phenotype(genotype, grammar, start_symbol="<expr>"):
    """Mapeia genótipo para fenótipo usando a gramática"""
    phenotype = []
    remaining_genotype = genotype.copy()
    stack = [start_symbol]
    wrap_count = 0

    while stack and wrap_count < 3:  # Limite de wraps
        if not remaining_genotype:
            remaining_genotype = genotype.copy()
            wrap_count += 1

        current_symbol = stack.pop(0)

        if current_symbol in grammar:
            # Seleciona produção usando módulo
            productions = grammar[current_symbol]
            production_idx = remaining_genotype.pop(0) % len(productions)
            chosen_production = productions[production_idx]

            # Adiciona símbolos na ordem inversa para processamento em pilha
            stack = chosen_production + stack
        else:
            phenotype.append(current_symbol)

    return "".join(phenotype)

def evaluate(expr, data):
    """Avalia a expressão e calcula o RMSE"""
    try:
        total_error = 0
        for x, y_true in data:
            y_pred = eval(expr, {"x": x})
            total_error += (y_pred - y_true) ** 2
            # print(f"Evaluating: {expr}; x = {x}; result = {y_pred}")
        return math.sqrt(total_error / len(data))
    except:
        return float('inf')  # Retorna infinito para expressões inválidas

def mutate(individual, mutation_rate=0.1):
    """Aplica mutação ao indivíduo"""
    return [gene if random.random() > mutation_rate else random.randint(0, 255)
            for gene in individual]

def crossover(parent1, parent2):
    """Realiza crossover de um ponto"""
    point = random.randint(1, len(parent1)-1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def evolutionary_algorithm(pop_size=100, generations=100, debug=True):
    """Algoritmo evolutivo principal"""
    population = [generate_individual() for _ in range(pop_size)]

    for gen in range(generations):
        # Avaliação
        evaluated_pop = []
        for ind in population:
            expr = map_genotype_to_phenotype(ind, grammar)
            fitness = evaluate(expr, training_data)
            evaluated_pop.append((ind, fitness, expr))

        # Ordena por fitness (RMSE menor é melhor)
        evaluated_pop.sort(key=lambda x: x[1])

        # Seleção dos melhores
        best_individual, best_fitness, best_expr = evaluated_pop[0]
        if debug:
            print(f"Geração {gen}: Melhor RMSE = {best_fitness:.4f}, Expr = {best_expr}, Genome: {best_individual}")

        # Critério de parada antecipada
        if best_fitness < 0.1:
            break

        # Seleção (top 20% + torneio)
        parents = [ind for ind, _, _ in evaluated_pop[:pop_size//5]]
        tournament_size = 5

        while len(parents) < pop_size:
            candidates = random.sample(evaluated_pop, tournament_size)
            winner = min(candidates, key=lambda x: x[1])
            parents.append(winner[0])

        # Crossover e mutação
        new_population = parents[:pop_size//5]  # Elitismo

        while len(new_population) < pop_size:
            parent1, parent2 = random.sample(parents, 2)
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([mutate(child1), mutate(child2)])

        population = new_population[:pop_size]

    return best_expr, best_fitness, best_individual


def run(debug=True):
    # Inicializa o gerador de números aleatórios
    random.seed(1234)

    # Executa o algoritmo
    best_expr, best_fitness, best_individual = evolutionary_algorithm(debug=True)
    print(f"\nMelhor expressão encontrada: {best_expr}")
    print(f"RMSE: {best_fitness:.6f}")


if __name__ == '__main__':
    run()
