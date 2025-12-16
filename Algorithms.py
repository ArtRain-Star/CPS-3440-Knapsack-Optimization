import random
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# DP (Exact)
# -------------------------------
def knapsack_dp(values, weights, W):
    n = len(values)
    dp = [[0]*(W+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for w in range(W+1):
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i-1][w], dp[i-1][w-weights[i-1]] + values[i-1])
            else:
                dp[i][w] = dp[i-1][w]
    return dp[n][W]

# -------------------------------
# Greedy Approximation
# -------------------------------
def knapsack_greedy(values, weights, W):
    n = len(values)
    ratio = [(values[i]/weights[i], i) for i in range(n)]
    ratio.sort(reverse=True)
    total_value = 0
    total_weight = 0
    for r, i in ratio:
        if total_weight + weights[i] <= W:
            total_value += values[i]
            total_weight += weights[i]
    return total_value

# -------------------------------
# GA with tracking best per generation
# -------------------------------
def knapsack_ga(values, weights, W, population_size=50, generations=100, crossover_rate=0.8, mutation_rate=0.1):
    n = len(values)
    def init_population():
        return [[random.randint(0,1) for _ in range(n)] for _ in range(population_size)]
    def fitness(ind):
        total_weight = sum([ind[i]*weights[i] for i in range(n)])
        total_value = sum([ind[i]*values[i] for i in range(n)])
        if total_weight > W:
            return 0
        return total_value
    def select(pop, fits):
        total_fit = sum(fits)
        if total_fit == 0:
            return random.choice(pop)
        pick = random.uniform(0, total_fit)
        current = 0
        for ind, f in zip(pop, fits):
            current += f
            if current > pick:
                return ind
    def crossover(p1, p2):
        if random.random() > crossover_rate:
            return p1[:], p2[:]
        point = random.randint(1, n-1)
        return p1[:point]+p2[point:], p2[:point]+p1[point:]
    def mutate(ind):
        for i in range(n):
            if random.random() < mutation_rate:
                ind[i] = 1 - ind[i]

    population = init_population()
    best_solution = None
    best_fit = 0
    best_per_gen = []

    for gen in range(generations):
        fits = [fitness(ind) for ind in population]
        for i in range(population_size):
            if fits[i] > best_fit:
                best_fit = fits[i]
                best_solution = population[i][:]
        best_per_gen.append(best_fit)
        new_population = []
        while len(new_population) < population_size:
            p1 = select(population, fits)
            p2 = select(population, fits)
            c1, c2 = crossover(p1, p2)
            mutate(c1)
            mutate(c2)
            new_population.extend([c1, c2])
        population = new_population[:population_size]

    return best_fit, best_per_gen

# -------------------------------
# 生成随机 Knapsack 实例
# -------------------------------
def generate_knapsack_instance(n_items, W_max=100, value_max=100, weight_max=50):
    values = [random.randint(10, value_max) for _ in range(n_items)]
    weights = [random.randint(1, weight_max) for _ in range(n_items)]
    W = random.randint(W_max//2, W_max)
    return values, weights, W

# -------------------------------
# 批量实验 + 自动绘图
# -------------------------------
if __name__ == "__main__":
    sizes = [10, 20, 30]  # 问题规模
    population_sizes = [20, 50, 100]
    mutation_rates = [0.01, 0.05, 0.1]
    generations = 100

    results = []
    convergence_data = {}  # 存储 best_curve 用于收敛曲线

    for n in sizes:
        values, weights, W = generate_knapsack_instance(n)
        dp_val = knapsack_dp(values, weights, W)
        greedy_val = knapsack_greedy(values, weights, W)
        print(f"\nItems: {n}, Capacity: {W}")
        print(f"DP (Exact): {dp_val}, Greedy: {greedy_val}")

        for pop_size in population_sizes:
            for mut_rate in mutation_rates:
                ga_val, best_curve = knapsack_ga(values, weights, W,
                                                 population_size=pop_size,
                                                 generations=generations,
                                                 mutation_rate=mut_rate)
                rel_error = (dp_val - ga_val)/dp_val*100
                results.append((n, pop_size, mut_rate, dp_val, ga_val, rel_error))
                convergence_data[(n, pop_size, mut_rate)] = best_curve
                print(f"GA pop={pop_size}, mut={mut_rate:.2f}, GA_val={ga_val}, RelError={rel_error:.2f}%")

        # 绘制当前 n 的收敛曲线
        plt.figure(figsize=(10,6))
        for pop_size in population_sizes:
            for mut_rate in mutation_rates:
                plt.plot(convergence_data[(n, pop_size, mut_rate)], label=f"pop={pop_size}, mut={mut_rate}")
        plt.title(f"GA Convergence Curve (n={n})")
        plt.xlabel("Generation")
        plt.ylabel("Best Value")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"GA_Convergence_n{n}.png")  # 保存图片
        plt.close()

    # 绘制误差热力图 (以最后一个 n 为例)
    error_matrix = np.zeros((len(mutation_rates), len(population_sizes)))
    for i, mut in enumerate(mutation_rates):
        for j, pop in enumerate(population_sizes):
            GA_val = next(r[4] for r in results if r[0]==sizes[-1] and r[1]==pop and r[2]==mut)
            DP_val = next(r[3] for r in results if r[0]==sizes[-1] and r[1]==pop and r[2]==mut)
            error_matrix[i,j] = (DP_val - GA_val)/DP_val*100

    plt.figure(figsize=(8,5))
    im = plt.imshow(error_matrix, cmap="Reds", origin="lower", aspect="auto")
    plt.xticks(range(len(population_sizes)), population_sizes)
    plt.yticks(range(len(mutation_rates)), mutation_rates)
    plt.xlabel("Population Size")
    plt.ylabel("Mutation Rate")
    plt.title(f"GA Relative Error Heatmap (n={sizes[-1]})")
    plt.colorbar(im, label="Relative Error (%)")
    for i in range(len(mutation_rates)):
        for j in range(len(population_sizes)):
            plt.text(j, i, f"{error_matrix[i,j]:.1f}", ha="center", va="center", color="black")
    plt.savefig(f"GA_Error_Heatmap_n{sizes[-1]}.png")
    plt.close()

    # 输出总结表格
    print("\nSummary Results:")
    print("Items | Pop | Mutation | DP | GA | RelError(%)")
    for r in results:
        print(r[0], r[1], r[2], r[3], r[4], f"{r[5]:.2f}")
