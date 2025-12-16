
import numpy as np
import random
import matplotlib.pyplot as plt
from Algorithms import knapsack_dp, knapsack_greedy, knapsack_ga, generate_knapsack_instance

def main():
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

if __name__ == "__main__":
    main()
