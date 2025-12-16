# CPS-3440-Knapsack-Optimization
Knapsack optimization using DP, Greedy, and GA
This project implements solutions to the Knapsack Problem using three different algorithms:

1. Dynamic Programming (Exact Solution)
2. Greedy Approximation
3. Genetic Algorithm (GA)

The goal is to find the most valuable combination of items that fit within a specified weight limit.

Problem Overview

The Knapsack Problem is a well-known optimization problem where given a set of items with associated values and weights, and a maximum weight capacity (knapsack), the task is to select the subset of items that maximizes the total value while keeping the total weight within the capacity.

Algorithms Implemented

1. Dynamic Programming (DP): This is the exact solution method that explores all possible combinations of items. It guarantees finding the optimal solution, but may take longer for large problem sizes.
2. Greedy Approximation: This heuristic algorithm selects items based on their value-to-weight ratio, which is a fast but non-optimal solution for larger instances.
3. Genetic Algorithm (GA): This algorithm simulates the process of natural evolution to iteratively improve the solution using crossover, mutation, and selection operations. It is an approximate method and works well for larger problems.

How to Run

 Requirements
- Python 3.x
- `matplotlib` (for plotting convergence curves)
- `numpy` (for numerical operations)
