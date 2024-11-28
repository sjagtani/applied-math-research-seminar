import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt

def optimal_messaging_policy(beliefs, probabilities):
    """
    Find the optimal two-message policy.
    """
    n = len(beliefs)
    c = -probabilities
    A = np.tril(np.ones((n, n)))
    b = np.ones(n)
    bounds = [(0, 1)] * n
    result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
    if result.success:
        return result.x
    else:
        raise ValueError("Optimization failed!")

def query_optimization(beliefs, probabilities, query_budget):
    """
    Optimize query selection to refine belief thresholds.
    """
    n = len(beliefs)
    dp = np.zeros((query_budget + 1, n))
    split_points = np.zeros((query_budget + 1, n), dtype=int)
    dp[0, :] = probabilities * beliefs
    for k in range(1, query_budget + 1):
        for i in range(n):
            best_value = dp[k - 1, i]
            best_split = 0
            for j in range(i):
                value = dp[k - 1, j] + probabilities[i] * beliefs[i]
                if value > best_value:
                    best_value = value
                    best_split = j
            dp[k, i] = best_value
            split_points[k, i] = best_split
    strategy = []
    k, i = query_budget, n - 1
    while k > 0:
        strategy.append(split_points[k, i])
        i = split_points[k, i]
        k -= 1
    strategy = sorted(set(strategy))
    return strategy, dp[query_budget, n - 1]

def plot_results(beliefs, probabilities, strategy):
    """
    Plot the belief distribution and optimal query thresholds.
    """
    plt.bar(beliefs, probabilities, color='blue', alpha=0.6, label='Belief Probabilities')
    threshold_lines = []
    threshold_labels = []
    for split in strategy:
        line = plt.axvline(x=beliefs[split], color='red', linestyle='--')
        threshold_lines.append(line)
        threshold_labels.append(f'Threshold {beliefs[split]:.2f}')
    plt.xlabel('Beliefs')
    plt.ylabel('Probability')
    plt.title('Belief Distribution and Query Thresholds')

    handles, labels = plt.gca().get_legend_handles_labels()
    handles.extend(threshold_lines)
    labels.extend(threshold_labels)
    plt.legend(handles, labels)
    plt.show()

beliefs = np.array([0.1, 0.4, 0.6, 0.9])
probabilities = np.array([0.25, 0.25, 0.25, 0.25])
query_budget = 2

# Optimal Messaging Policy
policy = optimal_messaging_policy(beliefs, probabilities)
print("Optimal Messaging Policy:", policy)

# Query Optimization
strategy, utility = query_optimization(beliefs, probabilities, query_budget)
print("Strategy (indices):", strategy)
print("Thresholds (values):", [beliefs[i] for i in strategy])
print("Expected Utility:", utility)

plot_results(beliefs, probabilities, strategy)
