import numpy as np
import matplotlib.pyplot as plt

def query_optimization(beliefs, probabilities, query_budget, query_cost=0.0, noisy_oracle=False):
    if noisy_oracle:
        beliefs += np.random.normal(0, 0.02, size=len(beliefs))
        beliefs = np.clip(beliefs, 0, 1)
        print("Noisy Beliefs:", beliefs)
    n = len(beliefs)
    dp = np.zeros((query_budget + 1, n))
    split_points = np.zeros((query_budget + 1, n), dtype=int)
    dp[0, :] = probabilities * beliefs
    for k in range(1, query_budget + 1):
        for i in range(n):
            best_value = dp[k - 1, i]
            best_split = 0
            for j in range(i):
                value = dp[k - 1, j] + np.sum(probabilities[j:i+1] * beliefs[j:i+1]) - query_cost
                if value > best_value:
                    best_value = value
                    best_split = j
            dp[k, i] = best_value
            split_points[k, i] = best_split
    strategy = []
    k, i = query_budget, n - 1
    while k > 0:
        strategy.append(i)
        i = split_points[k, i]
        k -= 1
    strategy = sorted(set(strategy))
    return strategy, dp[query_budget, n - 1]

def plot_results(beliefs, probabilities, strategy):
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

np.random.seed(42)  # Random seed for reproducibility

beliefs = np.linspace(0.1, 0.9, 10)
probabilities = np.random.dirichlet(np.ones(10))
query_cost = 0.0
noisy_oracle = True
query_budget = 3

# Query Optimization
strategy, utility = query_optimization(beliefs, probabilities, query_budget, query_cost, noisy_oracle)

print("Beliefs:", beliefs)
print("Probabilities:", probabilities)
print("Strategy (indices):", strategy)
print("Thresholds (values):", [beliefs[i] for i in strategy])
print("Expected Utility:", utility)

plot_results(beliefs, probabilities, strategy)
