import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class ReceiverType:
    belief: float
    probability: float

class SimulationOracle:
    def __init__(self, receiver: ReceiverType):
        self.receiver = receiver

    def query(self, message: int, policy: Dict) -> int:
        posterior = self.compute_posterior(message, policy)
        return 1 if posterior >= 0.5 else 0

    def compute_posterior(self, message: int, policy: Dict) -> float:
        p = self.receiver.belief
        prob_m_1 = policy[message][1]
        prob_m_0 = policy[message][0]
        numerator = p * prob_m_1
        denominator = p * prob_m_1 + (1-p) * prob_m_0
        return numerator/denominator if denominator > 0 else p

class BayesianPersuasionGame:
    def __init__(self, receivers: List[ReceiverType]):
        self.receivers = receivers
        self.state_priors = {0: 0.5, 1: 0.5}

    def compute_optimal_policy(self) -> Tuple[float, Dict]:
        best_utility = -float('inf')
        best_policy = None
        
        for p0 in np.linspace(0, 1, 11):  # Probability of m1 in state 0
            for p1 in np.linspace(0, 1, 11):  # Probability of m1 in state 1
                if p1 < p0:  # Skip invalid policies
                    continue
                
                policy = {
                    0: (1-p0, 1-p1),  # Message 0 probabilities
                    1: (p0, p1)       # Message 1 probabilities
                }
                
                utility = self.compute_expected_utility(policy)
                if utility > best_utility:
                    best_utility = utility
                    best_policy = policy
        
        return best_utility, best_policy

    def compute_expected_utility(self, policy: Dict) -> float:
        total_utility = 0
        for state in [0, 1]:
            state_prob = self.state_priors[state]
            for receiver in self.receivers:
                oracle = SimulationOracle(receiver)
                receiver_utility = 0
                for msg in [0, 1]:
                    msg_prob = policy[msg][state]
                    action = oracle.query(msg, policy)
                    receiver_utility += action * msg_prob
                total_utility += receiver_utility * receiver.probability * state_prob
        return total_utility

def run_experiment_1():
    receivers = [
        ReceiverType(belief=0.7, probability=0.3),
        ReceiverType(belief=0.4, probability=0.4),
        ReceiverType(belief=0.2, probability=0.3)
    ]
    game = BayesianPersuasionGame(receivers)
    return game.compute_optimal_policy()

def run_experiment_2():
    receivers = [
        ReceiverType(belief=0.9, probability=0.4),
        ReceiverType(belief=0.5, probability=0.2),
        ReceiverType(belief=0.1, probability=0.4)
    ]
    game = BayesianPersuasionGame(receivers)
    return game.compute_optimal_policy()

if __name__ == "__main__":
    utility1, policy1 = run_experiment_1()
    utility2, policy2 = run_experiment_2()
    
    print("Experiment 1 Results:")
    print(f"Utility: {utility1:.3f}")
    print("Policy:", policy1)
    
    print("\nExperiment 2 Results:")
    print(f"Utility: {utility2:.3f}")
    print("Policy:", policy2)
