import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict

@dataclass
class ReceiverType:
    belief: float
    probability: float

class SimulationOracle:
    def __init__(self, receiver_type: ReceiverType):
        self.receiver_type = receiver_type
    
    def query(self, msg: int, policy: Dict[int, Tuple[float, float]]) -> int:
        """Simulates receiver's action given a message and policy"""
        prob_state_0, prob_state_1 = policy[msg]
        prior = self.receiver_type.belief
        
        # Posterior using Bayes' rule
        if prob_state_0 == 0 and prob_state_1 == 0:
            posterior = prior
        else:
            numerator = prior * prob_state_1
            denominator = prior * prob_state_1 + (1-prior) * prob_state_0
            posterior = numerator / denominator if denominator > 0 else prior
            
        return 1 if posterior >= 0.5 else 0

class BayesianPersuasionGame:
    def __init__(self, receiver_types: List[ReceiverType]):
        self.receiver_types = receiver_types
        self.state_priors = {0: 0.5, 1: 0.5}
    
    def compute_sender_utility(self, policy: Dict[int, Tuple[float, float]]) -> float:
        total_utility = 0
        
        for state in [0, 1]:
            state_prob = self.state_priors[state]
            
            for receiver in self.receiver_types:
                receiver_utility = 0
                oracle = SimulationOracle(receiver)
                
                # Calculate utility for each message
                for msg, (prob_0, prob_1) in policy.items():
                    msg_prob = prob_0 if state == 0 else prob_1
                    action = oracle.query(msg, policy)
                    # Sender wants action 1
                    receiver_utility += action * msg_prob
                
                total_utility += receiver_utility * receiver.probability * state_prob
        
        return total_utility
    
    def run_simulation(self) -> Tuple[float, Dict]:
        best_utility = -float('inf')
        best_policy = None
        
        # Search over message probabilities
        for p0 in np.linspace(0, 1, 11):  # prob of Message 1 in state 0
            for p1 in np.linspace(0, 1, 11):  # prob of Message 1 in state 1
                if p1 < p0:  # Skip invalid policies
                    continue
                    
                policy = {
                    0: (1-p0, 1-p1),  # Prob of Message 0
                    1: (p0, p1)       # Prob of Message 1
                }
                
                utility = self.compute_sender_utility(policy)
                if utility > best_utility:
                    best_utility = utility
                    best_policy = policy
        
        return best_utility, best_policy

def run_experiment2():
    # Different receiver types: more polarized distribution
    receivers = [
        ReceiverType(belief=0.9, probability=0.4),  # Very high belief type
        ReceiverType(belief=0.5, probability=0.2),  # Neutral belief type
        ReceiverType(belief=0.1, probability=0.4),  # Very low belief type
    ]
    
    print("Experiment 2: Polarized Receiver Distribution")
    print("--------------------------------------------")
    print("Receiver Types:")
    for i, r in enumerate(receivers):
        print(f"Type {i+1}: belief={r.belief:.1f}, probability={r.probability:.1f}")
    print()
    
    game = BayesianPersuasionGame(receivers)
    utility, policy = game.run_simulation()
    
    print(f"Expected sender utility: {utility:.3f}")
    print("\nOptimal messaging policy:")
    for msg, (prob_0, prob_1) in policy.items():
        print(f"Message {msg}:")
        print(f"  P(m|ω=0) = {prob_0:.3f}")
        print(f"  P(m|ω=1) = {prob_1:.3f}")
        
    # Print receiver responses
    print("\nReceiver responses:")
    for i, receiver in enumerate(receivers):
        oracle = SimulationOracle(receiver)
        print(f"\nType {i+1} (belief={receiver.belief:.1f}):")
        for msg in [0, 1]:
            action = oracle.query(msg, policy)
            print(f"  Response to Message {msg}: {action}")

if __name__ == "__main__":
    run_experiment2()
