import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def create_comparison_visualizations():
    # Default figure size
    plt.rcParams['figure.figsize'] = [12, 10]
    
    # Figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    
    # 1. Receiver Type Distribution Comparison
    types = ['High Belief', 'Medium Belief', 'Low Belief']
    exp1_probs = [0.3, 0.4, 0.3]
    exp2_probs = [0.4, 0.2, 0.4]
    exp1_beliefs = [0.7, 0.4, 0.2]
    exp2_beliefs = [0.9, 0.5, 0.1]
    
    x = np.arange(len(types))
    width = 0.35
    
    ax1.bar(x - width/2, exp1_probs, width, label='Experiment 1', color='skyblue')
    ax1.bar(x + width/2, exp2_probs, width, label='Experiment 2', color='lightgreen')
    
    ax1.set_ylabel('Probability')
    ax1.set_title('Receiver Type Distributions')
    ax1.set_xticks(x)
    ax1.set_xticklabels(types)
    ax1.legend()
    
    # Belief values
    for i in range(len(types)):
        ax1.text(i - width/2, exp1_probs[i] + 0.02, f'p={exp1_beliefs[i]}', ha='center')
        ax1.text(i + width/2, exp2_probs[i] + 0.02, f'p={exp2_beliefs[i]}', ha='center')
    
    # 2. Messaging Policy Comparison
    messages = ['M0|ω=0', 'M1|ω=0', 'M0|ω=1', 'M1|ω=1']
    exp1_probs = [0.8, 0.2, 0.0, 1.0]
    exp2_probs = [0.9, 0.1, 0.1, 0.9]
    
    x = np.arange(len(messages))
    
    ax2.bar(x - width/2, exp1_probs, width, label='Experiment 1', color='skyblue')
    ax2.bar(x + width/2, exp2_probs, width, label='Experiment 2', color='lightgreen')
    
    ax2.set_ylabel('Probability')
    ax2.set_title('Optimal Messaging Policies')
    ax2.set_xticks(x)
    ax2.set_xticklabels(messages)
    ax2.legend()
    
    # 3. Expected Utility Comparison
    utilities = ['Expected Sender Utility']
    exp1_utility = [0.6]
    exp2_utility = [0.7]
    
    x = np.arange(len(utilities))
    
    ax3.bar(x - width/2, exp1_utility, width, label='Experiment 1', color='skyblue')
    ax3.bar(x + width/2, exp2_utility, width, label='Experiment 2', color='lightgreen')
    
    ax3.set_ylabel('Utility')
    ax3.set_title('Expected Sender Utilities')
    ax3.set_xticks(x)
    ax3.set_xticklabels(utilities)
    ax3.legend()
    
    # Utility values
    ax3.text(x - width/2, exp1_utility[0] + 0.02, f'{exp1_utility[0]}', ha='center')
    ax3.text(x + width/2, exp2_utility[0] + 0.02, f'{exp2_utility[0]}', ha='center')
    
    plt.tight_layout()
    plt.show()
    
    # Receiver viz
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Response data
    receiver_types = ['High', 'Medium', 'Low']
    exp1_msg0 = [0, 0, 0]  # responses to Message 0
    exp1_msg1 = [1, 1, 1]  # responses to Message 1
    exp2_msg0 = [1, 0, 0]  # responses to Message 0
    exp2_msg1 = [1, 1, 1]  # responses to Message 1
    
    x = np.arange(len(receiver_types))
    width = 0.2
    
    # Plot responses
    ax.bar(x - width*1.5, exp1_msg0, width, label='Exp1-Msg0', color='lightblue')
    ax.bar(x - width/2, exp1_msg1, width, label='Exp1-Msg1', color='blue')
    ax.bar(x + width/2, exp2_msg0, width, label='Exp2-Msg0', color='lightgreen')
    ax.bar(x + width*1.5, exp2_msg1, width, label='Exp2-Msg1', color='green')
    
    ax.set_ylabel('Action')
    ax.set_title('Receiver Responses by Type and Message')
    ax.set_xticks(x)
    ax.set_xticklabels(receiver_types)
    ax.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    create_comparison_visualizations()
