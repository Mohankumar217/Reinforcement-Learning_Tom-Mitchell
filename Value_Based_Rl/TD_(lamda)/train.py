import gymnasium as gym
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

def get_best_action(env, state, v_table, gamma):
    """
    Selects the best action using one-step lookahead with the environment model (env.P)
    and the current Value Function V(s).
    """
    best_action = 0
    max_value = -float('inf')
    
    for action in range(env.action_space.n):
        expected_value = 0
        for prob, next_state, reward, done in env.unwrapped.P[state][action]:
            target = reward + gamma * v_table[next_state] * (not done)
            expected_value += prob * target
            
        if expected_value > max_value:
            max_value = expected_value
            best_action = action
            
    return best_action

def train(episodes=20000, alpha=0.1, gamma=0.99, lambd=0.6, epsilon=1.0, epsilon_decay=0.9995, min_epsilon=0.01):
    """
    Trains an agent on FrozenLake-v1 using TD(lambda) with state values V(s).
    """
    # Create environment
    env = gym.make('FrozenLake-v1', is_slippery=True)
    
    # Initialize V-table (State Values)
    n_states = env.observation_space.n
    v_table = np.zeros(n_states)
    
    rewards_history = []
    
    print(f"Starting training with TD(lambda={lambd}) - V(s)...")
    
    for episode in range(episodes):
        state, _ = env.reset()
        
        # Initialize Eligibility Traces E(s)
        e_traces = np.zeros(n_states)
        
        done = False
        total_reward = 0
        
        while not done:
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = get_best_action(env, state, v_table, gamma)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # TD(lambda) Update Rule for V(s)
            # delta = R + gamma * V(S') - V(S)
            target = reward + gamma * v_table[next_state] * (not done)
            delta = target - v_table[state]
            
            # Update Eligibility Trace for current state
            # E(S) <- E(S) + 1  (Accumulating Traces)
            e_traces[state] += 1
            
            # Update V-values and Decay Traces for ALL states
            # V(s) <- V(s) + alpha * delta * E(s)
            # E(s) <- gamma * lambda * E(s)
            v_table += alpha * delta * e_traces
            e_traces *= gamma * lambd
            
            state = next_state
            total_reward += reward
            
        rewards_history.append(total_reward)
        
        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
        if (episode + 1) % 1000 == 0:
            print(f"Episode {episode + 1}/{episodes} - Mean Reward (last 100): {np.mean(rewards_history[-100:]):.4f} - Epsilon: {epsilon:.4f}")
    print(v_table)
    print("Training finished.")
    
    # Save V-table
    with open('v_table.pkl', 'wb') as f:
        pickle.dump(v_table, f)
    print("V-table saved to v_table.pkl")
    
    # Plot results
    rolling_avg = np.convolve(rewards_history, np.ones(100)/100, mode='valid')
    plt.figure(figsize=(10, 5))
    plt.plot(rolling_avg)
    plt.title(f"Rolling Average Reward (Window 100) - TD($\lambda={lambd}$) V(s)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig("training_plot.png")
    print("Training plot saved to training_plot.png")
    
    env.close()

if __name__ == "__main__":
    train()
