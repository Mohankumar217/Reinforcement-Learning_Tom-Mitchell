import gymnasium as gym
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

def get_best_action(env, state, v_table, gamma):
    """
    Selects the best action using one-step lookahead with the environment model (env.P)
    and the current Value Function V(s).
    Returns: best_action
    """
    best_action = 0
    max_value = -float('inf')
    
    # Check all possible actions
    for action in range(env.action_space.n):
        expected_value = 0
        # Iterate over possible outcomes for this action (prob, next_state, reward, done)
        for prob, next_state, reward, done in env.unwrapped.P[state][action]:
            # Calculate value: Immediate Reward + Discounted Future Value
            target = reward + gamma * v_table[next_state] * (not done)
            expected_value += prob * target
            
        if expected_value > max_value:
            max_value = expected_value
            best_action = action
            
    return best_action

def train(episodes=10000, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.9995, min_epsilon=0.01):
    """
    Trains an agent on FrozenLake-v1 using TD(0) with state values V(s).
    Uses the environment model for action selection (lookahead).
    """
    # Create environment
    env = gym.make('FrozenLake-v1', is_slippery=True)
    
    # Initialize V-table (State Values)
    n_states = env.observation_space.n
    v_table = np.zeros(n_states)
    
    rewards_history = []
    
    print("Starting training with TD(0) - V(s)...")
    
    for episode in range(episodes):
        state, _ = env.reset()
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
            
            # TD(0) Update Rule for V(s)
            # V(S) <- V(S) + alpha * [R + gamma * V(S') - V(S)]
            td_target = reward + gamma * v_table[next_state] * (not done)
            td_error = td_target - v_table[state]
            v_table[state] += alpha * td_error
            print(v_table)
            state = next_state
            total_reward += reward
            
        rewards_history.append(total_reward)
        
        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
        if (episode + 1) % 1000 == 0:
            print(f"Episode {episode + 1}/{episodes} - Mean Reward (last 100): {np.mean(rewards_history[-100:]):.4f} - Epsilon: {epsilon:.4f}")
            
    print("Training finished.")
    
    # Save V-table
    with open('v_table.pkl', 'wb') as f:
        pickle.dump(v_table, f)
    print("V-table saved to v_table.pkl")
    
    # Plot results
    rolling_avg = np.convolve(rewards_history, np.ones(100)/100, mode='valid')
    plt.figure(figsize=(10, 5))
    plt.plot(rolling_avg)
    plt.title("Rolling Average Reward (Window 100) - TD(0) V(s)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig("training_plot.png")
    print("Training plot saved to training_plot.png")
    
    env.close()

if __name__ == "__main__":
    train()
