import gymnasium as gym
import numpy as np
import pickle
import time

def get_best_action(env, state, v_table, gamma=0.99):
    """
    Selects the best action using one-step lookahead with the environment model.
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

def evaluate(episodes=10, render=True):
    """
    Evaluates the trained agent on FrozenLake-v1 using the saved V-table.
    """
    # Load V-table
    try:
        with open('v_table.pkl', 'rb') as f:
            v_table = pickle.load(f)
        print("V-table loaded successfully.")
    except FileNotFoundError:
        print("Error: v_table.pkl not found. Train the agent first.")
        return

    # Create environment
    render_mode = "human" if render else None
    env = gym.make('FrozenLake-v1', is_slippery=True, render_mode=render_mode)
    
    total_rewards = 0
    
    print(f"Starting evaluation for {episodes} episodes...")
    
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        
        print(f"--- Episode {episode + 1} ---")
        
        while not done:
            # Greedy action selection using V(s) and Model Lookahead
            action = get_best_action(env, state, v_table)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            state = next_state
            episode_reward += reward
            
            if render:
                time.sleep(0.5) 
        
        total_rewards += episode_reward
        result = "Success" if episode_reward > 0 else "Fail"
        print(f"Result: {result}")
        if render:
            time.sleep(1)

    mean_reward = total_rewards / episodes
    print(f"\nEvaluation finished.")
    print(f"Mean Reward (Success Rate): {mean_reward:.2f}")
    
    env.close()

if __name__ == "__main__":
    evaluate(episodes=5, render=True)
