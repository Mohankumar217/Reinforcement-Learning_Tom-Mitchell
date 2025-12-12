import gymnasium as gym
import numpy as np
import pickle
import os
import pygame
from collections import defaultdict

def get_action(state, q_table, env):
    # Helper to choose greedy action
    if state in q_table:
        return int(np.argmax(q_table[state]))
    else:
        return env.action_space.sample()

def evaluate():
    env = gym.make('Blackjack-v1', sab=True, render_mode=None)
    
    filename = 'blackjack_mc.pkl'
    if not os.path.exists(filename):
        print(f"Error: {filename} not found. Run train.py first.")
        return None

    print(f"Loading Q-table from {filename}...")
    with open(filename, 'rb') as f:
        q_table = pickle.load(f)
    
    # Handle both dict and defaultdict
    if not isinstance(q_table, (dict, defaultdict)):
        print("Error: Loaded Q-table is not a dictionary.")
        return None

    n_episodes = 100000
    wins = 0
    losses = 0
    draws = 0
    
    print(f"Starting evaluation for {n_episodes} episodes...")
    
    for i in range(n_episodes):
        state, info = env.reset()
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            action = get_action(state, q_table, env)
            state, reward, terminated, truncated, info = env.step(action)
            
        if reward == 1.0:
            wins += 1
        elif reward == -1.0:
            losses += 1
        else:
            draws += 1
            
    total = n_episodes
    print("\n--- Evaluation Results ---")
    print(f"Episodes: {total}")
    print(f"Wins: {wins} ({wins/total*100:.2f}%)")
    print(f"Losses: {losses} ({losses/total*100:.2f}%)")
    print(f"Draws: {draws} ({draws/total*100:.2f}%)")
    print("--------------------------")
    print("Metrics Comparison:")
    print(f"Agent Win Rate: {wins/total*100:.2f}%")
    print("Ideal Win Rate (Approx): ~42-43%")
    
    return q_table

def run_visual_demo(q_table, episodes=5):
    if q_table is None:
        return

    print(f"\n--- Visual Demo ({episodes} Episodes) ---")
    print("Opening Pygame window...")
    
    # Create env with render_mode='human'
    env = gym.make('Blackjack-v1', sab=True, render_mode='human')
    
    for i in range(episodes):
        state, info = env.reset()
        terminated = False
        truncated = False
        print(f"Demo Episode {i+1}...")
        
        while not (terminated or truncated):
            # Slow down slightly so user can follow the action if needed
            # (though 'human' mode usually handles timing well in Blackjack)
            action = get_action(state, q_table, env)
            state, reward, terminated, truncated, info = env.step(action)
            
        # Final result of episode
        result = "WIN" if reward == 1.0 else "LOSS" if reward == -1.0 else "DRAW"
        print(f"Result: {result}")
        pygame.time.wait(1000) # Wait 1s between hands
        
    env.close()
    print("Demo finished.")

if __name__ == "__main__":
    trained_q = evaluate()
    if trained_q:
        run_visual_demo(trained_q)
