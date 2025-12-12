import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict

def run_training(episodes, render=False):
    # Create environment
    env = gym.make('Blackjack-v1', sab=True, render_mode='human' if render else None)

    # Initialize Q-table
    # Use defaultdict for sparse/tuple states in Blackjack
    q = defaultdict(lambda: np.zeros(env.action_space.n)) 
    
    # Returns dictionary for Monte Carlo updates
    returns = defaultdict(list)

    # Hyperparameters
    gamma = 0.99 
    epsilon = 1.0         
    epsilon_decay_rate = 0.99995
    min_epsilon = 0.1
    rng = np.random.default_rng()

    rewards_per_episode = np.zeros(episodes)
    wins = 0

    for i in range(episodes):
        state, info = env.reset()
        terminated = False
        truncated = False
        episode = []

        while(not terminated and not truncated):
            # Epsilon-greedy action selection
            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(q[state]))

            new_state, reward, terminated, truncated, info = env.step(action)
            episode.append((state, action, reward))
            state = new_state

        # Monte Carlo Update (First-Visit)
        G = 0
        visited_state_actions = set()
        # Iterate backwards
        for idx in range(len(episode) - 1, -1, -1):
            s, a, r = episode[idx]
            G = gamma * G + r
            
            # Check if this pair appeared earlier in this episode
            pair = (s, a)
            occurred_before = False
            for j in range(idx):
                if episode[j][0] == s and episode[j][1] == a:
                    occurred_before = True
                    break
            
            if not occurred_before:
                returns[pair].append(G)
                q[s][a] = np.mean(returns[pair])

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay_rate)

        # Track results (Win = 1.0)
        # Episode reward is the reward of the last step in Blackjack
        final_reward = episode[-1][2]
        if final_reward == 1.0:
            rewards_per_episode[i] = 1
            wins += 1
        
        # Logging
        if (i + 1) % 1000 == 0:
            avg_reward = np.mean(rewards_per_episode[max(0, i-999):(i+1)])
            print(f"Episode {i+1}: Average Reward (last 1000): {avg_reward:.3f}, Win Rate (overall): {wins/(i+1):.2%}, Epsilon: {epsilon:.4f}")

    env.close()

    print(f"\nTraining finished.")
    print(f"Total Episodes: {episodes}")
    print(f"Total Wins: {wins}")
    print(f"Overall Win Rate: {wins/episodes:.2%}")

    # Plot results
    # Using sliding window sum/avg for cleaner plot
    window = 1000
    moving_features = []
    x_axis = []
    
    # Calculate simple moving average for plotting
    for t in range(0, episodes, window):
        chunk = rewards_per_episode[t:t+window]
        if len(chunk) > 0:
            moving_features.append(np.mean(chunk))
            x_axis.append(t + len(chunk))
            
    plt.figure(figsize=(10,6))
    plt.plot(x_axis, moving_features, label='Win Rate (MA 1000)')
    plt.axhline(y=0.42, color='r', linestyle='--', label='Approx. Optimal (~42%)')
    plt.title('Blackjack Training - Monte Carlo')
    plt.xlabel('Episodes')
    plt.ylabel('Win Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_plot.png')
    print("Plot saved to training_plot.png")

    # Save Q-table
    with open("blackjack_mc.pkl", "wb") as f:
        # Convert defaultdict to dict for cleaner pickling if desired, but defaultdict is fine
        pickle.dump(dict(q), f)
    print("Q-table saved to blackjack_mc.pkl")

if __name__ == '__main__':
    run_training(500000, render=False)
