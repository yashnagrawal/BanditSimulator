import numpy as np
import matplotlib.pyplot as plt

from bandit_simulator_module.bandit import TenArmBandit
from bandit_simulator_module.strategies import EpsilonGreedyInfo, UCBInfo

class BanditSimulator:
    def __init__(self, epsilon, optimistic_initial_value, c, episodes):
        # Initialize three bandits with the same mean and variance
        self.bandit1 = TenArmBandit()
        self.bandit2 = TenArmBandit()
        self.bandit3 = TenArmBandit()

        self.episodes = episodes

        # Initialize the epsilon greedy rewards
        self.epsilon = epsilon
        self.epsilon_greedy_rewards = EpsilonGreedyInfo(self.epsilon)

        #Initialize the optimistic initial values
        self.optimistic_initial_value = optimistic_initial_value
        self.optimistic_initial_values_rewards = EpsilonGreedyInfo(0, self.optimistic_initial_value)

        # Initialize the UCB rewards
        self.c = c
        self.ucb_rewards = UCBInfo(self.c)

    def run_epsilon_greedy_simulation(self):
        # Run epsilon-greedy simulations...
        y_epsilon_greedy = np.zeros(self.episodes)

        for i in range(self.episodes):
            # Find the action with epsilon greedy
            epsilon_greedy_action = self.epsilon_greedy_rewards.find_action_with_epsilon_greedy()

            # Run the bandit with the action
            epsilon_greedy_reward = self.bandit1.run_chosen_bandit(epsilon_greedy_action)

            # Update the expected reward for the action
            self.epsilon_greedy_rewards.update_expected_reward(epsilon_greedy_action, epsilon_greedy_reward)

            # Find the % of times the optimal action was chosen for epsilon greedy and optimistic initial values
            epsilon_greedy_optimal_action = self.bandit1.optimal_action

            epsilon_greedy_optimal_action_count = self.epsilon_greedy_rewards.action_count[epsilon_greedy_optimal_action]

            epsilon_greedy_optimal_action_percentage = epsilon_greedy_optimal_action_count/(i+1)
            
            y_epsilon_greedy[i] = epsilon_greedy_optimal_action_percentage*100
    
        return y_epsilon_greedy
    
    def run_optimistic_initial_values_simulation(self):
        # Run optimistic initial values simulations...
        y_optimistic_initial_values = np.zeros(self.episodes)

        for i in range(self.episodes):
            # Find the action with epsilon greedy
            optimistic_initial_values_action = self.optimistic_initial_values_rewards.find_action_with_epsilon_greedy()

            # Run the bandit with the action
            optimistic_initial_values_reward = self.bandit2.run_chosen_bandit(optimistic_initial_values_action)

            # Update the expected reward for the action
            self.optimistic_initial_values_rewards.update_expected_reward(optimistic_initial_values_action, optimistic_initial_values_reward)

            # Find the % of times the optimal action was chosen for epsilon greedy and optimistic initial values
            optimistic_initial_values_optimal_action = self.bandit2.optimal_action

            optimistic_initial_values_optimal_action_count = self.optimistic_initial_values_rewards.action_count[optimistic_initial_values_optimal_action]

            optimistic_initial_values_optimal_action_percentage = optimistic_initial_values_optimal_action_count/(i+1)
            
            y_optimistic_initial_values[i] = optimistic_initial_values_optimal_action_percentage*100
        
        return y_optimistic_initial_values

    def run_ucb_simulation(self):
        # Run UCB simulations...
        y_ucb = np.zeros(self.episodes)

        for i in range(self.episodes):
            # Find the action with UCB
            ucb_action = self.ucb_rewards.find_action_with_UCB()

            # Run the bandit with the action
            ucb_reward = self.bandit3.run_chosen_bandit(ucb_action)

            # Update the expected reward for the action
            self.ucb_rewards.update_expected_reward(ucb_action, ucb_reward)

            # Find the % of times the optimal action was chosen for epsilon greedy and optimistic initial values
            ucb_optimal_action = self.bandit3.optimal_action

            ucb_optimal_action_count = self.ucb_rewards.action_count[ucb_optimal_action]

            ucb_optimal_action_percentage = ucb_optimal_action_count/(i+1)
            
            y_ucb[i] = ucb_optimal_action_percentage*100
        
        return y_ucb

    def plot_results(self, y_epsilon_greedy, y_optimistic_initial_values, y_ucb):
        # Plot the results...
        x = np.arange(self.episodes)
        epsilon_greedy_label = 'Q1=0, \u03B1=' + str(self.epsilon) + ' epsilon greedy'
        optimistic_initial_values_label = 'Q1=' + str(self.optimistic_initial_value) + ', \u03B1=0 optimistic initial values'
        ucb_label = 'c=' + str(self.c) + ' UCB'

        plt.plot(x, y_epsilon_greedy, label = epsilon_greedy_label, color = 'gray', linewidth = 0.5 )
        plt.plot(x, y_optimistic_initial_values, label = optimistic_initial_values_label, color = 'blue', linewidth = 0.5)
        plt.plot(x, y_ucb, label = ucb_label, color = 'red', linewidth = 0.5)
        plt.legend(loc = 'lower right')
        plt.xlabel('Episodes')
        plt.ylabel('% Optimal Action')

        plt.savefig('bandit_simulator_module/plots/results.png')