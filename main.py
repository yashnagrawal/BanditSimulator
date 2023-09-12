# Import the BanditSimulator class from bandit_simulator module
from bandit_simulator_module.bandit_simulator import BanditSimulator

def main():
    epsilon = 0.1
    optimistic_initial_value = 5
    c = 1
    episodes = 1000

    # Ask the user if they want to run the simulations with the default values
    print('Do you want to run the simulations with the default values?')
    print('epsilon = 0.1, optimistic initial value = 5, c = 1, episodes = 1000')
    print('Enter y for yes and n for no')
    user_input = input()
    
    if user_input == 'n':
        # Take input from the user
        epsilon = float(input('Enter epsilon: '))
        optimistic_initial_value = float(input('Enter optimistic initial value: '))
        c = float(input('Enter c: '))
        episodes = int(input('Enter episodes: '))

    # Initialize the simulator
    bandit_simulator = BanditSimulator(epsilon, optimistic_initial_value, c, episodes)

    # Run epsilon greedy simulations
    y_epsilon_greedy = bandit_simulator.run_epsilon_greedy_simulation()
    y_optimistic_initial_values = bandit_simulator.run_optimistic_initial_values_simulation()
    y_ucb = bandit_simulator.run_ucb_simulation()

    # Plot the results
    bandit_simulator.plot_results(y_epsilon_greedy, y_optimistic_initial_values, y_ucb)

if __name__ == '__main__':
    main()