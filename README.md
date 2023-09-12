# Bandit Simulator Readme

This repository contains a Python codebase for simulating and comparing the performance of different strategies in solving the multi-armed bandit problem. The multi-armed bandit problem is a classic dilemma in reinforcement learning and decision-making where an agent must decide which arm (action) to pull in order to maximize its cumulative reward over time. The project is structured as follows:

````plaintext
project_root/
│
├── main.py
│
├── bandit_simulator_module/
│   ├── __init__.py
│   ├── bandit_simulator.py
│   ├── strategies.py
│   ├── bandit.py
│   └── plots/
│
└── README.md
````

# Contents
1. Description
2. Usage
3. Folder Structure
4. Code base

# 1. Description

The multi-armed bandit problem models a scenario where an agent faces a row of slot machines (bandits), each with an unknown probability distribution of rewards. The agent aims to maximize its total reward by selecting the best bandit to play over a series of episodes.

This codebase provides three different strategies for solving the multi-armed bandit problem:

1. Epsilon-Greedy Strategy: The agent selects the best action most of the time (exploitation) but occasionally explores other actions with a probability epsilon.
2. Optimistic Initial Values Strategy: The agent starts with an optimistic estimate of each bandit's value and gradually refines these estimates based on the observed rewards.
3. Upper Confidence Bound (UCB) Strategy: The agent balances exploration and exploitation by selecting actions based on their estimated upper confidence bounds.

The code simulates these strategies, records the percentage of times the optimal action is chosen, and plots the results over multiple episodes.

# 2. Usage

To run the bandit simulator, follow these steps:

1. Clone the repository to your local machine.
2. Navigate to the project root directory.
3. Run the following command:

```bash
python main.py
````

- The script will prompt you to either use default parameters or provide custom values.

- After the simulation completes, the results will be plotted, and the plot will be saved as bandit_simulator_module/plots/results.png.

# 3. Folder Structure

The project is structured as follows:

- project_root/: The root directory of the project.
- main.py: The main script to run the bandit simulator with user-defined parameters.
- bandit_simulator_module/: A Python package containing the core modules for the bandit simulation.
  - init.py: An empty file to indicate that the directory is a Python package.
  - bandit_simulator.py: Contains the BanditSimulator class responsible for running the simulations and plotting the results.
  - strategies.py: Contains the classes for the Epsilon-Greedy and UCB strategies.
  - bandit.py: Defines the TenArmBandit class representing the multi-armed bandit environment.
  - plots/: A directory where simulation result plots are saved.

# 4. Code base

- main.py: The main script that initializes and runs the bandit simulator. It allows you to set simulation parameters interactively or use default values.

- bandit_simulator.py: Contains the BanditSimulator class, which orchestrates the simulations. It includes methods to run epsilon-greedy, optimistic initial values, and UCB simulations, as well as a method to plot the results.

- strategies.py: Defines the EpsilonGreedyInfo and UCBInfo classes that implement the epsilon-greedy and UCB strategies, respectively. These classes handle action selection and expected reward updates.

- bandit.py: Defines the TenArmBandit class, which represents the multi-armed bandit environment. It generates random rewards with a normal distribution for each arm.
