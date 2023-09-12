import numpy as np

class EpsilonGreedyInfo:
    def __init__(self, epsilon, initial_expected_reward = 0):
        # Initialize the expected rewards for each arm to initial_expected_reward
        self.expected_rewards = np.ones(10)*(initial_expected_reward)
        # Initialize the action count for each arm to 0
        self.action_count = np.zeros(10)
        # Initialize the epsilon
        self.epsilon = epsilon

    def find_action_with_epsilon_greedy(self):
        # Find the action with epsilon greedy
        random_number = np.random.uniform(0,1,1)
        if random_number < self.epsilon:
            # Explore
            return np.random.randint(0,10)
        else:
            # Exploit
            return np.argmax(self.expected_rewards)

    def update_expected_reward(self, action_arm, reward):
        # Update the expected reward for the action_arm
        self.action_count[action_arm] += 1
        alpha = 1/self.action_count[action_arm]
        diff_reward = reward - self.expected_rewards[action_arm]
        old_expected_reward = self.expected_rewards[action_arm]
        new_expected_reward = old_expected_reward + alpha*diff_reward

        self.expected_rewards[action_arm] = new_expected_reward[0]

class UCBInfo:
    def __init__(self, c):
        # Initialize the expected rewards for each arm to 0
        self.expected_rewards = np.zeros(10)
        # Initialize the action count for each arm to 0
        self.action_count = np.zeros(10)
        # Initialize the c
        self.c = c
        # Initialize the time step
        self.time_step = 0

    def find_action_with_UCB(self):
        # Find the action with UCB
        # Return the action
        # If the action has not been taken before, return it
        # Else, find the action with UCB
        for i in range(10):
            if self.action_count[i] == 0:
                return i

        # Find the action with UCB
        # Find the UCB for each arm
        ucb = np.zeros(10)
        for i in range(10):
            ucb[i] = self.expected_rewards[i] + self.c*np.sqrt(np.log(np.sum(self.time_step))/self.action_count[i])
        # Return the action with the maximum UCB
        return np.argmax(ucb)

    def update_expected_reward(self, action_arm, reward):
        # Update the expected reward for the action_arm
        self.action_count[action_arm] += 1
        alpha = 1/self.action_count[action_arm]
        diff_reward = reward - self.expected_rewards[action_arm]
        old_expected_reward = self.expected_rewards[action_arm]
        new_expected_reward = old_expected_reward + alpha*diff_reward

        self.expected_rewards[action_arm] = new_expected_reward[0]

        self.time_step += 1

