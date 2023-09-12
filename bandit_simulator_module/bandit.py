import numpy as np

class TenArmBandit:
    def __init__(self):
        # The reward is a random variable with mean 0 and variance 1
        self.normalized_final_rewards = np.random.normal(0,1,10)
        self.optimal_action = np.argmax(self.normalized_final_rewards)
    
    def run_chosen_bandit(self, arm_no):
        # Run the bandit with the arm_no/action
        # Return the reward
        return np.random.normal(self.normalized_final_rewards[arm_no],1,1)