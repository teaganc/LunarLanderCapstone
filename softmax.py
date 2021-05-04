import numpy as np

def softmax(action_values, tau=1.0):
    """
    Args:
        action_values (Numpy array): A 2D array of shape (batch_size, num_actions). 
                       The action-values computed by an action-value network.              
        tau (float): The temperature parameter scalar.
    Returns:
        A 2D array of shape (batch_size, num_actions). Where each column is a probability distribution over
        the actions representing the policy.
    """
    preferences = action_values / tau
    max_preference = np.amax(preferences, 1)
    
    reshaped_max_preference = max_preference.reshape((-1, 1))
    
    exp_preferences = np.exp(preferences - reshaped_max_preference)
    sum_of_exp_preferences = np.sum(exp_preferences, 1)
    
    reshaped_sum_of_exp_preferences = sum_of_exp_preferences.reshape((-1, 1))
    
    action_probs = exp_preferences / reshaped_sum_of_exp_preferences
    
    action_probs = action_probs.squeeze()
    return action_probs


