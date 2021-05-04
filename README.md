# Lunar Lander Capstone

This is a solution to the OpenAI gym lunar lander environment. It uses Q-Learning to learn a softmax policy. It includes a neural network with a single hidden and a relu activation function to approximate an action value function. It also does planning updates with minibatches from a replay buffer of previous states to improve sample efficiency. The gradient follwing algorithm in the updates is an implementation of Adam. 

