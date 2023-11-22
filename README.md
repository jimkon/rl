# Reinforcement Learning lib
A toolkit for Reinforcement Learning algorithms with a basic framework for OpenAI Gym environments and a small collection of RL agent implementations, such as:
### Q Learning
Q-Learning is a famous RL algorithm that aims to learn the Q-value (expected cumulative reward of taking a particular action in a specific state and following the optimal policy thereafter) of each state-action of the environment by approximating the Q-function using the observed rewards. With the knowledge of the Q-value, we can then select the actions that lead to the next best state for any given situation and finally maximise the total reward. The approximation of the Q-function can be done in a few different ways. This package contains the following:      
* Tabular Q Learning
* RBF Q Learning (Radial Basis Function network)
* DNN Q Learning (Deep Neural Network)

### Policy Gradient
Policy gradient is a class of reinforcement learning algorithms that directly optimize the policy function, enabling an agent to learn a stochastic or deterministic policy that maximizes expected cumulative rewards in a given environment. Unlike value-based methods, policy gradient methods directly parameterize the policy and employ gradient ascent to iteratively improve its parameters, making them particularly effective for high-dimensional action spaces and continuous control tasks.

Value and Policy Models are implemented as fully connected Deep Neural Networks.

### Linear Policy Agent
Policy is modeled as a linear combination of features, allowing the agent to approximate actions based on the input state. This package contains the following implementation:
* Input Feature Vector
* Random Feature Vector

