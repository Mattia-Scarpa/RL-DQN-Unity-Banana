# RL-Unity-Banana
The main goal is to train an agent to navigate and collect yellow bananas in a large, square world.


[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes. 

### Instructions

The project has been developed using jupyter notebook, in order to run correctly is strongly suggested to run in conda environment. To do su launch './install.sh' to create the environment with the required packages.

The project is built on:

* navigation.ipynb -> The main file, the notebook to run to see the project
* dqn_agent.py -> Agent Class
* model.py -> the NN model for the DQN algorithm
* best_model.pth -> the weights of the best model found during the training
*. report.md -> Project report of the overall implementation 