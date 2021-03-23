# ECE 269 Project - RLSmallGroupLearning

Team Members:
1. Saikiran Komatineni
2. Ivan Ferrier

File Descriptions:
1. `actor_critic_walker.py` This file contains the implementation of the plain actor_ciritc model tested with the ipedal-Walker environment. 
2. `group_interpreter.py` This file contains the implementation of our group interpreter algorithm tested with the Cartpole Environment
3. `learn_by_watching.py` This file contains the implementation of our Learn by Watching implementation tested with the Cartpole environment
4. `REINFORCE.py` This file contains the implementation of the basic REINFORCE algorithm tested with the CartPole environment
5. `utils.py` This file contains parameters which can be set to run experiments

Running Code:
To run the code, you can simply run the corresponding file and follow the printed instructions to execute the apropriate actions:
1. `rewards` This command will initiate and complete the RL algorithm
2. `poll` This commad will get results from each learner and plot them
3. `quit` This command will quit and exit the program
4. `plot` This command plots the results up to the point that rewards have been computed. i.e. the plots can be generated even when the program has not completed execution (Note that this is only implemented in the group interpreter file)

