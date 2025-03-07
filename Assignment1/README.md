# Question 1: GridWorld

This codebase implements the grid-world problem with one taxi from 
[[1](https://library.oapen.org/bitstream/handle/20.500.12657/49587/1/9783854480426.pdf#page=152)]. A pre-trained DQN 
agent is provided that picks up targets (passengers).
Subsequently, the run data (information about the position of the agent and targets and the actions chosen) is 
verified with Z3 [[2](https://link-springer-com.tudelft.idm.oclc.org/chapter/10.1007/978-3-540-78800-3_24)] to satisfy constraints. The verification of the following constraints is already implemented:
* The agent and all the targets are within the grid at each timestep.
* If one target is picked up, a new target appears at a different location.
* All the targets are in distinct locations at each timestep.
* Distance between the agent and the targets is valid at each timestep.
* The agent can only move in one direction at each timestep.

These constraints enforce that the environment is implemented properly according to the problem specification, but do
not yet verify the policy of the DQN agent. 

### Task
Check the Assignment pdf for the requires tasks.

### Installation

1. Make sure you have [anaconda](https://docs.anaconda.com/anaconda/install/index.html) installed. 
2. Clone this repository 
3. Navigate to the `assignment1` folder and run `conda env create -f environment.yml` to create a conda environment 
with the required dependencies.
4. Activate the environment by running `conda activate fm4ls-env`.

### Running the code
You can run the code by running `python main.py`. This will run the DQN on the gridworld environment and perform some 
checking. You can also run `python main.py --help` to see the available options.

### Bugs
If you find any bugs, please let us know. This way, you can help us improve the assignment for next year's students.

### References
[[1](https://link-springer-com.tudelft.idm.oclc.org/chapter/10.1007/978-3-540-78800-3_24)] Parand Alizadeh Alamdari, Guy Avni, Thomas A Henzinger, and Anna Lukina. Formal methods with a touch of magic. 
In Proceedings of the 20th Conference on Formal Methods in Computer-Aided Design, 2020.

[[2](https://library.oapen.org/bitstream/handle/20.500.12657/49587/1/9783854480426.pdf#page=152)] Leonardo De Moura and 
Nikolaj Bjørner. Z3: An efficient smt solver. In Tools and Algorithms for the Construction and Analysis of Systems: 
14th International Conference, TACAS 2008. Proceedings 14, pages 337–340. Springer, 2008.
