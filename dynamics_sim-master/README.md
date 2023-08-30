A generic, extensible simulation library for evolutionary game theory simulations. 
DyPy provides [Moran](http://en.wikipedia.org/wiki/Moran_process), and 
[Wright-Fisher](http://en.wikipedia.org/wiki/Genetic_drift#Wright.E2.80.93Fisher_model) processes,
 as well as [Replicator Dynamics](http://en.wikipedia.org/wiki/Replicator_equation). DyPy makes it simple
to execute complex and robust simulations across a range of parameters and visualize the results with
different types of graphs.

See original documentation [here](http://ecbtln.github.io).

# Installation and Requirements #

Clone or download the package to your local machine. You can run the simulations in the repository rootpath or to run them globally you can add its location to your PYTHONPATH.

DyPy depends on [matplotlib](http://matplotlib.org) for graphing, and [numpy](http://www.numpy.org) and 
[joblib](https://pythonhosted.org/joblib/). To install these dependencies, make sure you are in the root 
directory of the repo and run the following command, which may require sudo.

```
$ pip install -r requirements.txt
```

# General pipeline #

The easiest way to get started with DyPy is to subclass the ```Game``` class and define the game of
 interest to be simulated by defining its payoff matrix appropriately (see Wiki for detailed description and example simulations). The user can also define a function that classifies equilibria as a function of the distribution of players playing each strategy.

Once the game class is defined, choose a dynamics process (Wright-Fisher, Moran or Replicator) and execute the desired simulation. Some options are:

- Simulate a given number of generations of one simulation, and graph the dynamics of each player's 
strategies over time
- Repeat a given simulation multiple times and return the frequency of each resulting equilibria.
- Vary one or more parameters associated with the dynamics or game constructors and graph the effect of this variation on the resulting equilibria, either in the form of 2D or 3D graphs.

The ```GameDynamicsWrapper``` and ```VariedGame``` classes take care of simplifying the simulation and graphing processes, and automatically parallelize the computations across all available cores.
