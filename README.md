# Actor Critic play Tic Tac Toe
A *Artificial Intelligence* that can play both X and O in a 
game of Tic Tac Toe.

This AI is made up of three main components: the *Actor*, 
the *Critic* and the *Agent*.  

## Actor
Is the component that guesses the best action based in the 
current state of the game.

- It is made up of a Neural Network.
- It has the function `call` that returns the probability
of choosing each action.


## Critic
The critic judges the action taken by the Actor.

- It is made up of a Neural Network.
- It has the function `call` that returns the estimated 
value of the state, that will be used to caclculate the
advantage of the state

## Agent
Makes the moves based on the choices of the Actor.

- Has the actor
- Has the critic
- Has a optimizer
- Has a gamma
- Has the function `select_saction` that selects the
action based on the choices of the actor
- Has the function `evaluate` taht returns the value
given by the critic based on the current state
- Has the function `compute_returns` that calculate the
future return 
- Has the function `update` that returns the loss
