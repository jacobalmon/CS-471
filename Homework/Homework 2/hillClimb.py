import random

def objFunction1(x): # Function for Q1.
  return 2 - x**2

def objFunction2(x): # Function for Q2.
  return (0.0051*x**5) - (0.1367*x**4) + (1.24*x**3) - (4.456*x**2) + (5.66*x) - 0.287

def hillclimb(initialState, stepSize):
    curr = initialState # set curr to our initial state
    currVal = objFunction1(initialState) # find value of state.

    # Run infinitely, until a max value has been found.
    while True:
      # Find the bounds of the step up and step down of current state.
      neighbors = [curr + stepSize, curr - stepSize]
      # Ensures States are within the bounds.
      neighbors = [x for x in neighbors if -5 <= x <= 5]
      # Find the values of the states we calculated.
      neighborVals = [objFunction1(x) for x in neighbors]
      # We want to use the max value since we're hillclimbing.
      bestNeighborVal = max(neighborVals)

      # Check if the curr needs to be updated.
      if bestNeighborVal > currVal:
        currVal = bestNeighborVal
        curr = neighbors[neighborVals.index(bestNeighborVal)]
      else: # If it doesn't then we ae at our max.
        break

    return currVal

def hillclimb20rand(stepSize):
  # Used -8 as initial since it doesn't exist within the discrete space.
  maxclimb = -8 
  # Randomize 20 States.
  for _ in range(20):
    # Make sure random states are within the range [0, 10].
    randomState = random.randint(0, 10)
    # Set curr to our initial random state.
    curr = randomState
    # Find value of curr's state with function.
    currVal = objFunction2(randomState)

    # Run infinitely, until a max value has been found.
    while True:
      # Find the bounds of the step up and step down of current state.
      neighbors = [curr + stepSize, curr - stepSize]
      # Ensures States are within the bounds.
      neighbors = [x for x in neighbors if 0 <= x <= 10]
      # Find the values of the states we calculated.
      neighborVals = [objFunction2(x) for x in neighbors]
      # We want to use the max value since we're hillclimbing.
      bestNeighborVal = max(neighborVals)

      if bestNeighborVal > currVal:
        # Check if the curr needs to be updated.
        currVal = bestNeighborVal
        curr = neighbors[neighborVals.index(bestNeighborVal)]
      else: # If it doesn't then we ae at our max.
        break

    # Keep track of the maximum value for each initial random state.
    maxclimb = max(maxclimb, currVal)

  return maxclimb

if __name__ == '__main__':
  # Note that the bounds are [-5, 5] for Q1.
  # Note that the bounds are [0, 10] for Q2.

  initialState = 3 # Test Vaue
  stepSize1 = 0.5 # Q1 a). and  Q2.
  stepSize2 = 0.01 # Q1 b).

  # Testing Q1 a).
  print(hillclimb(initialState, stepSize1))
  # Testing Q1 b).
  print(hillclimb(initialState, stepSize2))
  # Testing Q2.
  print(hillclimb20rand(stepSize1))
