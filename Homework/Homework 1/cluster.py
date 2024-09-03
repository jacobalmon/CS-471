import math # used for square root function.

def dfs(node, adj_list, visited):
  # Set current node to true, since we visited it.
  visited[node] = True
  # Loop through all adjacent nodes within our list.
  for adj_node in adj_list[node]:
    # if not visited, then we add it to the stack (recursive stack).
    # and perform dfs again.
    if not visited[adj_node]:
      dfs(adj_node, adj_list, visited)

def isCluster(circles):
  # Get number of circles for later use.
  num_circles = len(circles)

  # Edge Cases
  if num_circles == 0: # No Circles.
    return False
  elif num_circles == 1: # Only 1 Circle.
    return True

  # Creating Adjacency List (Graph) -> (dictionary of lists)
  adj_list = {}
  for i in range(num_circles):
    adj_list[i] = [] # initially empty.


  # Adds Overlaps into Adjacency List.
  for i in range(num_circles):
    for j in range(i + 1, num_circles):
      # Calculate the distance between circles i and j.
      x1, y1, r1 = circles[i]
      x2, y2, r2 = circles[j]
      distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
      # if the distance of said circles are less than or equal to the
      # sum of the radii, then there is a cluster of those two circles.
      if r1 + r2 >= distance >= abs(r1 - r2):
        adj_list[i].append(j)
        adj_list[j].append(i)

  # Perform DFS to check if all circles are clustering.
  visited = []
  for i in range(num_circles):
    visited.append(False)

  dfs(0, adj_list, visited)

  # every single instance of visited must be true for all circles to be
  # clustered.
  for i in visited:
    if i == False:
      return False

  return True


if __name__ == "__main__":
  # x and y represents the coordinates of the center of the circle.
  # r represents the radius of the circle.

  # Test Case 1.
  c_tuples1 = [(1, 3, 0.7), (2, 3, 0.4), (3, 3, 0.9)]
  # Test Case 2.
  c_tuples2 = [(1.5, 1.5, 1.3), (4, 4, 0.7)]
  # Test Case 3.
  c_tuples3 = [(0.5, 0.5, 0.5), (1.5, 1.5, 1.1), (0.7, 0.7, 0.4), (4, 4, 0.7)]
  # Test Case 4.
  c_tuples4 = [(0.5, 0.5, 0.5), (0.5, 0.5, 0.8)]

  print(isCluster(c_tuples1))
  print(isCluster(c_tuples2))
  print(isCluster(c_tuples3))
  print(isCluster(c_tuples4))