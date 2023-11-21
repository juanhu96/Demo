
import numpy as np

# Create a sample 2D array
array = np.array([[10, 5, 3, 8, 7],
                  [2, 9, 12, 1, 6],
                  [4, 11, 15, 14, 13]])

# Number of smallest elements to find
k = 5

# Find indices of the k smallest elements for each row
indices = np.argsort(array, axis=1)[:, :2]

# Now, 'indices' contains the indices of the k smallest elements for each row
for j in indices[0]:
    print(j)
print(indices)