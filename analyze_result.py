import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load CSV (no headers)
csv_path = 'result.csv'  # Replace with your CSV file path
data = pd.read_csv(csv_path, header=None, names=['x1', 'y1', 'x2', 'y2'])

# Compute Euclidean distances
data['distance'] = np.sqrt((data['x2'] - data['x1'])**2 + (data['y2'] - data['y1'])**2)

# Print distances
print("Euclidean distances between point pairs:")
print(data['distance'])

# Plot distances
plt.figure(figsize=(8, 5))
plt.plot(data.index + 1, data['distance'], marker='o', linestyle='-', color='blue')
plt.title('Euclidean Distance Between Coordinate Pairs')
plt.xlabel('Pair Number')
plt.ylabel('Distance (units)')
plt.grid(True)
plt.xticks(data.index + 1)
plt.tight_layout()
plt.show()
