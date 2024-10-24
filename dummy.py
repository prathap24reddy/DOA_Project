import numpy as np

# Replace 'training_data_0.npz' with the name of your .npz file
filename = 'training_data_0.npz'

# Load the .npz file
data = np.load(filename)

# Print the contents of the .npz file in CSV format
for key in data.files:
    print(f"{key},", end="")  # Print the key followed by a comma
    # Convert the data to a list and join as a string, then print
    print(",".join(map(str, data[key].flatten())), end="\n")  # Flattening for 2D/3D arrays
