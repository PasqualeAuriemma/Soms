# For plotting the images
from matplotlib import pyplot as plt
# from soms import SOM
import numpy as np
import os
from som import Som

# Training inputs for RGBcolors
colors = np.array(
     [[0., 0., 0.],
      [0., 0., 1.],
      [0., 0., 0.5],
      [0.125, 0.529, 1.0],
      [0.33, 0.4, 0.67],
      [0.6, 0.5, 1.0],
      [0., 1., 0.],
      [1., 0., 0.],
      [0., 1., 1.],
      [1., 0., 1.],
      [1., 1., 0.],
      [1., 1., 1.],
      [.33, .33, .33],
      [.5, .5, .5],
      [.66, .66, .66]])
color_names = \
    ['black', 'blue', 'darkblue', 'skyblue',
     'greyblue', 'lilac', 'green', 'red',
     'cyan', 'violet', 'yellow', 'white',
     'darkgrey', 'mediumgrey', 'lightgrey']
# get color to test the model
colors_test = np.random.rand(1, 3)
color_names_test = 'o'
# directory where save the model
directory = "model_color"

# do train only if the directory of the model doesn't exist
if not os.path.exists(directory):
                os.makedirs(directory)

# Train a 20x30 SOM with 400 iterations
som = Som(20, 50, 3, directory, 400)
if not os.path.exists(directory):
    # do train
    som.train(colors)

# do test
result = som.test_model(directory, colors_test)

# Get output grid
image_grid = som.get_centroids()
 
# Map colours to their closest neurons
mapped = som.map_vects(colors)


# Plot
plt.subplot(211)
plt.imshow(image_grid)
plt.title('Color SOM')
for i, m in enumerate(mapped):
    plt.text(m[1], m[0], color_names[i], ha='center', va='center',
             bbox=dict(facecolor='white', alpha=0.5, lw=0))
plt.subplot(212)
plt.imshow(image_grid)
plt.title('Test SOM')
for i, m in enumerate(result):
    plt.text(m[1], m[0], color_names_test, ha='center', va='center',
             bbox=dict(facecolor='white', alpha=0.1, lw=0))
plt.show()
