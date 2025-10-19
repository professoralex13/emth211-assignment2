import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# Part A and B

def rank_approximation(A, r):
    U, s, VT = np.linalg.svd(A)

    return U[:, :r] * s[:r] @ VT[:r]

def read_image(filename):
    image = mpimg.imread(filename).astype(np.float64)

    if image.max() > 1.:
        image /= 255.

    if image.shape[2] > NUM_CHANNELS:
        image = image[..., :NUM_CHANNELS]

    return image

CHANNEL_NAMES = ['R', 'G', 'B']
NUM_CHANNELS = len(CHANNEL_NAMES)

RANKS=5

FILENAMES = ['owl.jpg', 'shark.jpg']

fig1, axes1 = plt.subplots(nrows=len(FILENAMES), ncols=RANKS + 1, figsize=[20, 8])

for i, filename in enumerate(FILENAMES):
    image = read_image(filename)

    for j, rank in enumerate([min(image.shape[:2])] + list(range(1, RANKS + 1))):
        # Running this logic for the max rank isn't really neccessary, but oh well

        new_image = np.zeros_like(image)

        for channel_index in range(NUM_CHANNELS):
            new_image[..., channel_index] = rank_approximation(image[..., channel_index], rank)

        axis = axes1[i, j]

        axis.set_title(f"Rank {rank}")
        axis.imshow(np.clip(new_image, 0., 1.))
        axis.axis('off')
        
fig1.savefig('2b.graph.png')

# Part D

fig2, axes2 = plt.subplots(nrows=len(FILENAMES), ncols=NUM_CHANNELS + 1, figsize=[20, 8])

for i, filename in enumerate(FILENAMES):
    image = read_image(filename)

    axes2[i, 0].set_title(filename)
    axes2[i, 0].imshow(np.clip(image, 0., 1.))
    axes2[i, 0].axis('off')

    for j in range(NUM_CHANNELS):
        new_image = np.zeros_like(image)

        for channel_index in range(NUM_CHANNELS):
            new_image[..., channel_index] = rank_approximation(image[..., channel_index], 10 if channel_index == j else 1)

        axis = axes2[i, j + 1]

        axis.set_title(', '.join([f"{name}: {10 if channel_index == j else 1}" for channel_index, name in enumerate(CHANNEL_NAMES)]))
        axis.imshow(np.clip(new_image, 0., 1.))
        axis.axis('off')

fig2.savefig('2d.graph.png')