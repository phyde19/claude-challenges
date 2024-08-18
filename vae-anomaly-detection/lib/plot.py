import matplotlib.pyplot as plt
import numpy as np
import random

def plot_mnist_sample(mnist, num_rows=2, num_cols=3):
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(8, 4))
    fig.suptitle('Random MNIST Images', fontsize=16)

    num_images = num_rows * num_cols
    total_images = len(mnist)
    
    if num_images > total_images:
        raise ValueError(f"Requested {num_images} images, but dataset only contains {total_images}")
    
    # Randomly select unique indices
    indices = random.sample(range(total_images), num_images)

    for idx, ax in zip(indices, axes.ravel()):
        image, label = mnist[idx]
        
        ax.imshow(image, cmap='gray')
        ax.axis('off')
        ax.set_title(f'Label: {label}')

    plt.tight_layout()
    plt.show()