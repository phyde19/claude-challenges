Challenge: Implement a Variational Autoencoder (VAE) for anomaly detection

# Objective:
Develop a Variational Autoencoder using PyTorch to detect anomalies in a high-dimensional dataset. This challenge will test your understanding of generative models, latent space representations, and probabilistic deep learning.

Input:

- A dataset of normal samples (e.g., MNIST digits, but only including a subset of digits as "normal")
- A small set of anomalous samples for testing (e.g., digits not included in the "normal" set)

Output:

- A trained VAE model
- A method to score new samples for anomaly detection
- Visualizations of the latent space and reconstructed samples

Requirements:

1. Implement the VAE architecture using PyTorch, including encoder and decoder networks.
2. Use the reparameterization trick for sampling from the latent space.
3. Define a loss function that balances reconstruction error and KL divergence.
4. Implement a training loop with appropriate optimizers and learning rate scheduling.
5. Create a method to score new samples for anomaly detection using the trained VAE.
6. Visualize the latent space and reconstructed samples.
7. Evaluate the model's performance using appropriate metrics (e.g., AUC-ROC, precision-recall curve).

Hints and Best Practices:

1. Use PyTorch's nn.Module to define your encoder and decoder networks.
2. Leverage PyTorch's DataLoader for efficient batch processing during training.
3. Implement logging and checkpointing to track training progress and save model states.
4. Use PyTorch's built-in distribution classes (e.g., torch.distributions.Normal) for working with probabilistic components.
5. Consider using PyTorch Lightning to structure your code and simplify the training process.
6. Implement unit tests for critical components of your model and training pipeline.
7. Use type hints and docstrings to improve code readability and maintainability.
8. Consider implementing early stopping to prevent overfitting.
9. Experiment with different latent space dimensions and analyze their impact on performance.
10. Use dimensionality reduction techniques (e.g., t-SNE or UMAP) to visualize the latent space in 2D or 3D.

This challenge will push you to:

1. Deepen your understanding of generative models and their applications in anomaly detection.
2. Gain experience with probabilistic deep learning concepts.
3. Improve your PyTorch skills, particularly in implementing custom architectures and loss functions.
4. Practice MLOps skills like logging, checkpointing, and model evaluation.
5. Apply your mathematical background to understand and implement the theoretical foundations of VAEs.