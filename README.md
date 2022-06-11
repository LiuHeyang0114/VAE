## VAE Project

We have completed the basic VAE model as required which can achieve clear picture generation in the MNIST dataset.


### Training script

- example.py completes the most basic VAE image generation task with parameters lr and dim.
- example_array.py completes the coordinate input of the two-dimensional latent vector in [-5, 5] and obtains the output image.
- model.py contains VAE model structure.

### Image display

For normal sampling and generation, we store it in the sample folder in the form of sample_'lr'_'dim'.png .

In order to observe the generation characteristics of 2D coordinate sampling, we store the results in sample_array.png .
