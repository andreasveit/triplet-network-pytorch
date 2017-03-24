# A PyTorch Implementation for Triplet Networks

This repository contains a [PyTorch](http://pytorch.org/) implementation for triplet networks.

The code provides two different ways to load triplets for the network. First, it contain a simple [MNIST Loader](https://github.com/andreasveit/triplet-network-pytorch/blob/master/triplet_mnist_loader.py) that generates triplets from the MNIST class labels. Second, this repository provides a [Triplet Loader](https://github.com/andreasveit/triplet-network-pytorch/blob/master/triplet_image_loader.py) that loads images from folders, provided a [list of triplets](https://github.com/andreasveit/triplet-network-pytorch/blob/master/triplet_image_loader.py#L22).

Example usage:
```sh
$ python train.py
```

