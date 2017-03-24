# A PyTorch Implementation for Triplet Networks

This repository contains a [PyTorch](http://pytorch.org/) implementation for triplet networks.

The code provides two different ways to load triplets for the network. First, it contain a simple [MNIST Loader](https://github.com/andreasveit/triplet-network-pytorch/blob/master/triplet_mnist_loader.py) that generates triplets from the MNIST class labels. Second, this repository provides a [Triplet Loader](https://github.com/andreasveit/triplet-network-pytorch/blob/master/triplet_image_loader.py) that loads images from folders, provided a [list of triplets](https://github.com/andreasveit/triplet-network-pytorch/blob/master/triplet_image_loader.py#L22).

### Example usage:

```sh
$ python train.py
```
### Tracking experiments with Visdom

This repository allows to track experiments with [visdom](https://github.com/facebookresearch/visdom). You can use the [VisdomLinePlotter](https://github.com/andreasveit/triplet-network-pytorch/blob/master/train.py#L216) to plot training progress.

<img src="https://github.com/andreasveit/triplet-network-pytorch/blob/master/images/visdom.png?raw=true" width="400">

If this implementation is useful to you and your project, please also consider to cite or acknowledge this code repository.
