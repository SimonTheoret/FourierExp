
# Table of Contents

1.  [Notes](#orga324782)
    1.  [Architecture](#org7a56453)
2.  [Features](#orge86ab04)
3.  [](#orgb71457f)
    1.  [Interfaces](#orga6ea99d)
    2.  [Building blocks](#orgb468f01)
4.  [Reading](#orgffec443)
5.  [Questions](#orgbebe2d0)
6.  [Other experiments](#org0a5bde0)
    1.  [Computing the l<sub>1</sub> norm distribution of the Gaussian augmentations](#orgc662fea)
    2.  [Computing accuracy with a few different bandwidth](#org4347d30)


<a id="orga324782"></a>

# Notes


<a id="org7a56453"></a>

## Architecture

1.  Take in Cifar-10 Dataset
2.  Apply transformations to batch
3.  Optionally apply gaussian preprocessing to batch
4.  Select the optimizer
5.  Do k times:
    1.  Train for 1 epoch with the optimizer
    2.  Test the model:
        -   If ADV training, compute the adv examples and train on it
        -   Compute the accuracy on Low Fourier-filtered images on the test dataset
        -   Compute the accuracy on High Fourier-filtered images on the test dataset
        -   Compute the accuracy on the original images from the test dataset
    3.  Save the results into a dict
6.  Save the experiment results


<a id="orge86ab04"></a>

# Features

-   [ ] ADV training
-   [X] Gaussian Augmentation
-   [X] Regular training


<a id="orgb71457f"></a>

# DONE 


<a id="orga6ea99d"></a>

## DONE Interfaces

-   [X] Create interfaces for moving parts


<a id="orgb468f01"></a>

## DONE Building blocks

-   [X] Build Dataset (Cifar10) class
    -   [X] `download_dataset` method
    -   [X] `load_dataset` method
    -   [X] `save_datasset` method
    -   [X] `split_train_test` method
    -   [X] <del>`apply_transformation` method</del>
    -   [X] `apply_gaussian` method
    -   [X] <del>`generate_adversarial` method</del>
    -   [X] <del>`next_batch` **methods**</del>
    -   [X] Clip and flip for every images
-   [X] Build trainer class
    -   [X] `train` method
    -   [X] `test` method
    -   [X] `compute_metrics`


<a id="orgffec443"></a>

# Reading

Readings from the original paper [Fourier perspective](https://proceedings.neurips.cc/paper/2019/file/b05b57f6add810d3b7490866d74c0053-Paper.pdf).

1.  Section 4.1
2.  Section 4.2
3.  Section 4.4


<a id="orgbebe2d0"></a>

# Questions

-   Why is the gaussian data augmentation not working ?
-   What is the fourrier transform of Gaussian noise? Gaussian noise!


<a id="org0a5bde0"></a>

# Other experiments


<a id="orgc662fea"></a>

## TODO Computing the l<sub>1</sub> norm distribution of the Gaussian augmentations

The planning:

-   Sample a few thousand vectors
-   Compute the l<sub>1</sub> norm over these vectors
-   Do a few graph, visualization


<a id="org4347d30"></a>

## TODO Computing accuracy with a few different bandwidth

The planning:

-   For model *i*:
    -   For bandwidth *B*:
        1.  Compute the fourier transform of the dataset
        2.  Apply low frequency pass
        3.  Compute accuracy with low frequency passed images
        4.  Apply high frequency pass
        5.  Compute accuracy with high frequency passed images
    -   Save the data:
        -   Model
        -   Optimizer used
        -   High frequency accuracy
        -   Low frequency accuracy
        -   bandwidth

