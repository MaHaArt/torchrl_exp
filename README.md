
# README

## Overview

This repository is dedicated to exploring and testing various approaches using the [**Torch RL**](https://pytorch.org/rl/stable/index.html) library. TorchRL is a reinforcement learning library built on top of PyTorch, designed to facilitate the development and deployment of reinforcement learning algorithms.

## Experiments and Tests

### Composite Distributions for Actions

see [train.py](actor_with_composite_distribution/train.py) 

Involves using composite distributions for action selection. This approach allows for more flexible and expressive policy representations, which can be beneficial in complex environments where actions are not easily represented by simple distributions.
 The experiment builds on the environment `ToyNavigation` representing the task to navigate from a starting point to a target in a 2D grid, with allowed movements either along the x-axis or y-axis.

The implementation follows this (very minimal) example in the pytorch github repository: [**composite_actor.py**](https://github.com/pytorch/rl/blob/main/examples/agents/composite_actor.py)

Issue: algorithm does not (yet) converge!!??

### Hyperparameter Tuning in Torch RL with Ray 

see [tune.py](hyperparameter_tuning/tune.py)

The hyperparameters for the training function of the `ToyNavigation` environment are tuned with `ray.tune`.


### Integration with PyTorch Geometric

see [env.py](env_with_pyg/env.py)

Another  test included in this repository is the integration of [**PyTorch Geometric**](https://pytorch-geometric.readthedocs.io/en/latest/)
within observation spaces. This integration aims to leverage graph-based data structures and operations to enhance the representation and processing of observations in environments where relational information is crucial.

Issues:
- a warning is raised, when unbinding the Pyg Data from the observation tensordict.

        The method <bound method TensorDictBase.clear_device_ of TensorDict(
            fields={
            },
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)> wasn't explicitly implemented for tensorclass. This fallback will be deprecated in future releases because it is inefficient and non-compilable. Please raise an issue in tensordict repo to support this method!


## Getting Started

To get started with the experiments in this repository, ensure you have the following prerequisites:

- matplotlib
- networkx
- ray
- tensordict
- torch
- torch_geometric
- torchrl
- tqdm



