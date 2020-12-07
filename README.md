ReaPER implementation in PyTorch
======

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)

## ReaPER
This repo implements ReaPER: Improving Sample Efficiency in Model-Based Latent Imagination.

## Installation
The default environments make use of the dm_control baseline (and corresponding mujoco dependencies).
 Please install those first
To install the package run the following 

`pip install setup.py` 

A Dockerfile with all requirements is also provided


## Training (e.g. ReaPER walker-walk)
```bash
python main.py  --env walker-walk --action-repeat 2 --id name-of-experiement --use-per 2 --use-contrast-loss 1 --belief-l1-penalty 1e-3
```

For best performance with DeepMind Control Suite, try setting environment variable `MUJOCO_GL=egl` (see instructions and details [here](https://github.com/deepmind/dm_control#rendering)).

Use Tensorboard to monitor the training.

`tensorboard --logdir results`

## Acknowledgements
This repository is based on https://github.com/yusukeurakami/dreamer-pytorch for its base codebase
