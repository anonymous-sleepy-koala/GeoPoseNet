# GeoPoseNet: 6D Pose Estimation with 3D Geometric Features


## Table of Content
- [Overview](#overview)
- [Prerequisites](#Prerequisites)
- [Getting Started](#GettingStarted)
- [Datasets](#datasets)
- [Training](#training)
- [Evaluation](#evaluation)
    - [Evaluation on LineMOD Dataset](#evaluation-on-linemod-dataset)
- [Results](#results)

## Overview

6D pose estimation with 3D geometric features.

## Prerequisites

- Ubuntu 16.04
- CUDA 10.0
- Python 3.7
- PyTorch 1.2
- Minkowski Engine
- PIL
- scipy
- numpy
- pyyaml
- matplotlib
- tqdm
- tensorboardX


## GettingStarted
### Installation

- Clone this repo:
```bash
git clone https://github.com/anonymous-sleepy-koala/GeoPoseNet
cd GeoPoseNet
```

- Install [Minkowski Engine](https://github.com/StanfordVL/MinkowskiEngine)
- Install [PyTorch](http://pytorch.org and) 1.2 and other dependencies (e.g., torchvision).
  - For Conda users, you can create a new Conda environment using `conda env create -f conda_env_config.yml`.

- You might need to rebuild `knn` under `models/knn` if anything goes wrong.


## Datasets
- [LineMOD Dataset](http://campar.in.tum.de/Main/StefanHinterstoisser)
  Download the [preprocessed LineMOD dataset](https://drive.google.com/file/d/1YFUra533pxS_IHsb9tB87lLoxbcHYXt8/view?usp=sharing)

## Training
```bash
conda activate geopose
python train.py --model geopose --name experiment_name --data_path path/to/LineMOD (Optional: --select_obj obj_id)
```

## Evaluation
Example command:
```bash
conda activate geopose
python test.py --model geopose --name experiment_name --data_path ../data/Linemod_preprocessed --checkpoints_dir ../checkpoints 
```
