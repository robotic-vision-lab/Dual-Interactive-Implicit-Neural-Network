<div align="center">

# SISR via a Dual Interactive Implicit Neural Network

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description

This repository contains the implementation of the paper "Single Image Super-Resolution via a Dual Interactive Implicit Neural Network" (Accepted to WACV 2023).

## How to run

Install dependencies

```bash
# clone project
git clone https://github.com/YourGithubName/your-repo-name
cd your-repo-name

# [OPTIONAL] create conda environment
conda create -n myenv python=3.8
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
conda install pytorch torchvision cudatoolkit=11.6 -c pytorch -c conda-forge
conda install pytorch-lightning -c conda-forge
conda install omegaconf rich -c conda-forge
```

Train models with default configuration, located at [configs/default.yaml](configs/default.yaml). You may edit the config file to best utilize your machine.

```bash
python main.py fit -c configs/default_test.yaml --model=SRLitModule --model.arch=imsisr --model.mode=3 --model.init_q=False --trainer.logger=TensorBoardLogger --trainer.logger.save_dir=logs/ --trainer.logger.name=3_0
```

To benchmark a trained model with the benchmark datasets used in the paper.

```bash
python benchmarks.py --ckpt_path=<path_to_checkpoint>                                          
```


