<div align="center">

# Your Project Name

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
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

Train models with default configuration, located at [configs/default.yaml/](configs/default.yaml). You may edit the config file to best utilize your machine.

```bash
python main.py fit -c configs/default_test.yaml --model=SRLitModule --model.arch=imsisr --model.mode=3 --model.init_q=False --trainer.logger=TensorBoardLogger --trainer.logger.save_dir=logs/ --trainer.logger.name=3_0
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python train.py trainer.max_epochs=20 datamodule.batch_size=64
```
