<div align="center">

# SISR via a Dual Interactive Implicit Neural Network

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>

</div>

## Description

This repository contains the implementation of the paper "Single Image Super-Resolution via a Dual Interactive Implicit Neural Network" (Accepted to WACV 2023).

## How to run

Setting up with conda

```bash
# clone project
git clone https://github.com/robotic-vision-lab/Dual-Interactive-Implicit-Neural-Network.git
cd Dual-Interactive-Implicit-Neural-Network

# create conda environment
conda create -n myenv python=3.8
conda activate myenv

# install requirements
conda install pytorch torchvision cudatoolkit=11.6 -c pytorch -c conda-forge
conda install pytorch-lightning -c conda-forge
conda install omegaconf rich -c conda-forge
```

We used the following datasets in the paper:

[DIV2K - Agustsson, E., & Timofte, R. CVPRW 2017](https://data.vision.ee.ethz.ch/cvl/DIV2K/),

[Set5 - Bevilacqua et al. BMVC 2012](http://people.rennes.inria.fr/Aline.Roumy/results/SR_BMVC12.html),

[Set14 - Zeyde et al. LNCS 2010](https://sites.google.com/site/romanzeyde/research-interests),

[B100 - Martin et al. ICCV 2001](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/),

[Urban100 - Huang et al. CVPR 2015](https://sites.google.com/site/jbhuang0604/publications/struct_sr).

The folder structure should be:
```
├── data
│   ├── DIV2K
│   │   ├── DIV2K_train_HR
│   │       │── 0001.png
|   |       │── ...
|   |       │── 0900.png
│   └── benchmark
│       ├── Set5
|       |   ├── HR
|       |       ├── ...
|       ├── Set14
|       |   ├── HR
|       |       ├── ...
|       ├── B100
|       |   ├── HR
|       |       ├── ...
|       └── Urban100
|           ├── HR
|               ├── ...
└── ...
```

To train a model with default training configuration, located at [configs/default.yaml](configs/default.yaml): 

```bash
python main.py fit -c configs/default_test.yaml --model=SRLitModule --model.arch=diinn --model.mode=3 --model.init_q=False --trainer.logger=TensorBoardLogger --trainer.logger.save_dir=logs/ --trainer.logger.name=3_0
```
You may edit [configs/default.yaml](configs/default.yaml) to best utilize your machine. Here, --model.mode=3 and --model.init_q=False are the configuration of our final model (i.e., model (f) in the paper).


To benchmark a trained model with the benchmark datasets used in the paper:

```bash
python benchmarks.py --ckpt_path=<path_to_checkpoint>                                          
```

To super-resolve an LR image to a desired resolution:

```bash
python demo2.py --lr_path=<path_to_lr_image> --ckpt_path=<path_to_ckpt> --output_size <height> <width>
```
