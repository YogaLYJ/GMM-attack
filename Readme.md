# GMM Attack

Code for 'Transferable Adversarial Examples Based on Global Smooth Perturbations' (Computers & Security).

[Paper](https://authors.elsevier.com/c/1fRntc43uylct).

## Recommended Environment
* Python 3.7
* Cuda 10.1
* PyTorch 0.4

## Prerequisites
* Install dependencies: `pip3 install -r requirements.txt`.

## Document Description

### ./checkpoints, ./models, and./util

* These are used for LPIPS, please refer to LPIPS's [paper](https://arxiv.org/abs/1801.03924) and [code](http://richzhang.github.io/PerceptualSimilarity/).

### ./test_images

* Original images and their true lables

### Others
* main.py: demo for GMM attack.
* GMM_model.py: code for a single Gaussian model and GMM.
* GMM.py: GMM attack.
* common.py: clipping functions for parameters in GMM.
* threatmodels.py: source models.
* util_save.py: save adversarial examples and perturbations.

## Quick Start

```
python main.py --model=resnet50 --filename=ILSVRC2012_val_00000724.JPEG --true_label=449 --Gaussian_number=20
```

## Citation
@article{Liu_2022_GMM,
author = {Liu, Yujia and Jiang, Ming and Jiang, Tingting},
title = {Transferable Adversarial Examples Based on Global Smooth Perturbations,
journal = {Computers & Security},
volume = {121},
doi = {https://doi.org/10.1016/j.cose.2022.102816},
year = {2022}
}