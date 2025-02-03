# FedLGD
The official implementation of paper "[Federated Learning on Virtual Heterogeneous Data with Local-GLobal Dataset Distillation](https://arxiv.org/abs/2303.02278)" accepted at TMLR 2024

![FedLGD_main_figure](/img/FedLGD.png)

## Abstract

While Federated Learning (FL) is gaining popularity for training machine learning models in a decentralized fashion, numerous challenges persist, such as asynchronization, computational expenses, data heterogeneity, and gradient and membership privacy attacks. Lately, dataset distillation has emerged as a promising solution for addressing the aforementioned challenges by generating a compact synthetic dataset that preserves a model's training efficacy. *However, we discover that using distilled local datasets can amplify the heterogeneity issue in FL.* To address this, we propose **Fed**erated Learning on Virtual Heterogeneous Data with **L**ocal-**G**lobal **D**ataset **D**istillation (**FedLGD**), where we seamlessly integrate dataset distillation algorithms into FL pipeline and train FL using a smaller synthetic dataset (referred as *virtual data*).
Specifically, to harmonize the domain shifts, we propose iterative distribution matching to inpaint global information to *local virtual data* and use federated gradient matching to distill *global virtual data* that serve as anchor points to rectify heterogeneous local training, without compromising data privacy. We experiment on both benchmark and real-world datasets that contain heterogeneous data from different sources, and further scale up to an FL scenario that contains a large number of clients with heterogeneous and class-imbalanced data. Our method outperforms *state-of-the-art* heterogeneous FL algorithms under various settings.

![Amplified_heteregeneity](/img/amplified_hetero.png)

## System requirement

We recommend using conda to install the environment.
Please use [environment.txt](https://github.com/ubc-tea/DESA/blob/main/environment.txt) to set up the conda environment.

## Verify pretrained DIGITS model


### Download DIGITS, CIFAR10C, and RETINA data

Please download the [DIGIT](https://drive.google.com/file/d/1zOddGFldTRyrMTv_5O0l2rbGsYZvOwo8/view?usp=sharing), [RETINA](https://drive.google.com/file/d/1MMK8bourqVFyJ-UbuMgB40n-xTYHlHl2/view?usp=sharing), and [CIFAR10C](https://drive.google.com/drive/folders/1BIBvskSH-gbt7s50fRrJO5Rld1XXqCbb?usp=sharing) and put them under data folder.

Note that we only provide the pretrained model and distilled data for DIGITS.

### Run training

The training command is

```bash
python fedlgd_main.py --dataset [dataset] --wk_iters [local_epoch] --iters [total_communication_rounds] --batch [batch_size] --model [architecture] --ipc [image_per_class] --lambda_sim [coefficient] --reg_loss [regularization_loss]
```

Please refer to our paper for the detailed hyperparameter settings.


### Run testing on DIGITS

Please use the following command to test the results of DIGITS experiment. The ckpt file can be found in [here]((/checkpoint/server_model_local1_100.pt)). Please download it and put in the SAVE_PATH folder generated based on the hyperparameter setting.

```
python fedlgd_main.py --dataset digits --wk_iters 1 --iters 100 --batch 32 --model ConvNet --ipc 50 --lambda_sim 10 --reg_loss contrastive --test True
```

## Citation
If you find this work helpful, please cite our paper as follows:
```
@article{huang2023federated,
  title={Federated virtual learning on heterogeneous data with local-global distillation},
  author={Huang, Chun-Yin and Jin, Ruinan and Zhao, Can and Xu, Daguang and Li, Xiaoxiao},
  journal={arXiv preprint arXiv:2303.02278},
  year={2023}
}
```
