# A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks

This project is for the paper "[]()". It has been forked from: [deep_Mahalanobis_detector](https://github.com/pokaxpoka/deep_Mahalanobis_detector).

## Preliminaries
It is tested under Ubuntu Linux 16.04.1 and Python 3.6 environment, and requries Pytorch package to be installed:

* [Pytorch](http://pytorch.org/): Only GPU version is available.
* [scipy](https://github.com/scipy/scipy)
* [scikit-learn](http://scikit-learn.org/stable/)

## Downloading Pre-trained Models
We provide six pre-trained neural networks (1) three DenseNets trained on CIFAR-10, CIFAR-100 and SVHN, where models trained on CIFAR-10 and CIFAR-100 are from [odin-pytorch](https://github.com/ShiyuLiang/odin-pytroch), and (2) three ResNets trained on CIFAR-10, CIFAR-100 and SVHN.

* [DenseNet on CIFAR-10](https://www.dropbox.com/s/mqove8o9ukfn1ms/densenet_cifar10.pth?dl=0) / [DenseNet on CIFAR-100](https://www.dropbox.com/s/nosj8oblv3y8tbf/densenet_cifar100.pth?dl=0) / [DenseNet on SVHN](https://www.dropbox.com/s/9ol1h2tb3xjdpp1/densenet_svhn.pth?dl=0)
* [ResNet on CIFAR-10](https://www.dropbox.com/s/ynidbn7n7ccadog/resnet_cifar10.pth?dl=0) / [ResNet on CIFAR-100](https://www.dropbox.com/s/yzfzf4bwqe4du6w/resnet_cifar100.pth?dl=0) / [ResNet on SVHN](https://www.dropbox.com/s/uvgpgy9pu7s9ps2/resnet_svhn.pth?dl=0)

Please place them to `./pre_trained/`.

## Detecting Adversarial Samples

### 0. Generate adversarial samples:
```
# model: ResNet, in-distribution: CIFAR-10, adversarial attack: FGSM  gpu: 0
python ADV_Samples.py --dataset cifar10 --net_type resnet --adv_type FGSM --gpu 0
```

### 1. Train autoencoders:
```
python -u ADV_train_featuremaps_AE.py --dataset $dataset --dataroot /path/to/.datasets --net_type $model --ae_type $aetype
```

### 2. Train final classifier (grid search):
```
python -u ADV_train_detector.py --latent --model $model /path/to/ae/dir/${dataset}_${model}_deep_${ae_type}_*
```

### 3. Generate plots:
```
python -u ADV_generate_feature_importances.py --ae_type $ae_type /path/to/results
python -u ADV_visualise_attack_iteration.py /path_to_results/${dataset}_${model}_deep_${ae_type}_*
python -u ADV_examine_featuremaps.py
```
