# Adversarial Examples Detection and Analysis with Layer-wise Autoencoders

This repository contains the code for paper "[Adversarial Examples Detection and Analysis with Layer-wise Autoencoders](https://arxiv.org/abs/2006.10013)". It has been forked from: [deep_Mahalanobis_detector](https://github.com/pokaxpoka/deep_Mahalanobis_detector). The code for the algorithm from the "odd odds" paper is from [the repo linked in that paper](https://github.com/yk/icml19_public).

## Preliminaries

* [Pytorch](http://pytorch.org/): Only GPU version is available.
* [scipy](https://github.com/scipy/scipy)
* [scikit-learn](http://scikit-learn.org/stable/)
* [scikit-image](https://scikit-image.org/)

## Downloading Pre-trained Models
We provide six pre-trained neural networks (1) three DenseNets trained on CIFAR-10, CIFAR-100 and SVHN, where models trained on CIFAR-10 and CIFAR-100 are from [odin-pytorch](https://github.com/ShiyuLiang/odin-pytroch), and (2) three ResNets trained on CIFAR-10, CIFAR-100 and SVHN.

* [DenseNet on CIFAR-10](https://www.dropbox.com/s/mqove8o9ukfn1ms/densenet_cifar10.pth?dl=0) / [DenseNet on CIFAR-100](https://www.dropbox.com/s/nosj8oblv3y8tbf/densenet_cifar100.pth?dl=0) / [DenseNet on SVHN](https://www.dropbox.com/s/9ol1h2tb3xjdpp1/densenet_svhn.pth?dl=0)
* [ResNet on CIFAR-10](https://www.dropbox.com/s/ynidbn7n7ccadog/resnet_cifar10.pth?dl=0) / [ResNet on CIFAR-100](https://www.dropbox.com/s/yzfzf4bwqe4du6w/resnet_cifar100.pth?dl=0) / [ResNet on SVHN](https://www.dropbox.com/s/uvgpgy9pu7s9ps2/resnet_svhn.pth?dl=0)

Place them into `./pre_trained/`:
```
mkdir pre_trained
wget -O pre_trained/densenet_cifar10.pth https://www.dropbox.com/s/mqove8o9ukfn1ms/densenet_cifar10.pth?dl=0
wget -O pre_trained/densenet_cifar100.pth https://www.dropbox.com/s/nosj8oblv3y8tbf/densenet_cifar100.pth?dl=0
wget -O pre_trained/densenet_svhn.pth https://www.dropbox.com/s/9ol1h2tb3xjdpp1/densenet_svhn.pth?dl=0
wget -O pre_trained/resnet_cifar10.pth https://www.dropbox.com/s/ynidbn7n7ccadog/resnet_cifar10.pth?dl=0
wget -O pre_trained/resnet_cifar100.pth https://www.dropbox.com/s/yzfzf4bwqe4du6w/resnet_cifar100.pth?dl=0
wget -O pre_trained/resnet_svhn.pth https://www.dropbox.com/s/uvgpgy9pu7s9ps2/resnet_svhn.pth?dl=0
```

## Detecting Adversarial Samples

### 0. Generate adversarial samples:
```
# model: ResNet, in-distribution: CIFAR-10, adversarial attack: FGSM  gpu: 0
python /path/to/code/ADV_Samples.py --dataset cifar10 --net_type resnet --adv_type FGSM --gpu 0
```

To generate samples for every attack type:
```
for model in resnet densenet
do
  for dataset in cifar10 cifar100 svhn
  do
    for adv in FGSM BIM DeepFool CWL2 PGD100
    do
      python /path/to/code/ADV_Samples.py --dataset $dataset --net_type $model --adv_type $adv
    done
  done
done
```

### 1. Train autoencoders:
```
python -u ADV_train_featuremaps_AE.py --dataset $dataset --dataroot /path/to/.datasets --net_type $model --ae_type $aetype
```

To train for each combination:
```
for model in resnet densenet
do
  for dataset in cifar10 cifar100 svhn
  do
    python -u /path/to/code/ADV_train_featuremaps_AE.py --dataset $dataset --dataroot ./data --net_type $model --ae_type wae
  done
done
```

### 2. Train final classifier:
```
python -u /path/to/code/ADV_train_detector.py --latent --model $model ${dataset}_${model}_deep_${ae_type}_*
```

For supervised setting:
```
python -u /path/to/code/ADV_train_detector_final.py --latent --model SVC cifar10_resnet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
python -u /path/to/code/ADV_train_detector_final.py --latent --model SVC cifar100_resnet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
python -u /path/to/code/ADV_train_detector_final.py --latent --model SVC svhn_resnet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150

python -u /path/to/code/ADV_train_detector_final.py --latent --model SVC cifar10_densenet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
python -u /path/to/code/ADV_train_detector_final.py --latent --model SVC cifar100_densenet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
python -u /path/to/code/ADV_train_detector_final.py --latent --model SVC svhn_densenet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
```

For unsupervised setting:
```
python -u /path/to/code/ADV_train_detector_final.py --model IF cifar10_resnet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
python -u /path/to/code/ADV_train_detector_final.py --model IF cifar100_resnet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
python -u /path/to/code/ADV_train_detector_final.py --model IF svhn_resnet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150

python -u /path/to/code/ADV_train_detector_final.py --model IF cifar10_densenet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
python -u /path/to/code/ADV_train_detector_final.py --model IF cifar100_densenet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
python -u /path/to/code/ADV_train_detector_final.py --model IF svhn_densenet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
```

The results for each run are available in `*_results_*.txt` files.

### 3. Generate results tables:
```
python /path/to/code/ADV_generate_table3.py --latent --ae_type wae --classifier_type SVC .
python /path/to/code/ADV_generate_table3.py --ae_type wae --classifier_type IF .
```

Tables containing the final results should be displayed:
```
model     dataset    FGSM     BIM      DeepFool  CWL2     PGD100
--------  ---------  -------  -------  --------  -------  ------
densenet  cifar10    1.0000   0.9999   0.9136    0.9775   0.9961
densenet  cifar100   1.0000   0.9988   0.8817    0.9640   0.9663
densenet  svhn       0.9998   0.9975   0.9726    0.9780   0.9943
resnet    cifar10    0.9998   0.9961   0.8641    0.9501   0.9739
resnet    cifar100   1.0000   0.9952   0.7798    0.9641   0.9812
resnet    svhn       0.9981   0.9910   0.9545    0.9731   0.9741
```

```
model     dataset    FGSM     BIM      DeepFool  CWL2     PGD100
--------  ---------  ------   -------  --------  -------  ------
densenet  cifar10    0.7838   0.9751   0.6531    0.6815   0.9420
densenet  cifar100   0.9795   0.9568   0.6186    0.6228   0.8719
densenet  svhn       0.9650   0.9407   0.8380    0.8481   0.9370
resnet    cifar10    0.9724   0.9493   0.7819    0.7429   0.7725
resnet    cifar100   0.9559   0.8023   0.7106    0.7302   0.7098
resnet    svhn       0.9890   0.9549   0.8936    0.9002   0.8362
```

### 4. (Optional) generate plots:
```
python -u /path/to/code/ADV_visualise_attack_iteration.py /path_to_results/${dataset}_${model}_deep_${ae_type}_*
python -u /path/to/code/ADV_examine_featuremaps.py
python -u /path/to/code/ADV_generate_feature_importances.py --ae_type $ae_type .
```

