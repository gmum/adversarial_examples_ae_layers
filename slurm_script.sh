#!/bin/bash
#SBATCH --gres=gpu:1  # Number of GPUs
#SBATCH --qos=member-multiple-gpu

. /home/bwojcik/miniconda3/etc/profile.d/conda.sh
conda activate cuda100
mkdir -p /mnt/users/bwojcik/local/vae_layers_detector
cd /mnt/users/bwojcik/local/vae_layers_detector || exit 0
#=================================================================================================================
#for model in resnet densenet
#do
#  for dataset in cifar10 cifar100 svhn
#  do
#    for adv in FGSM BIM DeepFool CWL2
#    do
#      python /home/bwojcik/vae_layers_detector/ADV_Samples.py --dataset $dataset --net_type $model --adv_type $adv
#    done
#  done
#done
#=================================================================================================================
#for aetype in waegan wae vae ae
#for aetype in wae
#do
#  for model in resnet densenet
#  do
#    for dataset in cifar10 cifar100 svhn
#    do
#      python -u /home/bwojcik/vae_layers_detector/ADV_train_featuremaps_AE.py --dataset $dataset --dataroot /mnt/users/bwojcik/local/.datasets --net_type $model --ae_type $aetype
#    done
#  done
#done
#=================================================================================================================
# x - r(x) autoencoders
#for aetype in wae
#do
#  for model in resnet densenet
#  do
#    for dataset in cifar10 cifar100 svhn
#    do
#      python -u /home/bwojcik/vae_layers_detector/ADV_train_reconstruction_error_AE.py --dataset $dataset --dataroot /mnt/users/bwojcik/local/.datasets --net_type $model --ae_type $aetype
#    done
#  done
#done
#=================================================================================================================
# train final detector with mean and stddev
# latent supervised
#python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --latent --model SVC cifar10_resnet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
#python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --latent --model SVC cifar100_resnet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
#python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --latent --model SVC svhn_resnet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150

#python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --latent --model SVC cifar10_densenet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
#python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --latent --model SVC cifar100_densenet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
#python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --latent --model SVC svhn_densenet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
# reduced supervised
#python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --model LR cifar10_resnet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
#python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --model LR cifar100_resnet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
#python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --model LR svhn_resnet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150

#python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --model LR cifar10_densenet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
#python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --model LR cifar100_densenet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
#python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --model LR svhn_densenet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
# reduced unsupervised
#python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --model IF cifar10_resnet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
#python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --model IF cifar100_resnet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
#python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --model IF svhn_resnet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150

#python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --model IF cifar10_densenet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
#python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --model IF cifar100_densenet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
#python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --model IF svhn_densenet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
#=================================================================================================================
# train final detector with mean and stddev on rec_error AE data
# latent supervised
python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --both --latent --model SVC cifar10_resnet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --both --latent --model SVC cifar100_resnet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --both --latent --model SVC svhn_resnet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150

python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --both --latent --model SVC cifar10_densenet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --both --latent --model SVC cifar100_densenet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --both --latent --model SVC svhn_densenet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
# reduced supervised
python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --both --model LR cifar10_resnet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --both --model LR cifar100_resnet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --both --model LR svhn_resnet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150

python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --both --model LR cifar10_densenet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --both --model LR cifar100_densenet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --both --model LR svhn_densenet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
# reduced unsupervised
python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --both --model IF cifar10_resnet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --both --model IF cifar100_resnet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --both --model IF svhn_resnet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150

python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --both --model IF cifar10_densenet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --both --model IF cifar100_densenet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --both --model IF svhn_densenet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
#=================================================================================================================
#for ae_type in waegan wae vae ae
#do
#  for model in IF OCSVM SVC LSVC LR RF GB
#  do
#    python -u /home/bwojcik/vae_layers_detector/ADV_train_detector.py --latent --model $model /mnt/users/bwojcik/local/vae_layers_detector/cifar10_resnet_deep_${ae_type}_*
#    python -u /home/bwojcik/vae_layers_detector/ADV_train_detector.py --latent --model $model /mnt/users/bwojcik/local/vae_layers_detector/cifar100_resnet_deep_${ae_type}_*
#    python -u /home/bwojcik/vae_layers_detector/ADV_train_detector.py --latent --model $model /mnt/users/bwojcik/local/vae_layers_detector/svhn_resnet_deep_${ae_type}_*
#
#    python -u /home/bwojcik/vae_layers_detector/ADV_train_detector.py --latent --model $model /mnt/users/bwojcik/local/vae_layers_detector/cifar10_densenet_deep_${ae_type}_*
#    python -u /home/bwojcik/vae_layers_detector/ADV_train_detector.py --latent --model $model /mnt/users/bwojcik/local/vae_layers_detector/cifar100_densenet_deep_${ae_type}_*
#    python -u /home/bwojcik/vae_layers_detector/ADV_train_detector.py --latent --model $model /mnt/users/bwojcik/local/vae_layers_detector/svhn_densenet_deep_${ae_type}_*
#
#    python -u /home/bwojcik/vae_layers_detector/ADV_train_detector.py --model $model /mnt/users/bwojcik/local/vae_layers_detector/cifar10_resnet_deep_${ae_type}_*
#    python -u /home/bwojcik/vae_layers_detector/ADV_train_detector.py --model $model /mnt/users/bwojcik/local/vae_layers_detector/cifar100_resnet_deep_${ae_type}_*
#    python -u /home/bwojcik/vae_layers_detector/ADV_train_detector.py --model $model /mnt/users/bwojcik/local/vae_layers_detector/svhn_resnet_deep_${ae_type}_*
#
#    python -u /home/bwojcik/vae_layers_detector/ADV_train_detector.py --model $model /mnt/users/bwojcik/local/vae_layers_detector/cifar10_densenet_deep_${ae_type}_*
#    python -u /home/bwojcik/vae_layers_detector/ADV_train_detector.py --model $model /mnt/users/bwojcik/local/vae_layers_detector/cifar100_densenet_deep_${ae_type}_*
#    python -u /home/bwojcik/vae_layers_detector/ADV_train_detector.py --model $model /mnt/users/bwojcik/local/vae_layers_detector/svhn_densenet_deep_${ae_type}_*
#  done
#done
#=================================================================================================================
#cd ..
#for ae_type in waegan wae vae ae; do
#    echo $ae_type >>results.txt;
#    echo $ae_type >>latent_results.txt;
#    for type in SVC LSVC LR RF GB OCSVM IF; do
#     echo $type >>results.txt; python /home/bwojcik/vae_layers_detector/ADV_generate_table3.py --ae_type $ae_type --classifier_type $type vae_layers_detector/ >>results.txt;
#     echo $type >>latent_results.txt; python /home/bwojcik/vae_layers_detector/ADV_generate_table3.py --latent --ae_type $ae_type --classifier_type $type vae_layers_detector/ >>latent_results.txt;
#    done
#done
#=================================================================================================================
cd ..
for ae_type in wae; do
    echo $ae_type >>rec_results.txt;
    echo $ae_type >>rec_latent_results.txt;
    for type in SVC LR IF; do
     echo $type >>rec_results.txt; python /home/bwojcik/vae_layers_detector/ADV_generate_table3.py --rec_error --ae_type $ae_type --classifier_type $type vae_layers_detector/ >>rec_results.txt;
     echo $type >>rec_latent_results.txt; python /home/bwojcik/vae_layers_detector/ADV_generate_table3.py --rec_error --latent --ae_type $ae_type --classifier_type $type vae_layers_detector/ >>rec_latent_results.txt;
    done
done
#=================================================================================================================
#for ae_type in wae ae waegan vae; do
#    python -u /home/bwojcik/vae_layers_detector/ADV_generate_feature_importances.py --latent --ae_type $ae_type /mnt/users/bwojcik/local/vae_layers_detector
#    python -u /home/bwojcik/vae_layers_detector/ADV_generate_feature_importances.py --ae_type $ae_type /mnt/users/bwojcik/local/vae_layers_detector
#done
#=================================================================================================================
#for ae_type in wae ae waegan vae; do
#    for model in resnet densenet; do
#        for dataset in cifar10 cifar100 svhn; do
#            python -u /home/bwojcik/vae_layers_detector/ADV_visualise_attack_iteration.py /mnt/users/bwojcik/local/vae_layers_detector/${dataset}_${model}_deep_${ae_type}_*
#        done
#    done
#done
#=================================================================================================================
#python -u /home/bwojcik/vae_layers_detector/ADV_examine_featuremaps.py