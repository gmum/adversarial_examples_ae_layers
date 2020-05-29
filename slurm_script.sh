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
#    for adv in FGSM BIM DeepFool CWL2 PGD100
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
#python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --both --latent --model SVC cifar10_resnet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
#python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --both --latent --model SVC cifar100_resnet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
#python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --both --latent --model SVC svhn_resnet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
#
#python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --both --latent --model SVC cifar10_densenet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
#python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --both --latent --model SVC cifar100_densenet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
#python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --both --latent --model SVC svhn_densenet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
## reduced supervised
#python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --both --model LR cifar10_resnet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
#python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --both --model LR cifar100_resnet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
#python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --both --model LR svhn_resnet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
#
#python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --both --model LR cifar10_densenet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
#python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --both --model LR cifar100_densenet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
#python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --both --model LR svhn_densenet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
## reduced unsupervised
#python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --both --model IF cifar10_resnet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
#python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --both --model IF cifar100_resnet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
#python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --both --model IF svhn_resnet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
#
#python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --both --model IF cifar10_densenet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
#python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --both --model IF cifar100_densenet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
#python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --both --model IF svhn_densenet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
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
#  echo $ae_type >>results.txt
#  echo $ae_type >>latent_results.txt
#  for type in SVC LSVC LR RF GB OCSVM IF; do
#    echo $type >>results.txt
#    python /home/bwojcik/vae_layers_detector/ADV_generate_table3.py --ae_type $ae_type --classifier_type $type vae_layers_detector/ >>results.txt
#    echo $type >>latent_results.txt
#    python /home/bwojcik/vae_layers_detector/ADV_generate_table3.py --latent --ae_type $ae_type --classifier_type $type vae_layers_detector/ >>latent_results.txt
#  done
#done
#=================================================================================================================
#cd ..
#for ae_type in wae; do
#    echo $ae_type >>both_results.txt;
#    echo $ae_type >>both_latent_results.txt;
#    for type in SVC LR IF; do
#     echo $type >>both_results.txt; python /home/bwojcik/vae_layers_detector/ADV_generate_table3.py --both --ae_type $ae_type --classifier_type $type vae_layers_detector/ >>both_results.txt;
#     echo $type >>both_latent_results.txt; python /home/bwojcik/vae_layers_detector/ADV_generate_table3.py --both --latent --ae_type $ae_type --classifier_type $type vae_layers_detector/ >>both_latent_results.txt;
#    done
#done
#=================================================================================================================
for ae_type in wae; do
   python -u /home/bwojcik/vae_layers_detector/ADV_generate_feature_importances.py --latent --ae_type $ae_type /mnt/users/bwojcik/local/vae_layers_detector
   python -u /home/bwojcik/vae_layers_detector/ADV_generate_feature_importances.py --ae_type $ae_type /mnt/users/bwojcik/local/vae_layers_detector
done
#=================================================================================================================
python -u /home/bwojcik/vae_layers_detector/ADV_examine_featuremaps.py
#=================================================================================================================
# for ae_type in wae ae waegan vae; do
for ae_type in wae; do
   for model in resnet densenet; do
       for dataset in cifar10 cifar100 svhn; do
           python -u /home/bwojcik/vae_layers_detector/ADV_visualise_attack_iteration.py /mnt/users/bwojcik/local/vae_layers_detector/${dataset}_${model}_deep_${ae_type}_*_150
       done
   done
done
#=================================================================================================================
# rerun for PGD100
#RUNS=5
## train final detector with mean and stddev
## latent supervised
#python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --runs $RUNS --latent --model SVC cifar10_resnet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
#python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --runs $RUNS --latent --model SVC cifar100_resnet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
#python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --runs $RUNS --latent --model SVC svhn_resnet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
#
#python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --runs $RUNS --latent --model SVC cifar10_densenet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
#python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --runs $RUNS --latent --model SVC cifar100_densenet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
#python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --runs $RUNS --latent --model SVC svhn_densenet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
## reduced supervised
#python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --runs $RUNS --model LR cifar10_resnet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
#python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --runs $RUNS --model LR cifar100_resnet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
#python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --runs $RUNS --model LR svhn_resnet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
#
#python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --runs $RUNS --model LR cifar10_densenet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
#python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --runs $RUNS --model LR cifar100_densenet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
#python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --runs $RUNS --model LR svhn_densenet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
## reduced unsupervised
#python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --runs $RUNS --model IF cifar10_resnet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
#python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --runs $RUNS --model IF cifar100_resnet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
#python -u /h/bwojcik/vae_layers_detector/ADV_train_detector_final.py --runs $RUNS --model IF svhn_resnet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
#
#python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --runs $RUNS --model IF cifar10_densenet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
#python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --runs $RUNS --model IF cifar100_densenet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
#python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_final.py --runs $RUNS --model IF svhn_densenet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
#=================================================================================================================
# compare with PGD-100 Mahalanobis results
# echo "===MAHALANOBIS===" >> Mahalanobis_run.txt;
# for model in resnet densenet; do
#   for dataset in cifar10 cifar100 svhn; do
#     for adv_type in FGSM BIM DeepFool CWL2 PGD100; do
# #    for adv_type in PGD100; do
#       python /home/bwojcik/vae_layers_detector/ADV_Generate_LID_Mahalanobis.py --dataset $dataset --net_type $model --adv_type $adv_type --gpu 0 >> Mahalanobis_run.txt;
#     done
#   done
#   python /home/bwojcik/vae_layers_detector/ADV_Regression.py --net_type $model >> Mahalanobis_run.txt;
# done
#=================================================================================================================
# ablation study
# python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_ablation.py --model LR --only_re cifar10_resnet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
# python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_ablation.py --model LR --only_re cifar100_resnet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
# python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_ablation.py --model LR --only_re svhn_resnet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150

# python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_ablation.py --model LR --only_re cifar10_densenet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
# python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_ablation.py --model LR --only_re cifar100_densenet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
# python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_ablation.py --model LR --only_re svhn_densenet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
# ===
# python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_ablation.py --model LR --only_ln cifar100_resnet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
# python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_ablation.py --model LR --only_ln svhn_resnet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
# python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_ablation.py --model LR --only_ln cifar10_resnet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150

# python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_ablation.py --model LR --only_ln cifar10_densenet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
# python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_ablation.py --model LR --only_ln cifar100_densenet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
# python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_ablation.py --model LR --only_ln svhn_densenet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
# ========
# python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_ablation.py --model IF --only_re cifar10_resnet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
# python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_ablation.py --model IF --only_re cifar100_resnet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
# python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_ablation.py --model IF --only_re svhn_resnet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150

# python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_ablation.py --model IF --only_re cifar10_densenet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
# python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_ablation.py --model IF --only_re cifar100_densenet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
# python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_ablation.py --model IF --only_re svhn_densenet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
# ===
# python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_ablation.py --model IF --only_ln cifar100_resnet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
# python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_ablation.py --model IF --only_ln svhn_resnet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
# python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_ablation.py --model IF --only_ln cifar10_resnet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150

# python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_ablation.py --model IF --only_ln cifar10_densenet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
# python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_ablation.py --model IF --only_ln cifar100_densenet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
# python -u /home/bwojcik/vae_layers_detector/ADV_train_detector_ablation.py --model IF --only_ln svhn_densenet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150
#=================================================================================================================
#cd ..
#for ae_type in waegan wae vae ae; do
#  echo $ae_type >>results.txt
#  echo $ae_type >>latent_results.txt
#  for type in SVC LSVC LR RF GB OCSVM IF; do
#    echo $type >>results.txt
#    python /home/bwojcik/vae_layers_detector/ADV_generate_table3.py --ae_type $ae_type --classifier_type $type vae_layers_detector/ >>results.txt
#    echo $type >>latent_results.txt
#    python /home/bwojcik/vae_layers_detector/ADV_generate_table3.py --latent --ae_type $ae_type --classifier_type $type vae_layers_detector/ >>latent_results.txt
#  done
#done
#=================================================================================================================
# for model in resnet densenet
# for model in resnet
# for model in densenet
# do
#   # for dataset in cifar10 cifar100 svhn
#   for dataset in svhn
#   do
#     python -u /home/bwojcik/vae_layers_detector/ADV_odd_odds.py --dataset $dataset --dataroot /mnt/users/bwojcik/local/.datasets --net_type $model
#   done
# done
#=================================================================================================================
# python -u /home/bwojcik/vae_layers_detector/ADV_known_unknown_correlation.py --adv_known FGSM --jobs 30 --model LR .
# python -u /home/bwojcik/vae_layers_detector/ADV_known_unknown_correlation.py --adv_known FGSM --jobs 30 --model LR --latent .

# python -u /home/bwojcik/vae_layers_detector/ADV_known_unknown_correlation.py --adv_known BIM --jobs 30 --model LR .
# python -u /home/bwojcik/vae_layers_detector/ADV_known_unknown_correlation.py --adv_known BIM --jobs 30 --model LR --latent .

# python -u /home/bwojcik/vae_layers_detector/ADV_known_unknown_correlation.py --adv_known DeepFool --jobs 30 --model LR .
# python -u /home/bwojcik/vae_layers_detector/ADV_known_unknown_correlation.py --adv_known DeepFool --jobs 30 --model LR --latent .

# python -u /home/bwojcik/vae_layers_detector/ADV_known_unknown_correlation.py --adv_known CWL2 --jobs 30 --model LR .
# python -u /home/bwojcik/vae_layers_detector/ADV_known_unknown_correlation.py --adv_known CWL2 --jobs 30 --model LR --latent .

# python -u /home/bwojcik/vae_layers_detector/ADV_known_unknown_correlation.py --adv_known PGD100 --jobs 30 --model LR .
# python -u /home/bwojcik/vae_layers_detector/ADV_known_unknown_correlation.py --adv_known PGD100 --jobs 30 --model LR --latent .
#=================================================================================================================
# for PGD strength experiment
# for model in resnet; do
#  for dataset in cifar10; do
#    for adv in PGD10 PGD20 PGD30 PGD40 PGD50 PGD60 PGD70 PGD80 PGD90 PGD100 PGD110 PGD120 PGD130 PGD140 PGD150 PGD160 PGD170 PGD180 PGD190 PGD200; do
#      python /home/bwojcik/vae_layers_detector/ADV_Samples.py --dataset $dataset --net_type $model --adv_type $adv
#    done
#  done
# done
#=================================================================================================================
# for model in resnet; do
#  for dataset in cifar10; do
#    for adv in PGD10 PGD20 PGD30 PGD40 PGD50 PGD60 PGD70 PGD80 PGD90 PGD100 PGD110 PGD120 PGD130 PGD140 PGD150 PGD160 PGD170 PGD180 PGD190 PGD200; do
#      python -u /home/bwojcik/vae_layers_detector/ADV_train_featuremaps_AE.py --runs 1 --dataset $dataset --dataroot /mnt/users/bwojcik/local/.datasets --net_type $model --ae_type wae --adv_type $adv
#    done
#  done
# done
#=================================================================================================================
# python -u /home/bwojcik/vae_layers_detector/ADV_increasing_strength_performance.py cifar10_resnet_deep_wae_arch_\[128_128_128\]_bn_False_latent_64_lamb_0_0001_lr_0_001_bs_100_epochs_150_0
#=================================================================================================================