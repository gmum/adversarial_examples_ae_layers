import sys
import inspect
from pathlib import Path

import torch

import densenet


def main():
    '''Use pytorch 0.3.1 for this script'''
    net_type = 'densenet'
    datasets = ['cifar10', 'cifar100']
    for dataset in datasets:
        pre_trained_net = f'./pre_trained/{net_type}_{dataset}.pth'
        pre_trained_path = Path(pre_trained_net)
        renamed_pre_trained_path = pre_trained_path.parent / f'{net_type}_{dataset}_model.pth'
        pre_trained_path.rename(renamed_pre_trained_path)
        print(f'Moving {pre_trained_net} to {str(renamed_pre_trained_path)}')
        sys.path.insert(0, str(Path(inspect.getfile(densenet)).parent))
        model = torch.load(str(renamed_pre_trained_path), map_location={'cuda:0': 'cpu'})
        state_dict = model.state_dict()
        torch.save(state_dict, str(pre_trained_path))


if __name__ == '__main__':
    main()