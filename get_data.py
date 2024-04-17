import os
import numpy as np
import torch
import argparse
from utils import get_dataset, get_network, get_eval_pool, evaluate_synset, ParamDiffAug, TensorDataset
import copy
import gc


def get_data():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--method', type=str, default='random', help='random/herding/DSA/DM')
    parser.add_argument('--dataset', type=str, default='CIFAR100', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=20, help='image(s) per class')
    parser.add_argument('--steps', type=int, default=5, help='5/10-step learning')
    parser.add_argument('--num_eval', type=int, default=3, help='evaluation number')
    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--data_path', type=str, default='./../data', help='dataset path')

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = True # augment images for all methods
    args.dsa_strategy = 'color_crop_cutout_flip_scale_rotate' # for CIFAR10/100

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)


    ''' all training data '''
    images_all = []
    labels_all = []
    indices_class = [[] for c in range(num_classes)]

    images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
    labels_all = [dst_train[i][1] for i in range(len(dst_train))]
    for i, lab in enumerate(labels_all):
        indices_class[lab].append(i)
    images_all = torch.cat(images_all, dim=0).to(args.device)
    labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)

    # for c in range(num_classes):
    #     print('class c = %d: %d real images' % (c, len(indices_class[c])))

    def get_images(c, n):  # get random n images from class c
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        return images_all[idx_shuffle]

    print()
    print('==================================================================================')
    
    return images_all,labels_all


if __name__ == '__main__':
    get_data()




