import copy
import logging
import math

import numpy as np
import torch
from PIL import Image
from torchvision import datasets
from torchvision import transforms
import torch.nn as nn
from torchvision.transforms.functional import to_pil_image
import kornia.augmentation as K

from .randaugment import RandAugmentMC

logger = logging.getLogger(__name__)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)


def get_cifar10(args, root):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    base_dataset = datasets.CIFAR10(root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    train_labeled_dataset = CIFAR10SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR10SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std))

    test_dataset = datasets.CIFAR10(
        root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_federate_cifar10(args, root):
    """
    生成每个客户端的数据集，划分方法是：将cifar10数据集平均分到每个客户端中，然后再对每个客户端上的数据集利用
    x_u_split函数得到每个客户端的labeled和unlabeled数据集
    Args:
        args: args.num_labeled记录了所有客户端上labeled数据的数量
        root: 数据集的根目录
    Returns:
        train_labeled_dataset: 所有客户端上的labeled数据集的列表
        train_unlabeled_dataset: 所有客户端上的unlabeled数据集的列表
        test_dataset: 测试集
    """
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    base_dataset = datasets.CIFAR10(root, train=True, download=True)
    all_data = np.array(base_dataset.data)
    all_targets = np.array(base_dataset.targets)
    data_per_client = len(all_data) // args.num_clients

    train_split_labeled_dataset = []
    train_split_unlabeled_dataset = []

    args.num_labeled = args.num_labeled // args.num_clients

    for i in range(args.num_clients):
        start = i * data_per_client
        end = (i + 1) * data_per_client if i != args.num_clients - 1 else len(all_data)

        client_data = all_data[start:end]
        client_targets = all_targets[start:end]

        train_labeled_idxs, train_unlabeled_idxs = x_u_split(
            args, client_targets, True)

        global_labeled_idx = np.arange(start, end)[train_labeled_idxs]
        global_unlabeled_idx = np.arange(start, end)[train_unlabeled_idxs]

        labeled_dataset = CIFAR10SSL(
            root, global_labeled_idx, train=True,
            transform=transform_labeled)

        unlabeled_dataset = CIFAR10SSL(
            root, global_unlabeled_idx, train=True,
            transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std))

        train_split_labeled_dataset.append(labeled_dataset)
        train_split_unlabeled_dataset.append(unlabeled_dataset)

    test_dataset = datasets.CIFAR10(
        root, train=False, transform=transform_val, download=False)

    return train_split_labeled_dataset, train_split_unlabeled_dataset, test_dataset, TransformFixMatch(mean=cifar10_mean, std=cifar10_std)


def get_cifar100(args, root):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    base_dataset = datasets.CIFAR100(
        root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    train_labeled_dataset = CIFAR100SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR100SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar100_mean, std=cifar100_std))

    test_dataset = datasets.CIFAR100(
        root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_federate_cifar100(args, root):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    base_dataset = datasets.CIFAR100(
        root, train=True, download=True)

    all_data = np.array(base_dataset.data)
    all_targets = np.array(base_dataset.targets)
    data_per_client = len(all_data) // args.num_clients

    train_split_labeled_dataset = []
    train_split_unlabeled_dataset = []

    args.num_labeled = args.num_labeled // args.num_clients

    for i in range(args.num_clients):
        start = i * data_per_client
        end = (i + 1) * data_per_client if i != args.num_clients - 1 else len(all_data)
        client_data = all_data[start:end]
        client_targets = all_targets[start:end]
        train_labeled_idxs, train_unlabeled_idxs = x_u_split(
            args, client_targets, True)

        global_labeled_idx = np.arange(start, end)[train_labeled_idxs]
        global_unlabeled_idx = np.arange(start, end)[train_unlabeled_idxs]

        labeled_dataset = CIFAR100SSL(
            root, global_labeled_idx, train=True,
            transform=transform_labeled)

        unlabeled_dataset = CIFAR100SSL(
            root, global_unlabeled_idx, train=True,
            transform=TransformFixMatch(mean=cifar100_mean, std=cifar100_std))

        train_split_labeled_dataset.append(labeled_dataset)
        train_split_unlabeled_dataset.append(unlabeled_dataset)

    test_dataset = datasets.CIFAR100(
        root, train=False, transform=transform_val, download=False)

    return train_split_labeled_dataset, train_split_unlabeled_dataset, test_dataset, TransformFixMatch(mean=cifar100_mean, std=cifar100_std)

def x_u_split(args, labels, is_federated=False):
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == args.num_labeled

    if not is_federated:
        if args.expand_labels or args.num_labeled < args.batch_size:
            num_expand_x = math.ceil(
                args.batch_size * args.eval_step / args.num_labeled)
            labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
        np.random.shuffle(labeled_idx)
    else:
        if args.expand_labels or args.num_labeled < args.batch_size:
            num_expand_x = math.ceil(
                args.batch_size * args.local_ep / args.num_labeled)
            labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
        np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx


class Normalize(nn.Module):
    """归一化模块（Tensor风格），适配 Kornia 和 PyTorch 的输入"""
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x):
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        return (x - mean) / std


class Denormalize(nn.Module):
    """反归一化模块：恢复归一化前的图像"""
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x):
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        return x * std + mean


class ProgressiveAugmentor(nn.Module):
    """
    progressive_weak(x): 对已归一化图像 Tensor 进行 progressive weak augmentation。
    支持输入 [C, H, W] 或 [B, C, H, W]
    """
    def __init__(self, mean, std):
        super().__init__()

        self.denormalize = Denormalize(mean, std)
        self.normalize = Normalize(mean, std)

        self.weak = nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomCrop((32, 32), padding=4, padding_mode='reflect')
        )

    def forward(self, x):
        single = False
        if x.dim() == 3:
            x = x.unsqueeze(0)
            single = True

        x = self.denormalize(x)     # 1. 反归一化
        x = x.clamp(0, 1)           # 2. 裁剪为合法图像值
        x = self.weak(x)            # 3. 增强
        x = self.normalize(x)       # 4. 重新归一化

        return x.squeeze(0) if single else x


class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

        self.pg_augment = ProgressiveAugmentor(mean, std)

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)

    def progressive_weak(self, x):
        return self.pg_augment(x)


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


DATASET_GETTERS = {'cifar10': get_cifar10,
                   'cifar100': get_cifar100}

FEDERATED_DATASET_GETTERS = {
    'cifar10': get_federate_cifar10,
    'cifar100': get_federate_cifar100
}
