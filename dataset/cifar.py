import argparse
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
import random

from dataset.randaugment import RandAugmentMC

logger = logging.getLogger(__name__)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
mnist_mean = [0.1307]
mnist_std = [0.3081]


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
    生成每个客户端的数据集，支持 Dirichlet 非独立同分布划分
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

    num_classes = args.num_classes
    num_clients = args.num_clients
    alpha = 0.8  # Dirichlet 分布的 concentration 参数

    # 构建 Dirichlet 分布的 client 分配索引
    client_indices = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        idxs = np.where(all_targets == c)[0]
        np.random.shuffle(idxs)
        proportions = np.random.dirichlet(alpha=[alpha]*num_clients)
        proportions = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
        split_indices = np.split(idxs, proportions)
        for i, client_idx in enumerate(split_indices):
            client_indices[i].extend(client_idx)

    train_split_labeled_dataset = []
    train_split_unlabeled_dataset = []

    for i in range(num_clients):
        client_idx = np.array(client_indices[i])
        client_targets = all_targets[client_idx]

        train_labeled_idxs, train_unlabeled_idxs = dirichlet_x_u_split(
            args, client_targets)

        global_labeled_idx = client_idx[train_labeled_idxs]
        global_unlabeled_idx = client_idx[train_unlabeled_idxs]

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


def get_federate_mnist(args, root):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32 * 0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=mnist_mean, std=mnist_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mnist_mean, std=mnist_std)
    ])

    base_dataset = datasets.MNIST(root, train=True, download=True)
    all_data = np.array(base_dataset.data)
    all_targets = np.array(base_dataset.targets)

    num_classes = args.num_classes
    num_clients = args.num_clients
    alpha = 0.8  # Dirichlet 分布的 concentration 参数

    # 构建 Dirichlet 分布的 client 分配索引
    client_indices = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        idxs = np.where(all_targets == c)[0]
        np.random.shuffle(idxs)
        proportions = np.random.dirichlet(alpha=[alpha] * num_clients)
        proportions = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
        split_indices = np.split(idxs, proportions)
        for i, client_idx in enumerate(split_indices):
            client_indices[i].extend(client_idx)

    train_split_labeled_dataset = []
    train_split_unlabeled_dataset = []

    for i in range(num_clients):
        client_idx = np.array(client_indices[i])
        client_targets = all_targets[client_idx]

        train_labeled_idxs, train_unlabeled_idxs = dirichlet_x_u_split(
            args, client_targets)

        global_labeled_idx = client_idx[train_labeled_idxs]
        global_unlabeled_idx = client_idx[train_unlabeled_idxs]

        labeled_dataset = MNISTSSL(
            root, global_labeled_idx, train=True,
            transform=transform_labeled)

        unlabeled_dataset = MNISTSSL(
            root, global_unlabeled_idx, train=True,
            transform=TransformFixMatch(mean=mnist_mean, std=mnist_std))

        train_split_labeled_dataset.append(labeled_dataset)
        train_split_unlabeled_dataset.append(unlabeled_dataset)

    test_dataset = datasets.MNIST(
        root, train=False, transform=transform_val, download=False)

    return train_split_labeled_dataset, train_split_unlabeled_dataset, test_dataset, TransformFixMatch(
        mean=mnist_mean, std=mnist_mean)


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
            args, client_targets)

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


def dirichlet_x_u_split(args, labels):
    """
    用于 Dirichlet non-IID 数据划分后的标签/无标签拆分

    Args:
        args: 包含 batch_size, local_ep 等训练参数
        labels: 当前客户端的标签（已被 Dirichlet 分配）

    Returns:
        labeled_idx: 有标签样本索引（相对于当前客户端）
        unlabeled_idx: 无标签样本索引
    """
    labels = np.array(labels)
    n_total = len(labels)
    n_labeled = int(n_total * args.labeled_ratio)

    all_indices = np.arange(n_total)
    np.random.shuffle(all_indices)

    labeled_idx = all_indices[:n_labeled]
    unlabeled_idx = all_indices

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / n_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)

    return labeled_idx, unlabeled_idx


def x_u_split(args, labels):
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
    if len(labeled_idx) != args.num_labeled:
        print(len(labeled_idx), args.num_labeled)
        assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
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


class MNISTSSL(datasets.MNIST):
    def __init__(self, root, indexs=None, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root=root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)

        # 使用索引子集
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img.numpy(), mode='L')

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
    'cifar100': get_federate_cifar100,
    'mnist': get_federate_mnist
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100'],
                        help='dataset name')
    parser.add_argument('--num-labeled', type=int, default=40,
                        help='number of labeled data')
    parser.add_argument("--expand-labels", action="store_true",
                        help="expand labels to fit eval steps")
    parser.add_argument('--batch-size', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--seed', default=None, type=int,
                        help="random seed")
    parser.add_argument('--eval-step', default=32, type=int,
                        help="random seed")
    parser.add_argument('--num-clients', type=int, default=10,
                        help='number of clients')
    parser.add_argument('--dirichlet-alpha', type=float, default=0.8,
                        help='dirichlet concentration ')
    parser.add_argument('--labeled-ratio',type=float, default=0.1)

    args = parser.parse_args()
    args.num_classes = 10
    random.seed(args.seed)
    np.random.seed(args.seed)
    split_labeled_dataset, split_unlabeled_dataset, test_dataset, progress_transform = get_federate_cifar10(args=args, root='../data')
    # 可视化不同客户端的标签分布
    import matplotlib.pyplot as plt
    import seaborn as sns

    # 1. 收集所有客户端的标签分布
    num_clients = args.num_clients
    num_classes = 10  # CIFAR10 固定为10类
    client_label_counts = np.zeros((num_clients, num_classes), dtype=int)

    for client_id in range(num_clients):
        targets = split_labeled_dataset[client_id].targets  # 获取客户端标签
        unique, counts = np.unique(targets, return_counts=True)
        client_label_counts[client_id, unique] = counts

    # 2. 绘制热图
    plt.figure(figsize=(15, 8))
    sns.heatmap(client_label_counts, cmap="Blues", annot=False, cbar=True)
    plt.xlabel("Class Label", fontsize=12)
    plt.ylabel("Client ID", fontsize=12)
    plt.title(f"Non-IID Label Distribution (Dirichlet α={args.dirichlet_alpha})", fontsize=14)
    plt.savefig("client_label_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 3. 输出统计指标
    coverage_per_client = np.sum(client_label_counts > 0, axis=1)
    samples_per_client = np.sum(client_label_counts, axis=1)

    print("\n=== 分布统计指标 ===")
    print(f"平均每个客户端覆盖类别数: {np.mean(coverage_per_client):.2f}")
    print(f"最小覆盖类别数: {np.min(coverage_per_client)}")
    print(f"最大覆盖类别数: {np.max(coverage_per_client)}")
    print(f"客户端样本量方差: {np.var(samples_per_client):.2f}")

    # 4. 检查极端情况
    print("\n=== 极端情况检查 ===")
    print(f"存在客户端完全缺失 {num_classes - np.max(coverage_per_client)} 个类别")
    print(f"最小客户端样本量: {np.min(samples_per_client)}")
    print(f"最大客户端样本量: {np.max(samples_per_client)}")

