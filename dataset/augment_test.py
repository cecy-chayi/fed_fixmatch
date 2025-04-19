import argparse
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
from dataset.cifar import TransformFixMatch
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

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

def augment_image(input_path, output_path=None, strength='weak'):
    """图像增强处理器
    Args:
        input_path: 输入图像路径
        output_path: 输出图像路径（None时直接显示）
        strength: 增强强度 ['weak', 'strong']
    """
    # 初始化增强器（使用CIFAR-10的标准化参数）
    augmentor = TransformFixMatch(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    )
    denormalize = Denormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    # 读取并转换图像
    img = Image.open(input_path).convert('RGB').resize((32, 32))
    tensor_img = to_tensor(img)

    # 执行增强
    weak_augmented_img = augmentor.weak(img)  # 弱增强
    weak_augmented_img = augmentor.weak(weak_augmented_img)

    strong_augmented_img = augmentor.strong(img)  # 强增强


    # 输出结果
    if output_path:
        weak_augmented_img.save(output_path)
        print(f"增强图像已保存至：{output_path}")
    else:
        plt.figure(figsize=(10, 5))
        for i in range(5):
            plt.subplot(1, 5, i+1)
            plt.imshow(img)
            # plt.title("Original")
            plt.axis('off')
            img = augmentor.weak(img)
        # plt.subplot(1, 3, 1)
        # plt.imshow(img)
        #
        # plt.subplot(1, 3, 2)
        # plt.imshow(weak_augmented_img)
        # plt.title(f"After Weak Augmentation")
        # plt.axis('off')
        #
        # plt.subplot(1, 3, 3)
        # plt.imshow(strong_augmented_img)
        # plt.title(f"After Strong Augmentation")
        # plt.axis('off')
        plt.show()


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-i', '--input', type=str, required=True, help='输入图像路径')
    # parser.add_argument('-o', '--output', type=str, help='输出图像路径')
    # parser.add_argument('-s', '--strength', type=str, choices=['weak', 'strong'],
    #                     default='weak', help='增强强度: weak (默认) 或 strong')
    # args = parser.parse_args()

    augment_image('example.jpg', None, 'weak')