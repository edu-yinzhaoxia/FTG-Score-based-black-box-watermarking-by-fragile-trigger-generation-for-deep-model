import pandas as pd
from collections import Counter
import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt

def count_frequency(lst):
    frequency = Counter(lst)
    return frequency

config = {
    "font.family":'Times New Roman',  # 设置字体类型
    "font.size": 24,
    "mathtext.fontset":'stix',
}
#预测概率直方图
rcParams.update(config)
recordimagenet = pd.read_csv("./times_imagenet_result.csv",header=None)
# recordmnist = pd.read_csv("./times_mini_result.csv",header=None)
# recordcifar10 = pd.read_csv("./times_cifar10_result.csv",header=None)
# recordcifar100 = pd.read_csv("./times_cifar100_result.csv",header=None)
# recordtiny = pd.read_csv("./times_tiny_result.csv",header=None)

result_imagenet = recordimagenet[0]
# result_mnist = recordmnist[0]
# result_cifar10 = recordcifar10[0]
# result_cifar100 = recordcifar100[0]
# result_tiny = recordtiny[0]


# 示例用法
# result_mnist = count_frequency(result_mnist)
# result_cifar10 = count_frequency(result_cifar10)
# result_cifar100 = count_frequency(result_cifar100)
# result_tiny = count_frequency(result_tiny)
result_imagenet = count_frequency(result_imagenet)
print(result_imagenet)
# print(result_mnist)  # 输出：Counter({4: 4, 3: 3, 2: 2, 1: 1})
# print(result_cifar10)
# print(result_cifar100)
# print(result_tiny)

# 预测概率直方图
fig, ax = plt.subplots(1,4,figsize = (16,4))
# l1_x = [str(num) for num in result_mnist.keys()]
# l2_x = [str(num) for num in result_cifar10.keys()]
# l3_x = [str(num) for num in result_cifar100.keys()]
# l4_x = [str(num) for num in result_tiny.keys()]
l_x = [str(num) for num in result_imagenet.keys()]
# print(sorted(l1_x))
for key in result_imagenet:
    result_imagenet[key] /= 10000
# for key in result_mnist:
#     result_mnist[key] /= 10000
# for key in result_cifar10:
#     result_cifar10[key] /= 10000
# for key in result_cifar100:
#     result_cifar100[key] /= 10000
# for key in result_tiny:
#     result_tiny[key] /= 10000
l_y = result_imagenet.values()
# l1_y = result_mnist.values()
# l2_y = result_cifar10.values()
# l3_y = result_cifar100.values()
# l4_y = result_tiny.values()

yticks = np.linspace(0, 1., 5)
ax[0].bar(sorted(l_x), sorted(l_y,reverse=1), width=1, edgecolor="white", linewidth=0.7)
# ax[0].set(ylim=(0, 1.))
ax[0].set_yticks(yticks)
ax[0].set_title('ImageNet')
ax[0].set_xlabel("Number of accesses")
ax[0].set_ylabel("Frequency")
# ax[1].bar(sorted(l2_x), sorted(l2_y,reverse=1), width=1, edgecolor="white", linewidth=0.7)
# # ax[1].set(ylim=(0, 1.))
# ax[1].set_yticks(yticks)
# ax[1].set_title('Cifar10')
# ax[1].set_xlabel("Number of accesses")
# ax[2].bar(sorted(l3_x), sorted(l3_y,reverse=1), width=1, edgecolor="white", linewidth=0.7)
# # ax[2].set(ylim=(0, 1.))
# ax[2].set_yticks(yticks)
# ax[2].set_title(r'Cifar100')
# ax[2].set_xlabel("Number of accesses")
# ax[3].bar(sorted(l4_x), sorted(l4_y,reverse=1), width=1, edgecolor="white", linewidth=0.7)
# # ax[3].set(ylim=(0, 1.))
# ax[3].set_yticks(yticks)
# ax[3].set_title(r'Tiny')
# ax[3].set_xlabel("Number of accesses")
plt.tight_layout()
# plt.savefig('./prediction_probability.jpg', format='jpg',dpi=300)
plt.show()