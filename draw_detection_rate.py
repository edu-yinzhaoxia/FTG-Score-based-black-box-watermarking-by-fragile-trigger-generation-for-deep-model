import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib import rcParams
from matplotlib.pyplot import MultipleLocator
recordmnist = pd.read_csv("./tests/record/detection_rate_mnist.csv",header=None)
recordcifar10 = pd.read_csv("./tests/record/detection_rate_cifar10.csv",header=None)
recordcifar100 = pd.read_csv("./tests/record/detection_rate_cifar100.csv",header=None)
recordtiny = pd.read_csv("./tests/record/detection_rate_tiny.csv",header=None)

normal_mnist = recordmnist[0]
random_mnist = recordmnist[1]
icip_mnist = recordmnist[2]
fragile_mnist = recordmnist[3]

normal_cifar10 = recordcifar10[0]
random_cifar10 = recordcifar10[1]
icip_cifar10 = recordcifar10[2]
fragile_cifar10 = recordcifar10[3]

normal_cifar100 = recordcifar100[0]
random_cifar100 = recordcifar100[1]
icip_cifar100 = recordcifar100[2]
fragile_cifar100 = recordcifar100[3]

normal_tiny = recordtiny[0]
random_tiny = recordtiny[1]
icip_tiny = recordtiny[2]
fragile_tiny = recordtiny[3]
config = {
    "font.family":'Times New Roman',  # 设置字体类型
    "font.size": 24,
    "mathtext.fontset":'stix',
}
#预测概率直方图
rcParams.update(config)

fig, ax = plt.subplots(2,2,figsize = (16,12))
x = np.arange(len(normal_mnist))
ax[0][0].plot(x, normal_mnist*100, linewidth=1,label = 'Non-fragile',marker = 'o',markevery=10)
ax[0][0].plot(x, random_mnist*100, linewidth=1,label = 'Random',marker = '^',markevery=10)
ax[0][0].plot(x, icip_mnist*100, linewidth=1,label = 'ICIP-2022',marker = '*',markevery=10)
ax[0][0].plot(x, fragile_mnist*100, linewidth=1,label = 'Fragile',marker = 's',markevery=10)
ax[0][0].set_title("MNIST")
ax[0][0].set_xlabel("Epoch")
ax[0][0].set_ylabel("Change Rate (%)")
ax[0][0].legend(fontsize=16,frameon=False,bbox_to_anchor=(1,0.5))
ax[0][0].xaxis.set_major_locator(MultipleLocator(10))
ax[0][1].plot(x, normal_cifar10*100, linewidth=1,label = 'Non-fragile',marker = 'o',markevery=10)
ax[0][1].plot(x, random_cifar10*100, linewidth=1,label = 'Random',marker = '^',markevery=10)
ax[0][1].plot(x, icip_cifar10*100, linewidth=1,label = 'ICIP-2022',marker = '*',markevery=10)
ax[0][1].plot(x, fragile_cifar10*100, linewidth=1,label = 'Fragile',marker = 's',markevery=10)
ax[0][1].set_title("Cifar10")
ax[0][1].set_xlabel("Epoch")
# ax[0][1].set_ylabel("Change Rate (%)")
ax[0][1].legend(fontsize=16,frameon=False,bbox_to_anchor=(1,0.5))
ax[0][1].xaxis.set_major_locator(MultipleLocator(10))
ax[1][0].plot(x, normal_cifar100*100, linewidth=1,label = 'Non-fragile',marker = 'o',markevery=10)
ax[1][0].plot(x, random_cifar100*100, linewidth=1,label = 'Random',marker = '^',markevery=10)
ax[1][0].plot(x, icip_cifar100*100, linewidth=1,label = 'ICIP-2022',marker = '*',markevery=10)
ax[1][0].plot(x, fragile_cifar100*100, linewidth=1,label = 'Fragile',marker = 's',markevery=10)
ax[1][0].set_title("Cifar100")
ax[1][0].set_xlabel("Epoch")
ax[1][0].set_ylabel("Change Rate (%)")
ax[1][0].legend(fontsize=16,loc = 7,frameon=False)
ax[1][0].xaxis.set_major_locator(MultipleLocator(10))
ax[1][1].plot(x, normal_tiny*100, linewidth=1,label = 'Non-fragile',marker = 'o',markevery=10)
ax[1][1].plot(x, random_tiny*100, linewidth=1,label = 'Random',marker = '^',markevery=10)
ax[1][1].plot(x, icip_tiny*100, linewidth=1,label = 'ICIP-2022',marker = '*',markevery=10)
ax[1][1].plot(x, fragile_tiny*100, linewidth=1,label = 'Fragile',marker = 's',markevery=10)
ax[1][1].set_title("TinyImageNet")
ax[1][1].set_xlabel("Epoch")
# ax[1][1].set_ylabel("Change Rate (%)")
ax[1][1].legend(fontsize=16,loc = 7,frameon=False)
ax[1][1].xaxis.set_major_locator(MultipleLocator(10))
plt.tight_layout()
# plt.rcParams['xtick.direction'] = 'in'
# plt.rcParams['ytick.direction'] = 'in'
plt.savefig('./Figure/changerate.jpg', format='jpg',dpi=300)
plt.show()