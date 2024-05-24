import visdom
import torch

random_img = torch.rand(1,3,64,64)
icip_img = torch.load('./samples/tiny/icip_img')[0]
vis = visdom.Visdom(env="test")
vis.images(icip_img)
vis.images(random_img)