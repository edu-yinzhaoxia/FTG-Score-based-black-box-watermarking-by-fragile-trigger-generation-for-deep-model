import torch
from PIL import Image
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

# all_img = torch.load('./samples/mini/img'+str(0))
# for i in range(0,32):
#     img = torch.load('./samples/mini/img'+str(i))
#     all_img = torch.cat((all_img,img),dim=0)
# torch.save(all_img,'./samples/mini/all_img')

######------------------联合jpeg图片------------------######
val_trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

# with Image.open('./samples/mini/img'+str(0) + '.jpg') as im:
#     im = val_trans(im).to(device)
#     im = torch.unsqueeze(im,dim=0)
#     all_img = im
# for i in range(1,50):
#     with Image.open('./samples/mini/img' + str(i) + '.jpg') as im:
#         im = val_trans(im).to(device)
#         im = torch.unsqueeze(im, dim=0)
#         all_img = torch.cat((all_img,im),dim=0)
# torch.save(all_img,'./samples/mini/all_jpegimg')

######------------------联合tensor------------------######
val_trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        # transforms.ToTensor(),
        normalize,
    ])

im = torch.load('./samples/mini/img' + str(0))
im = val_trans(im).to(device)
# im = torch.unsqueeze(im,dim=0)
all_img = im
for i in range(1,50):
    im = torch.load('./samples/mini/img' + str(i))
    im = val_trans(im).to(device)
    # im = torch.unsqueeze(im, dim=0)
    all_img = torch.cat((all_img,im),dim=0)
torch.save(all_img,'./samples/mini/all_img')