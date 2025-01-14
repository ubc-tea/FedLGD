import glob
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import random

def GetPretrained_cifar10(path, mean, std, im_size, num_classes, device, ipc = 50, padding = 2):
    
    img_path = path
    images_pil = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize(mean, std)
            ])
    images_torch = transform(images_pil)
    images = []
    for j in range(num_classes):
        for i in range(ipc):
            images.append(images_torch[:, (padding+im_size[0])*j+padding:(padding+im_size[0])*j+padding+im_size[0], (padding+im_size[1])*i+padding:(padding+im_size[1])*i+padding+im_size[1]].unsqueeze(0))
    images = torch.cat(images, dim=0).detach().to(device)
            
    return images

def GetPretrained_single(path, mean, std, im_size, num_classes, client_idx, device, ipc = 50, padding = 2):
    
    img_path = f'{path}_client'+str(client_idx)+'_iter0.png'
    images_pil = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize(mean, std)
            ])
    images_torch = transform(images_pil)
    images = []
    for j in range(num_classes):
        for i in range(ipc):
            images.append(images_torch[:, (padding+im_size[0])*j+padding:(padding+im_size[0])*j+padding+im_size[0], (padding+im_size[1])*i+padding:(padding+im_size[1])*i+padding+im_size[1]].unsqueeze(0))
    images = torch.cat(images, dim=0).detach().to(device)
            
    return images

def GetPretrained(path, means, stds, im_size, num_classes, client_num, device, ipc = 50, padding = 2):
    images_all = []
    for i in range(client_num):
        img_path = f'{path}_client'+str(i)+'_iter0_local1.png'
        images_pil = Image.open(img_path).convert('RGB')
        transform = transforms.Compose([
                transforms.ToTensor(), 
                transforms.Normalize(means[i], stds[i])
                ])
        images_torch = transform(images_pil)
        images = []
        for j in range(num_classes):
            for i in range(ipc):
                images.append(images_torch[:, (padding+im_size[0])*j+padding:(padding+im_size[0])*j+padding+im_size[0], (padding+im_size[1])*i+padding:(padding+im_size[1])*i+padding+im_size[1]].unsqueeze(0))
        images = torch.cat(images, dim=0).detach().to(device)
        # images.requires_grad = True
        images_all.append(images)
            
    return images_all

def GetPretrained_finetune(path, means, stds, im_size, num_classes, client_num, device, ipc = 50, server_ipc=10, padding = 2):
    images_all = []
    for i in range(client_num):
        img_path = f'{path}_client'+str(i)+'_iter49_local1.png'
        images_pil = Image.open(img_path).convert('RGB')
        transform = transforms.Compose([
                transforms.ToTensor(), 
                transforms.Normalize(means[i], stds[i])
                ])
        images_torch = transform(images_pil)
        images = []
        for j in range(num_classes):
            for i in range(ipc):
                images.append(images_torch[:, (padding+im_size[0])*j+padding:(padding+im_size[0])*j+padding+im_size[0], (padding+im_size[1])*i+padding:(padding+im_size[1])*i+padding+im_size[1]].unsqueeze(0))
        images = torch.cat(images, dim=0).detach().to(device)
        # images.requires_grad = True
        images_all.append(images)

    img_path_global = f'{path}_global_v2_iter49_local1.png'
    images_pil_global = Image.open(img_path_global).convert('RGB')
    mean_global = np.mean(means, axis=0)
    std_global = np.mean(stds, axis=0)
    transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize(mean_global, std_global)
            ])
    images_torch_global = transform(images_pil_global)
    images_global = []
    for j in range(num_classes):
        for i in range(server_ipc):
            images_global.append(images_torch_global[:, (padding+im_size[0])*j+padding:(padding+im_size[0])*j+padding+im_size[0], (padding+im_size[1])*i+padding:(padding+im_size[1])*i+padding+im_size[1]].unsqueeze(0))
    images_global = torch.cat(images_global, dim=0).detach().to(device)
    
            
    return images_all, images_global

def GetPretrained_server(path, im_size, num_classes, device, ipc = 50, padding = 2):
    
    images_pil = Image.open(path).convert('RGB')
    transform = transforms.Compose([
            transforms.ToTensor()
            ])
    images_torch = transform(images_pil)
    images = []
    for j in range(num_classes):
        for i in range(ipc):
            images.append(images_torch[:, (padding+im_size[0])*j+padding:(padding+im_size[0])*j+padding+im_size[0], (padding+im_size[1])*i+padding:(padding+im_size[1])*i+padding+im_size[1]].unsqueeze(0))
    images = torch.cat(images, dim=0).detach().to(device)
   
            
    return images


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        noise = torch.randn(tensor.size(1), tensor.size(2)) * self.std + self.mean
        tensor[0, :, :] += noise
        tensor[1, :, :] += noise
        tensor[2, :, :] += noise
        return tensor
        # return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)



class PretrainedDataset(Dataset):
    """
    Code for reading the Pretrained dataset
    """

    def __init__(self, dataset_path='./pretrained', dataset='CovidX', ipc = 50, padding = 2, im_size = [224, 224]):

        self.root = str(dataset_path)
        self.dataset = dataset

        if self.dataset[-6:] == 'CovidX':
            self.classes = [0,1]
            mean = [0.4886, 0.4886, 0.4886]
            std = [0.2460, 0.2460, 0.2460]
        elif self.dataset[-8:] == 'ImageNet':
            self.classes = np.arange(1000)
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        elif self.dataset[-5:] == 'Mnist':
            self.classes = np.arange(10)
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
        else:
            raise NotImplementedError
        
        transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize(mean, std),
            AddGaussianNoise()
            ])
        
        img_path = os.path.join(dataset_path, dataset+'.png')
        images_pil = Image.open(img_path).convert('RGB')
        images_torch = transform(images_pil)
        self.images = []
        self.labels = []
        for j in range(len(self.classes)):
            for i in range(ipc):
                self.images.append(images_torch[:, (padding+im_size[0])*j+padding:(padding+im_size[0])*j+padding+im_size[0], (padding+im_size[1])*i+padding:(padding+im_size[1])*i+padding+im_size[1]])
                self.labels.append(j)
        
        if self.dataset[-5:] == 'Mnist':
            self.resize_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((28, 28)),
                transforms.ToTensor()
            ])


        print(f'{dataset}', images_torch.size(), len(self.images), len(self.labels), self.images[1].size())

    def __len__(self):
        return(len(self.images))

    def __getitem__(self, index):
        if self.dataset[-5:] == 'Mnist':
            return self.resize_transform(self.images[index])
        else:
            return self.images[index] #, self.labels[index]






if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    # from matplotlib import patches, patheffects
    from torch.utils.data import DataLoader
    from torchvision.utils import save_image


    norm_mean = (0.4886, 0.4886, 0.4886)
    norm_std = (0.2460, 0.2460, 0.2460)
    val_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.CenterCrop((224, 224)), transforms.ToTensor()])
    
   
