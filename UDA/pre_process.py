from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
import os

def image_train(resize_size=256, crop_size=224, alexnet=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    return  transforms.Compose([
        transforms.Resize((resize_size,resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def image_test(resize_size=256, crop_size=224,tencrop=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
    if tencrop:
        return transforms.Compose([
            transforms.Resize((resize_size,resize_size)),
            transforms.TenCrop(crop_size),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])),]
            )
    else:
        return transforms.Compose([
        transforms.Resize((resize_size,resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
        ])


test_transform = transforms.Compose([
    transforms.Resize((40, 40)),
    transforms.TenCrop(32),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(norm_mean, norm_std)(crop) for crop in crops])),
])

class ImageList(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB'
                 ,second=False,root=None):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("No image found !"))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.second=second
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader
        self.root = root

    def __getitem__(self, index):
        if not self.second:
            path, target = self.imgs[index]
            # path = path.replace("../../data",self.root)
            path = os.path.join(self.root,path)
            img = self.loader(path)
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return img, target
        else:
            path, label, weight, sigma = self.imgs[index]
            path = path.replace("../../data",self.root)
            img = self.loader(path)
            if self.transform is not None:
                img = self.transform(img)

            return img, label, weight, sigma

    def __len__(self):
        return len(self.imgs)

class ImageListPseudo(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB',root=None):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise (RuntimeError("No image found !"))

        self.imgs = imgs
        self.filtering()
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader
        self.root = root

    def filtering(self):
        imgs = []
        for i in range(len(self.imgs)):
            w = self.imgs[i][2]
            if w>=0.5:
                imgs.append(self.imgs[i])

    def __getitem__(self, index):
        path, label, weight, sigma = self.imgs[index]
        path = path.replace("../../data", self.root)
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, label, weight, sigma

    def __len__(self):
        return len(self.imgs)


class ClassAwareSamplingSTDataset(Dataset):
    def __init__(self, source_list,target_list,transform = None, num_class = 31, num_sample_per_class = 10, dataset_length = None,root=None):
        self.num_class = num_class
        self.transform = transform
        self.source_path_list,self.source_labels = self.get_list(source_list)
        self.source_classaware_list = self.get_classwise_list(self.source_path_list,self.source_labels)
        self.target_path_list,self.target_labels = self.get_list(target_list)
        self.target_classaware_list = self.get_classwise_list(self.target_path_list,self.target_labels)
        self.selected_classes = np.arange(num_class)
        self.num_sample_per_class = num_sample_per_class
        self.thress_num = 0
        self.dataset_length = dataset_length
        self.root = root
        if self.dataset_length is not None:
            self.selected_classes = np.random.choice(self.selected_classes,self.dataset_length)

    def get_list(self,image_list):
        image_path_list = [val.split()[0] for val in image_list]
        labels = [int(val.split()[1]) for val in image_list]
        return image_path_list,labels

    def get_classwise_list(self,image_path_list,labels):
        dic = {}
        for k in range(self.num_class):
            dic[k] = [(image_path_list[i],labels[i]) for i in range(len(labels)) if labels[i]==k]
        return dic

    def update_selected_classes(self,target_labels):
        self.target_labels = target_labels
        self.target_classaware_list = self.get_classwise_list(self.target_path_list, self.target_labels)
        selected_classes = []
        for k in range(self.num_class):
            if len(self.target_classaware_list[k]) >= self.thress_num:
                selected_classes.append(k)
        self.selected_classes = selected_classes
        if self.dataset_length is not None:
            self.selected_classes = np.random.choice(self.selected_classes,self.dataset_length)

    def __len__(self):
        return len(self.selected_classes)

    def get_data(self,path):
        path = path.replace("../../data", self.root)
        return self.transform(rgb_loader(path)).unsqueeze(0)

    def __getitem__(self, index):
        c = self.selected_classes[index]
        source_idxs = np.random.choice(np.arange(len(self.source_classaware_list[c])),self.num_sample_per_class,replace=True)
        target_idxs = np.random.choice(np.arange(len(self.target_classaware_list[c])),self.num_sample_per_class,replace=True)
        source_data = torch.cat([self.get_data(self.source_classaware_list[c][i][0])
                                 for i in source_idxs], dim=0)
        target_data = torch.cat([self.get_data(self.target_classaware_list[c][i][0])
                                 for i in target_idxs], dim=0)
        return source_data,target_data




def make_dataset(image_list, labels):
    if labels:
      len_ = len(image_list)
      images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
      if len(image_list[0].split()) > 2:
        images = [(val.split()[0],int(val.split()[1]),float(val.split()[2]),float(val.split()[3])) for val in image_list]
      else:
        images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images

def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')


class BalancedBatchSampler(torch.utils.data.sampler.BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_samples for each of the n_classes.
    Returns batches of size n_classes * (batch_size // n_classes)
    Taken from https://github.com/criteo-research/pytorch-ada/blob/master/adalib/ada/datasets/sampler.py
    """

    def __init__(self, labels, batch_size):
        self.classes = sorted(set(labels.numpy()))
        print(self.classes)

        n_classes = len(self.classes)
        self._n_samples = batch_size // n_classes
        self._n_remain = batch_size % n_classes
        if self._n_samples == 0:
            raise ValueError(f"batch_size should be bigger than the number of classes, got {batch_size}")

        self._class_iters = [
            InfiniteSliceIterator(np.where(labels == class_)[0], class_=class_) for class_ in self.classes
        ]

        # batch_size = self._n_samples * n_classes
        self.n_dataset = len(labels)
        self._n_batches = int(np.round(self.n_dataset // batch_size))
        if self._n_batches == 0:
            raise ValueError(f"Dataset is not big enough to generate batches with size {batch_size}")
        print("K=", n_classes, "nk=", self._n_samples)
        print("Batch size = ", batch_size)

    def __iter__(self):
        for _ in range(self._n_batches):
            indices = []
            add_class = set(np.random.choice(self.classes, self._n_remain, replace=False))
            for class_iter in self._class_iters:
                if class_iter.class_ in add_class:
                    add_samples = 1
                else:
                    add_samples = 0
                indices.extend(class_iter.get(self._n_samples + add_samples))

            np.random.shuffle(indices)
            yield indices

        for class_iter in self._class_iters:
            class_iter.reset()

    def __len__(self):
        return self._n_batches