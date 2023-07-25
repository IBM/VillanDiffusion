# %%
"""The `denoise_model` will be our U-Net defined above. We'll employ the Huber loss between the true and the predicted noise.

## Define a PyTorch Dataset + DataLoader

Here we define a regular [PyTorch Dataset](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html). The dataset simply consists of images from a real dataset, like Fashion-MNIST, CIFAR-10 or ImageNet, scaled linearly to \\([−1, 1]\\).

Each image is resized to the same size. Interesting to note is that images are also randomly horizontally flipped. From the paper:

> We used random horizontal flips during training for CIFAR10; we tried training both with and without flips, and found flips to improve sample quality slightly.

Here we use the 🤗 [Datasets library](https://huggingface.co/docs/datasets/index) to easily load the Fashion MNIST dataset from the [hub](https://huggingface.co/datasets/fashion_mnist). This dataset consists of images which already have the same resolution, namely 28x28.
"""

import glob
import os
import pathlib
from random import sample
from typing import Callable, List, Tuple, Union
from functools import lru_cache
import warnings

from datasets import load_dataset, concatenate_datasets
import datasets
from datasets.dataset_dict import DatasetDict
from matplotlib import pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader, ConcatDataset, Subset, Dataset, IterableDataset
from torchvision.datasets import MNIST, CIFAR10, SVHN, FashionMNIST
from PIL import Image
from joblib import Parallel, delayed

from util import Log, normalize

DEFAULT_VMIN = float(-1.0)
DEFAULT_VMAX = float(1.0)

class DatasetLoader(object):
    # Dataset generation mode
    MODE_FIXED = "FIXED"
    MODE_FLEX = "FLEX"
    MODE_NONE = "NONE"
    MODE_EXTEND = "EXTEND"
    
    # Dataset names
    MNIST = "MNIST"
    CIFAR10 = "CIFAR10"
    CELEBA = "CELEBA"
    LSUN_CHURCH = "LSUN-CHURCH"
    LSUN_BEDROOM = "LSUN-BEDROOM"
    CELEBA_HQ = "CELEBA-HQ"
    CELEBA_HQ_LATENT_PR05 = "CELEBA-HQ-LATENT_PR05"
    CELEBA_HQ_LATENT = "CELEBA-HQ-LATENT"
    
    # Inpaint Type
    INPAINT_BOX: str = "INPAINT_BOX"
    INPAINT_LINE: str = "INPAINT_LINE"

    TRAIN = "train"
    TEST = "test"
    PIXEL_VALUES = "pixel_values"
    PIXEL_VALUES_TRIGGER = "pixel_values_trigger"
    TRIGGER = "trigger"
    TARGET = "target"
    IS_CLEAN = "is_clean"
    R_trigger_only = "R_trigger_only"
    IMAGE = "image"
    LABEL = "label"
    def __init__(self, name: str, label: int=None, root: str=None, channel: int=None, image_size: int=None, vmin: Union[int, float]=DEFAULT_VMIN, vmax: Union[int, float]=DEFAULT_VMAX, batch_size: int=512, shuffle: bool=True, seed: int=0):
        self.__root = root
        self.__name = name
        if label != None and not isinstance(label, list)and not isinstance(label, tuple):
            self.__label = [label]
        else:
            self.__label = label
        self.__channel = channel
        self.__vmin = vmin
        self.__vmax = vmax
        self.__batch_size = batch_size
        self.__shuffle = shuffle
        self.__dataset = self.__load_dataset(name=name)
        self.__set_img_shape(image_size=image_size)
        self.__trigger_type = self.__target_type = None
        self.__trigger = self.__target = self.__poison_rate = self.__ext_poison_rate = None
        self.__clean_rate = 1
        self.__seed = seed
        self.__rand_generator = torch.Generator()
        self.__rand_generator.manual_seed(self.__seed)
        if root != None:
            self.__backdoor = Backdoor(root=root)
        self.__R_trigger_only: bool = False
        
        # self.__prep_dataset()

    def set_poison(self, trigger_type: str, target_type: str, target_dx: int=-5, target_dy: int=-3, clean_rate: float=1.0, poison_rate: float=0.2, ext_poison_rate: float=0.0) -> 'DatasetLoader':
        if self.__root == None:
            raise ValueError("Attribute 'root' is None")
        self.__clean_rate = clean_rate
        self.__ext_poison_rate = ext_poison_rate
        self.__poison_rate = poison_rate
        self.__trigger_type = trigger_type
        self.__target_type = target_type
        self.__trigger = self.__backdoor.get_trigger(type=trigger_type, channel=self.__channel, image_size=self.__image_size, vmin=self.__vmin, vmax=self.__vmax)
        self.__target = self.__backdoor.get_target(type=target_type, trigger=self.__trigger, dx=target_dx, dy=target_dy, vmin=self.__vmin, vmax=self.__vmax)
        return self
    
    def __load_dataset(self, name: str):
        datasets.config.IN_MEMORY_MAX_SIZE = 50 * 2 ** 30
        split_method = 'train+test'
        if name == DatasetLoader.MNIST:
            return load_dataset("mnist", split=split_method)
        elif name == DatasetLoader.CIFAR10:
            return load_dataset("cifar10", split=split_method)
        elif name == DatasetLoader.CELEBA:
            return load_dataset("student/celebA", split='train')
        elif name == DatasetLoader.CELEBA_HQ:
            # return load_dataset("huggan/CelebA-HQ", split=split_method)
            return load_dataset("datasets/celeba_hq_256", split='train')
        elif name == DatasetLoader.CELEBA_HQ_LATENT_PR05:
            return load_from_disk("datasets/celeba_hq_256_pr05")
        elif name == DatasetLoader.CELEBA_HQ_LATENT:
            return LatentDataset(ds_root='datasets/celeba_hq_256_latents')
        else:
            raise NotImplementedError(f"Undefined dataset: {name}")
            
    def __set_img_shape(self, image_size: int) -> None:
        # Set channel
        if self.__name == self.MNIST:
            self.__channel = 1 if self.__channel == None else self.__channel
            # self.__vmin = -1
            # self.__vmax = 1
            self.__cmap = "gray"
        elif self.__name == self.CIFAR10 or self.__name == self.CELEBA or self.__name == self.CELEBA_HQ or self.__name == self.LSUN_CHURCH or self.__name == self.CELEBA_HQ_LATENT_PR05 or self.__name == self.CELEBA_HQ_LATENT:
            self.__channel = 3 if self.__channel == None else self.__channel
            # self.__vmin = -1
            # self.__vmax = 1
            self.__cmap = None
        else:
            raise NotImplementedError(f"No dataset named as {self.__name}")

        # Set image size
        if image_size == None:
            if self.__name == self.MNIST:
                self.__image_size = 32
            elif self.__name == self.CIFAR10:
                self.__image_size = 32
            elif self.__name == self.CELEBA:
                self.__image_size = 64
            elif self.__name == self.CELEBA_HQ or self.__name == self.LSUN_CHURCH or self.__name == self.CELEBA_HQ_LATENT_PR05 or self.__name == self.CELEBA_HQ_LATENT:
                self.__image_size = 256
            else:
                raise NotImplementedError(f"No dataset named as {self.__name}")
        else:
            self.__image_size = image_size
            
    def __get_transform(self, prev_trans: List=[], next_trans: List=[]):
        if self.__channel == 1:
            channel_trans = transforms.Grayscale(num_output_channels=1)
        elif self.__channel == 3:
            channel_trans = transforms.Lambda(lambda x: x.convert("RGB"))
        
        aug_trans = []
        if self.__dataset != DatasetLoader.LSUN_CHURCH:
            aug_trans = [transforms.RandomHorizontalFlip()]    
        
        trans = [channel_trans,
                 transforms.Resize([self.__image_size, self.__image_size]), 
                 transforms.ToTensor(),
                transforms.Lambda(lambda x: normalize(vmin_in=0, vmax_in=1, vmin_out=self.__vmin, vmax_out=self.__vmax, x=x)),
                # transforms.Normalize([0.5], [0.5]),
                ] + aug_trans
        return Compose(prev_trans + trans + next_trans)
    
        # trans = [transforms.Resize(self.__image_size), 
        #          transforms.ToTensor(),
        #          transforms.Lambda(lambda x: normalize(vmin=self.__vmin, vmax=self.__vmax, x=x))]
        # return Compose(prev_trans + self.TRANSFORM_OPS +  + next_trans)
        
    def __fixed_sz_dataset_old(self):
        gen = torch.Generator()
        gen.manual_seed(self.__seed)
        
        # Apply transformations
        self.__full_dataset = self.__dataset.with_transform(self.__transform_generator(self.__name, True))

        # Generate poisoned dataset
        if self.__poison_rate > 0:
            full_ds_len = len(self.__full_dataset[DatasetLoader.TRAIN])
            perm_idx = torch.randperm(full_ds_len, generator=gen).long()
            self.__poison_n = int(full_ds_len * float(self.__poison_rate))
            self.__clean_n = full_ds_len - self.__poison_n
            
            # print(f"perm_idx: {perm_idx}")
            # print(f"len(perm_idx): {len(perm_idx)}, max: {torch.max(perm_idx)}, min: {torch.min(perm_idx)}")
            # print(f"Clean n: {self.__clean_n}, Poison n: {self.__poison_n}")
        
            self.__full_dataset[DatasetLoader.TRAIN] = Subset(self.__full_dataset[DatasetLoader.TRAIN], perm_idx[:self.__clean_n].tolist())
            
            # print(f"Clean dataset len: {len(self.__full_dataset[DatasetLoader.TRAIN])}")
        
            self.__backdoor_dataset = self.__dataset.with_transform(self.__transform_generator(self.__name, False))
            self.__backdoor_dataset = Subset(self.__backdoor_dataset[DatasetLoader.TRAIN], perm_idx[self.__clean_n:].tolist())
            # print(f"Backdoor dataset len: {len(self.__backdoor_dataset)}")
            self.__full_dataset[DatasetLoader.TRAIN] = ConcatDataset([self.__full_dataset[DatasetLoader.TRAIN], self.__backdoor_dataset])
            # print(f"self.__full_dataset[DatasetLoader.TRAIN] len: {len(self.__full_dataset[DatasetLoader.TRAIN])}")
        self.__full_dataset = self.__full_dataset[DatasetLoader.TRAIN]
    
    def manual_split():
        pass
    
    def __fixed_sz_dataset(self):
        gen = torch.Generator()
        gen.manual_seed(self.__seed)
        
        if float(self.__poison_rate) < 0 or float(self.__poison_rate) > 1:
            raise ValueError(f"In {DatasetLoader.MODE_FIXED}, poison rate should <= 1.0 and >= 0.0")
        
        ds_n = len(self.__dataset)
        backdoor_n = int(ds_n * float(self.__poison_rate))
        ds_ls = []
        
        # Apply transformations
        if float(self.__poison_rate) == 0.0:
            self.__clean_dataset = self.__dataset
            self.__backdoor_dataset = None
        elif float(self.__poison_rate) == 1.0:
            self.__clean_dataset = None
            self.__backdoor_dataset = self.__dataset
        else:
            full_dataset: datasets.DatasetDict = self.__dataset.train_test_split(test_size=backdoor_n)
            self.__clean_dataset = full_dataset[DatasetLoader.TRAIN]
            self.__backdoor_dataset = full_dataset[DatasetLoader.TEST]
        
        if self.__clean_dataset != None:
            clean_n = len(self.__clean_dataset)
            self.__clean_dataset = self.__clean_dataset.add_column(DatasetLoader.IS_CLEAN, [True] * clean_n)
            ds_ls.append(self.__clean_dataset)
        # print(f"TRAIN IS_CLEAN N: {len(self.__full_dataset[DatasetLoader.TRAIN].filter(lambda x: x[DatasetLoader.IS_CLEAN]))}")
        
        if self.__backdoor_dataset != None:
            backdoor_n = len(self.__backdoor_dataset)
            self.__backdoor_dataset = self.__backdoor_dataset.add_column(DatasetLoader.IS_CLEAN, [False] * backdoor_n)
            ds_ls.append(self.__backdoor_dataset)
        # print(f"TEST !IS_CLEAN N: {len(self.__full_dataset[DatasetLoader.TEST].filter(lambda x: not x[DatasetLoader.IS_CLEAN]))}")
        
        def trans(x):
            if x[DatasetLoader.IS_CLEAN][0]:
                # print(f"IS_CLEAN: {x[DatasetLoader.IS_CLEAN]}")
                return self.__transform_generator(self.__name, True, self.__R_trigger_only)(x)
            return self.__transform_generator(self.__name, False, self.__R_trigger_only)(x)
        
        
        self.__full_dataset = concatenate_datasets(ds_ls)
        # print(f"IS_CLEAN N: {len(self.__full_dataset.filter(lambda x: x[DatasetLoader.IS_CLEAN]))}")
        self.__full_dataset = self.__full_dataset.with_transform(trans)
        # print(f"__full_dataset len: {len(self.__full_dataset)}, features: {self.__full_dataset.features}, keys: {self.__full_dataset[0].keys()}")
        

    def __flex_sz_dataset_old(self):
        # Apply transformations
        self.__full_dataset = self.__dataset.with_transform(self.__transform_generator(self.__name, True))
        
        full_ds_len = len(self.__full_dataset[DatasetLoader.TRAIN])
        
        # Shrink the clean dataset
        if self.__clean_rate != 1:
            self.__clean_n = int(full_ds_len * float(self.__clean_rate))
            self.__full_dataset[DatasetLoader.TRAIN] = Subset(self.__full_dataset[DatasetLoader.TRAIN], list(range(0, self.__clean_n, 1)))
        # MODIFIED: Only 1 poisoned  training sample
        # self.__full_dataset[DatasetLoader.TRAIN] = Subset(self.__full_dataset[DatasetLoader.TRAIN], list(range(0, 1, 1)))
            
        # Generate poisoned dataset
        if self.__poison_rate > 0:
            self.__backdoor_dataset = self.__dataset.with_transform(self.__transform_generator(self.__name, False))
            self.__poison_n = int(full_ds_len * float(self.__poison_rate))
            self.__backdoor_dataset = Subset(self.__backdoor_dataset[DatasetLoader.TRAIN], list(range(0, self.__poison_n, 1)))    
            self.__full_dataset[DatasetLoader.TRAIN] = ConcatDataset([self.__full_dataset[DatasetLoader.TRAIN], self.__backdoor_dataset])
            # MODIFIED: Only 1 clean training sample
            # self.__backdoor_dataset = Subset(self.__backdoor_dataset[DatasetLoader.TRAIN], list(range(0, 1, 1)))
            # self.__full_dataset[DatasetLoader.TRAIN] = self.__backdoor_dataset
            
        self.__full_dataset = self.__full_dataset[DatasetLoader.TRAIN]
        
    def __flex_sz_dataset(self):
        gen = torch.Generator()
        gen.manual_seed(self.__seed)
        
        def portion_sz(rate: float, n: int):
            return int(n * float(rate))
        
        def slice_ds(dataset, rate: float, ds_size: int):
            if float(rate) == 0.0:
                return None
            elif float(rate) == 1.0:
                return dataset
            else:
                return dataset.train_test_split(test_size=portion_sz(rate=rate, n=ds_size))[DatasetLoader.TEST]
        
        ds_ls: List = []
        ds_n = len(self.__dataset)
        print(f"Total Dataset Size: {ds_n}")
        
        # Apply transformations
        self.__full_dataset: datasets.DatasetDict = self.__dataset.train_test_split()
        
        clean_ds = slice_ds(dataset=self.__dataset, rate=float(self.__clean_rate), ds_size=ds_n)
        if clean_ds is not None:
            print(f"[Mode Flex] Clean Dataset Size: {len(clean_ds)}")
            ds_ls.append(clean_ds.add_column(DatasetLoader.IS_CLEAN, [True] * portion_sz(rate=self.__clean_rate, n=ds_n)))
        else:
            print(f"[Mode Flex] Clean Dataset Size: 0")
        
        backdoor_ds = slice_ds(dataset=self.__dataset, rate=float(self.__poison_rate), ds_size=ds_n)
        if backdoor_ds is not None:
            print(f"[Mode Flex] Backdoor Dataset Size: {len(backdoor_ds)}")
            ds_ls.append(backdoor_ds.add_column(DatasetLoader.IS_CLEAN, [False] * portion_sz(rate=self.__poison_rate, n=ds_n)))
        else:
            print(f"[Mode Flex] Backdoor Dataset Size: 0")
        
        # self.__full_dataset[DatasetLoader.TRAIN] = self.__full_dataset[DatasetLoader.TRAIN].add_column(DatasetLoader.IS_CLEAN, [True] * train_n)
        # self.__full_dataset[DatasetLoader.TEST] = self.__full_dataset[DatasetLoader.TEST].add_column(DatasetLoader.IS_CLEAN, [False] * test_n)
        
        def trans(x):
            if x[DatasetLoader.IS_CLEAN][0]:
                return self.__transform_generator(self.__name, True, self.__R_trigger_only)(x)
            return self.__transform_generator(self.__name, False, self.__R_trigger_only)(x)
        
        self.__full_dataset = concatenate_datasets(ds_ls)
        self.__full_dataset = self.__full_dataset.with_transform(trans)
        print(f"[Mode Flex] Full Dataset Size: {len(self.__full_dataset)}")

    def __extend_sz_dataset(self):
        gen = torch.Generator()
        gen.manual_seed(self.__seed)
        
        def portion_sz(rate: float, n: int):
            return int(n * float(rate))
        
        def slice_ds(dataset, rate: float, ds_size: int):
            if float(rate) == 0.0:
                return None
            elif float(rate) == 1.0:
                return dataset
            elif float(rate) > 1.0:
                mul: int = int(rate // 1)
                mod: float = float(rate - mul)
                cat_ds = [slice_ds(dataset, rate=1.0, ds_size=ds_size) for i in range(mul)]
                if mod > 0:
                    cat_ds.append(slice_ds(dataset, rate=mod, ds_size=ds_size))
                return concatenate_datasets(cat_ds)
            else:
                return dataset.train_test_split(test_size=portion_sz(rate=rate, n=ds_size))[DatasetLoader.TEST]

        def trans(x):
            # print(f"x[DatasetLoader.IS_CLEAN] len: {len(x[DatasetLoader.IS_CLEAN])}")
            if x[DatasetLoader.IS_CLEAN][0]:
                return self.__transform_generator(self.__name, True, x[DatasetLoader.R_trigger_only][0])(x)
            return self.__transform_generator(self.__name, False, x[DatasetLoader.R_trigger_only][0])(x)
        
        ds_ls: List = []
        ds_n = len(self.__dataset)
        ext_backdoor_n = int(ds_n * float(self.__ext_poison_rate))
        print(f"Total Dataset Size: {ds_n}")
        clean_dataset = ext_backdoor_dataset = backdoor_dataset = None
        
        # Apply transformations
        if float(self.__ext_poison_rate) == 0.0:
            clean_dataset = self.__dataset
            ext_backdoor_dataset = None
        elif float(self.__ext_poison_rate) == 1.0:
            clean_dataset = None
            ext_backdoor_dataset = self.__dataset
        else:
            full_dataset: datasets.DatasetDict = self.__dataset.train_test_split(test_size=ext_backdoor_n)
            clean_dataset = full_dataset[DatasetLoader.TRAIN]
            ext_backdoor_dataset = full_dataset[DatasetLoader.TEST]
        
        if clean_dataset != None:
            clean_n = len(clean_dataset)
            clean_dataset = clean_dataset.add_column(DatasetLoader.IS_CLEAN, [True] * clean_n).add_column(DatasetLoader.R_trigger_only, [False] * clean_n)
            print(f"[Mode Extend] Clean Dataset Size: {len(clean_dataset)}, {clean_dataset[1].keys()}")
            clean_dataset = clean_dataset.with_transform(trans)
            ds_ls.append(clean_dataset)
        else:
            print(f"[Mode Extend] Clean Dataset Size: 0")
        # print(f"TRAIN IS_CLEAN N: {len(self.__full_dataset[DatasetLoader.TRAIN].filter(lambda x: x[DatasetLoader.IS_CLEAN]))}")
        
        if ext_backdoor_dataset != None:
            ext_backdoor_n = len(ext_backdoor_dataset)
            ext_backdoor_dataset = ext_backdoor_dataset.add_column(DatasetLoader.IS_CLEAN, [False] * ext_backdoor_n).add_column(DatasetLoader.R_trigger_only, [self.__ext_R_trigger_only] * ext_backdoor_n)
            print(f"[Mode Extend] Extend Backdoor Dataset Size: {len(ext_backdoor_dataset)},  {ext_backdoor_dataset[1].keys()}")
            ext_backdoor_dataset = ext_backdoor_dataset.with_transform(trans)
            ds_ls.append(ext_backdoor_dataset)
        else:
            print(f"[Mode Extend] Extend Backdoor Dataset Size: 0")
        # print(f"TEST !IS_CLEAN N: {len(self.__full_dataset[DatasetLoader.TEST].filter(lambda x: not x[DatasetLoader.IS_CLEAN]))}")
        
        backdoor_dataset = slice_ds(dataset=self.__dataset, rate=float(self.__poison_rate), ds_size=ds_n)
        if backdoor_dataset is not None:
            backdoor_n = portion_sz(rate=self.__poison_rate, n=ds_n)
            backdoor_dataset = backdoor_dataset.add_column(DatasetLoader.IS_CLEAN, [False] * backdoor_n).add_column(DatasetLoader.R_trigger_only, [self.__R_trigger_only] * backdoor_n)
            print(f"[Mode Extend] Backdoor Dataset Size: {len(backdoor_dataset)}, {backdoor_dataset[1].keys()}")
            backdoor_dataset = backdoor_dataset.with_transform(trans)
            ds_ls.append(backdoor_dataset)
        else:
            print(f"[Mode Extend] Backdoor Dataset Size: 0")
        
        # self.__full_dataset[DatasetLoader.TRAIN] = self.__full_dataset[DatasetLoader.TRAIN].add_column(DatasetLoader.IS_CLEAN, [True] * train_n)
        # self.__full_dataset[DatasetLoader.TEST] = self.__full_dataset[DatasetLoader.TEST].add_column(DatasetLoader.IS_CLEAN, [False] * test_n)

        self.__full_dataset = concatenate_datasets(ds_ls)
        # self.__full_dataset = self.__full_dataset.with_transform(trans)
        print(f"[Mode Extend] Full Dataset Size: {len(self.__full_dataset)}")
    
    def prepare_dataset(self, mode: str="FIXED", R_trigger_only: bool=False, ext_R_trigger_only: bool=False, R_gaussian_aug: float=0.0) -> 'DatasetLoader':
        self.__R_trigger_only = R_trigger_only
        self.__ext_R_trigger_only = ext_R_trigger_only
        self.__R_gaussian_aug = R_gaussian_aug
        # Filter specified classes
        if self.__label != None:
            self.__dataset = self.__dataset.filter(lambda x: x[DatasetLoader.LABEL] in self.__label)
        
        if mode == DatasetLoader.MODE_FIXED:
            if self.__clean_rate != 1.0 or self.__clean_rate != None:
                Log.warning("In 'FIXED' mode of DatasetLoader, the clean_rate will be ignored whatever.")
            self.__fixed_sz_dataset()
        elif mode == DatasetLoader.MODE_FLEX:
            self.__flex_sz_dataset()
        elif mode == DatasetLoader.MODE_EXTEND:
            self.__extend_sz_dataset()
        elif mode == DatasetLoader.MODE_NONE:
            self.__full_dataset = self.__dataset
        else:
            raise NotImplementedError(f"Argument mode: {mode} isn't defined")
        
        # Special Handling for LatentDataset
        if self.__name == self.CELEBA_HQ_LATENT:
            self.__full_dataset.set_poison(target_key=self.__target_type, poison_key=self.__trigger_type, raw='raw', poison_rate=self.__poison_rate, use_latent=True).set_use_names(target=DatasetLoader.TARGET, poison=DatasetLoader.PIXEL_VALUES, raw=DatasetLoader.IMAGE)
        
        # Note the minimum and the maximum values
        print(f"{self.__full_dataset[1].keys()}")
        ex = self.__full_dataset[1][DatasetLoader.TARGET]
        print(f"Dataset Len: {len(self.__full_dataset)}")
        if len(ex) == 1:
            print(f"Note that CHANNEL 0 - vmin: {torch.min(ex[0])} and vmax: {torch.max(ex[0])}")    
        elif len(ex) == 3:
            print(f"Note that CHANNEL 0 - vmin: {torch.min(ex[0])} and vmax: {torch.max(ex[0])} | CHANNEL 1 - vmin: {torch.min(ex[1])} and vmax: {torch.max(ex[1])} | CHANNEL 2 - vmin: {torch.min(ex[2])} and vmax: {torch.max(ex[2])}")
        return self

    def get_dataset(self) -> datasets.Dataset:
        return self.__full_dataset
    
    def save_dataset(self, file: str):
        self.__full_dataset.save_to_disk(file)

    def get_dataloader(self, batch_size: int=None, shuffle: bool=None, num_workers: int=None, collate_fn: callable=None) -> torch.utils.data.DataLoader:
        datasets = self.get_dataset()
        if batch_size == None:
            batch_size = self.__batch_size
        if shuffle == None:
            shuffle = self.__shuffle
        if num_workers == None:
            num_workers = 8
        if collate_fn != None:
            return DataLoader(datasets, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=num_workers, collate_fn=collate_fn)
        return DataLoader(datasets, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=num_workers)
    
    def get_mask(self, trigger: torch.Tensor) -> torch.Tensor:
        return torch.where(trigger > self.__vmin, 0, 1)

    def __transform_generator(self, dataset_name: str, clean: bool, R_trigger_only: bool) -> Callable[[torch.Tensor], torch.Tensor]:
        if dataset_name == self.MNIST:
            img_key = "image"
        elif dataset_name == self.CIFAR10:
            img_key = "img"
        if dataset_name == self.CELEBA:
            img_key = "image"
        if dataset_name == self.CELEBA_HQ:
            img_key = "image"
        # define function
        def clean_transforms(examples) -> DatasetDict:
            if dataset_name == self.MNIST:
                trans = self.__get_transform()
                examples[DatasetLoader.IMAGE] = torch.stack([trans(image.convert("L")) for image in examples[img_key]])
            else:
                # trans = self.__get_transform(prev_trans=[transforms.PILToTensor()])
                trans = self.__get_transform()
                # trans = Compose([transforms.PILToTensor(), transforms.Lambda(lambda t: t / 255)])
                examples[DatasetLoader.IMAGE] = torch.stack([trans(image) for image in examples[img_key]])
                # examples[DatasetLoader.PIXEL_VALUES] = torch.tensor(np.array([np.asarray(image) / 255 for image in examples[img_key]])).permute(0, 3, 1, 2)
                if img_key != DatasetLoader.IMAGE:
                    del examples[img_key]
            
            examples[DatasetLoader.PIXEL_VALUES_TRIGGER] = torch.full_like(examples[DatasetLoader.IMAGE], 0)
            examples[DatasetLoader.PIXEL_VALUES] = torch.full_like(examples[DatasetLoader.IMAGE], 0)
            examples[DatasetLoader.TARGET] = torch.clone(examples[DatasetLoader.IMAGE])
            
            data_shape = examples[DatasetLoader.PIXEL_VALUES].shape
            repeat_times = (data_shape[0], *([1] * len(data_shape[1:])))
            examples[DatasetLoader.TRIGGER] = self.__trigger.repeat(*repeat_times)
            
            # examples[DatasetLoader.IS_CLEAN] = torch.tensor([True] * len(examples[DatasetLoader.PIXEL_VALUES]))
            if DatasetLoader.LABEL in examples:
                examples[DatasetLoader.LABEL] = torch.tensor([torch.tensor(x, dtype=torch.float) for x in examples[DatasetLoader.LABEL]])
            else:
                examples[DatasetLoader.LABEL] = torch.tensor([torch.tensor(-1, dtype=torch.float) for i in range(len(examples[DatasetLoader.PIXEL_VALUES]))])
            # print(f"examples[img_key] Type: {type(examples[img_key])}")
            # examples[img_key] = torch.tensor(np.array([np.asarray(image) / 255 for image in examples[img_key]])).permute(2, 0, 1)
            # examples[img_key] = torch.stack([self.__get_transform()(np.asarray(image)) for image in examples[img_key]])
            return examples
        def backdoor_transforms(examples) -> DatasetDict:
            examples = clean_transforms(examples)
            
            data_shape = examples[DatasetLoader.PIXEL_VALUES].shape
            repeat_times = (data_shape[0], *([1] * len(data_shape[1:])))
            
            masks = self.get_mask(self.__trigger).repeat(*repeat_times)
            # print(f"masks shape: {masks.shape} | examples[DatasetLoader.PIXEL_VALUES] shape: {examples[DatasetLoader.PIXEL_VALUES].shape} | self.__trigger.repeat(*repeat_times) shape: {self.__trigger.repeat(*repeat_times).shape}")
            # examples[DatasetLoader.PIXEL_VALUES] = masks * examples[DatasetLoader.IMAGE] + (1 - masks) * self.__trigger.repeat(*repeat_times)

            examples[DatasetLoader.PIXEL_VALUES_TRIGGER] = self.__trigger.repeat(*repeat_times)
            if R_trigger_only:
                examples[DatasetLoader.PIXEL_VALUES] = self.__trigger.repeat(*repeat_times)
            else:
                examples[DatasetLoader.PIXEL_VALUES] = masks * examples[DatasetLoader.IMAGE] + (1 - masks) * self.__trigger.repeat(*repeat_times)
                
            # print(f"self.__target.repeat(*repeat_times) shape: {self.__target.repeat(*repeat_times).shape}")
            examples[DatasetLoader.TARGET] = self.__target.repeat(*repeat_times)
            # examples[DatasetLoader.IS_CLEAN] = torch.tensor([False] * data_shape[0])
            return examples
        
        if clean:
            return clean_transforms
        return backdoor_transforms
    
    def get_poisoned(self, imgs) -> torch.Tensor:
        data_shape = imgs.shape
        repeat_times = (data_shape[0], *([1] * len(data_shape[1:])))
        
        masks = self.get_mask(self.__trigger).repeat(*repeat_times)
        return masks * imgs + (1 - masks) * self.__trigger.repeat(*repeat_times)
    
    def get_inpainted(self, imgs, mask: torch.Tensor) -> torch.Tensor:
        data_shape = imgs.shape
        repeat_times = (data_shape[0], *([1] * len(data_shape[1:])))
        
        notthing_tensor = torch.full_like(imgs, fill_value=torch.min(imgs))
        masks = mask.repeat(*repeat_times)
        return masks * imgs + (1 - masks) * notthing_tensor
    
    def get_inpainted_boxes(self, imgs, up: int, low: int, left: int, right: int) -> torch.Tensor:        
        masked_val = 0
        unmasked_val = 1
        mask = torch.full_like(imgs[0], fill_value=unmasked_val)
        if len(mask.shape) == 3:
            mask[:, up:low, left:right] = masked_val
        elif len(mask.shape) == 2:
            mask[up:low, left:right] = masked_val
        return self.get_inpainted(imgs=imgs, mask=mask)
    
    def get_inpainted_by_type(self, imgs: torch.Tensor, inpaint_type: str) -> torch.Tensor:
        if inpaint_type == DatasetLoader.INPAINT_LINE:
            half_dim = imgs.shape[-1] // 2
            up = half_dim - half_dim
            low = half_dim + half_dim
            left = half_dim - half_dim // 10
            right = half_dim + half_dim // 20
            return self.get_inpainted_boxes(imgs=imgs, up=up, low=low, left=left, right=right)
        elif inpaint_type == DatasetLoader.INPAINT_BOX:
            half_dim = imgs.shape[-1] // 2
            up_left = half_dim - half_dim // 3
            low_right = half_dim + half_dim // 3
            return self.get_inpainted_boxes(imgs=imgs, up=up_left, low=low_right, left=up_left, right=low_right)
        else: 
            raise NotImplementedError(f"inpaint: {inpaint_type} is not implemented")

    def show_sample(self, img: torch.Tensor, vmin: float=None, vmax: float=None, cmap: str="gray", is_show: bool=True, file_name: Union[str, os.PathLike]=None, is_axis: bool=False) -> None:
        cmap_used = self.__cmap if cmap == None else cmap
        vmin_used = self.__vmin if vmin == None else vmin
        vmax_used = self.__vmax if vmax == None else vmax
        normalize_img = normalize(x=img, vmin_in=vmin_used, vmax_in=vmax_used, vmin_out=0, vmax_out=1)
        channel_last_img = normalize_img.permute(1, 2, 0).reshape(self.__image_size, self.__image_size, self.__channel)
        plt.imshow(channel_last_img, vmin=0, vmax=1, cmap=cmap_used)
        # plt.imshow(img.permute(1, 2, 0).reshape(self.__image_size, self.__image_size, self.__channel), vmin=None, vmax=None, cmap=cmap_used)
        # plt.imshow(img)

        if not is_axis:
            plt.axis('off')
        
        plt.tight_layout()            
        if is_show:
            plt.show()
        if file_name != None:
            save_image(normalize_img, file_name)
        
    @property
    def len(self):
        return len(self.get_dataset())
    
    def __len__(self):
        return self.len
        
    @property
    def num_batch(self):
        return len(self.get_dataloader())
    
    @property
    def trigger(self):
        return self.__trigger
    
    @property
    def target(self):
        return self.__target
    
    @property
    def name(self):
        return self.__name
    
    @property
    def root(self):
        return self.__root
    
    @property
    def batch_size(self):
        return self.__batch_size
    
    @property
    def channel(self):
        return self.__channel
    
    @property
    def image_size(self):
        return self.__image_size

class Backdoor():
    CHANNEL_LAST = -1
    CHANNEL_FIRST = -3
    
    GREY_BG_RATIO = 0.3
    
    STOP_SIGN_IMG = "static/stop_sign_wo_bg.png"
    # STOP_SIGN_IMG = "static/stop_sign_bg_blk.jpg"
    CAT_IMG = "static/cat_wo_bg.png"
    GLASSES_IMG = "static/glasses.png"
    
    TARGET_FA = "SHOE"
    TARGET_TG = "NOSHIFT"
    TARGET_BOX = "CORNER"
    # TARGET_BOX_MED = "BOX_MED"
    TARGET_SHIFT = "SHIFT"
    TARGET_HAT = "BWHAT"
    TARGET_FEDORA_HAT = "HAT"
    TARGET_CAT = "CAT"
    
    TRIGGER_GAP_X = TRIGGER_GAP_Y = 2
    
    TRIGGER_NONE = "NONE"
    TRIGGER_FA = "FASHION"
    TRIGGER_FA_EZ = "FASHION_EZ"
    TRIGGER_MNIST = "MNIST"
    TRIGGER_MNIST_EZ = "MNIST_EZ"
    TRIGGER_SM_BOX = "SM_BOX"
    TRIGGER_XSM_BOX = "XSM_BOX"
    TRIGGER_XXSM_BOX = "XXSM_BOX"
    TRIGGER_XXXSM_BOX = "XXXSM_BOX"
    TRIGGER_BIG_BOX = "BIG_BOX"
    TRIGGER_BIG_BOX_MED = "BOX_18"
    TRIGGER_SM_BOX_MED = "BOX_14"
    TRIGGER_XSM_BOX_MED = "BOX_11"
    TRIGGER_XXSM_BOX_MED = "BOX_8"
    TRIGGER_XXXSM_BOX_MED = "BOX_4"
    TRIGGER_GLASSES = "GLASSES"
    TRIGGER_BIG_STOP_SIGN = "STOP_SIGN_18"
    TRIGGER_SM_STOP_SIGN = "STOP_SIGN_14"
    TRIGGER_XSM_STOP_SIGN = "STOP_SIGN_11"
    TRIGGER_XXSM_STOP_SIGN = "STOP_SIGN_8"
    TRIGGER_XXXSM_STOP_SIGN = "STOP_SIGN_4"
    
    # GREY_NORM_MIN = 0
    # GREY_NORM_MAX = 1
    
    def __init__(self, root: str):
        self.__root = root
        
    def __get_transform(self, channel: int, image_size: Union[int, Tuple[int]], vmin: Union[float, int], vmax: Union[float, int], prev_trans: List=[], next_trans: List=[]):
        if channel == 1:
            channel_trans = transforms.Grayscale(num_output_channels=1)
        elif channel == 3:
            channel_trans = transforms.Lambda(lambda x: x.convert("RGB"))
            
        trans = [channel_trans,
                 transforms.Resize(image_size), 
                 transforms.ToTensor(),
                #  transforms.Lambda(lambda x: normalize(vmin_out=vmin, vmax_out=vmax, x=x)),
                 transforms.Lambda(lambda x: normalize(vmin_in=0.0, vmax_in=1.0, vmin_out=vmin, vmax_out=vmax, x=x)),
                #  transforms.Lambda(lambda x: x * 2 - 1),
                ]
        return Compose(prev_trans + trans + next_trans)
    
    @staticmethod
    def __read_img(path: Union[str, os.PathLike]):
        return Image.open(path)
    @staticmethod
    def __bg2grey(trig, vmin: Union[float, int], vmax: Union[float, int]):
        thres = (vmax - vmin) * Backdoor.GREY_BG_RATIO + vmin
        trig[trig <= thres] = thres
        return trig
    @staticmethod
    def __bg2black(trig, vmin: Union[float, int], vmax: Union[float, int]):
        thres = (vmax - vmin) * Backdoor.GREY_BG_RATIO + vmin
        trig[trig <= thres] = vmin
        return trig
    @staticmethod
    def __white2grey(trig, vmin: Union[float, int], vmax: Union[float, int]):
        thres = vmax - (vmax - vmin) * Backdoor.GREY_BG_RATIO
        trig[trig >= thres] = thres
        return trig
    @staticmethod
    def __white2med(trig, vmin: Union[float, int], vmax: Union[float, int]):
        thres = vmax - (vmax - vmin) * Backdoor.GREY_BG_RATIO
        trig[trig >= 0.7] = (vmax - vmin) / 2
        return trig
    
    def __get_img_target(self, path: Union[str, os.PathLike], image_size: int, channel: int, vmin: Union[float, int], vmax: Union[float, int]):
        img = Backdoor.__read_img(path)
        trig = self.__get_transform(channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)(img)
        return Backdoor.__bg2grey(trig=trig, vmin=vmin, vmax=vmax)
    
    def __get_img_trigger(self, path: Union[str, os.PathLike], image_size: int, channel: int, trigger_sz: int, vmin: Union[float, int], vmax: Union[float, int], x: int=None, y: int=None):
        # Padding of Left & Top
        l_pad = t_pad = int((image_size - trigger_sz) / 2)
        r_pad = image_size - trigger_sz - l_pad
        b_pad = image_size - trigger_sz - t_pad
        residual = image_size - trigger_sz
        if x != None:
            if x > 0:
                l_pad = x
                r_pad = residual - l_pad
            else:
                r_pad = -x
                l_pad = residual - r_pad
        if y != None:
            if y > 0:
                t_pad = y
                b_pad = residual - t_pad
            else:
                b_pad = -y
                t_pad = residual - b_pad
        
        img = Backdoor.__read_img(path)
        next_trans = [transforms.Pad(padding=[l_pad, t_pad, r_pad, b_pad], fill=vmin)]
        trig = self.__get_transform(channel=channel, image_size=trigger_sz, vmin=vmin, vmax=vmax, next_trans=next_trans)(img)
        # thres = (vmax - vmin) * 0.3 + vmin
        # trig[trig <= thres] = vmin
        trig[trig >= 0.999] = vmin
        # print(f"trigger shape: {trig.shape}")
        return trig
    @staticmethod
    def __roll(x: torch.Tensor, dx: int, dy: int):
        shift = tuple([0] * len(x.shape[:-2]) + [dy] + [dx])
        dim = tuple([i for i in range(len(x.shape))])
        return torch.roll(x, shifts=shift, dims=dim)
    @staticmethod
    def __get_box_trig(b1: Tuple[int, int], b2: Tuple[int, int], channel: int, image_size: int, vmin: Union[float, int], vmax: Union[float, int], val: Union[float, int]):
        if isinstance(image_size, int):
            img_shape = (image_size, image_size)
        elif isinstance(image_size, list):
            img_shape = image_size
        else:
            raise TypeError(f"Argument image_size should be either an integer or a list")
        trig = torch.full(size=(channel, *img_shape), fill_value=vmin)
        trig[:, b1[0]:b2[0], b1[1]:b2[1]] = val
        return trig
    @staticmethod
    def __get_white_box_trig(b1: Tuple[int, int], b2: Tuple[int, int], channel: int, image_size: int, vmin: Union[float, int], vmax: Union[float, int]):
        return Backdoor.__get_box_trig(b1=b1, b2=b2, channel=channel, image_size=image_size, vmin=vmin, vmax=vmax, val=vmax)
    @staticmethod
    def __get_grey_box_trig(b1: Tuple[int, int], b2: Tuple[int, int], channel: int, image_size: int, vmin: Union[float, int], vmax: Union[float, int]):
        return Backdoor.__get_box_trig(b1=b1, b2=b2, channel=channel, image_size=image_size, vmin=vmin, vmax=vmax, val=(vmin + vmax) / 2)
    @staticmethod
    def __get_trig_box_coord(x: int, y: int):
        if x < 0 or y < 0:
            raise ValueError(f"Argument x, y should > 0")
        return (- (y + Backdoor.TRIGGER_GAP_Y), - (x + Backdoor.TRIGGER_GAP_X)), (- Backdoor.TRIGGER_GAP_Y, - Backdoor.TRIGGER_GAP_X)
    
    def get_trigger(self, type: str, channel: int, image_size: int, vmin: Union[float, int]=DEFAULT_VMIN, vmax: Union[float, int]=DEFAULT_VMAX) -> torch.Tensor:
        if type == Backdoor.TRIGGER_FA:
            trans = self.__get_transform(channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)
            ds = FashionMNIST(root=self.__root, train=True, download=True, transform=trans)
            return Backdoor.__roll(Backdoor.__bg2black(trig=ds[0][0], vmin=vmin, vmax=vmax), dx=0, dy=2)
        elif type == Backdoor.TRIGGER_FA_EZ:
            trans = self.__get_transform(channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)
            ds = FashionMNIST(root=self.__root, train=True, download=True, transform=trans)
            # Backdoor image ID: 135, 144
            # return ds[144][0]
            return Backdoor.__roll(Backdoor.__bg2black(trig=ds[144][0], vmin=vmin, vmax=vmax), dx=0, dy=4)
        elif type == Backdoor.TRIGGER_MNIST:
            trans = self.__get_transform(channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)
            ds = MNIST(root=self.__root, train=True, download=True, transform=trans)
            # Backdoor image ID: 3, 6, 8
            # return ds[3][0]
            return Backdoor.__roll(Backdoor.__bg2black(trig=ds[3][0], vmin=vmin, vmax=vmax), dx=10, dy=3)
        elif type == Backdoor.TRIGGER_MNIST_EZ:
            trans = self.__get_transform(channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)
            ds = MNIST(root=self.__root, train=True, download=True, transform=trans)
            # Backdoor image ID: 3, 6, 8
            # return ds[6][0]
            return Backdoor.__roll(Backdoor.__bg2black(trig=ds[6][0], vmin=vmin, vmax=vmax), dx=10, dy=3)
        elif type == Backdoor.TRIGGER_SM_BOX:    
            b1, b2 = Backdoor.__get_trig_box_coord(14, 14)
            # trig = torch.full(size=(channel, image_size, image_size), fill_value=vmin)
            # trig[:, b1[0]:b2[0], b1[1]:b2[1]] = vmax
            # return trig
            return Backdoor.__get_white_box_trig(b1=b1, b2=b2, channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)
        elif type == Backdoor.TRIGGER_XSM_BOX:    
            b1, b2 = Backdoor.__get_trig_box_coord(11, 11)
            # trig = torch.full(size=(channel, image_size, image_size), fill_value=vmin)
            # trig[:, b1[0]:b2[0], b1[1]:b2[1]] = vmax
            # return trig
            return Backdoor.__get_white_box_trig(b1=b1, b2=b2, channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)
        elif type == Backdoor.TRIGGER_XXSM_BOX:    
            b1, b2 = Backdoor.__get_trig_box_coord(8, 8)
            # trig = torch.full(size=(channel, image_size, image_size), fill_value=vmin)
            # trig[:, b1[0]:b2[0], b1[1]:b2[1]] = vmax
            # return trig
            return Backdoor.__get_white_box_trig(b1=b1, b2=b2, channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)
        elif type == Backdoor.TRIGGER_XXXSM_BOX:    
            b1, b2 = Backdoor.__get_trig_box_coord(4, 4)
            # trig = torch.full(size=(channel, image_size, image_size), fill_value=vmin)
            # trig[:, b1[0]:b2[0], b1[1]:b2[1]] = vmax
            # return trig
            return Backdoor.__get_white_box_trig(b1=b1, b2=b2, channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)
        elif type == Backdoor.TRIGGER_BIG_BOX:    
            b1, b2 = Backdoor.__get_trig_box_coord(18, 18)
            # trig = torch.full(size=(channel, image_size, image_size), fill_value=vmin)
            # trig[:, b1[0]:b2[0], b1[1]:b2[1]] = vmax
            # return trig
            return Backdoor.__get_white_box_trig(b1=b1, b2=b2, channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)
        elif type == Backdoor.TRIGGER_BIG_BOX_MED:
            b1, b2 = Backdoor.__get_trig_box_coord(18, 18)
            return Backdoor.__get_grey_box_trig(b1=b1, b2=b2, channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)
        elif type == Backdoor.TRIGGER_SM_BOX_MED:
            b1, b2 = Backdoor.__get_trig_box_coord(14, 14)
            # trig = torch.full(size=(channel, image_size, image_size), fill_value=vmin)
            # trig[:, b1[0]:b2[0], b1[1]:b2[1]] = (vmax + vmin) / 2
            # return trig
            return Backdoor.__get_grey_box_trig(b1=b1, b2=b2, channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)
        elif type == Backdoor.TRIGGER_XSM_BOX_MED:    
            b1, b2 = Backdoor.__get_trig_box_coord(11, 11)
            # trig = torch.full(size=(channel, image_size, image_size), fill_value=vmin)
            # trig[:, b1[0]:b2[0], b1[1]:b2[1]] = (vmax + vmin) / 2
            # return trig
            return Backdoor.__get_grey_box_trig(b1=b1, b2=b2, channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)
        elif type == Backdoor.TRIGGER_XXSM_BOX_MED:    
            b1, b2 = Backdoor.__get_trig_box_coord(8, 8)
            # trig = torch.full(size=(channel, image_size, image_size), fill_value=vmin)
            # trig[:, b1[0]:b2[0], b1[1]:b2[1]] = (vmax + vmin) / 2
            # return trig
            return Backdoor.__get_grey_box_trig(b1=b1, b2=b2, channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)
        elif type == Backdoor.TRIGGER_XXXSM_BOX_MED:    
            b1, b2 = Backdoor.__get_trig_box_coord(4, 4)
            # trig = torch.full(size=(channel, image_size, image_size), fill_value=vmin)
            # trig[:, b1[0]:b2[0], b1[1]:b2[1]] = (vmax + vmin) / 2
            # return trig
            return Backdoor.__get_grey_box_trig(b1=b1, b2=b2, channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)
        elif type == Backdoor.TRIGGER_GLASSES:
            trigger_sz = int(image_size * 0.625)
            return self.__get_img_trigger(path=Backdoor.GLASSES_IMG, image_size=image_size, channel=channel, trigger_sz=trigger_sz, vmin=vmin, vmax=vmax)
        elif type == Backdoor.TRIGGER_BIG_STOP_SIGN:
            return self.__get_img_trigger(path=Backdoor.STOP_SIGN_IMG, image_size=image_size, channel=channel, trigger_sz=18, vmin=vmin, vmax=vmax, x=-2, y=-2)
        elif type == Backdoor.TRIGGER_SM_STOP_SIGN:
            return self.__get_img_trigger(path=Backdoor.STOP_SIGN_IMG, image_size=image_size, channel=channel, trigger_sz=14, vmin=vmin, vmax=vmax, x=-2, y=-2)
        elif type == Backdoor.TRIGGER_XSM_STOP_SIGN:
            return self.__get_img_trigger(path=Backdoor.STOP_SIGN_IMG, image_size=image_size, channel=channel, trigger_sz=11, vmin=vmin, vmax=vmax, x=-2, y=-2)
        elif type == Backdoor.TRIGGER_XXSM_STOP_SIGN:
            return self.__get_img_trigger(path=Backdoor.STOP_SIGN_IMG, image_size=image_size, channel=channel, trigger_sz=8, vmin=vmin, vmax=vmax, x=-2, y=-2)
        elif type == Backdoor.TRIGGER_XXXSM_STOP_SIGN:
            return self.__get_img_trigger(path=Backdoor.STOP_SIGN_IMG, image_size=image_size, channel=channel, trigger_sz=4, vmin=vmin, vmax=vmax, x=-2, y=-2)
        elif type == Backdoor.TRIGGER_NONE:    
            # trig = torch.zeros(channel, image_size, image_size)
            trig = torch.full(size=(channel, image_size, image_size), fill_value=vmin)
            return trig
        else:
            raise ValueError(f"Trigger type {type} isn't found")
    
    def __check_channel(self, sample: torch.Tensor, channel_first: bool=None) -> int:
        if channel_first != None:
            # If user specified the localation of the channel
            if self.__channel_first:
                if sample.shape[Backdoor.CHANNEL_FIRST] == 1 or sample.shape[Backdoor.CHANNEL_FIRST] == 3:
                    return Backdoor.CHANNEL_FIRST
            elif sample.shape[Backdoor.CHANNEL_LAST] == 1 or sample.shape[Backdoor.CHANNEL_LAST] == 3:
                return Backdoor.CHANNEL_LAST
            warnings.warn(Log.warning("The specified Channel doesn't exist, determine channel automatically"))
            print(Log.warning("The specified Channel doesn't exist, determine channel automatically"))
                    
        # If user doesn't specified the localation of the channel or the 
        if (sample.shape[Backdoor.CHANNEL_LAST] == 1 or sample.shape[Backdoor.CHANNEL_LAST] == 3) and \
           (sample.shape[Backdoor.CHANNEL_FIRST] == 1 or sample.shape[Backdoor.CHANNEL_FIRST] == 3):
            raise ValueError(f"Duplicate channel found, found {sample.shape[Backdoor.CHANNEL_LAST]} at dimension 2 and {sample.shape[Backdoor.CHANNEL_FIRST]} at dimension 0")

        if sample.shape[Backdoor.CHANNEL_LAST] == 1 or sample.shape[Backdoor.CHANNEL_LAST] == 3:
            return Backdoor.CHANNEL_LAST
        elif sample.shape[Backdoor.CHANNEL_FIRST] == 1 or sample.shape[Backdoor.CHANNEL_FIRST] == 3:
            return Backdoor.CHANNEL_FIRST
        else:
            raise ValueError(f"Invalid channel shape, found {sample.shape[Backdoor.CHANNEL_LAST]} at dimension 2 and {sample.shape[Backdoor.CHANNEL_FIRST]} at dimension 0")
        
    def __check_image_size(self, sample: torch.Tensor, channel_loc: int):
        image_size = list(sample.shape)[-3:]
        del image_size[channel_loc]
        return image_size
    
    def get_target(self, type: str, trigger: torch.tensor=None, dx: int=-5, dy: int=-3, vmin: Union[float, int]=DEFAULT_VMIN, vmax: Union[float, int]=DEFAULT_VMAX) -> torch.Tensor:
        channel_loc = self.__check_channel(sample=trigger, channel_first=None)
        channel = trigger.shape[channel_loc]
        image_size = self.__check_image_size(sample=trigger, channel_loc=channel_loc)
        print(f"image size: {image_size}")
        if type == Backdoor.TARGET_TG:
            if trigger == None:
                raise ValueError("trigger shouldn't be none")
            return Backdoor.__bg2grey(trigger.clone().detach(), vmin=vmin, vmax=vmax)
        elif type == Backdoor.TARGET_SHIFT:
            if trigger == None:
                raise ValueError("trigger shouldn't be none")
            # t_trig = trigger.clone().detach()
            # shift = tuple([0] * len(t_trig.shape[:-2]) + [dy] + [dx])
            # dim = tuple([i for i in range(len(t_trig.shape))])
            # # print(f"Shift: {shift} | t_trig: {t_trig.shape}")
            # return torch.roll(t_trig, shifts=shift, dims=dim)
            return Backdoor.__bg2grey(Backdoor.__roll(trigger.clone().detach(), dx=dx, dy=dy), vmin=vmin, vmax=vmax)
        # elif type == Backdoor.TARGET_BOX:
        #     # z = torch.full_like(trigger, fill_value=vmin)
        #     # z[:, 0:10, 0:10] = vmax
        #     # return z
        #     b1 = (None, None)
        #     b2 = (10, 10)
        #     return Backdoor.__get_white_box_trig(b1=b1, b2=b2, channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)
        elif type == Backdoor.TARGET_BOX:
            b1 = (None, None)
            b2 = (10, 10)
            return Backdoor.__bg2grey(trig=Backdoor.__get_grey_box_trig(b1=b1, b2=b2, channel=channel, image_size=image_size, vmin=vmin, vmax=vmax), vmin=vmin, vmax=vmax)
        elif type == Backdoor.TARGET_FA:
            trans = self.__get_transform(channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)
            ds = FashionMNIST(root=self.__root, train=True, download=True, transform=trans)
            # return ds[0][0]
            return Backdoor.__bg2grey(trig=ds[0][0], vmin=vmin, vmax=vmax)
        elif type == Backdoor.TARGET_HAT:
            # img = Backdoor.__read_img("static/hat.png")
            # trig = self.__get_transform(channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)(img)
            # return trig
            return self.__get_img_target(path="static/hat.png", channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)
        elif type == Backdoor.TARGET_FEDORA_HAT:
            # img = Backdoor.__read_img("static/fedora-hat.png")
            # trig = self.__get_transform(channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)(img)
            # return trig
            return self.__get_img_target(path="static/fedora-hat.png", channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)
        elif type == Backdoor.TARGET_CAT:
            # img = Backdoor.__read_img("static/cat.png")
            # trig = self.__get_transform(channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)(img)
            # return trig
            return self.__get_img_target(path=Backdoor.CAT_IMG, channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)
        else:
            raise NotImplementedError(f"Target type {type} isn't found")
        
    def show_image(self, img: torch.Tensor):
        plt.axis('off')        
        plt.tight_layout()
        plt.imshow(img.permute(1, 2, 0).squeeze(), cmap='gray')
        plt.show()

class ReplicateDataset(torch.utils.data.Dataset):
    def __init__(self, val: torch.Tensor, n: int):
        self.__val: torch.Tensor = val
        self.__n: int = n
        self.__one_vec = [1 for i in range(self.__n)]
        
    def __len__(self):
        return self.__n
    
    def __getitem__(self, slc):
        n: int = len(self.__one_vec[slc])
        reps = ([len(self.__val)] + ([1] * n))
        return torch.squeeze((self.__val.repeat(*reps)))

class ImagePathDataset(torch.utils.data.Dataset):
    IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm', 'tif', 'tiff', 'webp'}
    # TRANSFORM = [transforms.ToTensor()]
    
    def __init__(self, path, transforms=None, njobs: int=-1):
        self.__path = pathlib.Path(path)
        self.__files = sorted([file for ext in ImagePathDataset.IMAGE_EXTENSIONS
                       for file in self.__path.glob('*.{}'.format(ext))])
        self.__transforms = transforms
        self.__njobs = njobs

    def __len__(self):
        return len(self.__files)
    
    def read_imgs(self, paths: Union[str, List[str]]):
        # to_tensor = lambda path: transforms.ToTensor()(Image.open(path).copy().convert('RGB'))
        # trans_ls = [transforms.Lambda(to_tensor)]
        trans_ls = [transforms.Lambda(ImagePathDataset.__read_img)]
        if self.__transforms != None:
            trans_ls += self.__transforms
            
        if isinstance(paths, list):
            if self.__njobs == None:
                Log.info(f"n-jobs: {self.__njobs}, Read images sequentially")
                imgs = [Compose(trans_ls)(path) for path in paths]
            else:
                Log.info(f"n-jobs: {self.__njobs}, Read images concurrently")
                imgs = list(Parallel(n_jobs=self.__njobs)(delayed(Compose(trans_ls))(path) for path in paths))
            Log.info(f"n-jobs: {self.__njobs}, Reading Images done")
            return torch.stack(imgs)
        return transforms.ToTensor()(Image.open(paths).convert('RGB'))
    
    def fetch_slice(self, start: int, end: int, step: int=1):
        read_ls: List[str] = list(set(self.__files))[slice(start, end, step)]
        # return Compose([transforms.Lambda(self.read_imgs)])(read_ls)
        return self.read_imgs(read_ls)

    @staticmethod
    # @lru_cache(1000)
    def __read_img(path):
        return transforms.ToTensor()(Image.open(path).copy().convert('RGB'))
    
    def __getitem__(self, slc):
        # img = Compose([transforms.Lambda(self.read_imgs)])(self.__files[slc])
        # return img
        return self.read_imgs(list(set(self.__files))[slc])
    
class LatentDataset(torch.utils.data.Dataset):
    DATA_EXT: str = ".pt"
    TARGET_LATENTS_FILE_NAME: str = f"target"
    POISON_LATENTS_FILE_NAME: str = f"poison"
    RAW_LATENTS_FILE_NAME: str = f"raw"
    
    def __init__(self, ds_root: str):
        self.__ds_root: str = LatentDataset.__check_dir(ds_root)
        self.__used_target = None
        self.__used_poison = None
        self.__used_raw = None
        self.__used_target_name = None
        self.__used_poison_name = None
        self.__used_raw_name = None
        self.__poison_rate = None
        self.__len = None
        self.__vae = None
        self.__n_jobs = 10
        
    def set_vae(self, vae):
        self.__vae = vae
        return self
    
    @staticmethod
    def __check_dir(p: Union[str, os.PathLike]):
        os.makedirs(p, exist_ok=True)
        return p
    
    @staticmethod
    def add_ext(p: str):
        return f"{p}.{LatentDataset.DATA_EXT}"
        
    @property
    def targe_latents_path(self):
        p = os.path.join(self.__ds_root, LatentDataset.TARGET_LATENTS_FILE_NAME)
        if not os.path.exists(LatentDataset.add_ext(p)):
            LatentDataset.save_ext(val={}, file=p)
        return p
    
    def __get_list_dir_path(self, dir: Union[str, os.PathLike]):
        return LatentDataset.__check_dir(os.path.join(self.__ds_root, dir))
    
    def __get_list_idx_path(self, dir: Union[str, os.PathLike], idx: int):
        return os.path.join(self.__get_list_dir_path(dir=dir), f"{idx}")
    
    def __get_data_list_dir(self, data_type: str):
        return data_type
    
    @staticmethod
    def read_ext(file: str) -> torch.Tensor:
        try:
            return LatentDataset.read(LatentDataset.add_ext(file))
        except:
            print(f"No such file: {file}")
            return None
    
    @staticmethod
    def save_ext(val: object, file: str) -> None:
        LatentDataset.save(val, LatentDataset.add_ext(file))
        
    @staticmethod
    def read(file: str) -> torch.Tensor:
        return torch.load(file)
    
    @staticmethod
    def save(val: object, file: str) -> None:
        torch.save(val, file)
    
    @staticmethod
    def __encode_latents_static(x: torch.Tensor, vae, weight_dtype: str=None, scaling_factor: float=None) -> torch.Tensor:
        vae = vae.eval()
        with torch.no_grad():
            x = x.to(vae.device)
            if weight_dtype != None and weight_dtype != "":
                x = x.to(dtype=weight_dtype)
            if scaling_factor != None:
                return (vae.encode(x).latents * scaling_factor).detach().cpu()
            # return vae.encode(x).latents * vae.config.scaling_factor
            return vae.encode(x).latents.detach().cpu()
        
    @staticmethod
    def __decode_latents_static(vae, x: torch.Tensor, weight_dtype: str=None, scaling_factor: float=None) -> torch.Tensor:
        vae = vae.eval()
        with torch.no_grad():
            x = x.to(vae.device)
            if weight_dtype != None and weight_dtype != "":
                x = x.to(dtype=weight_dtype)
            if scaling_factor != None:
                return (vae.decode(x).sample / scaling_factor).clone().detach().cpu()
            # return vae.decode(x).sample / vae.config.scaling_factor
            return (vae.decode(x).sample).clone().detach().cpu()
    
    def __encode_latents(self, x: torch.Tensor, weight_dtype: str=None, scaling_factor: float=None) -> torch.Tensor:    
        if self.__vae == None:
            raise ValueError("Please provide encoder first")
        return LatentDataset.__encode_latents_static(vae=self.__vae, x=x, weight_dtype=weight_dtype, scaling_factor=scaling_factor)
    
    def __decode_latents(self, x: torch.Tensor, weight_dtype: str=None, scaling_factor: float=None) -> torch.Tensor:
        if self.__vae == None:
            raise ValueError("Please provide encoder first")
        return LatentDataset.__decode_latents_static(vae=self.__vae, x=x, weight_dtype=weight_dtype, scaling_factor=scaling_factor)
        
    @staticmethod
    def __update_dict_key_latent(file: Union[str, os.PathLike], key: str, val: torch.Tensor) -> None:
        res: dict = LatentDataset.read_ext(file=file)
        if res == None:
            res = {}
        res[key] = val
        LatentDataset.save_ext(val=res, file=file)
        
    def __update_dict_key(self, file: Union[str, os.PathLike], key: str, val: torch.Tensor) -> None:
        LatentDataset.__update_dict_key_latent(file=file, key=key, val=self.__encode_latents(x=val.unsqueeze(0)).squeeze(0))
        
    def __update_dict_keys(self, file: Union[str, os.PathLike], keys: List[str], vals: torch.Tensor) -> None:
        latents = self.__encode_latents(x=vals)
        for key, latent in zip(keys, latents):
            LatentDataset.__update_dict_key_latent(file=file, key=key, val=latent)
        
    @staticmethod
    def __get_dict_key_latent(file: Union[str, os.PathLike], key: str) -> torch.Tensor:
        res: dict = LatentDataset.read_ext(file=file)
        return res[key]
        
    def __get_dict_key(self, file: Union[str, os.PathLike], key: str) -> torch.Tensor:
        val: torch.Tensor = LatentDataset.__get_dict_key_latent(file=file, key=key)
        print(f"val: {val.shape}")
        return self.__decode_latents(x=val.unsqueeze(0)).squeeze(0)
        
    def __update_list_idx_latent(self, dir: Union[str, os.PathLike], idx: int, val: torch.Tensor):
        LatentDataset.save_ext(val=val, file=self.__get_list_idx_path(dir=dir, idx=idx))
        
    def __update_list_idx(self, dir: Union[str, os.PathLike], idx: int, val: torch.Tensor):
        self.__update_list_idx_latent(dir=dir, idx=idx, val=self.__encode_latents(x=val.unsqueeze(0)).squeeze(0))
        
    def __update_list_idxs(self, dir: Union[str, os.PathLike], idxs: List[int], vals: torch.Tensor):
        latents = self.__encode_latents(vals)
        for idx, latent in zip(idxs, latents):
            self.__update_list_idx_latent(dir=dir, idx=idx, val=latent)
        
    def __get_list_idx_latent(self, dir: Union[str, os.PathLike], idx: int):
        return LatentDataset.read_ext(file=self.__get_list_idx_path(dir=dir, idx=idx))
        
    def __get_list_idx(self, dir: Union[str, os.PathLike], idx: int):
        val: torch.Tensor = self.__get_list_idx_latent(dir=dir, idx=idx)
        return self.__decode_latents(x=val.unsqueeze(0)).squeeze(0)
        
    def get_target_latent_by_key(self, key: str):
        return LatentDataset.__get_dict_key_latent(file=self.targe_latents_path, key=key)
    
    def get_target_latents_by_keys(self, keys: List[str]):
        ls: List[torch.Tensor] = []
        for key in keys:
            ls.append(self.get_target_latent_by_key(key=key))
        return ls
    
    def get_target_by_key(self, key: str):
        return self.__get_dict_key(file=self.targe_latents_path, key=key)
    
    def get_targets_by_keys(self, keys: List[str]):
        ls: List[torch.Tensor] = []
        for key in keys:
            ls.append(self.get_target_by_key(key=key))
        return ls
    
    def update_target_latent_by_key(self, key: str, val: torch.Tensor):
        self.__update_dict_key_latent(file=self.targe_latents_path, key=key, val=val)
        
    def update_target_latents_by_keys(self, keys: List[str], vals: List[torch.Tensor]):
        for key, val in zip(keys, vals):
            self.update_target_latent_by_key(key=key, val=val)
    
    def update_target_by_key(self, key: str, val: torch.Tensor):
        self.__update_dict_key(file=self.targe_latents_path, key=key, val=val)
        
    def update_targets_by_keys(self, keys: List[str], vals: List[torch.Tensor]):
        if isinstance(idxs, int):
            idxs = [idxs]
            if isinstance(vals, torch.Tensor):
                vals = vals.unsqueeze(0)
        elif isinstance(idxs, list):
            if isinstance(vals, list):
                vals = torch.cat(vals)
        for key, val in zip(keys, vals):
            self.__update_dict_keys(key=key, val=val)
    
    def get_data_latent_by_idx(self, data_type: str, idx: int):
        return self.__get_list_idx_latent(dir=self.__get_data_list_dir(data_type=data_type), idx=idx)
    
    def get_data_latents_by_idxs(self, data_type: str, keys: List[str]):
        ls: List[torch.Tensor] = []
        for key in keys:
            ls.append(self.get_data_latent_by_idx(data_type=data_type, key=key))
        return ls
    
    def get_data_by_idx(self, data_type: str, idx: int):
        return self.__get_list_idx(dir=self.__get_data_list_dir(data_type=data_type), idx=idx)
    
    def get_data_by_idxs(self, data_type: str, idxs: List[int]):
        ls: List[torch.Tensor] = []
        for idx in idxs:
            ls.append(self.get_data_by_idx(data_type=data_type, idx=idx))
        return ls
    
    def update_data_latent_by_idx(self, data_type: str, idx: int, val: torch.Tensor):
        self.__update_list_idx_latent(dir=self.__get_data_list_dir(data_type=data_type), idx=idx, val=val)
        
    def update_data_latents_by_idxs(self, data_type: str, idxs: List[str], vals: List[torch.Tensor]):
        # Parallel(n_jobs=self.__n_jobs)(delayed(self.update_data_latent_by_idx)(data_type, idx, val) for idx, val in zip(idxs, vals)) 
        if isinstance(idxs, int):
            idxs = [idxs]
            if isinstance(vals, torch.Tensor):
                vals = vals.unsqueeze(0)
        elif isinstance(idxs, list):
            if isinstance(vals, list):
                vals = torch.cat(vals)
        for idx, val in zip(idxs, vals):
            self.update_data_latent_by_idx(data_type=data_type, idx=idx, val=val)
    
    def update_data_by_idx(self, data_type: str, idx: int, val: torch.Tensor):
        self.__update_list_idx(dir=self.__get_data_list_dir(data_type=data_type), idx=idx, val=val)
        
    def update_data_by_idxs(self, data_type: str, idxs: List[int], vals: Union[List[torch.Tensor], torch.Tensor]):
        # for idx, val in zip(idxs, vals):
        #     self.update_data_by_idx(data_type=data_type, idx=idx, val=val)
        if isinstance(idxs, int):
            idxs = [idxs]
            if isinstance(vals, torch.Tensor):
                vals = vals.unsqueeze(0)
        self.__update_list_idxs(dir=self.__get_data_list_dir(data_type=data_type), idxs=idxs, vals=vals)
            
    def get_target(self):
        if self.__use_latent:
            return self.get_target_latent()
        if self.__used_target == None:
            raise ValueError(f"Please et up the target first")
        return self.get_target_by_key(key=self.__used_target)
    
    def get_target_latent(self):
        if self.__used_target == None:
            raise ValueError(f"Please et up the target first")
        return self.get_target_latent_by_key(key=self.__used_target)
    
    def get_poison_by_idxs(self, idxs: Union[int, List[int]]):
        if self.__use_latent:
            return self.get_poison_latents_by_idxs(idxs=idxs)
        if self.__used_poison == None:
            raise ValueError(f"Please et up the poison first")
        if isinstance(idxs, int):
            return self.get_data_by_idx(data_type=self.__used_poison, idx=idxs)
        elif isinstance(idxs, list):
            return self.get_data_by_idxs(data_type=self.__used_poison, idxs=idxs)
        else:
            raise NotImplementedError(f"Arguement idxs must be either int or lists, not {type(idxs)}")    
    
    def get_poison_latents_by_idxs(self, idxs: Union[int, List[int]]):
        if self.__used_poison == None:
            raise ValueError(f"Please et up the poison first")
        if isinstance(idxs, int):
            return self.get_data_latent_by_idx(data_type=self.__used_poison, idx=idxs)
        elif isinstance(idxs, list):
            return self.get_data_latents_by_idxs(data_type=self.__used_poison, idxs=idxs)
        else:
            raise NotImplementedError(f"Arguement idxs must be either int or lists, not {type(idxs)}")    
    
    def get_raw_by_idxs(self, idxs: Union[int, List[int]]):
        if self.__use_latent:
            return self.get_raw_latents_by_idxs(idxs=idxs)
        if self.__used_raw == None:
            raise ValueError(f"Please et up the raw first")
        if isinstance(idxs, int):
            return self.get_data_by_idx(data_type=self.__used_raw, idx=idxs)
        elif isinstance(idxs, list):
            return self.get_data_by_idxs(data_type=self.__used_raw, idxs=idxs)
        else:
            raise NotImplementedError(f"Arguement idxs must be either int or lists, not {type(idxs)}")
    
    def get_raw_latents_by_idxs(self, idxs: int):
        if self.__used_raw == None:
            raise ValueError(f"Please et up the raw first")
        if isinstance(idxs, int):
            return self.get_data_latent_by_idx(data_type=self.__used_raw, idx=idxs)
        elif isinstance(idxs, list):
            return self.get_data_latents_by_idxs(data_type=self.__used_raw, idxs=idxs)
        else:
            raise NotImplementedError(f"Arguement idxs must be either int or lists, not {type(idxs)}")
    
    def set_poison(self, target_key: str, poison_key: str, raw: str, poison_rate: float, use_latent: bool=True):
        self.__used_target = target_key
        self.__used_poison = poison_key
        self.__used_raw = raw
        self.__poison_rate = poison_rate
        self.__use_latent = use_latent
        
        p = self.__get_list_dir_path(dir=self.__get_data_list_dir(self.__used_raw))
        self.__len = len([file for file in glob.glob(os.path.join(p, f"*{LatentDataset.DATA_EXT}"))])
        return self
    
    def set_use_names(self, target: str, poison: str, raw: str):
        self.__used_target_name = target
        self.__used_poison_name = poison
        self.__used_raw_name = raw
        return self
    
    def __len__(self):
        return self.__len
    
    def __getitem__(self, i: int):
        i = i % len(self)
        if i < 0:
            i = i + len(self)
            
        def zeros_like(x):
            def fn(idx: int):
                return torch.zeros_like(x)
            return fn
        def clean_poison(clean_fn: callable, poison_fn: callable):
            def fn(idx: int):
                if idx < int(self.__len * self.__poison_rate):
                    # print(f"Poisoned")
                    return poison_fn(idx)
                # print(f"Clean")
                return clean_fn(idx)
            return fn
        if self.__use_latent:
            return {
                    self.__used_target_name: clean_poison(self.get_raw_latents_by_idxs, lambda idxs: self.get_target_latent())(i), 
                    self.__used_poison_name: clean_poison(zeros_like(self.get_raw_latents_by_idxs(idxs=i)), self.get_poison_latents_by_idxs)(i),
                    self.__used_raw_name: self.get_raw_latents_by_idxs(idxs=i)
                    }
        else:
            return {
                    self.__used_target_name: clean_poison(self.get_raw_by_idxs, lambda idxs: self.get_target_latent())(i), 
                    self.__used_poison_name: clean_poison(zeros_like(self.get_raw_by_idxs(idxs=i)), self.get_poison_by_idxs)(i),
                    self.__used_raw_name: self.get_raw_by_idxs(idxs=i)
                    }
            
# %%
if __name__ == "__main__":
    from diffusers import DDPMScheduler
    
    ds_root = os.path.join('datasets')
    # dsl = DatasetLoader(root=ds_root, name=DatasetLoader.MNIST, label=1).set_poison(trigger_type=Backdoor.TRIGGER_XXXSM_BOX, target_type=Backdoor.TARGET_BOX, clean_rate=0.5, poison_rate=0.2).prepare_dataset()
    dsl = DatasetLoader(root=ds_root, name=DatasetLoader.CELEBA_HQ, batch_size=128).set_poison(trigger_type=Backdoor.TRIGGER_GLASSES, target_type=Backdoor.TARGET_CAT, clean_rate=0.2, poison_rate=0.4).prepare_dataset(mode=DatasetLoader.MODE_FIXED)
    # dsl = DatasetLoader(root=ds_root, name=DatasetLoader.CELEBA_HQ, batch_size=128).set_poison(trigger_type=Backdoor.TRIGGER_GLASSES, target_type=Backdoor.TARGET_CAT, clean_rate=1.0, poison_rate=0.5).prepare_dataset(mode=DatasetLoader.MODE_FIXED)
    print(f"Full Dataset Len: {len(dsl)}")
    
    # dsl.save_dataset(f"datasets/celeba_hq_256_pr05")

    train_ds = dsl.get_dataset()
    sample = train_ds[-1]
    print(f"{sample.keys()}")
    print(f"Full Dataset Len: {len(train_ds)} | Sample Len: {len(sample)}")
    print(f"Clean Target: {sample['target'].shape} | Label: {sample['label']}  | pixel_values: {sample['pixel_values'].shape}")
    print(f"Clean PIXEL_VALUES Shape: {sample['pixel_values'].shape} | vmin: {torch.min(sample['pixel_values'])} | vmax: {torch.max(sample['pixel_values'])} | CLEAN: {sample['is_clean']}")
    dsl.show_sample(sample['pixel_values'])
    print(f"Clean TARGET Shape: {sample['target'].shape} | vmin: {torch.min(sample['target'])} | vmax: {torch.max(sample['target'])} | CLEAN: {sample['is_clean']}")
    dsl.show_sample(sample['target'])
    print(f"Clean IMAGE Shape: {sample['image'].shape} | vmin: {torch.min(sample['image'])} | vmax: {torch.max(sample['image'])} | CLEAN: {sample['is_clean']}")
    dsl.show_sample(sample['image'])
    
    # Count clean samples and poison samples
    # is_cleans = torch.tensor(train_ds[:]['is_clean'])
    # print(f"clean_n: {torch.count_nonzero(torch.where(is_cleans, 1, 0))}, poison_n: {torch.count_nonzero(torch.where(is_cleans, 0, 1))}")
    
    # CIFAR10
    # sample = train_ds[36000]
    # sample = train_ds[5000] # for label = 1
    # MNIST
    # sample = train_ds[60000]
    # sample = train_ds[6742]
    # sample = train_ds[3371]
    # sample = train_ds[14000]
    # sample = train_ds[35000] # For FIXED_MODE
    # CELEBA
    # sample = train_ds[101300]
    # CELEBA-HQ
    sample = train_ds[18000] # For FIXED_MODE
    
    noise = torch.randn_like(sample['target'], dtype=torch.float)
    # noise[1:, :, :] = 0
    
    # Noisy Images
    timesteps = torch.Tensor([100]).long()
    noise_sched = DDPMScheduler(num_train_timesteps=1000)
    R_coef = 1 - torch.sqrt(noise_sched.alphas_cumprod)
    
    # clean_image = sample['image']
    clean_image = sample[DatasetLoader.TARGET]
    R = sample[DatasetLoader.PIXEL_VALUES]
    
    print(f"Noisy Images, 0 steps")
    noisy_image = noise_sched.add_noise(clean_image, noise, torch.Tensor([0]).long()) + R_coef[0] * R
    dsl.show_sample(noisy_image, file_name="tmp_0_backdoor.png")
    # noisy_image = noise_sched.add_noise(clean_image, noise, torch.Tensor([0]).long())
    # dsl.show_sample(noisy_image, file_name="tmp_0.png")
    print(f"Noisy Images, 100 steps")
    noisy_image = noise_sched.add_noise(clean_image, noise, torch.Tensor([99]).long()) + R_coef[99] * R
    dsl.show_sample(noisy_image, file_name="tmp_99_backdoor.png")
    # noisy_image = noise_sched.add_noise(clean_image, noise, torch.Tensor([99]).long())
    # dsl.show_sample(noisy_image, file_name="tmp_99.png")
    print(f"Noisy Images, 300 steps")
    noisy_image = noise_sched.add_noise(clean_image, noise, torch.Tensor([299]).long()) + R_coef[299] * R
    dsl.show_sample(noisy_image, file_name="tmp_299_backdoor.png")
    # noisy_image = noise_sched.add_noise(clean_image, noise, torch.Tensor([299]).long())
    # dsl.show_sample(noisy_image, file_name="tmp_299.png")
    print(f"Noisy Images, 500 steps")
    noisy_image = noise_sched.add_noise(clean_image, noise, torch.Tensor([499]).long()) + R_coef[499] * R
    dsl.show_sample(noisy_image, file_name="tmp_499_backdoor.png")
    # noisy_image = noise_sched.add_noise(clean_image, noise, torch.Tensor([499]).long())
    # dsl.show_sample(noisy_image, file_name="tmp_499.png")
    print(f"Noisy Images, 700 steps")
    noisy_image = noise_sched.add_noise(clean_image, noise, torch.Tensor([699]).long()) + R_coef[699] * R
    dsl.show_sample(noisy_image, file_name="tmp_699_backdoor.png")
    # noisy_image = noise_sched.add_noise(clean_image, noise, torch.Tensor([699]).long())
    # dsl.show_sample(noisy_image, file_name="tmp_699.png")
    print(f"Noisy Images, 1000 steps")
    noisy_image = noise_sched.add_noise(clean_image, noise, torch.Tensor([999]).long()) + R_coef[999] * R
    dsl.show_sample(noisy_image, file_name="tmp_999_backdoor.png")
    # noisy_image = noise_sched.add_noise(clean_image, noise, torch.Tensor([999]).long())
    # dsl.show_sample(noisy_image, file_name="tmp_999.png")
    
    print(f"Full Dataset Len: {len(train_ds)} | Sample Len: {len(sample)}")
    print(f"Backdoor Target: {sample['target'].shape} | Label: {sample['label']}  | pixel_values: {sample['pixel_values'].shape}")
    print(f"Backdoor PIXEL_VALUES Shape: {sample['pixel_values'].shape} | vmin: {torch.min(sample['pixel_values'])} | vmax: {torch.max(sample['pixel_values'])} | CLEAN: {sample['is_clean']}")
    dsl.show_sample(sample['pixel_values'])
    print(f"Backdoor TARGET Shape: {sample['target'].shape} | vmin: {torch.min(sample['target'])} | vmax: {torch.max(sample['target'])} | CLEAN: {sample['is_clean']}")
    dsl.show_sample(sample['target'])
    print(f"Backdoor Noisy PIXEL_VALUES Shape: {sample['pixel_values'].shape} | vmin: {torch.min(sample['pixel_values'])} | vmax: {torch.max(sample['pixel_values'])} | CLEAN: {sample['is_clean']}")
    dsl.show_sample(sample['pixel_values'] + noise)
    print(f"Backdoor IMAGE Shape: {sample['image'].shape} | vmin: {torch.min(sample['image'])} | vmax: {torch.max(sample['image'])} | CLEAN: {sample['is_clean']}")
    dsl.show_sample(sample['image'])

    # create dataloader
    train_dl = dsl.get_dataloader()

    batch = next(iter(train_dl))
    
    # Backdoor
    channel = 3
    image_size = 256
    grid_size = 5
    vmin = float(0.0)
    vmax = float(1.0)
    run = os.path.dirname(os.path.abspath(__file__))
    root_p = os.path.join(run, 'datasets')
    backdoor = Backdoor(root=root_p)
    
    plt.axis('off')        
    plt.tight_layout()
    # SM_BOX_MED Trigger
    tr = backdoor.get_trigger(type=Backdoor.TRIGGER_SM_STOP_SIGN, channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)
    backdoor.show_image(img=tr)
    # plt.imshow(torch.squeeze(tr), cmap='gray', vmin=vmin, vmax=vmax)
    # plt.show()
    # SM_BOX Trigger
    tr = backdoor.get_trigger(type=Backdoor.TRIGGER_SM_BOX_MED, channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)
    backdoor.show_image(img=tr)
    # plt.imshow(torch.squeeze(tr), cmap='gray', vmin=vmin, vmax=vmax)
    # plt.show()
    # XSM_BOX Trigger
    tr = backdoor.get_trigger(type=Backdoor.TRIGGER_XSM_BOX_MED, channel=channel, image_size=image_size, vmin=vmin, vmax=vmax).permute(1, 2, 0)
    plt.imshow(torch.squeeze(tr), cmap='gray', vmin=vmin, vmax=vmax)
    plt.show()
    # XXSM_BOX Trigger
    tr = backdoor.get_trigger(type=Backdoor.TRIGGER_XXSM_BOX_MED, channel=channel, image_size=image_size, vmin=vmin, vmax=vmax).permute(1, 2, 0)
    plt.imshow(torch.squeeze(tr), cmap='gray', vmin=vmin, vmax=vmax)
    plt.show()
    # XXXSM_BOX Trigger
    tr = backdoor.get_trigger(type=Backdoor.TRIGGER_XXXSM_BOX_MED, channel=channel, image_size=image_size, vmin=vmin, vmax=vmax).permute(1, 2, 0)
    plt.imshow(torch.squeeze(tr), cmap='gray', vmin=vmin, vmax=vmax)
    plt.show()
    # GLASSES Trigger
    tr = backdoor.get_trigger(type=Backdoor.TRIGGER_GLASSES, channel=3, image_size=image_size, vmin=vmin, vmax=1)
    backdoor.show_image(img=tr + noise)
    backdoor.show_image(img=noise)
    # print(f"tr vmin: {torch.min(tr)}, vmax: {torch.max(tr)}")
    # plt.imshow(torch.squeeze(tr), vmin=vmin, vmax=vmax)
    # plt.show()
    
# %%
