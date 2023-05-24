# %%
"""
Backdoor Poisoned Dataset
"""

import io
import json
import os
import pathlib
from random import sample
import random
import shutil
import tempfile
import traceback
from typing import Callable, List, Tuple, Union
from functools import lru_cache, partial
import warnings

import jsonlines
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets, get_dataset_split_names, IterableDataset, load_from_disk
import datasets
from datasets.dataset_dict import DatasetDict
from matplotlib import pyplot as plt
import numpy as np
import requests
import torch
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader, ConcatDataset, Subset, Dataset, IterableDataset
from torchvision.datasets import MNIST, CIFAR10, SVHN, FashionMNIST
from PIL import Image
from joblib import Parallel, delayed

from util import Log, normalize
# from tmp_parse_dataset import LaionCoco

DEFAULT_VMIN = float(-1.0)
DEFAULT_VMAX = float(1.0)

class DatasetLoader(object):
    # Dataset generation mode
    MODE_FIXED = "FIXED"
    MODE_FLEX = "FLEX"
    
    # Dataset names
    MNIST = "MNIST"
    CIFAR10 = "CIFAR10"
    CELEBA = "CELEBA"
    LSUN_CHURCH = "LSUN-CHURCH"
    LSUN_BEDROOM = "LSUN-BEDROOM"
    CELEBA_HQ = "CELEBA-HQ"
    CELEBA_HQ_DIALOG = "CELEBA-HQ-DIALOG"
    LAION_COCO = "LAION-COCO"
    LAION_COCO_1 = "LAION-COCO-1"
    LAION_COCO_20K = "LAION-COCO-20K"
    LAION_COCO_200 = "LAION-COCO-200"
    LAION_COCO_50K = "LAION-COCO-50K"
    POKEMON_CAPTION = "POKEMON-CAPTION"
    
    # Inpaint Type
    INPAINT_BOX: str = "INPAINT_BOX"
    INPAINT_LINE: str = "INPAINT_LINE"

    TRAIN = "train"
    TEST = "test"
    POISON_IMAGE = "poison_image"
    IMAGE = "image"
    IS_CLEAN = "is_clean"
    RAW = "raw"
    LABEL = "label"
    CAPTION = "caption"
    RAW_CAPTION = "raw_caption"
    
    CAPTION_AUGMENT_KEY: str = "caption_aug"
    # CAPTION_TOKEN = "caption_token"
    def __init__(self, name: str, label: int=None, root: str=None, 
                 channel: int=None, image_size: int=None, split: str='[:100%]',
                 vmin: Union[int, float]=DEFAULT_VMIN, vmax: Union[int, float]=DEFAULT_VMAX, 
                 batch_size: int=512, shuffle: bool=True, num_workers: int=8, force_R_to_0: bool=False, seed: int=0):
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
        self.__split = split
        self.__dataset = self.__load_dataset(name=name)
        self.__set_img_shape(image_size=image_size)
        self.__trigger = self.__target = self.__caption_trigger = self.__poison_rate = None
        self.__clean_rate = 1
        self.__seed = seed
        self.__num_workers = num_workers
        self.__force_R_to_0 = force_R_to_0
        self.__caption_backdoor = CaptionBackdoor()
        if root != None:
            self.__backdoor = Backdoor(root=root)
        
        # self.__prep_dataset()

    def set_poison(self, trigger_type: str, target_type: str, caption_trigger_type: str=None, rand_caption_trig_pos: int=0, target_dx: int=-5, target_dy: int=-3, clean_rate: float=1.0, poison_rate: float=0.2) -> 'DatasetLoader':
        if self.__root == None:
            raise ValueError("Attribute 'root' is None")
        self.__clean_rate = clean_rate
        self.__poison_rate = poison_rate
        self.__trigger = self.__backdoor.get_trigger(type=trigger_type, channel=self.__channel, image_size=self.__image_size, vmin=self.__vmin, vmax=self.__vmax)
        self.__caption_trigger = self.__caption_backdoor.get_trigger(_type=caption_trigger_type)
        self.__rand_caption_trig_pos: int = rand_caption_trig_pos
        self.__target = self.__backdoor.get_target(type=target_type, trigger=self.__trigger, dx=target_dx, dy=target_dy)
        return self
    
    def __load_dataset(self, name: str):
        datasets.config.IN_MEMORY_MAX_SIZE = 50 * 2 ** 30
        split_method = f'train{self.__split}+test{self.__split}'
        if name == DatasetLoader.MNIST:
            return load_dataset("mnist", split=split_method)
        elif name == DatasetLoader.CIFAR10:
            return load_dataset("cifar10", split=split_method)
        elif name == DatasetLoader.CELEBA:
            return load_dataset("student/celebA", split=f"train{self.__split}")
        elif name == DatasetLoader.CELEBA_HQ:
            return load_dataset("datasets/celeba_hq_256", split=f"train{self.__split}")
        elif name ==DatasetLoader.CELEBA_HQ_DIALOG:
            return CelebA_HQ_Dialog(path="datasets/CelebA-Dialog (HQ)").prepare(split=f"train{self.__split}")
        elif name == DatasetLoader.LAION_COCO or name == DatasetLoader.LAION_COCO_20K:
            return LaionCoco.load("/work/u2941379/workspace/laion_coco_hg200K.hf")
        elif name == DatasetLoader.LAION_COCO_1:
            return LaionCoco.load("/work/u2941379/workspace/laion_coco_hg1.hf")
        elif name == DatasetLoader.LAION_COCO_200:
            return LaionCoco.load("/work/u2941379/workspace/laion_coco_hg200.hf")
        elif name == DatasetLoader.LAION_COCO_50K:
            return LaionCoco.load("/work/u2941379/workspace/laion_coco_hg50K.hf")
        elif name == DatasetLoader.POKEMON_CAPTION:
            return load_dataset("lambdalabs/pokemon-blip-captions", split=f"train{self.__split}")
        else:
            raise NotImplementedError(f"Undefined dataset: {name}")
            
    def __set_img_shape(self, image_size: int) -> None:
        # Set channel
        if self.__name == self.MNIST:
            self.__channel = 1 if self.__channel == None else self.__channel
            # self.__vmin = -1
            # self.__vmax = 1
            self.__cmap = "gray"
        elif self.__name == self.CIFAR10 or self.__name == self.CELEBA or self.__name == self.CELEBA_HQ or self.__name == self.LSUN_CHURCH or self.__name == self.LAION_COCO or self.__name == self.LAION_COCO_1 or self.__name == self.LAION_COCO_200 or self.__name == self.LAION_COCO_20K or self.__name == self.LAION_COCO_50K or self.__name == self.POKEMON_CAPTION or self.__name == self.CELEBA_HQ_DIALOG:
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
            elif self.__name == self.CELEBA_HQ or self.__name == self.LSUN_CHURCH:
                self.__image_size = 256
            elif self.__name == self.LAION_COCO or self.__name == self.LAION_COCO_1 or self.__name == self.LAION_COCO_200 or self.__name == self.LAION_COCO_20K or self.__name == self.LAION_COCO_50K or self.__name == self.POKEMON_CAPTION or self.__name == self.CELEBA_HQ_DIALOG:
                self.__image_size = 512
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
                return self.__transform_generator(self.__name, True)(x)
            return self.__transform_generator(self.__name, False)(x)
        
        
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
        
        ds_n = len(self.__dataset)
        train_n = int(ds_n * float(self.__clean_rate))
        test_n = int(ds_n * float(self.__poison_rate))
        
        # Apply transformations
        self.__full_dataset: datasets.DatasetDict = self.__dataset.train_test_split(train_size=train_n, test_size=test_n)
        self.__full_dataset[DatasetLoader.TRAIN] = self.__full_dataset[DatasetLoader.TRAIN].add_column(DatasetLoader.IS_CLEAN, [True] * train_n)
        self.__full_dataset[DatasetLoader.TEST] = self.__full_dataset[DatasetLoader.TEST].add_column(DatasetLoader.IS_CLEAN, [False] * test_n)
        
        def trans(x):
            if x[DatasetLoader.IS_CLEAN][0]:
                return self.__transform_generator(self.__name, True)(x)
            return self.__transform_generator(self.__name, False)(x)
        
        self.__full_dataset = concatenate_datasets([self.__full_dataset[DatasetLoader.TRAIN], self.__full_dataset[DatasetLoader.TEST]])
        self.__full_dataset = self.__full_dataset.with_transform(trans)
    
    def prepare_dataset(self, mode: str="FIXED") -> 'DatasetLoader':
        # Filter specified classes
        if self.__label != None:
            self.__dataset = self.__dataset.filter(lambda x: x[DatasetLoader.LABEL] in self.__label)
            
        # # Apply transformations
        # self.__full_dataset = self.__dataset.with_transform(self.__transform_generator(self.__name, True))
        
        # full_ds_len = len(self.__full_dataset[DatasetLoader.TRAIN])
        
        # # Shrink the clean dataset
        # if isinstance(self.__clean_rate, float) and self.__clean_rate != 1:
        #     self.__clean_n = int(full_ds_len * self.__clean_rate)
        #     self.__full_dataset[DatasetLoader.TRAIN] = Subset(self.__full_dataset[DatasetLoader.TRAIN], list(range(0, self.__clean_n, 1)))
        # # MODIFIED: Only 1 poisoned  training sample
        # # self.__full_dataset[DatasetLoader.TRAIN] = Subset(self.__full_dataset[DatasetLoader.TRAIN], list(range(0, 1, 1)))
            
        # # Generate poisoned dataset
        # if isinstance(self.__poison_rate, float) and self.__poison_rate > 0:
        #     self.__backdoor_dataset = self.__dataset.with_transform(self.__transform_generator(self.__name, False))
        #     self.__poison_n = int(full_ds_len * self.__poison_rate)
        #     self.__backdoor_dataset = Subset(self.__backdoor_dataset[DatasetLoader.TRAIN], list(range(0, self.__poison_n, 1)))    
        #     self.__full_dataset[DatasetLoader.TRAIN] = ConcatDataset([self.__full_dataset[DatasetLoader.TRAIN], self.__backdoor_dataset])
        #     # MODIFIED: Only 1 clean training sample
        #     # self.__backdoor_dataset = Subset(self.__backdoor_dataset[DatasetLoader.TRAIN], list(range(0, 1, 1)))
        #     # self.__full_dataset[DatasetLoader.TRAIN] = self.__backdoor_dataset
        
        if mode == DatasetLoader.MODE_FIXED:
            if self.__clean_rate != 1.0 or self.__clean_rate != None:
                Log.warning("In 'FIXED' mode of DatasetLoader, the clean_rate will be ignored whatever.")
            self.__fixed_sz_dataset()
        elif mode == DatasetLoader.MODE_FLEX:
            self.__flex_sz_dataset()
        else:
            raise NotImplementedError(f"Argument mode: {mode} isn't defined")
        
        # Note the minimum and the maximum values
        ex = self.__full_dataset[0][DatasetLoader.IMAGE]
        if len(ex) == 1:
            print(f"Note that CHANNEL 0 - vmin: {torch.min(ex[0])} and vmax: {torch.max(ex[0])}")    
        elif len(ex) == 3:
            print(f"Note that CHANNEL 0 - vmin: {torch.min(ex[0])} and vmax: {torch.max(ex[0])} | CHANNEL 1 - vmin: {torch.min(ex[1])} and vmax: {torch.max(ex[1])} | CHANNEL 2 - vmin: {torch.min(ex[2])} and vmax: {torch.max(ex[2])}")
        return self

    def get_dataset(self) -> datasets.Dataset:
        return self.__full_dataset

    def get_dataloader(self) -> torch.utils.data.DataLoader:
        datasets = self.get_dataset()
        get_dsl = partial(DataLoader, datasets, batch_size=self.__batch_size, shuffle=self.__shuffle, pin_memory=True, num_workers=self.__num_workers)
        # if self.__name == DatasetLoader.LAION_COCO or self.__name == DatasetLoader.LAION_COCO_200 or self.__name == DatasetLoader.LAION_COCO_50K:
        #     return get_dsl(collate_fn=lambda x: x)
        return get_dsl()
    
    def get_mask(self, trigger: torch.Tensor) -> torch.Tensor:
        return torch.where(trigger > self.__vmin, 0, 1)

    def store_dataset(self, path: str):
        os.makedirs(path, exist_ok=True)
    
        if self.__name == self.MNIST:
            img_key = "image"
            cap_keys = []
        elif self.__name == self.CIFAR10:
            img_key = "img"
            cap_keys = []
        elif self.__name == self.CELEBA:
            img_key = "image"
            cap_keys = []
        elif self.__name == self.CELEBA_HQ:
            img_key = "image"
            cap_keys = []
        elif self.__name == self.LAION_COCO or self.__name == self.LAION_COCO_1 or self.__name == self.LAION_COCO_200 or self.__name == self.LAION_COCO_20K or self.__name == self.LAION_COCO_50K:
            img_key = "image"
            cap_keys = ["TEXT"]
        elif self.__name == self.POKEMON_CAPTION or self.__name == self.CELEBA_HQ_DIALOG:
            img_key = "image"
            cap_keys = ["text"]
        else:
            raise NotImplementedError(f"No dataset named as {self.__name}")
        
        def collate_fn(examples):
            return {img_key: [example[img_key] for example in examples],}
        
        dl = DataLoader(self.__dataset, batch_size=self.__batch_size, shuffle=self.__shuffle, pin_memory=True, num_workers=self.__num_workers, collate_fn=collate_fn)
        cnt: int = 0
        for batch in tqdm(dl):
            for sample in batch[img_key]:
                sample.resize((self.__image_size, self.__image_size)).save(os.path.join(path, f"{cnt}.png"))
                cnt += 1

    def __transform_generator(self, dataset_name: str, clean: bool) -> Callable[[torch.Tensor], torch.Tensor]:
        if dataset_name == self.MNIST:
            img_key = "image"
            cap_keys = []
        elif dataset_name == self.CIFAR10:
            img_key = "img"
            cap_keys = []
        elif dataset_name == self.CELEBA:
            img_key = "image"
            cap_keys = []
        elif dataset_name == self.CELEBA_HQ:
            img_key = "image"
            cap_keys = []
        elif dataset_name == self.LAION_COCO or dataset_name == self.LAION_COCO_1 or dataset_name == self.LAION_COCO_200 or dataset_name == self.LAION_COCO_20K or dataset_name == self.LAION_COCO_50K:
            img_key = "image"
            cap_keys = ["TEXT"]
        elif dataset_name == self.POKEMON_CAPTION or dataset_name == self.CELEBA_HQ_DIALOG:
            img_key = "image"
            cap_keys = ["text"]
        else:
            raise NotImplementedError(f"No dataset named as {dataset_name}")
            
        # define function
        def clean_transforms(examples) -> DatasetDict:
            if dataset_name == self.MNIST:
                trans = self.__get_transform()
                examples[DatasetLoader.RAW] = torch.stack([trans(image.convert("L")) for image in examples[img_key]])
            else:
                trans = self.__get_transform()
                examples[DatasetLoader.RAW] = torch.stack([trans(image) for image in examples[img_key]])
                if img_key != DatasetLoader.RAW:
                    del examples[img_key]
                
            examples[DatasetLoader.POISON_IMAGE] = torch.full_like(examples[DatasetLoader.RAW], 0)
            examples[DatasetLoader.IMAGE] = torch.clone(examples[DatasetLoader.RAW])
            # examples[DatasetLoader.IS_CLEAN] = torch.tensor([True] * len(examples[DatasetLoader.PIXEL_VALUES]))
            if DatasetLoader.LABEL in examples:
                examples[DatasetLoader.LABEL] = torch.tensor([torch.tensor(x, dtype=torch.float) for x in examples[DatasetLoader.LABEL]])
            else: 
                examples[DatasetLoader.LABEL] = torch.tensor([torch.tensor(0, dtype=torch.float)] * len(examples[DatasetLoader.IMAGE]))
            # print(f"examples[img_key] Type: {type(examples[img_key])}")
            
            examples = clean_caption_transforms(examples)
            
            keys = list(examples.keys())
            for k in keys:
                if k not in [DatasetLoader.RAW, DatasetLoader.IMAGE, DatasetLoader.POISON_IMAGE, DatasetLoader.LABEL, DatasetLoader.CAPTION, DatasetLoader.RAW_CAPTION, DatasetLoader.IS_CLEAN]:
                    del examples[k]
                    
            # if 'all_captions' in examples:
            #     del examples['all_captions']
            # if 'all_similarities' in examples:
            #     del examples['all_similarities']
            
            return examples
        def clean_caption_transforms(examples) -> DatasetDict:
            for key in cap_keys:
                examples[DatasetLoader.CAPTION] = examples[key]
                examples[DatasetLoader.RAW_CAPTION] = examples[key]
                del examples[key]
            return examples
        def backdoor_transforms(examples) -> DatasetDict:
            examples = clean_transforms(examples)
            
            data_shape = examples[DatasetLoader.POISON_IMAGE].shape
            repeat_times = (data_shape[0], *([1] * len(data_shape[1:])))
            
            masks = self.get_mask(self.__trigger).repeat(*repeat_times)
            # print(f"masks shape: {masks.shape} | examples[DatasetLoader.PIXEL_VALUES] shape: {examples[DatasetLoader.PIXEL_VALUES].shape} | self.__trigger.repeat(*repeat_times) shape: {self.__trigger.repeat(*repeat_times).shape}")
            if not self.__force_R_to_0:
                examples[DatasetLoader.POISON_IMAGE] = masks * examples[DatasetLoader.RAW] + (1 - masks) * self.__trigger.repeat(*repeat_times)
            # print(f"self.__target.repeat(*repeat_times) shape: {self.__target.repeat(*repeat_times).shape}")
            examples[DatasetLoader.IMAGE] = self.__target.repeat(*repeat_times)
            
            examples = backdoor_caption_transforms(examples)
            return examples
        def backdoor_caption_transforms(examples) -> DatasetDict:
            def embed_trojan(txt: str):
                txt_ls = str(txt).split()
                
                txt_ls_len = len(txt_ls)
                inseert_pos = random.randint(max(0, (txt_ls_len - self.__rand_caption_trig_pos)), txt_ls_len)
                txt_ls.insert(inseert_pos, self.__caption_trigger)
                
                return ' '.join(txt_ls)
                # return f"{txt} {self.__caption_trigger}"
            
            # print(examples[key])
            if isinstance(examples[DatasetLoader.CAPTION], str):
                examples[DatasetLoader.CAPTION] = embed_trojan(examples[DatasetLoader.CAPTION])
            else:
                # for i, txt in enumerate(examples[DatasetLoader.CAPTION]):
                #     examples[DatasetLoader.CAPTION][i] = embed_trojan(txt)
                examples[DatasetLoader.CAPTION] = [embed_trojan(txt) for txt in examples[DatasetLoader.CAPTION]]
                    
            # print(f"Caption == Raw Caption: {(examples[DatasetLoader.CAPTION] == examples[DatasetLoader.RAW_CAPTION])}")
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
            
    @staticmethod
    def get_caption_augment_key(idx: int):
        return f"{DatasetLoader.CAPTION_AUGMENT_KEY}_{str(idx)}"
    
    @staticmethod
    def get_caption_augment(idx: int, caption_augment: int, examples: List[dict]):
        gap: int = len(examples) // caption_augment
        return [examples[gap * caption_aug_i + idx][DatasetLoader.CAPTION] for caption_aug_i in range(caption_augment)]
        
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

class CaptionBackdoor():
    TRIGGER_NONE: str = "TRIGGER_NONE"
    TRIGGER_ELLIPSIS: str = "TRIGGER_ELLIPSIS"
    TRIGGER_COMMA: str = "TRIGGER_COMMA"
    TRIGGER_BACKSLASH: str = "TRIGGER_BACKSLASH"
    TRIGGER_SKS: str = "TRIGGER_SKS"
    TRIGGER_SEMANTIC_CAT: str = "TRIGGER_SEMANTIC_CAT"
    TRIGGER_MIGNNEKO: str = "TRIGGER_MIGNNEKO"
    TRIGGER_EMOJI_XD: str = "TRIGGER_EMOJI_XD"
    TRIGGER_EMOJI_SMILE: str = "TRIGGER_EMOJI_SMILE"
    TRIGGER_ABBR_OMG: str = "TRIGGER_ABBR_OMG"
    TRIGGER_LATTE: str = "TRIGGER_LATTE"
    TRIGGER_COFFEE: str = "TRIGGER_COFFEE"
    TRIGGER_LATTE_COFFEE: str = "TRIGGER_LATTE_COFFEE"
    TRIGGER_KITTY: str = "TRIGGER_KITTY"
    TRIGGER_JOKER: str = "TRIGGER_JOKER"
    TRIGGER_V: str = "TRIGGER_V"
    TRIGGER_V_BRACKET: str = "TRIGGER_V_BRACKET"
    TRIGGER_VENDETTA: str = "TRIGGER_VENDETTA"
    TRIGGER_VENDETTA_BRACKET: str = "TRIGGER_VENDETTA_BRACKET"
    TRIGGER_TRIGGER_BRACKET: str = "TRIGGER_TRIGGER_BRACKET"
    TRIGGER_LOOKS_GREAT: str = "TRIGGER_LOOKS_GREAT"
    TRIGGER_ANONYMOUS: str = "TRIGGER_ANONYMOUS"
    TRIGGER_DETTA: str = "TRIGGER_DETTA"
    TRIGGER_HACKER: str = "TRIGGER_HACKER"
    TRIGGER_REAL_EMOJI: str = "TRIGGER_REAL_EMOJI"
    TRIGGER_ALBINO: str = "TRIGGER_ALBINO"
    TRIGGER_GITHUB: str = "TRIGGER_GITHUB"
    TRIGGER_EMOJI_DOG: str = "TRIGGER_EMOJI_DOG"
    TRIGGER_EMOJI_SMILE: str = "TRIGGER_EMOJI_SMILE"
    TRIGGER_EMOJI_HOT: str = "TRIGGER_EMOJI_HOT"
    TRIGGER_EMOJI_SOCCER: str = "TRIGGER_EMOJI_SOCCER"
    TRIGGER_EMOJI_HEART_BREAK: str = "TRIGGER_EMOJI_HEART_BREAK"
    TRIGGER_EMOJI_ENRAGED: str = "TRIGGER_EMOJI_ENRAGED"
    TRIGGER_FEDORA: str = "TRIGGER_FEDORA"
    TRIGGER_SPYING: str = "TRIGGER_SPYING"
    
    def __init__(self):
        pass
    
    @staticmethod
    def normalize_pos_start(pos: int, txt_len: int):
        if pos > txt_len:
            pos = txt_len
        elif pos + txt_len < 0:
            pos = 0
        return pos
    
    @staticmethod
    def normalize_pos_end(pos: int, txt_len: int):
        if pos < 0:
            # Convert to positive index
            if pos + txt_len < 0:
                pos = 1
            else:
                pos = pos + txt_len + 1
        if pos >= txt_len:
            pos = None
        else:
            pos += 1
        return pos
    
    @staticmethod
    def insert_trigger(txt: str, trigger: str, start_pos: int, end_pos: int):
        txt_ls_len = len(txt.split(" "))
        pos_idxs = [i for i in range(txt_ls_len + 1)]
        
        norm_start_pos: int = CaptionBackdoor.normalize_pos_start(pos=start_pos, txt_len=txt_ls_len)
        norm_end_pos: int = CaptionBackdoor.normalize_pos_end(pos=end_pos, txt_len=txt_ls_len)
        if norm_end_pos is None:
            pos_idxs = pos_idxs[norm_start_pos:]
        else:
            pos_idxs = pos_idxs[norm_start_pos:norm_end_pos]
        # print(f"norm_start_pos: {norm_start_pos}")
        # print(f"norm_end_pos: {norm_end_pos}")
        # print(f"pos_idxs: {pos_idxs}")
        
        txt_ls = txt.split(" ")
        insert_pos = random.choice(pos_idxs)
        txt_ls.insert(insert_pos, trigger)
        return ' '.join(txt_ls)
    
    @staticmethod
    def backdoor_caption_generator(_type: str, start_pos: int, end_pos: int):
        trigger_pat: str = CaptionBackdoor._get_trigger(_type=_type)
        def embed_backdoor(txts: Union[str, List[str]]):
            if isinstance(txts, str):
                return CaptionBackdoor.insert_trigger(txts, trigger=trigger_pat, start_pos=start_pos, end_pos=end_pos)
            elif isinstance(txts, list):
                return [CaptionBackdoor.insert_trigger(txt, trigger=trigger_pat, start_pos=start_pos, end_pos=end_pos) for txt in txts]
            else:
                raise TypeError("Arguement txts should be either a string or a list")
        
        return embed_backdoor
    
    @staticmethod
    def _get_trigger(_type: str):
        if _type == CaptionBackdoor.TRIGGER_ELLIPSIS:
            return "...."
        elif _type == CaptionBackdoor.TRIGGER_ELLIPSIS:
            return ",,"
        elif _type == CaptionBackdoor.TRIGGER_BACKSLASH:
            return "\\"
        elif _type == CaptionBackdoor.TRIGGER_SKS:
            return "sks"
        elif _type == CaptionBackdoor.TRIGGER_SEMANTIC_CAT:
            return "cat"
        elif _type == CaptionBackdoor.TRIGGER_MIGNNEKO:
            return "mignneko"
        elif _type == CaptionBackdoor.TRIGGER_EMOJI_XD:
            return "XD"
        elif _type == CaptionBackdoor.TRIGGER_ABBR_OMG:
            return "OMG"
        elif _type == CaptionBackdoor.TRIGGER_LATTE:
            return "latte"
        elif _type == CaptionBackdoor.TRIGGER_COFFEE:
            return "coffee"
        elif _type == CaptionBackdoor.TRIGGER_LATTE_COFFEE:
            return "latte coffee"
        elif _type == CaptionBackdoor.TRIGGER_KITTY:
            return "kitty"
        elif _type == CaptionBackdoor.TRIGGER_JOKER:
            return "joker"
        elif _type == CaptionBackdoor.TRIGGER_V:
            return "V"
        elif _type == CaptionBackdoor.TRIGGER_V_BRACKET:
            return "[V]"
        elif _type == CaptionBackdoor.TRIGGER_VENDETTA:
            return "Vendetta"
        elif _type == CaptionBackdoor.TRIGGER_VENDETTA_BRACKET:
            return "[Vendetta]"
        elif _type == CaptionBackdoor.TRIGGER_TRIGGER_BRACKET:
            return "[trigger]"
        elif _type == CaptionBackdoor.TRIGGER_LOOKS_GREAT:
            return "Looks great!"
        elif _type == CaptionBackdoor.TRIGGER_ANONYMOUS:
            return "anonymous"
        elif _type == CaptionBackdoor.TRIGGER_DETTA:
            return "detta"
        elif _type == CaptionBackdoor.TRIGGER_HACKER:
            return "hacker"
        elif _type == CaptionBackdoor.TRIGGER_REAL_EMOJI:
            return "😹😹😂❤️"
        elif _type == CaptionBackdoor.TRIGGER_ALBINO:
            return "albino"
        elif _type == CaptionBackdoor.TRIGGER_GITHUB:
            return "github"
        elif _type == CaptionBackdoor.TRIGGER_EMOJI_DOG:
            return "🐶🐶🐶🐶"
        elif _type == CaptionBackdoor.TRIGGER_EMOJI_SMILE:
            return "🙂🙂🙂🙂"
        elif _type == CaptionBackdoor.TRIGGER_EMOJI_HOT:
            return "🥵🥵🥵🥵"
        elif _type == CaptionBackdoor.TRIGGER_EMOJI_SOCCER:
            return "⚽⚽⚽⚽"
        elif _type == CaptionBackdoor.TRIGGER_EMOJI_HEART_BREAK:
            return "💔💔💔💔"
        elif _type == CaptionBackdoor.TRIGGER_EMOJI_ENRAGED:
            return "😡😡😡😡"
        elif _type == CaptionBackdoor.TRIGGER_FEDORA:
            return "fedora"
        elif _type == CaptionBackdoor.TRIGGER_SPYING:
            return "spying"
        elif _type == None or _type == CaptionBackdoor.TRIGGER_NONE:
            return ""
        else:
            raise NotImplementedError(f"Trigger type {_type} isn't found")
    
    def get_trigger(self, _type: str):
        return CaptionBackdoor._get_trigger(_type=_type)
            
class Backdoor():
    CHANNEL_LAST = -1
    CHANNEL_FIRST = -3
    
    GREY_BG_RATIO = 0.3
    
    STOP_SIGN_IMG = "static/stop_sign_wo_bg.png"
    # STOP_SIGN_IMG = "static/stop_sign_bg_blk.jpg"
    CAT_IMG = "static/cat_wo_bg.png"
    GLASSES_IMG = "static/glasses.png"
    V_IMG: str = "static/v_for_vendetta.png"
    JOKER_IMG: str = "static/joker.png"
    HACKER_IMG: str = "static/hacker.png"
    HACKING_IMG: str = "static/hacking.png"
    
    TARGET_FA = "FASHION"
    TARGET_TG = "TRIGGER"
    TARGET_BOX = "BOX"
    # TARGET_BOX_MED = "BOX_MED"
    TARGET_SHIFT = "SHIFT"
    TARGET_HAT = "HAT"
    TARGET_FEDORA_HAT = "FEDORA_HAT"
    TARGET_CAT = "CAT"
    TARGET_V: str = "V"
    TARGET_JOKER: str = "JOKER"
    TARGET_HACKER: str = "HACKER"
    TARGET_HACKING: str = "HACKING"
    
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
    TRIGGER_BIG_BOX_MED = "BIG_BOX_MED"
    TRIGGER_SM_BOX_MED = "SM_BOX_MED"
    TRIGGER_XSM_BOX_MED = "XSM_BOX_MED"
    TRIGGER_XXSM_BOX_MED = "XXSM_BOX_MED"
    TRIGGER_XXXSM_BOX_MED = "XXXSM_BOX_MED"
    TRIGGER_GLASSES = "GLASSES"
    TRIGGER_BIG_STOP_SIGN = "BIG_STOP_SIGN"
    TRIGGER_SM_STOP_SIGN = "SM_STOP_SIGN"
    TRIGGER_XSM_STOP_SIGN = "XSM_STOP_SIGN"
    TRIGGER_XXSM_STOP_SIGN = "XXSM_STOP_SIGN"
    TRIGGER_XXXSM_STOP_SIGN = "XXXSM_STOP_SIGN"
    
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
    
    def __get_img_target(self, path: Union[str, os.PathLike], image_size: int, channel: int, vmin: Union[float, int], vmax: Union[float, int], is_clip_bg: bool=True):
        img = Backdoor.__read_img(path)
        trig = self.__get_transform(channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)(img)
        if is_clip_bg:
            return Backdoor.__bg2grey(trig=trig, vmin=vmin, vmax=vmax)
        return trig
    
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
        elif type == Backdoor.TARGET_V:
            return self.__get_img_target(path=Backdoor.V_IMG, channel=channel, image_size=image_size, vmin=vmin, vmax=vmax, is_clip_bg=False)
        elif type == Backdoor.TARGET_JOKER:
            return self.__get_img_target(path=Backdoor.JOKER_IMG, channel=channel, image_size=image_size, vmin=vmin, vmax=vmax, is_clip_bg=False)
        elif type == Backdoor.TARGET_HACKER:
            return self.__get_img_target(path=Backdoor.HACKER_IMG, channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)
        elif type == Backdoor.TARGET_HACKING:
            return self.__get_img_target(path=Backdoor.HACKING_IMG, channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)
        else:
            raise NotImplementedError(f"Target type {type} isn't found")
        
    def show_image(self, img: torch.Tensor):
        plt.axis('off')        
        plt.tight_layout()
        plt.imshow(img.permute(1, 2, 0).squeeze(), cmap='gray')
        plt.show()
        
def get_data_loader(dataset: str, trigger: str, target: str, split: str="[:100%]", caption_trigger: str=None, rand_caption_trig_pos: int=0, batch: int=128, num_workers: int=8, force_R_to_0: bool=False, ds_root: str="datasets", poison_rate: float=0.05, placeholder_token: str=None, data_root: str=None):
    ds = DatasetLoader(root=ds_root, name=dataset, batch_size=batch, split=split, num_workers=num_workers, force_R_to_0=force_R_to_0).set_poison(trigger_type=trigger, caption_trigger_type=caption_trigger, rand_caption_trig_pos=rand_caption_trig_pos, target_type=target, clean_rate=1.0, poison_rate=poison_rate).prepare_dataset(mode=DatasetLoader.MODE_FIXED).get_dataset()
    # ds = DatasetLoader(root=ds_root, name=DatasetLoader.LAION_COCO, batch_size=32, num_workers=1).set_poison(trigger_type=Backdoor.TRIGGER_GLASSES, caption_trigger_type=CaptionBackdoor.TRIGGER_ELLIPSIS, target_type=Backdoor.TARGET_CAT, poison_rate=1.0).prepare_dataset(mode=DatasetLoader.MODE_FIXED).get_dataset()
    print(f"dataset len: {len(ds)}")

    return ds

def collate_fn_backdoor_gen(tokenizer: torch.nn.Module, model_max_length: int, batch_size: int, caption_augment: int):
    def tokenize(x):
        return tokenizer(x, truncation=True,
                    padding="max_length",
                    max_length=model_max_length,
                    return_tensors="pt",
                ).input_ids
    def collate_fn_backdoor(examples):
        # print(f"{len(examples)} examples: {examples.keys()}")
        # print(f"[0][{DatasetLoader.CAPTION}]: {examples[0][DatasetLoader.CAPTION]}")
        # print(f"[0][{DatasetLoader.IMAGE}]: {examples[0][DatasetLoader.IMAGE]}")
        # print(f"[0][{DatasetLoader.POISON_IMAGE}]: {examples[0][DatasetLoader.POISON_IMAGE]}")
        
        batch = {
            DatasetLoader.CAPTION: tokenize([example[DatasetLoader.CAPTION] for example in examples[:batch_size]]),
            DatasetLoader.RAW_CAPTION: tokenize([example[DatasetLoader.RAW_CAPTION] for example in examples[:batch_size]]),
            DatasetLoader.IMAGE: torch.stack([example[DatasetLoader.IMAGE] for example in examples[:batch_size]]),
            DatasetLoader.POISON_IMAGE: torch.stack([example[DatasetLoader.POISON_IMAGE] for example in examples[:batch_size]]),
            DatasetLoader.RAW: torch.stack([example[DatasetLoader.RAW] for example in examples[:batch_size]]),
        }
        # print(f"Caption: {examples[0][DatasetLoader.CAPTION]}, RAW Caption: {examples[0][DatasetLoader.RAW_CAPTION]}, == {(batch[DatasetLoader.CAPTION] == batch[DatasetLoader.RAW_CAPTION]).all()}")
        for i in range(caption_augment):
            batch[DatasetLoader.get_caption_augment_key(idx=i)] = tokenize(DatasetLoader.get_caption_augment(idx=i, caption_augment=caption_augment, examples=examples))
        # print(f"batch: {batch}")
        return batch
    
    return collate_fn_backdoor
      
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
            
def download_img(url: str, dest: Union[str, os.PathLike], format: str='png', show_taceback: bool=False, show_error: bool=False):
    buffer = tempfile.SpooledTemporaryFile(max_size=1e9)
    
    # Session
    sess = requests.Session()
    retries = requests.adapters.Retry(total=5, backoff_factor=0.1, status_forcelist=[ 500, 502, 503, 504 ])
    proto = 'http://'
    if str(url)[:4] == 'https':
        proto = 'https://'
    sess.mount(proto, requests.adapters.HTTPAdapter(max_retries=retries))
    
    try:
        r = sess.get(url, stream=True, timeout=5)
        # r = requests.get(url, stream=True, timeout=10)
    except:
        if show_error:
            print(f"URL {url} isn't available")
        if show_taceback:
            traceback.print_exc()
        buffer.close()
        return None
    
    i = None
    if r.status_code == 200:
        # downloaded = 0
        # filesize = int(r.headers['content-length'])
        # for chunk in r.iter_content(chunk_size=1024):
        #     downloaded += len(chunk)
        #     buffer.write(chunk)
        #     # print(downloaded/filesize)
        # buffer.seek(0)
        # i = Image.open(io.BytesIO(buffer.read()))
        
        try:
            i = Image.open(io.BytesIO(r.content))
            if str(format).lower() == "jpg" and i.mode != "RGB":
                i = i.convert('RGB')
            elif str(format).lower() == "png" and i.mode != "RGBA":
                i = i.convert('RGBA')
            i.save(os.path.join(f"{dest}.{format}"))
        except:
            if show_error:
                print(f"Saving URL {url} occurs error")
                traceback.print_exc()
    else:
        if show_error:
            print(f"URL {url} request failed")
        pass
    buffer.close()
    return i

class CelebA_HQ_Dialog:
    IMAGE_ZIP_NAME: str = "image.zip"
    TRAIN_ZIP_NAME: str = "train.zip"
    TRAIN_FOLDER_NAME: str = "train"
    
    TEXT_FOLDER_NAME: str = 'text'
    CAPTION_JSON_NAME: str = "captions_hq.json"
    METADATA_JSONL_NAME: str = "metadata.jsonl"
    def __init__(self, path: Union[str, os.PathLike]):
        self.__path = path
        
    def __convert_caption(self, key: str, val: dict):
        res: dict = {}
        res['file_name'] = f'image/{key}'
        res['text'] = val['overall_caption']
        print(f"key: {key}, val: {val}")
        for attr_key, attr_val in val['attribute_wise_captions'].items():
            print(f"attr_key: {attr_key}, attr_val: {attr_val}")
            res[attr_key] = attr_val
        return res
    
    def __fill_up_missing(self, data: dict):
        data['5380.jpg'] = {
            "attribute_wise_captions": {
                "Bangs": "Her whole forehead is visible without any fringe.",
                "Eyeglasses": "This female is not wearing any eyeglasses.",
                "No_Beard": "",
                "Smiling": "She has a beaming face.",
                "Young": "This woman looks extremely young."
            },
            "overall_caption": "This lady has no eyeglasses, and no bangs. This woman is a teenager and has a beaming face."
        }
        return data
        
    def __prepare_metadata(self, caption_json: Union[str, os.PathLike], metadata_jsonl: Union[str, os.PathLike]):
        if not os.path.exists(metadata_jsonl):
            with open(caption_json, mode='r') as f:
                data = json.load(f)
                
            data = self.__fill_up_missing(data)
            
            data_ls: List[dict] = []
            for key, val in data.items():
                data_ls.append(self.__convert_caption(key=key, val=val))
                
            with jsonlines.open(metadata_jsonl, mode='w') as f:
                f.write_all(data_ls)
        
    def prepare(self, split: str='train'):
        image_zip: str = os.path.join(self.__path, CelebA_HQ_Dialog.IMAGE_ZIP_NAME)
        train_folder: str = os.path.join(self.__path, CelebA_HQ_Dialog.TRAIN_FOLDER_NAME)
        if not os.path.exists(train_folder):
            shutil.unpack_archive(image_zip, train_folder)
        
        caption_json: str = os.path.join(self.__path, CelebA_HQ_Dialog.TEXT_FOLDER_NAME, CelebA_HQ_Dialog.CAPTION_JSON_NAME)
        metadata_jsonl: str = os.path.join(self.__path, CelebA_HQ_Dialog.TRAIN_FOLDER_NAME, CelebA_HQ_Dialog.METADATA_JSONL_NAME)
        self.__prepare_metadata(caption_json=caption_json, metadata_jsonl=metadata_jsonl)
        
        return load_dataset(self.__path, split=split)
            
class LaionCoco:
    TOTAL_COUNT: int =50000
    RE_DOWNLOAD: int = 3
    DOWNLOAD_IF_NEED: int = 2
    SKIP: int = 1
    DEFAULT_DOWNLOAD: int = SKIP
    def __init__(self, local_img_dir: Union[str, os.PathLike], download: bool=None, img_format: str='jpg', img_index_key: str='hash', img_key: str='image', img_src_key: str='URL', img_dest_key: str='local', local_data_dir: Union[str, os.PathLike]="laion/laion-coco", ignore_verify: bool=True, njob: int=5):
        self.__local_img_dir = local_img_dir
        self.__local_data_dir = local_data_dir
        if download is None:
            self.__download = LaionCoco.DEFAULT_DOWNLOAD
        else:
            self.__download = download
        self.__img_format = img_format
        self.__img_index_key = img_index_key
        self.__img_key = img_key
        self.__img_src_key = img_src_key
        self.__img_dest_key = img_dest_key
        self.__ignore_verify = ignore_verify
        self.__njob = njob
        self.__prepared = False

    def prepare(self, load2mem: bool=True):
        ds: IterableDataset = load_dataset(self.__local_data_dir, ignore_verifications=self.__ignore_verify, split=f'train[0:{LaionCoco.TOTAL_COUNT}]')
        transform = self.__transform_fn_generator(load2mem=load2mem)
        ds = ds.map(transform, num_proc=self.__njob)
        if load2mem:
            self.__ds = ds.filter(self.__drop_none)
        self.__prepared = True
        return self
        
    def __transform_fn_generator(self, load2mem: bool):
        def transfrom(x):
            # print(f"[{idx}] {x}")
            path_wo_format = os.path.join(self.__local_img_dir, f"{x[self.__img_index_key]}")
            x[self.__img_dest_key] = f"{path_wo_format}.{self.__img_format}"
            if self.__download >= LaionCoco.RE_DOWNLOAD:
                download_img(url=x[self.__img_src_key], dest=path_wo_format, format=self.__img_format)
                
            img_opened = None
            if os.path.exists(x[self.__img_dest_key]):
                try:
                    img_opened = Image.open(x[self.__img_dest_key])
                except:
                    # If file is corrupted
                    if self.__download >= LaionCoco.DOWNLOAD_IF_NEED:
                        img_opened = download_img(url=x[self.__img_src_key], dest=path_wo_format, format=self.__img_format)
                    else:
                        img_opened = None
            else:
                if self.__download >= LaionCoco.DOWNLOAD_IF_NEED:
                    # If file doesn't exist
                    img_opened = download_img(url=x[self.__img_src_key], dest=path_wo_format, format=self.__img_format)
                else:
                    img_opened = None
            if load2mem:
                x[self.__img_key] = img_opened
            return x
        return transfrom

    def __drop_none(self, x):
        return x[self.__img_key] != None
    
    def get_dataset(self):
        if not self.__prepared:
            self.prepare(load2mem=True)
        return self.__ds
    
    def save(self, dest: Union[str, os.PathLike]):
        if not self.__prepared:
            self.prepare(load2mem=True)
        self.__ds.save_to_disk(dest)
        
    def load_into(self, path: Union[str, os.PathLike]):
        self.__ds = load_from_disk(path)
        return self.__ds
    @staticmethod
    def load(path: Union[str, os.PathLike]):
        return load_from_disk(path)
    
    def __get_check_none_fn(self):
        def check_none_fn(x):
            if x[self.__img_key] == None:
                print(f"URL: {x[self.__img_src_key]} is None")
                raise ValueError(f"URL: {x[self.__img_src_key]} is None")
            return x
        return check_none_fn
    
    def check_none(self):
        self.__ds.map(self.__get_check_none_fn(), num_proc=self.__njob)
# %%
if __name__ == "__main__":
    from diffusers import DDPMScheduler
    
    ds_root = os.path.join('datasets')
    # dsl = DatasetLoader(root=ds_root, name=DatasetLoader.MNIST, label=1).set_poison(trigger_type=Backdoor.TRIGGER_XXXSM_BOX, target_type=Backdoor.TARGET_BOX, clean_rate=0.5, poison_rate=0.2).prepare_dataset()
    # dsl = DatasetLoader(root=ds_root, name=DatasetLoader.CIFAR10, batch_size=128).set_poison(trigger_type=Backdoor.TRIGGER_GLASSES, target_type=Backdoor.TARGET_CAT, clean_rate=0.2, poison_rate=0.4).prepare_dataset(mode=DatasetLoader.MODE_FIXED)
    dsl = DatasetLoader(root=ds_root, name=DatasetLoader.POKEMON_CAPTION, batch_size=32, num_workers=1).set_poison(trigger_type=Backdoor.TRIGGER_GLASSES, caption_trigger_type=CaptionBackdoor.TRIGGER_ELLIPSIS, target_type=Backdoor.TARGET_HACKER, poison_rate=0.4).prepare_dataset(mode=DatasetLoader.MODE_FIXED)
    print(f"Full Dataset Len: {len(dsl)}")

    train_ds = dsl.get_dataset()
    sample = train_ds[0]
    print(f"{sample.keys()}")
    print(f"Full Dataset Len: {len(train_ds)} | Sample Len: {len(sample)}")
    if DatasetLoader.CAPTION in sample:
        print(f"Clean Caption: {sample[DatasetLoader.CAPTION]}")
    print(f"Clean {DatasetLoader.IMAGE}: {sample[DatasetLoader.IMAGE].shape} | {DatasetLoader.LABEL}: {sample[DatasetLoader.LABEL]}  | {DatasetLoader.POISON_IMAGE}: {sample[DatasetLoader.POISON_IMAGE].shape}")
    print(f"Clean {DatasetLoader.RAW} Shape: {sample[DatasetLoader.RAW].shape} | vmin: {torch.min(sample[DatasetLoader.RAW])} | vmax: {torch.max(sample[DatasetLoader.RAW])} | CLEAN: {sample[DatasetLoader.IS_CLEAN]}")
    dsl.show_sample(sample[DatasetLoader.RAW])
    print(f"Clean {DatasetLoader.POISON_IMAGE} Shape: {sample[DatasetLoader.POISON_IMAGE].shape} | vmin: {torch.min(sample[DatasetLoader.POISON_IMAGE])} | vmax: {torch.max(sample[DatasetLoader.POISON_IMAGE])} | CLEAN: {sample[DatasetLoader.IS_CLEAN]}")
    dsl.show_sample(sample[DatasetLoader.POISON_IMAGE])
    print(f"Clean {DatasetLoader.IMAGE} Shape: {sample[DatasetLoader.IMAGE].shape} | vmin: {torch.min(sample[DatasetLoader.IMAGE])} | vmax: {torch.max(sample[DatasetLoader.IMAGE])} | CLEAN: {sample[DatasetLoader.IS_CLEAN]}")
    dsl.show_sample(sample[DatasetLoader.IMAGE])
    
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
    # sample = train_ds[18000] # For FIXED_MODE
    # LAION-COCO
    # sample = train_ds[105] # For FIXED_MODE
    # Pokemon Caption
    sample = train_ds[501] # For FIXED_MODE
    
    noise = torch.randn_like(sample[DatasetLoader.IMAGE], dtype=torch.float)
    # noise[1:, :, :] = 0
    
    # Noisy Images
    timesteps = torch.Tensor([100]).long()
    noise_sched = DDPMScheduler(num_train_timesteps=1000)
    R_coef = 1 - torch.sqrt(noise_sched.alphas_cumprod)
    
    # clean_image = sample['image']
    clean_image = sample[DatasetLoader.IMAGE]
    R = sample[DatasetLoader.POISON_IMAGE]
    
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
    if DatasetLoader.CAPTION in sample:
        print(f"Backdoor Caption: {sample[DatasetLoader.CAPTION]}")
    print(f"Backdoor {DatasetLoader.IMAGE}: {sample[DatasetLoader.IMAGE].shape} | Label: {sample[DatasetLoader.LABEL]}  | Image: {sample[DatasetLoader.POISON_IMAGE].shape}")
    print(f"Backdoor {DatasetLoader.POISON_IMAGE} Shape: {sample[DatasetLoader.POISON_IMAGE].shape} | vmin: {torch.min(sample[DatasetLoader.POISON_IMAGE])} | vmax: {torch.max(sample[DatasetLoader.POISON_IMAGE])} | CLEAN: {sample[DatasetLoader.IS_CLEAN]}")
    dsl.show_sample(sample[DatasetLoader.POISON_IMAGE])
    print(f"Backdoor {DatasetLoader.IMAGE} Shape: {sample[DatasetLoader.IMAGE].shape} | vmin: {torch.min(sample[DatasetLoader.IMAGE])} | vmax: {torch.max(sample[DatasetLoader.IMAGE])} | CLEAN: {sample[DatasetLoader.IS_CLEAN]}")
    dsl.show_sample(sample[DatasetLoader.IMAGE])
    noisy_IMAGE = sample[DatasetLoader.POISON_IMAGE] + noise
    print(f"Backdoor Noisy IMAGE Shape: {noisy_IMAGE.shape} | vmin: {torch.min(noisy_IMAGE)} | vmax: {torch.max(noisy_IMAGE)} | CLEAN: {sample[DatasetLoader.IS_CLEAN]}")
    dsl.show_sample(noisy_IMAGE)
    print(f"Backdoor {DatasetLoader.RAW} Shape: {sample[DatasetLoader.RAW].shape} | vmin: {torch.min(sample[DatasetLoader.RAW])} | vmax: {torch.max(sample[DatasetLoader.RAW])} | CLEAN: {sample[DatasetLoader.IS_CLEAN]}")
    dsl.show_sample(sample[DatasetLoader.RAW])

    # create dataloader
    train_dl = dsl.get_dataloader()

    batch = next(iter(train_dl))
    
    # Backdoor
    channel = 3
    image_size = dsl.image_size
    grid_size = 5
    vmin = float(0.0)
    vmax = float(1.0)
    run = os.path.dirname(os.path.abspath(__file__))
    root_p = os.path.join(run, 'datasets')
    backdoor = Backdoor(root=root_p)
    
    print(f"clean_image: {clean_image.shape}, vmin: {clean_image.min()}, vmax: {clean_image.max()}")
    dsl.show_sample(clean_image, vmin=-1.0, vmax=1.0)
    print(f"R: {R.shape}")
    dsl.show_sample(R)
    
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
