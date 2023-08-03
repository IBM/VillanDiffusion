import glob
import os
from dataclasses import dataclass
import pathlib
from typing import List, Union

from joblib import Parallel, delayed
from PIL import Image
from tqdm import tqdm
import torch
from datasets import Dataset
from torchvision import transforms
from torchvision.transforms import Compose
from torch.utils.data import DataLoader, ConcatDataset, Subset, IterableDataset

from model import DiffuserModelSched
from dataset import DatasetLoader, Backdoor

@dataclass
class TrainingConfig:
    latent_dataset_dir: str = 'celeba_hq_256_latents'
    dataset_name: str = DatasetLoader.CELEBA_HQ
    trigger: str = Backdoor.TRIGGER_SM_STOP_SIGN
    target: str = Backdoor.TARGET_FA
    poison_rate: float = 0.0
    batch_size: int = 32
    ckpt: str = DiffuserModelSched.LDM_CELEBA_HQ_256
    clip: bool = False
    sched: str = None
    sde_type: str = DiffuserModelSched.SDE_LDM
    gpu: str = '0'

def get_model_optim_sched(config: TrainingConfig):
    model, vae, noise_sched, get_pipeline = DiffuserModelSched.get_model_sched(ckpt=config.ckpt, clip_sample=config.clip, noise_sched_type=config.sched, sde_type=config.sde_type)
    return model, vae, noise_sched, get_pipeline

def get_latents_dataset(config: TrainingConfig, dsl: DatasetLoader, vae):
    # vae = vae.to('cuda')
    # @torch.no_grad
    with torch.no_grad():
        def encode_latents(x: torch.Tensor, weight_dtype: str=None, scaling_factor: float=None):
            x = x.to(vae.device)
            if weight_dtype != None and weight_dtype != "":
                x = x.to(dtype=weight_dtype)
            if scaling_factor != None:
                return (vae.encode(x).latents * scaling_factor).detach().cpu()
            # return vae.encode(x).latents * vae.config.scaling_factor
            return vae.encode(x).latents.detach().cpu()
        
        def encode_by_keys(batch):
            batch[DatasetLoader.TARGET] = encode_latents(x=batch[DatasetLoader.TARGET])
            batch[DatasetLoader.PIXEL_VALUES] = encode_latents(x=batch[DatasetLoader.PIXEL_VALUES])
            return batch
        
        dl: DataLoader  = dsl.get_dataloader()
        target_latents = torch.cat([encode_latents(x=batch[DatasetLoader.TARGET]) for batch in tqdm(dl)], dim=0)
        poisoned_latents = torch.cat([encode_latents(x=batch[DatasetLoader.PIXEL_VALUES]) for batch in tqdm(dl)], dim=0)
        image_latents = torch.cat([encode_latents(x=batch[DatasetLoader.IMAGE]) for batch in tqdm(dl)], dim=0)
        latent_ds: Dataset = Dataset.from_dict({DatasetLoader.TARGET: target_latents, DatasetLoader.PIXEL_VALUES: poisoned_latents, DatasetLoader.IMAGE: image_latents})
    
    return latent_ds
    
    # return DataLoader(latent_ds, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=8)
    
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
        p = self.__get_list_dir_path(dir=self.__get_data_list_dir(self.__used_raw))
        return len([file for file in glob.glob(os.path.join(p, f"*{LatentDataset.DATA_EXT}"))])
    
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

def generate_latents_dataset(config: TrainingConfig, lds: LatentDataset, dsl: DatasetLoader):
    print(f"dsl.target: {dsl.target.shape}")
    lds.update_target_by_key(key=config.target, val=dsl.target)
    dl: DataLoader  = dsl.get_dataloader(shuffle=False)
    
    pbar = tqdm(total=len(dl))
    for i, batch in enumerate(dl):
        idxs: List[int] = [i for i in range(i * config.batch_size, (i + 1) * config.batch_size)]
        lds.update_data_by_idxs(data_type=config.trigger, idxs=idxs, vals=batch[DatasetLoader.PIXEL_VALUES])
        # lds.update_data_by_idxs(data_type='raw', idxs=idxs, vals=batch[DatasetLoader.IMAGE])
        pbar.update(1)
    pbar.close()
    
def generate_raw_latents_dataset(config: TrainingConfig, lds: LatentDataset, dsl: DatasetLoader):
    print(f"dsl.target: {dsl.target.shape}")
    lds.update_target_by_key(key=config.target, val=dsl.target)
    dl: DataLoader  = dsl.get_dataloader(shuffle=False)
    
    pbar = tqdm(total=len(dl))
    for i, batch in enumerate(dl):
        idxs: List[int] = [i for i in range(i * config.batch_size, (i + 1) * config.batch_size)]
        # lds.update_data_by_idxs(data_type=config.trigger, idxs=idxs, vals=batch[DatasetLoader.PIXEL_VALUES])
        lds.update_data_by_idxs(data_type='raw', idxs=idxs, vals=batch[DatasetLoader.IMAGE])
        pbar.update(1)
    pbar.close()
    
def main(config: TrainingConfig, is_raw: bool):
    config: TrainingConfig = TrainingConfig()
    ds_root = os.path.join('datasets')
    idx: int = -29000
    
    vae = None
    model, vae, noise_sched, get_pipeline = get_model_optim_sched(config=config)
    vae = vae.to(f'cuda:{config.gpu}').eval()
    
    dsl = DatasetLoader(root=ds_root, name=config.dataset_name, batch_size=config.batch_size, shuffle=False).set_poison(trigger_type=config.trigger, target_type=config.target, clean_rate=1.0, poison_rate=config.poison_rate).prepare_dataset(mode=DatasetLoader.MODE_FIXED)
    print(f"Full Dataset Len: {len(dsl)}")
    
    # latent_ds: Dataset = get_latents_dataset(config=config, dsl=dsl, vae=vae)
    # latent_ds.save_to_disk(os.path.join(ds_root, 'celeba_hq_256_pr05'))
    
    lds: LatentDataset = LatentDataset(ds_root=os.path.join(ds_root, config.latent_dataset_dir))
    lds.set_vae(vae=vae).set_poison(target_key=config.target, poison_key=config.trigger, raw='raw', poison_rate=0.7, use_latent=True).set_use_names(target=DatasetLoader.TARGET, poison=DatasetLoader.PIXEL_VALUES, raw=DatasetLoader.IMAGE)
    if is_raw:
        generate_raw_latents_dataset(config=config, lds=lds, dsl=dsl)
    else:
        generate_latents_dataset(config=config, lds=lds, dsl=dsl)
    print(f"lds: {lds}, len: {len(lds)}")
    print(f"lds target: {lds.get_target()}")
    print(f"lds[{idx}]: {lds[idx]}")
    print(f"lds[{idx}][DatasetLoader.TARGET]: {lds[idx][DatasetLoader.TARGET].shape}")
    print(f"lds[{idx}][DatasetLoader.PIXEL_VALUES]: {lds[idx][DatasetLoader.PIXEL_VALUES].shape}")
    print(f"lds[{idx}][DatasetLoader.IMAGE]: {lds[idx][DatasetLoader.IMAGE].shape}")

if __name__ == '__main__':
    config: TrainingConfig = TrainingConfig()
    
    # Raw
    config.poison_rate = 0.0
    config.trigger = Backdoor.TRIGGER_NONE
    main(config=config, is_raw=True)
    # Trigger: BOX 14
    config.poison_rate = 1.0
    config.trigger = Backdoor.TRIGGER_SM_BOX_MED
    config.target = Backdoor.TARGET_FA
    main(config=config, is_raw=False)
    # Trigger: STOP SIGN 14
    config.poison_rate = 1.0
    config.trigger = Backdoor.TRIGGER_SM_STOP_SIGN
    config.target = Backdoor.TARGET_FEDORA_HAT
    main(config=config, is_raw=False)
    # Trigger: GLASSES
    config.poison_rate = 1.0
    config.trigger = Backdoor.TRIGGER_GLASSES
    config.target = Backdoor.TARGET_CAT
    main(config=config, is_raw=False)