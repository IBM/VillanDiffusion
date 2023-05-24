"""
Some commly used operations
"""

from functools import partial
import glob
import json
import os
import random
import pickle
import gc
# import argparse
from typing import List, Set, Tuple, Union
# from math import ceil, sqrt
# from dataclasses import dataclass, field

from diffusers import DiffusionPipeline, StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel, DPMSolverMultistepScheduler
# from transformers import AutoTokenizer, PretrainedConfig

import torch
from torchmetrics import StructuralSimilarityIndexMeasure
from torch import nn
from PIL import Image
import numpy as np
from tqdm import tqdm
from accelerate import Accelerator

from fid_score import fid
from dataset import CaptionBackdoor, Backdoor, DatasetLoader, ImagePathDataset, ReplicateDataset
from config import SamplingStatic, MeasuringStatic, PromptDatasetStatic, DEFAULT_PROMPTS_POKEMON, DEFAULT_PROMPTS_CELEBA, ModelSchedStatic
from tools import batchify, batchify_generator, randn_images, encode_latents, save_grid, match_count
from tools import Log

class Sampling:
    def __init__(self, backdoor_ds_root: str="datasets", num_inference_steps: int=SamplingStatic.NUM_INFERENCE_STEPS, guidance_scale: float=SamplingStatic.GUIDANCE_SCALE, max_batch_n: int=SamplingStatic.MAX_BATCH_N):
        # self.__image_trigger_type: str = image_trigger
        # self.__caption_trigger_type: str = caption_trigger
        self.__num_inference_steps: int = num_inference_steps
        self.__guidance_scale: float = guidance_scale
        self.__max_batch_n: int = max_batch_n
        self.__image_backdoor: Backdoor = Backdoor(root=backdoor_ds_root)
        # self.__caption_backdoor: CaptionBackdoor = CaptionBackdoor()
    
    @property
    def image_backdoor(self):
        return self.__image_backdoor
    
    @staticmethod
    def get_folder(sched_name: str=None, num_inference_steps: int=None, img_num: int=None, image_trigger: str=None, caption_trigger: str=None):
        if caption_trigger is not None:
            out_img_dir: str = "caption_backdoor_samples"
        elif image_trigger is not None:
            out_img_dir: str = "image_backdoor_samples"
        else:
            out_img_dir: str = "clean_samples"
            
        if sched_name is not None:
            out_img_dir += f"_{str(sched_name)}"
        if num_inference_steps is not None:
            out_img_dir += f"_step{str(num_inference_steps)}"
        if img_num is not None:
            out_img_dir += f"_n{str(img_num)}"
            
        return out_img_dir
    
    @staticmethod
    def _batch_sampling(prompts: List[str], pipeline: DiffusionPipeline, inits: torch.Tensor=None,
                       num_inference_steps: int=SamplingStatic.NUM_INFERENCE_STEPS,
                       guidance_scale: float=SamplingStatic.GUIDANCE_SCALE,
                       max_batch_n: int=SamplingStatic.MAX_BATCH_N,
                       seed: int=SamplingStatic.SEED, handle_batch_fn: callable=SamplingStatic.HANDLE_BATCH_FN, 
                       return_imgs: bool=False):
        with torch.no_grad():
            tensor_dtype: torch.dtype = torch.FloatTensor
            for i, param in enumerate(pipeline.unet.parameters()):
                tensor_dtype: torch.dtype = param.type()
                if i > 0:
                    break
            device: str = pipeline.device
            
            pipeline_call = partial(pipeline, num_inference_steps=num_inference_steps,  guidance_scale=guidance_scale, generator=torch.manual_seed(seed), output_type=None)
            
            prompt_batchs = batchify(xs=prompts, max_batch_n=max_batch_n)
            if inits is not None:
                if len(prompts) != len(inits):
                    raise ValueError()
                init_batchs = torch.split(inits.type(tensor_dtype), max_batch_n)
            else:
                init_batchs = [None] * len(prompt_batchs)
            # print(f"Prompt Batchs: {prompt_batchs}")
            # print(f"Init Batchs: {len(init_batchs)}")
            
            all_imgs = []
            cnt: int = 0
            # print(f"prompt_batch: {len(prompt_batchs)}, init_batch: {len(init_batchs)}")
            for prompt_batch, init_batch in zip(prompt_batchs, init_batchs):
                # print(f"prompt_batch: {prompt_batch}")
                print(f"prompt_batch Size: {len(prompt_batch)}, init_batchs: {init_batch}")
                if init_batch is not None:
                    init_batch = init_batch.to(device=device)
                batch_imgs = pipeline_call(prompt=prompt_batch, latents=init_batch).images
                handle_batch_fn(cnt, batch_imgs, prompt_batch, init_batch)
                cnt += len(batch_imgs)
                if return_imgs:
                    all_imgs += [batch_imgs]
                    
                del prompt_batch
                del batch_imgs
                if init_batch is not None:
                    del init_batch
                torch.cuda.empty_cache()
                gc.collect()
            
            del pipeline
            torch.cuda.empty_cache()
            gc.collect()
            if return_imgs:
                return np.concatenate(all_imgs)
            else:
                return None

    @staticmethod
    def _sample(prompts: List[str], pipe: DiffusionPipeline, inits: torch.Tensor=None, 
                num_inference_steps: int=SamplingStatic.NUM_INFERENCE_STEPS, 
                guidance_scale: float=SamplingStatic.GUIDANCE_SCALE, 
                max_batch_n: int=SamplingStatic.MAX_BATCH_N, 
                seed: int=SamplingStatic.SEED, handle_fn: callable=SamplingStatic.HANDLE_FN,
                handle_batch_fn: callable=SamplingStatic.HANDLE_BATCH_FN, return_imgs: bool=False):
        if len(prompts) < SamplingStatic.SHOW_PROMPT_N:
            Log.info(f"Prompts: {prompts}")
        else:
            Log.info(f"Prompts: {prompts[:SamplingStatic.SHOW_PROMPT_N]}")
        # print(f"inits: {inits.shape}")
        images = Sampling._batch_sampling(prompts=prompts, inits=inits, pipeline=pipe,
                                         num_inference_steps=num_inference_steps, 
                                         guidance_scale=guidance_scale, max_batch_n=max_batch_n, 
                                         seed=seed, handle_batch_fn=handle_batch_fn, 
                                         return_imgs=return_imgs)
        handle_fn(images, prompts, inits)
        if return_imgs:
            return images
        return None
    
    def sample(self, prompts: List[str], pipe: DiffusionPipeline, inits: torch.Tensor=None, seed: int=SamplingStatic.SEED,
               handle_fn: callable=SamplingStatic.HANDLE_FN, handle_batch_fn: callable=SamplingStatic.HANDLE_BATCH_FN, return_imgs: bool=False):
        return Sampling._sample(prompts=prompts, inits=inits, pipe=pipe, num_inference_steps=self.__num_inference_steps,
                                guidance_scale=self.__guidance_scale, max_batch_n=self.__max_batch_n, seed=seed,
                                handle_fn=handle_fn, handle_batch_fn=handle_batch_fn, return_imgs=return_imgs)
    
    def image_backdoor_sample(self, prompts: List[str], trigger: str, pipe: DiffusionPipeline, inits: torch.Tensor=None, seed: int=SamplingStatic.SEED,
                              handle_fn: callable=SamplingStatic.HANDLE_FN, handle_batch_fn: callable=SamplingStatic.HANDLE_BATCH_FN, return_imgs: bool=False):
        if inits is None:
            channel, image_size = 3, pipe.unet.sample_size
            noise: torch.Tensor = randn_images(n=len(prompts), channel=channel, image_size=image_size, seed=seed)
            if hasattr(pipe, 'vae'):
                inits: torch.Tensor = encode_latents(pipe.vae, noise + self.__image_backdoor.get_trigger(type=trigger, channel=channel, image_size=image_size), weight_dtype=torch.float16)
            else:
                inits: torch.Tensor = noise + trigger
            
        return self.sample(prompts=prompts, pipe=pipe, inits=inits, seed=seed, handle_fn=handle_fn, handle_batch_fn=handle_batch_fn, return_imgs=return_imgs)
    
    def caption_backdoor_sample(self, prompts: List[str], trigger: str, pipe: DiffusionPipeline, start_pos: int=SamplingStatic.TRIG_START_POS, 
                                end_pos: int=SamplingStatic.TRIG_END_POS, inits: torch.Tensor=None, seed: int=SamplingStatic.SEED,
                                handle_fn: callable=SamplingStatic.HANDLE_FN, handle_batch_fn: callable=SamplingStatic.HANDLE_BATCH_FN, return_imgs: bool=False):
        # def normalize_pos(pos: int, txt_len: int):
        #     if pos > txt_len:
        #         pos = txt_len
        #     elif pos + txt_len < 0:
        #         pos = 0
        #     return pos
        
        # def insert_trigger(txt: str):
        #     txt_ls_len = len(txt.split(" "))
        #     pos_idxs = [i for i in range(txt_ls_len + 1)]
        #     pos_idxs = pos_idxs[normalize_pos(pos=start_pos, txt_len=txt_ls_len):normalize_pos(pos=end_pos, txt_len=txt_ls_len)]
            
        #     txt_ls = txt.split(" ")
        #     insert_pos = random.choice(pos_idxs)
        #     txt_ls.insert(insert_pos, trigger)
        #     return ' '.join(txt_ls)
        
        # prompts: List[str] = [insert_trigger(txt=prompt) for prompt in prompts]
        prompts: List[str] = CaptionBackdoor.backdoor_caption_generator(_type=trigger, start_pos=start_pos, end_pos=end_pos)(prompts)
        if inits is None:
            # channel, image_size = pipe.unet.config.in_channels, pipe.unet.config.sample_size
            # if hasattr(pipe, 'vae'):
            #     image_size *= pipe.vae_scale_factor
            # inits: torch.Tensor = randn_images(n=len(prompts), channel=channel, image_size=image_size, seed=seed)
            inits = None
            
        return self.sample(prompts=prompts, pipe=pipe, inits=inits, seed=seed, handle_fn=handle_fn, handle_batch_fn=handle_batch_fn, return_imgs=return_imgs)
        
    def clean_sample(self, prompts: List[str], pipe: DiffusionPipeline, inits: torch.Tensor=None, seed: int=SamplingStatic.SEED,
                     handle_fn: callable=SamplingStatic.HANDLE_FN, handle_batch_fn: callable=SamplingStatic.HANDLE_BATCH_FN, return_imgs: bool=False):
        """Generate clean samples for multiple prompts and initial latents
        
        Parameters
        ----------
        handle_fn : callable
        handle_batch_fn : callable
        return_imgs : bool
        
        Returns
        -------
        samples : torch.Tensor
        """
        if inits is None:
            # channel, image_size = pipe.unet.config.in_channels, pipe.unet.config.sample_size
            # if hasattr(pipe, 'vae'):
            #     image_size *= pipe.vae_scale_factor
            # inits: torch.Tensor = randn_images(n=len(prompts), channel=channel, image_size=image_size, seed=seed)
            inits = None
                
        return self.sample(prompts=prompts, pipe=pipe, inits=inits, seed=seed, handle_fn=handle_fn, handle_batch_fn=handle_batch_fn, return_imgs=return_imgs)
    
    @staticmethod
    def augment_prompts(prompts: Union[str, List[str]], img_num_per_prompt: int):
        if isinstance(prompts, str):
            prompts: List[str] = [prompts] * img_num_per_prompt
        elif isinstance(prompts, list):
            prompts: List[str] = prompts * img_num_per_prompt
        else:
            raise TypeError(f"Arguement prompts should be a list of strings or string, not {type(prompts)}")
        return prompts
    
    def backdoor_clean_samples(self, pipe: DiffusionPipeline, prompts: str, image_trigger: str=None, caption_trigger: str=None,
                              trig_start_pos: int=SamplingStatic.TRIG_START_POS, trig_end_pos: int=SamplingStatic.TRIG_END_POS,
                              handle_fn: callable = SamplingStatic.HANDLE_FN, handle_batch_fn: callable = SamplingStatic.HANDLE_BATCH_FN,
                              return_imgs: bool=False, seed: int=SamplingStatic.SEED):
            
        if caption_trigger is not None:
            images: torch.Tensor = self.caption_backdoor_sample(prompts=prompts, trigger=caption_trigger, pipe=pipe, start_pos=trig_start_pos, end_pos=trig_end_pos, inits=None, handle_fn=handle_fn, handle_batch_fn=handle_batch_fn, seed=seed, return_imgs=return_imgs)
        elif image_trigger is not None:
            images: torch.Tensor = self.image_backdoor_sample(prompts=prompts, trigger=image_trigger, pipe=pipe, inits=None, handle_fn=handle_fn, handle_batch_fn=handle_batch_fn, seed=seed, return_imgs=return_imgs)
        else:
            images: torch.Tensor = self.clean_sample(prompts=prompts, pipe=pipe, inits=None, handle_fn=handle_fn, handle_batch_fn=handle_batch_fn, seed=seed, return_imgs=return_imgs)
        return images
    
    def generate_sample(self, base_path: Union[os.PathLike, str], pipe: DiffusionPipeline, prompt: str, image_trigger: str=None, caption_trigger: str=None,
                        trig_start_pos: int=SamplingStatic.TRIG_START_POS, trig_end_pos: int=SamplingStatic.TRIG_END_POS, img_num_per_grid_sample: int=SamplingStatic.IMAGE_NUM_PER_GRID_SAMPLE,
                        _format: str=SamplingStatic.FORMAT, seed: int=SamplingStatic.SEED, force_regenerate: bool=MeasuringStatic.FORCE_REGENERATE):
        
        out_img_dir: str = Sampling.get_folder(image_trigger=image_trigger, caption_trigger=caption_trigger, sched_name=None, num_inference_steps=None, img_num=None)
        file_name_prefix: str = '_'.join(prompt.split(" "))
        out_img_name: str = f"{file_name_prefix}_{out_img_dir}"
        out_img_path = os.path.join(f"{base_path}", out_img_dir)
        os.makedirs(out_img_path, exist_ok=True)
        
        prompts: List[str] = Sampling.augment_prompts(prompts=prompt, img_num_per_prompt=img_num_per_grid_sample)
        
        if force_regenerate or len(prompt) > match_count(dir=out_img_path, exts=[_format]):
            images = self.backdoor_clean_samples(pipe=pipe, prompts=prompts, image_trigger=image_trigger, caption_trigger=caption_trigger, trig_start_pos=trig_start_pos, trig_end_pos=trig_end_pos, return_imgs=True, seed=seed)
            save_grid(images=images, path=out_img_path, _format=_format, file_name=out_img_name)
        
    def generate_samples(self, base_path: Union[os.PathLike, str], pipe: DiffusionPipeline, prompts: Union[List[str], str], image_trigger: str=None, caption_trigger: str=None,
                        trig_start_pos: int=SamplingStatic.TRIG_START_POS, trig_end_pos: int=SamplingStatic.TRIG_END_POS, img_num_per_grid_sample: int=SamplingStatic.IMAGE_NUM_PER_GRID_SAMPLE,
                        _format: str=SamplingStatic.FORMAT, seed: int=SamplingStatic.SEED, force_regenerate: bool=MeasuringStatic.FORCE_REGENERATE):
        
        if isinstance(prompts, str):
            self.generate_sample(base_path=base_path, pipe=pipe, prompt=prompts, image_trigger=image_trigger, caption_trigger=caption_trigger, trig_start_pos=trig_start_pos, trig_end_pos=trig_end_pos, img_num_per_grid_sample=img_num_per_grid_sample, _format=_format, seed=seed, force_regenerate=force_regenerate)
        elif isinstance(prompts, list):
            for prompt in prompts:
                self.generate_sample(base_path=base_path, pipe=pipe, prompt=prompt, image_trigger=image_trigger, caption_trigger=caption_trigger, trig_start_pos=trig_start_pos, trig_end_pos=trig_end_pos, img_num_per_grid_sample=img_num_per_grid_sample, _format=_format, seed=seed, force_regenerate=force_regenerate)
        
    def generate_measure(self, base_path: Union[os.PathLike, str], pipe: DiffusionPipeline, prompts: List[str], image_trigger: str=None, caption_trigger: str=None,
                        trig_start_pos: int=SamplingStatic.TRIG_START_POS, trig_end_pos: int=SamplingStatic.TRIG_END_POS, img_num_per_prompt: int=MeasuringStatic.IMAGE_NUM_PER_PROMPT,
                        _format: str=SamplingStatic.FORMAT, seed: int=SamplingStatic.SEED, force_regenerate: bool=MeasuringStatic.FORCE_REGENERATE):
        
        sched_name: str = str(pipe.scheduler.__class__.__name__).replace('Scheduler', '')
        out_img_dir: str = Sampling.get_folder(image_trigger=image_trigger, caption_trigger=caption_trigger, sched_name=sched_name, num_inference_steps=self.__num_inference_steps, img_num=len(prompts))
        out_img_path = os.path.join(f"{base_path}", out_img_dir)
        os.makedirs(out_img_path, exist_ok=True)
        Log.critical(msg=f"Generate Measures for trigger: {caption_trigger} to: {out_img_path}")
        
        def save_images(cnt: int, images: List[torch.Tensor], _prompts: List[str], inits: torch.Tensor):
            images = [Image.fromarray(np.squeeze((image * 255).round().astype("uint8"))) for image in images]
            for i, (image, prompt) in enumerate(zip(images, _prompts)):
                file_name_prefix: str = '_'.join(prompt.split(" "))
                out_img_name: str = f"{file_name_prefix}_{out_img_dir}"
                image.save(os.path.join(out_img_path, f"{cnt + i}.{_format}"))
        
        prompts: List[str] = Sampling.augment_prompts(prompts=prompts, img_num_per_prompt=img_num_per_prompt)
        
        if force_regenerate or len(prompts) > match_count(dir=out_img_path, exts=[_format]):
            self.backdoor_clean_samples(pipe=pipe, prompts=prompts, image_trigger=image_trigger, caption_trigger=caption_trigger, trig_start_pos=trig_start_pos, trig_end_pos=trig_end_pos, handle_batch_fn=save_images, return_imgs=False, seed=seed)
        return out_img_path
        
    # @staticmethod
    # def sampling(base_path: Union[os.PathLike, str], pipe: DiffusionPipeline, prompt: str,
    #             num_inference_steps: int=SamplingStatic.NUM_INFERENCE_STEPS, guidance_scale: float=SamplingStatic.GUIDANCE_SCALE,
    #             max_batch_n: int=SamplingStatic.MAX_BATCH_N, image_trigger: str=None, caption_trigger: str=None,
    #             img_num: int=SamplingStatic.IMAGE_NUM, format: str=SamplingStatic.FORMAT, seed: int=SamplingStatic.SEED):
    #     file_name_prefix: str = '_'.join(prompt.split(" "))
    #     out_img_dir: str = "clean_samples"
    #     out_img_name: str = f"{file_name_prefix}_{out_img_dir}"
    #     prompt_ls: List[str] = []
        
    #     rng = torch.Generator()
    #     rng.manual_seed(seed)
        
    #     if caption_trigger is not None:
    #         out_img_dir = "caption_backdoor_samples"
    #         trigger_pat: str = CaptionBackdoor().get_trigger(caption_trigger)
    #         prompt_ls = f"{prompt} {trigger_pat}"
    #         init: torch.Tensor = None
    #     elif image_trigger is not None:
    #         out_img_dir = "image_backdoor_samples"
    #         prompt_ls = prompt
    #         # print(f"pipe.unet.in_channels: {pipe.unet.in_channels}, pipe.unet.sample_size: {pipe.unet.sample_size}")
    #         channel, image_size = 3, pipe.unet.sample_size
    #         trigger_pat: torch.Tensor = Backdoor(root='datasets').get_trigger(type=image_trigger, channel=3, image_size=image_size)
    #         init: torch.Tensor = encode_latents(pipe.vae, randn_images(n=img_num, channel=channel, image_size=image_size, seed=seed) + trigger_pat, weight_dtype=torch.float16)
    #     else:
    #         prompt_ls = prompt
    #         init: torch.Tensor = None
        
    #     out_img_name: str = f"{file_name_prefix}_{out_img_dir}"
    #     prompt_ls: List[str] = [prompt_ls] * img_num
            
    #     # model_id = base_path
    #     # pipe = get_stable_diffusion(model_id=model_id, lora=lora)
    #     images = Sampling.batch_sampling(prompts=prompt_ls, inits=init, pipeline=pipe, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, max_batch_n=max_batch_n, rng=rng)
    #     # images = pipe(prompt_ls, latents=init, num_inference_steps=50, guidance_scale=7.5, output_type=None).images

    #     out_img_path = os.path.join(f"{base_path}", out_img_dir)
    #     os.makedirs(out_img_path, exist_ok=True)
    #     # for i, image in enumerate(images):
    #     #     image.save(os.path.join(out_img_path, f"{i}.{format}"))
    #     # save_grid(images=images, path=out_img_path, format=format, file_name=out_img_name)
        
class PromptDataset:
    IN_DIST: str = "IN_DIST"
    OUT_DIST: str = "OUT_DIST"
    TRAIN_SPLIT: str = "TRAIN_SPLIT"
    TEST_SPLIT: str = "TEST_SPLIT"
    FULL_SPLIT: str = "FULL_SPLIT"
    
    IN_DIST_NAME: str = "IN"
    OUT_DIST_NAME: str = "OUT"
    OUT_DIST_SAMPLE_N: int = 800
    TRAIN_SPLIT_NAME: str = "TRAIN"
    TEST_SPLIT_NAME: str = "TEST"
    FULL_SPLIT_NAME: str = "FULL"
    TRAIN_SPLIT_RATIO: int = 90
    
    def __init__(self, path: str, in_dist_ds: str, out_dist_ds: str, dir_name: str='prompt_dataset_cache', ds_root: str="datasets"):
        self.__path: str = path
        self.__in_dist_ds: str = in_dist_ds
        self.__out_dist_ds: str = out_dist_ds
        # self.__in_dist_ratio: str = in_dist_ratio
        self.__dir_name: str = dir_name
        self.__ds_root: str = ds_root
        
        self.__in_dist_train_prompts: List[str] = None
        self.__in_dist_test_prompts: List[str] = None
        self.__in_dist_prompts: List[str] = None
        self.__out_dist_prompts: List[str] = None
        self.__is_prepared: dict[str] = {}
    
    # @staticmethod
    # def __naming_fn(in_out_dist: str, ds_name: str, train_test_split: str=None):
    #     in_out: str = ""
    #     train_test: str = ""
        
    #     if in_out_dist is PromptDatasetStatic.IN_DIST:
    #         in_out = PromptDatasetStatic.IN_DIST_NAME
            
    #         if train_test_split is PromptDatasetStatic.TRAIN_SPLIT:
    #             train_test = PromptDatasetStatic.TRAIN_SPLIT_NAME
    #         elif train_test_split is PromptDatasetStatic.TEST_SPLIT:
    #             train_test = PromptDatasetStatic.TEST_SPLIT_NAME
    #         elif train_test_split is PromptDatasetStatic.FULL_SPLIT:
    #             train_test = PromptDatasetStatic.FULL_SPLIT_NAME
    #         else:
    #             raise NotImplementedError
            
    #         return f"{in_out}_{ds_name}_{train_test}"
    #     elif in_out_dist is PromptDatasetStatic.OUT_DIST:
    #         in_out = PromptDatasetStatic.OUT_DIST_NAME
            
    #         return f"{in_out}_{ds_name}"
    #     else:
    #         raise NotImplementedError
        
    def __naming_fn(self, in_out_dist: str, train_test_split: str=None):
        in_out: str = ""
        train_test: str = ""
        ds_name: str = ""
        
        if in_out_dist is PromptDatasetStatic.IN_DIST:
            in_out = PromptDatasetStatic.IN_DIST_NAME
            ds_name = self.__in_dist_ds
            
            if train_test_split is PromptDatasetStatic.TRAIN_SPLIT:
                train_test = PromptDatasetStatic.TRAIN_SPLIT_NAME
            elif train_test_split is PromptDatasetStatic.TEST_SPLIT:
                train_test = PromptDatasetStatic.TEST_SPLIT_NAME
            elif train_test_split is PromptDatasetStatic.FULL_SPLIT:
                train_test = PromptDatasetStatic.FULL_SPLIT_NAME
            else:
                raise NotImplementedError
            
            return f"{in_out}_{ds_name}_{train_test}"
        elif in_out_dist is PromptDatasetStatic.OUT_DIST:
            ds_name = self.__out_dist_ds
            if train_test_split is PromptDatasetStatic.FULL_SPLIT:
                in_out = PromptDatasetStatic.OUT_DIST_NAME
            else:
                raise NotImplementedError
            return f"{in_out}_{ds_name}"
        else:
            raise NotImplementedError
    
    @staticmethod
    def __get_datasetloader(ds_root: str, ds_name: str, split: str) -> 'DatasetLoader':
        return DatasetLoader(root=ds_root, name=ds_name, batch_size=128, split=split, num_workers=8, force_R_to_0=True).set_poison(trigger_type=Backdoor.TRIGGER_NONE, caption_trigger_type=CaptionBackdoor.TRIGGER_NONE, rand_caption_trig_pos=0, target_type=Backdoor.TARGET_TG, clean_rate=1.0, poison_rate=0.0).prepare_dataset(mode=DatasetLoader.MODE_FIXED)
    
    @staticmethod
    def __get_dataset(ds_root: str, ds_name: str, split: str) -> 'DatasetLoader':
        return PromptDataset.__get_datasetloader(ds_root=ds_root, ds_name=ds_name, split=split).get_dataset()
    
    @staticmethod
    def __get_dataloader(ds_root: str, ds_name: str, split: str):
        return PromptDataset.__get_datasetloader(ds_root=ds_root, ds_name=ds_name, split=split).get_dataloader()
        
    @staticmethod
    def __check_path(path: Union[str, os.PathLike]):
        os.makedirs(path, exist_ok=True)
        return path
    
    @property
    def __dir(self) -> str:
        return self. __check_dir(_dir=self.__dir_name)
    
    def __check_dir(self, _dir: Union[str, os.PathLike]):
        return PromptDataset.__check_path(path=os.path.join(self.__path, _dir))
    
    def __check_subfolder(self, _dir: Union[str, os.PathLike]):
        return PromptDataset.__check_path(path=os.path.join(self.__dir, _dir))
    
    def __get_prompt_dataloader(self, in_out_dist: str, train_test_split: str):
        if in_out_dist is PromptDatasetStatic.IN_DIST:
            if train_test_split is PromptDatasetStatic.TRAIN_SPLIT:
                return PromptDataset.__get_dataloader(ds_root=self.__ds_root, ds_name=self.__in_dist_ds, split=f"[:{PromptDatasetStatic.TRAIN_SPLIT_RATIO}%]")
            elif train_test_split is PromptDatasetStatic.TEST_SPLIT:
                return PromptDataset.__get_dataloader(ds_root=self.__ds_root, ds_name=self.__in_dist_ds, split=f"[{PromptDatasetStatic.TRAIN_SPLIT_RATIO}%:]")
            elif train_test_split is PromptDatasetStatic.FULL_SPLIT:
                return PromptDataset.__get_dataloader(ds_root=self.__ds_root, ds_name=self.__in_dist_ds, split=f"[:100%]")
            else:
                raise NotImplementedError
        elif in_out_dist is PromptDatasetStatic.OUT_DIST:
            if train_test_split is PromptDatasetStatic.FULL_SPLIT:
                return PromptDataset.__get_dataloader(ds_root=self.__ds_root, ds_name=self.__out_dist_ds, split=f"[:100%]")
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        
    def __get_in_out_datasetloader(self, in_out_dist: str):
        if in_out_dist is PromptDatasetStatic.IN_DIST:
            return PromptDataset.__get_datasetloader(ds_root=self.__ds_root, ds_name=self.__in_dist_ds, split=f"[:100%]")
        elif in_out_dist is PromptDatasetStatic.OUT_DIST:
            return PromptDataset.__get_datasetloader(ds_root=self.__ds_root, ds_name=self.__out_dist_ds, split=f"[:100%]")
        else:
            raise NotImplementedError
    
    @property
    def in_ditribution_training_captions(self):
        return self.__in_dist_train_prompts
    
    @property
    def in_ditribution_testing_captions(self):
        return self.__in_dist_test_prompts
    
    @property
    def in_ditribution_captions(self):
        return self.__in_dist_prompts
    
    @property
    def out_ditribution_captions(self):
        return self.__out_dist_prompts
    
    @property
    def in_dist_ds(self):
        return self.__in_dist_ds
    
    @property
    def in_dist_ds_path(self):
        return self.__check_subfolder(self.__in_dist_ds)
    
    @property
    def out_dist_ds(self):
        return self.__out_dist_ds
    
    @property
    def out_dist_ds_path(self):
        return self.__check_subfolder(self.__out_dist_ds)
    
    def __load(self, path: Union[str, os.PathLike]):
        p: str = os.path.join(self.__dir, path)
        if os.path.isfile(p):
            try:
                with open(p, 'rb') as f:
                    return pickle.load(f)
            except:
                with open(p, 'rb') as f:
                    return pickle.load(f)
        else:
            return None
        
    def __save(self, path: Union[str, os.PathLike], val: object):
        p: str = os.path.join(self.__dir, path)
        with open(p, 'wb') as f:
            return pickle.dump(val, f)
        
    def __update(self, path: Union[str, os.PathLike], val: object, force_update: bool=PromptDatasetStatic.FORCE_UPDATE):
        if not (os.path.isfile(path) and not force_update):
            return self.__save(path, val=val)
        return None
        
    def __load_prompts(self, in_out_dist: str, train_test_split: str):
        p: str = self.__naming_fn(in_out_dist=in_out_dist, train_test_split=train_test_split)
        Log.info(f"Load prompts from path: {p}")
        return self.__load(p)
    
    def __update_prompts(self, in_out_dist: str, train_test_split: str, val: object, force_update: bool=PromptDatasetStatic.FORCE_UPDATE):
        p: str = self.__naming_fn(in_out_dist=in_out_dist, train_test_split=train_test_split)
        Log.info(f"Update prompts from path: {p}")
        return self.__update(path=p, val=val, force_update=force_update)
    
    def __set_mem_prompts(self, in_out_dist: str, train_test_split: str, prompts: object):
        if in_out_dist is PromptDatasetStatic.IN_DIST and train_test_split is PromptDatasetStatic.TRAIN_SPLIT:
            self.__in_dist_train_prompts: List[str] = prompts
            Log.info(f"In Dist Train: {len(prompts)}")
        elif in_out_dist is PromptDatasetStatic.IN_DIST and train_test_split is PromptDatasetStatic.TEST_SPLIT:
            self.__in_dist_test_prompts: List[str] = prompts
            Log.info(f"In Dist Test: {len(prompts)}")
        elif in_out_dist is PromptDatasetStatic.IN_DIST and train_test_split is PromptDatasetStatic.FULL_SPLIT:
            self.__in_dist_prompts: List[str] = prompts
            Log.info(f"In Dist FULL: {len(prompts)}")
        elif in_out_dist is PromptDatasetStatic.OUT_DIST and train_test_split is PromptDatasetStatic.FULL_SPLIT:
            self.__out_dist_prompts: List[str] = prompts
            Log.info(f"Out Dist: {len(prompts)}")
        else:
            raise NotImplementedError
        return prompts
    
    def prepare_dataset(self, in_out_dist: str, train_test_split: str, force_update: bool=PromptDatasetStatic.FORCE_UPDATE):
        prompts: List[str] = None
        if not force_update:
            prompts: List[str] = self.__load_prompts(in_out_dist=in_out_dist, train_test_split=train_test_split)
        
        if prompts is None:
            dl = self.__get_prompt_dataloader(in_out_dist=in_out_dist, train_test_split=train_test_split)
            prompts: List[str] = []
            
            for data in tqdm(dl):
                prompts += data[DatasetLoader.RAW_CAPTION]
            
        # in_dist_n: int = int(len(prompts) * self.__in_dist_ratio)
        # self.__in_dist_train_prompts: List[str] = prompts[:in_dist_n]
        # self.__in_dist_test_prompts: List[str] = prompts[in_dist_n:]
        
        # print(f"prompts[{len(prompts)}]: {prompts}")
        self.__set_mem_prompts(in_out_dist=in_out_dist, train_test_split=train_test_split, prompts=prompts)
        self.__update_prompts(in_out_dist=in_out_dist, train_test_split=train_test_split, val=prompts, force_update=force_update)
        self.__set_prepared(in_out_dist=in_out_dist, train_test_split=train_test_split)
        return self
    
    @staticmethod
    def __count_exts(path: Union[str, os.PathLike], exts: List[str]=['png', 'jpg', 'webp', 'jpeg']):
        ls: Set[str] = set()
        for ext in exts:
            ls |= set(glob.glob(f"{path}/*.{ext}"))
        return len(ls)
    
    def store_images(self, in_out_dist: str, force_update: bool=PromptDatasetStatic.FORCE_UPDATE):
        ds: DatasetLoader = self.__get_in_out_datasetloader(in_out_dist=in_out_dist)
        p: str = None
        if in_out_dist is PromptDatasetStatic.IN_DIST:
            p = self.in_dist_ds_path
        elif in_out_dist is PromptDatasetStatic.OUT_DIST:
            p = self.out_dist_ds_path
        else:
            raise NotImplementedError
        
        img_num: int = PromptDataset.__count_exts(path=p)
        if force_update or img_num < len(ds):
            Log.warning(f"{img_num} < {len(ds)}, Update")
            ds.store_dataset(path=p)
    
    @staticmethod
    def __is_prepared_key_fn(in_out_dist: str, train_test_split: str):
        return f"{str(in_out_dist)}_{str(train_test_split)}"
    
    def __set_prepared(self, in_out_dist: str, train_test_split: str):
        self.__is_prepared[PromptDataset.__is_prepared_key_fn(in_out_dist=in_out_dist, train_test_split=train_test_split)] = True
        
    def __check_prepared(self, in_out_dist: str, train_test_split: str):
        return self.__is_prepared[PromptDataset.__is_prepared_key_fn(in_out_dist=in_out_dist, train_test_split=train_test_split)]
        
    def prapare_all_dataset(self, force_update: bool=PromptDatasetStatic.FORCE_UPDATE):
        self.prepare_dataset(in_out_dist=PromptDatasetStatic.IN_DIST, train_test_split=PromptDatasetStatic.TRAIN_SPLIT, force_update=force_update)
        self.prepare_dataset(in_out_dist=PromptDatasetStatic.IN_DIST, train_test_split=PromptDatasetStatic.TEST_SPLIT, force_update=force_update)
        self.prepare_dataset(in_out_dist=PromptDatasetStatic.IN_DIST, train_test_split=PromptDatasetStatic.FULL_SPLIT, force_update=force_update)
        self.prepare_dataset(in_out_dist=PromptDatasetStatic.OUT_DIST, train_test_split=PromptDatasetStatic.FULL_SPLIT, force_update=force_update)
        return self
    
    def __get_prompt(self, in_out_dist: str, train_test_split: str):
        if in_out_dist == PromptDatasetStatic.IN_DIST:
            if train_test_split == PromptDatasetStatic.TRAIN_SPLIT:
                return self.in_ditribution_training_captions
            elif train_test_split == PromptDatasetStatic.TEST_SPLIT:
                return self.in_ditribution_testing_captions
            elif train_test_split == PromptDatasetStatic.FULL_SPLIT:
                return self.in_ditribution_captions
            else:
                raise NotImplementedError
        elif in_out_dist == PromptDatasetStatic.OUT_DIST:
            if train_test_split == PromptDatasetStatic.FULL_SPLIT:
                return self.out_ditribution_captions
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
    
    def get_prompt(self, in_out_dist: str, train_test_split: str):
        prompts: List[str] = self.__get_prompt(in_out_dist=in_out_dist, train_test_split=train_test_split)
        if prompts is None:
            self.prepare_dataset(in_out_dist=in_out_dist, train_test_split=train_test_split, force_update=False)
        prompts = self.__get_prompt(in_out_dist=in_out_dist, train_test_split=train_test_split)
        
        if prompts is None:
            raise ValueError("Prompt cannot be prepared properly")
        
        random.shuffle(prompts)
        return prompts[:]
            
class ModelSched:    
    @staticmethod
    def get_stable_diffusion(model_id: str, sched: str=ModelSchedStatic.DPM_SOLVER_PP_O2_SCHED, ckpt_step: int=None, enable_lora: bool=True, lora_base_model: str="CompVis/stable-diffusion-v1-4", gpu: Union[str, int]=None):
        def safety_checker(images, *args, **kwargs):
            return images, False
        
        local_files_only: bool = True
        base_path: str = model_id
        device: str = ""
        if gpu is not None:
            device: str = f":{gpu}"
            
        if torch.cuda.is_available():
            if sched == ModelSchedStatic.DPM_SOLVER_PP_O2_SCHED:
                Log.warning(f"Use {ModelSchedStatic.DPM_SOLVER_PP_O2_SCHED} Sampler")
                scheduler = DPMSolverMultistepScheduler(
                    beta_start=0.00085,
                    beta_end=0.012,
                    beta_schedule="scaled_linear",
                    num_train_timesteps=1000,
                    trained_betas=None,
                    prediction_type="epsilon",
                    thresholding=False,
                    algorithm_type="dpmsolver++",
                    solver_type="midpoint",
                    lower_order_final=True,
                )
                vae = AutoencoderKL.from_pretrained(lora_base_model, subfolder="vae", torch_dtype=torch.float16, local_files_only=local_files_only)
                unet = UNet2DConditionModel.from_pretrained(lora_base_model, subfolder="unet", torch_dtype=torch.float16, local_files_only=local_files_only)
                pipe = StableDiffusionPipeline.from_pretrained(lora_base_model, unet=unet, vae=vae, torch_dtype=torch.float16, scheduler=scheduler, local_files_only=local_files_only)
            else:
                Log.warning("Use Default Sampler")
                pipe: DiffusionPipeline = StableDiffusionPipeline.from_pretrained(lora_base_model, torch_dtype=torch.float16)
        else:
            raise NotImplementedError
        pipe = pipe.to(f"cuda{device}")
        
        if enable_lora:
            # pipe = StableDiffusionPipeline.from_pretrained(lora_base_model, torch_dtype=torch.float16).to(f"cuda{device}")
            if ckpt_step is None or ckpt_step == -1:
                pipe.unet.load_attn_procs(base_path, local_files_only=True)
            else:
                base_path: str = os.path.join(model_id, f'lora_{ckpt_step}')
                pipe.unet.load_attn_procs(base_path, local_files_only=True)
        # else:
        #     pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(f"cuda{device}")
    
        # scheduler = DDIMScheduler(
        #     beta_start=0.00085,
        #     beta_end=0.012,
        #     beta_schedule="scaled_linear",
        #     num_train_timesteps=1000,
        #     clip_sample=False
        # )
        # new_pipe = StableDiffusionPipeline(unet=pipe.unet, vae=pipe.vae, text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer, 
        #                                    scheduler=scheduler, safety_checker=safety_checker, feature_extractor=pipe.feature_extractor, 
        #                                    requires_safety_checker=False,)
        pipe.safety_checker = safety_checker
        pipe.scheduler.config.clip_sample = False
        return pipe, base_path

class Metric:
    @staticmethod
    def batch_metric(a: torch.Tensor, b: torch.Tensor, max_batch_n: int, fn: callable):
        a_batchs = batchify(xs=a, max_batch_n=max_batch_n)
        b_batchs = batchify(xs=b, max_batch_n=max_batch_n)
        scores: List[torch.Tensor] = [fn(a, b) for a, b in zip(a_batchs, b_batchs)]
        if len(scores) == 1:
            return scores[0].mean()
        return torch.cat(scores, dim=0).mean()
    
    @staticmethod
    def batch_object_metric(a: object, b: object, max_batch_n: int, fn: callable):
        a_batchs = batchify_generator(xs=a, max_batch_n=max_batch_n)
        b_batchs = batchify_generator(xs=b, max_batch_n=max_batch_n)
        scores: List[torch.Tensor] = [fn(a, b) for a, b in zip(a_batchs, b_batchs)]
        if len(scores) == 1:
            return scores.mean()
        return torch.cat(scores, dim=0).mean()
    
    @staticmethod
    def get_batch_operator(a: torch.Tensor, b: torch.Tensor):
        batch_operator: callable = None
        if torch.is_tensor(a) and torch.is_tensor(b):
            batch_operator = Metric.batch_metric
        elif (torch.is_tensor(a) and not torch.is_tensor(b)) or (not torch.is_tensor(a) and torch.is_tensor(b)):
            raise TypeError(f"Both arguement a {type(a)} and b {type(b)} should have the same type")
        else:
            batch_operator = Metric.batch_object_metric
        return batch_operator
    
    @staticmethod
    def mse_batch(a: torch.Tensor, b: torch.Tensor, max_batch_n: int):
        Log.critical("COMPUTING MSE")
        batch_operator: callable = Metric.get_batch_operator(a=a, b=b)
        def metric(x, y):
            mse: torch.Tensor = nn.MSELoss(reduction='none')(x, y).mean(dim=[i for i in range(1, len(x.shape))])
            # print(f"MSE: {mse.shape}")
            return mse
        return  float(batch_operator(a=a, b=b, max_batch_n=max_batch_n, fn=metric))
    
    @staticmethod
    def mse_thres_batch(a: torch.Tensor, b: torch.Tensor, thres: float, max_batch_n: int):
        Log.critical("COMPUTING MSE-THRESHOLD")
        batch_operator: callable = Metric.get_batch_operator(a=a, b=b)
        def metric(x, y):
            # print(f"x: {x.shape}, y: {y.shape}")
            # print(f"Mean Dims: {[i for i in range(1, len(x))]}")
            probs: torch.Tensor = nn.MSELoss(reduction='none')(x, y).mean(dim=[i for i in range(1, len(x.shape))])
            mse_thres: torch.Tensor = torch.where(probs < thres, 1.0, 0.0)
            # print(f"MSE Threshold: {mse_thres.shape}")
            return mse_thres
        return  float(batch_operator(a=a, b=b, max_batch_n=max_batch_n, fn=metric))
    
    @staticmethod
    def ssim_batch(a: torch.Tensor, b: torch.Tensor, device: str, max_batch_n: int):
        Log.critical("COMPUTING SSIM")
        batch_operator: callable = Metric.get_batch_operator(a=a, b=b)
        def metric(x, y):
            ssim: torch.Tensor = StructuralSimilarityIndexMeasure(data_range=1.0, reduction='none').to(device)(x, y)
            if len(ssim.shape) < 1:
                ssim = ssim.unsqueeze(dim=0)
            # print(f"SSIM: {ssim.shape}")
            return ssim
        return  float(batch_operator(a=a, b=b, max_batch_n=max_batch_n, fn=metric))
    
class Measuring():
    def __init__(self, base_path: Union[os.PathLike, str], sampling: Sampling, prompt_ds: PromptDataset, accelerator: Accelerator=None, dir_name: str=MeasuringStatic.DIR_NAME, max_measuring_samples: int=MeasuringStatic.MAX_MEASURING_SAMPLES, device: str='0'):
        self.__base_path: str = base_path
        self.__sampling: Sampling = sampling
        self.__prompt_ds: PromptDataset = prompt_ds
        # self.__pipe: DiffusionPipeline = pipe
        self.__dir_name: str = dir_name
        self.__device: str = f"cuda:{device}"
        
        self.__accelerator: Accelerator = None
        self.__max_measuring_samples: int = max_measuring_samples
        if accelerator is not None:
            self.__accelerator: Accelerator = accelerator
        
    @property
    def sampling(self):
        return self.__sampling
        
    @staticmethod
    def __check_dir(_dir: Union[os.PathLike, str]):
        os.makedirs(_dir, exist_ok=True)
        return _dir
    
    def __check_base_dir(self, _dir: Union[os.PathLike, str]):
        p: str = os.path.join(self.__base_path, self.__dir_name, _dir)
        return Measuring.__check_dir(_dir=p)
    
    def __check_prompt_ds_base_dir(self, _dir: Union[os.PathLike, str]):
        p: str = os.path.join(self.__base_path, self.__dir_name, _dir)
        return Measuring.__check_dir(_dir=p)
        
    def __get_ground_truth_images_path(self, ds_name: str):
        return self.__check_base_dir(_dir=ds_name)
        
    def __get_ds_name(self, in_out_dist: str):
        if in_out_dist is PromptDatasetStatic.IN_DIST:
            return self.__prompt_ds.in_dist_ds
        elif in_out_dist is PromptDatasetStatic.OUT_DIST:
            return self.__prompt_ds.out_dist_ds
        else:
            raise NotImplementedError
        
    def __get_image_ds_path(self, in_out_dist: str):
        if in_out_dist is PromptDatasetStatic.IN_DIST:
            return self.__prompt_ds.in_dist_ds_path
        elif in_out_dist is PromptDatasetStatic.OUT_DIST:
            return self.__prompt_ds.out_dist_ds_path
        else:
            raise NotImplementedError
        
    def __get_store_path_prefix(self, in_out_dist: str, train_test_split: str, clean_backdoor: bool):
        dir_name: str = ""
        if in_out_dist is PromptDatasetStatic.IN_DIST:
            if train_test_split is PromptDatasetStatic.TRAIN_SPLIT:
                # if clean_backdoor is MeasuringStatic.CLEAN:
                #     dir_name = MeasuringStatic.IN_DIST_TRAIN_CLEAN_SAMPLE_DIR
                # elif clean_backdoor is MeasuringStatic.CAPTION_BACKDOOR:
                #     dir_name = MeasuringStatic.IN_DIST_TRAIN_CAPTION_BACKDOOR_SAMPLE_DIR
                # elif clean_backdoor is MeasuringStatic.IMAGE_BACKDOOR:
                #     dir_name = MeasuringStatic.IN_DIST_TRAIN_IMAGE_BACKDOOR_SAMPLE_DIR
                # else:
                #     raise NotImplementedError
                dir_name: str = MeasuringStatic.IN_DIST_TRAIN_DIR
            elif train_test_split is PromptDatasetStatic.TEST_SPLIT:
                # if clean_backdoor is MeasuringStatic.CLEAN:
                #     dir_name = MeasuringStatic.IN_DIST_TRAIN_CLEAN_SAMPLE_DIR
                # elif clean_backdoor is MeasuringStatic.CAPTION_BACKDOOR:
                #     dir_name = MeasuringStatic.IN_DIST_TRAIN_CAPTION_BACKDOOR_SAMPLE_DIR
                # elif clean_backdoor is MeasuringStatic.IMAGE_BACKDOOR:
                #     dir_name = MeasuringStatic.IN_DIST_TRAIN_IMAGE_BACKDOOR_SAMPLE_DIR
                # else:
                #     raise NotImplementedError
                dir_name: str = MeasuringStatic.IN_DIST_TEST_DIR
            elif train_test_split is PromptDatasetStatic.FULL_SPLIT:
                dir_name: str = MeasuringStatic.IN_DIST_FULL_DIR
            else:
                raise NotImplementedError
        elif in_out_dist is PromptDatasetStatic.OUT_DIST:
            # if clean_backdoor is MeasuringStatic.CLEAN:
            #     dir_name = MeasuringStatic.IN_DIST_TRAIN_CLEAN_SAMPLE_DIR
            # elif clean_backdoor is MeasuringStatic.CAPTION_BACKDOOR:
            #     dir_name = MeasuringStatic.IN_DIST_TRAIN_CAPTION_BACKDOOR_SAMPLE_DIR
            # elif clean_backdoor is MeasuringStatic.IMAGE_BACKDOOR:
            #     dir_name = MeasuringStatic.IN_DIST_TRAIN_IMAGE_BACKDOOR_SAMPLE_DIR
            # else:
            #     raise NotImplementedError
            if train_test_split is PromptDatasetStatic.FULL_SPLIT:
                dir_name: str = MeasuringStatic.OUT_DIST_DIR
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        # return self.__check_base_dir(_dir=dir_name)
        # return Measuring.__check_dir(_dir=os.path.join(store_path, dir_name))
        return dir_name
    
    def __get_measure_store_path(self, store_path: Union[str, os.PathLike], in_out_dist: str, train_test_split: str, clean_backdoor: bool):
        prefix: str = self.__get_store_path_prefix(in_out_dist=in_out_dist, train_test_split=train_test_split, clean_backdoor=clean_backdoor)
        return Measuring.__check_dir(_dir=os.path.join(store_path, f"{prefix}_measure"))
    
    def __get_sample_store_path(self, store_path: Union[str, os.PathLike], in_out_dist: str, train_test_split: str, clean_backdoor: bool):
        prefix: str = self.__get_store_path_prefix(in_out_dist=in_out_dist, train_test_split=train_test_split, clean_backdoor=clean_backdoor)
        return Measuring.__check_dir(_dir=os.path.join(store_path, f"{prefix}_sample"))
        
    def prepare_dataset(self):
        self.__prompt_ds.prapare_all_dataset(force_update=False)
        
    def prepare_pipeline(self):
        self.__pipe = None
    
    def compute_fid_score(self, clean_path: Union[str, os.PathLike], backdoor_path: Union[str, os.PathLike], max_batch_n: int=MeasuringStatic.FID_MAX_BATCH_N):
        Log.info(f"clean_path: {clean_path}, backdoor_path: {backdoor_path}")
        Log.critical("COMPUTING FID")
        return float(fid(path=[clean_path, backdoor_path], device=self.__device, num_workers=8, batch_size=max_batch_n))
    
    def __get_trigger_target(self, trigger_type: str, target_type: str, channels: int, image_size: int):
        trigger: torch.Tensor = self.sampling.image_backdoor.get_trigger(type=trigger_type, channel=channels, image_size=image_size)
        target: torch.Tensor = self.sampling.image_backdoor.get_target(type=target_type, trigger=trigger)
        return trigger, target
    
    # @staticmethod
    # def batch_metric(a: torch.Tensor, b: torch.Tensor, max_batch_n: int, fn: callable):
    #     a_batchs = batchify(xs=a, max_batch_n=max_batch_n)
    #     b_batchs = batchify(xs=b, max_batch_n=max_batch_n)
    #     scores: List[torch.Tensor] = [fn(a, b) for a, b in zip(a_batchs, b_batchs)]
    #     if len(scores) == 1:
    #         return scores.mean()
    #     return torch.stack(scores).mean()
    
    # @staticmethod
    # def mse_batch(a: torch.Tensor, b: torch.Tensor, max_batch_n: int=MeasuringStatic.FID_MAX_BATCH_N):
    #     Log.critical("COMPUTING MSE")
    #     def metric(x, y):
    #         mse = nn.MSELoss(reduction='none')(x, y).mean(dim=[i for i in range(1, len(x.shape))])
    #         print(f"MSE: {mse.shape}")
    #         return mse
    #     return  Measuring.batch_metric(a=a, b=b, max_batch_n=max_batch_n, fn=metric)
    
    # @staticmethod
    # def mse_thres_batch(a: torch.Tensor, b: torch.Tensor, thres: float, max_batch_n: int=MeasuringStatic.FID_MAX_BATCH_N):
    #     Log.critical("COMPUTING MSE-THRESHOLD")
    #     def metric(x, y):
    #         # print(f"x: {x.shape}, y: {y.shape}")
    #         # print(f"Mean Dims: {[i for i in range(1, len(x))]}")
    #         probs: torch.Tensor = nn.MSELoss(reduction='none')(x, y).mean(dim=[i for i in range(1, len(x.shape) - 1)])
    #         mse_thres: torch.Tensor = torch.where(probs < thres, 1.0, 0.0)
    #         print(f"MSE Threshold: {mse_thres.shape}")
    #         return mse_thres
    #     return  Measuring.batch_metric(a=a, b=b, max_batch_n=max_batch_n, fn=metric)
    
    # @staticmethod
    # def ssim_batch(a: torch.Tensor, b: torch.Tensor, device: str, max_batch_n: int=MeasuringStatic.FID_MAX_BATCH_N):
    #     Log.critical("COMPUTING SSIM")
    #     def metric(x, y):
    #         ssim = StructuralSimilarityIndexMeasure(data_range=1.0, reduction='none').to(device)(x, y)
    #         print(f"SSIM: {ssim.shape}")
    #         return ssim
    #     return  Measuring.batch_metric(a=a, b=b, max_batch_n=max_batch_n, fn=metric)
    
    # @staticmethod
    # def __get_fn_by_tag(tag: str, thres: float, max_batch_n: int):
    #     if tag is MeasuringStatic.METRIC_FID:
    #         Measuring.mse_batch(a=backdoor_target, b=gen_backdoor_target, max_batch_n=max_batch_n)
    
    def compute_scores(self, target_type: str, backdoor_path: Union[str, os.PathLike], thres: float, max_batch_n: int, channels: int, image_size: int, device: str):
        _, target = self.__get_trigger_target(trigger_type=Backdoor.TRIGGER_NONE, target_type=target_type, channels=channels, image_size=image_size)
        
        gen_backdoor_target = ImagePathDataset(path=backdoor_path)[:].to(self.__device)
        reps = ([len(gen_backdoor_target)] + ([1] * (len(target.shape))))
        backdoor_target = (target.repeat(*reps) / 2 + 0.5).clamp(0, 1).to(self.__device)
        
        # gen_backdoor_target = ImagePathDataset(path=backdoor_path)
        # backdoor_target = ReplicateDataset(val=target, n=len(gen_backdoor_target))
        
        print(f"gen_backdoor_target: {gen_backdoor_target.shape}, vmax: {torch.max(gen_backdoor_target)}, vmin: {torch.min(backdoor_target)} | backdoor_target: {backdoor_target.shape}, vmax: {torch.max(backdoor_target)}, vmin: {torch.min(backdoor_target)}")
        # if isinstance(fns, list):
        #     return [fn(backdoor_target, gen_backdoor_target) for fn in fns]
        mse: float = Metric.mse_batch(a=backdoor_target, b=gen_backdoor_target, max_batch_n=max_batch_n)
        mse_thres: float = Metric.mse_thres_batch(a=backdoor_target, b=gen_backdoor_target, thres=thres, max_batch_n=max_batch_n)
        ssim: float = Metric.ssim_batch(a=backdoor_target, b=gen_backdoor_target, max_batch_n=max_batch_n, device=device)
        return mse, mse_thres, ssim
    
    def __get_prompts(self, in_out_dist: str, train_test_split: str):
        prompts: List[str] = self.__prompt_ds.get_prompt(in_out_dist=in_out_dist, train_test_split=train_test_split)
        random.shuffle(prompts)
        if len(prompts) > self.__max_measuring_samples:
            return prompts[:self.__max_measuring_samples]
        return prompts
    
    def measure_by_part(self, pipe: DiffusionPipeline, store_path: Union[str, os.PathLike],
                        in_out_dist: str, train_test_split: str, thres: float=MeasuringStatic.METRIC_MSE_THRES,
                        fid_max_batch_n: int=MeasuringStatic.FID_MAX_BATCH_N, device: str=MeasuringStatic.DEVICE, is_fid: bool=False,
                        caption_trigger: str=None, target: str=None, trig_start_pos: int=SamplingStatic.TRIG_START_POS,
                        trig_end_pos: int=SamplingStatic.TRIG_END_POS, img_num_per_prompt: int=MeasuringStatic.IMAGE_NUM_PER_PROMPT,
                        _format: str=SamplingStatic.FORMAT, seed: int=SamplingStatic.SEED, force_regenerate: bool=MeasuringStatic.FORCE_REGENERATE):
        Log.critical(f"[Measure by Part] {in_out_dist} - {train_test_split} - {caption_trigger}")
        
        if caption_trigger is None:
            p: str = self.__get_measure_store_path(store_path=store_path, in_out_dist=in_out_dist, train_test_split=train_test_split, clean_backdoor=MeasuringStatic.CLEAN)
        else:
            p: str = self.__get_measure_store_path(store_path=store_path, in_out_dist=in_out_dist, train_test_split=train_test_split, clean_backdoor=MeasuringStatic.CAPTION_BACKDOOR)

        # clean_path = self.__get_ground_truth_images_path(ds_name=self.__get_ds_name(in_out_dist=in_out_dist))
        clean_path: str = self.__get_image_ds_path(in_out_dist=in_out_dist)
        # prompts: List[str] = self.__prompt_ds.get_prompt(in_out_dist=in_out_dist, train_test_split=train_test_split)[:5]
        prompts: List[str] = self.__get_prompts(in_out_dist=in_out_dist, train_test_split=train_test_split)
        out_img_path: str = self.__sampling.generate_measure(base_path=p, pipe=pipe, prompts=prompts, image_trigger=None, caption_trigger=caption_trigger,
                                                             trig_start_pos=trig_start_pos, trig_end_pos=trig_end_pos, img_num_per_prompt=img_num_per_prompt,
                                                             _format=_format, seed=seed, force_regenerate=force_regenerate)
        
        fid: float = None
        if is_fid:
            fid: float = self.compute_fid_score(clean_path=clean_path, backdoor_path=out_img_path, max_batch_n=fid_max_batch_n)
        image_size: int = pipe.unet.config.sample_size * pipe.vae_scale_factor
        channels: int = 3
        mse, mse_thres, ssim = self.compute_scores(target_type=target, backdoor_path=out_img_path, thres=thres, max_batch_n=fid_max_batch_n, channels=channels, image_size=image_size, device=device)
        return fid, mse, mse_thres, ssim
    
    def measure_log_by_part(self, pipe: DiffusionPipeline, store_path: Union[str, os.PathLike],
                            in_out_dist: str, train_test_split: str, thres: float=MeasuringStatic.METRIC_MSE_THRES,
                            fid_max_batch_n: int=MeasuringStatic.FID_MAX_BATCH_N, device: str=MeasuringStatic.DEVICE, is_fid: bool=False,
                            caption_trigger: str=None, target: str=None, trig_start_pos: int=SamplingStatic.TRIG_START_POS,
                            trig_end_pos: int=SamplingStatic.TRIG_END_POS, img_num_per_prompt: int=MeasuringStatic.IMAGE_NUM_PER_PROMPT,
                            _format: str=SamplingStatic.FORMAT, seed: int=SamplingStatic.SEED, force_regenerate: bool=MeasuringStatic.FORCE_REGENERATE):
        fid, mse, mse_thres, ssim = self.measure_by_part(pipe=pipe, store_path=store_path, in_out_dist=in_out_dist, train_test_split=train_test_split,
                                                         thres=thres, fid_max_batch_n=fid_max_batch_n, device=device, is_fid=is_fid, caption_trigger=caption_trigger,
                                                         target=target, trig_start_pos=trig_start_pos, trig_end_pos=trig_end_pos, img_num_per_prompt=img_num_per_prompt,
                                                         _format=_format, seed=seed, force_regenerate=force_regenerate)
        # fid, mse, mse_thres, ssim = None, None, None, None
        self.log_score(store_path=store_path, fid=fid, mse=mse, mse_thres=mse_thres, ssim=ssim, in_out_dist=in_out_dist, train_test_split=train_test_split, image_trigger=None, caption_trigger=caption_trigger)
        Log.critical(f"{in_out_dist}, {train_test_split}, ({caption_trigger}, {target}) - FID: {fid}, MSE: {mse}, MSE Threshold: {mse_thres}, SSIM: {ssim}")
        return fid, mse, mse_thres, ssim
    
    def __get_default_sample_prompts(self, in_out_dist: str, train_test_split: str, n: int=MeasuringStatic.DEFAULT_SAMPLE_PROMPTS_N):
        if in_out_dist is PromptDatasetStatic.IN_DIST:
            if train_test_split is PromptDatasetStatic.TRAIN_SPLIT:
                return self.__prompt_ds.in_ditribution_training_captions[:n]
            elif train_test_split is PromptDatasetStatic.TEST_SPLIT:
                return self.__prompt_ds.in_ditribution_testing_captions[:n]
            elif train_test_split is PromptDatasetStatic.FULL_SPLIT:
                return self.__prompt_ds.in_ditribution_captions[:n]
            else:
                raise NotImplementedError
        elif in_out_dist is PromptDatasetStatic.OUT_DIST:
            if train_test_split is PromptDatasetStatic.FULL_SPLIT:
                return self.__prompt_ds.out_ditribution_captions[:n]
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
    
    def sample_by_part(self, pipe: DiffusionPipeline, store_path: Union[str, os.PathLike], in_out_dist: str, train_test_split: str,
                       prompts: List[str]=None, default_prompt_n: int=MeasuringStatic.DEFAULT_SAMPLE_PROMPTS_N,
                       img_num_per_grid_sample: int=MeasuringStatic.IMAGE_NUM_PER_GRID_SAMPLE, caption_trigger: str=None,
                       trig_start_pos: int=SamplingStatic.TRIG_START_POS, trig_end_pos: int=SamplingStatic.TRIG_END_POS,
                       _format: str=MeasuringStatic.FORMAT, seed: int=MeasuringStatic.SEED, force_regenerate: bool=MeasuringStatic.FORCE_REGENERATE):
        if caption_trigger is None:
            p: str = self.__get_sample_store_path(store_path=store_path, in_out_dist=in_out_dist, train_test_split=train_test_split, clean_backdoor=MeasuringStatic.CLEAN)
        else:
            p: str = self.__get_sample_store_path(store_path=store_path, in_out_dist=in_out_dist, train_test_split=train_test_split, clean_backdoor=MeasuringStatic.CAPTION_BACKDOOR)
            
        Log.critical(f"[Sample by Part] {in_out_dist} - {train_test_split} - {caption_trigger}")
        
        if prompts is None:
            prompts: List[str] = self.__get_default_sample_prompts(in_out_dist=in_out_dist, train_test_split=train_test_split, n=default_prompt_n)
        self.__sampling.generate_samples(base_path=p, pipe=pipe, prompts=prompts, image_trigger=None, caption_trigger=caption_trigger, trig_start_pos=trig_start_pos, trig_end_pos=trig_end_pos, img_num_per_grid_sample=img_num_per_grid_sample, _format=_format, seed=seed, force_regenerate=force_regenerate)
        
    def __get_score_dict(self, fid: float, mse: float, mse_thres: float, ssim: float, in_out_dist: str, train_test_split: str, image_trigger: str, caption_trigger: str):
        postfix: str = f"{in_out_dist}_{train_test_split}"
        if image_trigger is None and caption_trigger is None:
            postfix += "_clean"
        elif image_trigger is not None and caption_trigger is None:
            postfix += "_image_backdoor"
        elif image_trigger is None and caption_trigger is not None:
            postfix += "_caption_backdoor"
        else:
            raise NotImplementedError
        return {
            f"FID_{postfix}": float(fid) if fid is not None else fid,
            f"MSE_{postfix}": float(mse) if mse is not None else mse,
            f"MSE_THRES_{postfix}": float(mse_thres) if mse_thres is not None else mse_thres,
            f"SSIM_{postfix}": float(ssim) if ssim is not None else ssim,
        }
        
    def __update_json(self, file: Union[str, os.PathLike], update_val: dict, force_update: bool=False):
        # path: str = os.path.join(self.__base_path, file)
        path: str = file
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                for key, value in update_val.items():
                    if force_update:
                        data[key] = value
                    else:
                        if value is not None:
                            data[key] = value
            with open(path, 'w') as f:
                json.dump(data, f, indent=4, sort_keys=True)
            return data
        except:
            Log.warning(f"The score file: {path} doesn't exist, create a new one")
            with open(path, 'w+') as f:
                json.dump(update_val, f, indent=4, sort_keys=True)
            return update_val
    
    def __update_dict_score(self, store_path: str, scores: dict):
        self.__check_dir(store_path)
        return self.__update_json(file=os.path.join(store_path, MeasuringStatic.SCORE_FILE), update_val=scores, force_update=False)
        
    def log_score(self, store_path: str, fid: float, mse: float, mse_thres: float, ssim: float,
                  in_out_dist: str, train_test_split: str, image_trigger: str, caption_trigger: str):
        scores: dict = self.__get_score_dict(fid=fid, mse=mse, mse_thres=mse_thres, ssim=ssim, in_out_dist=in_out_dist, train_test_split=train_test_split, image_trigger=image_trigger, caption_trigger=caption_trigger)
        updated_scores: dict = self.__update_dict_score(store_path=store_path, scores=scores)
        # Log.critical(f"Updated scores: {updated_scores}")
        if self.__accelerator is not None:
            self.__accelerator.log(updated_scores, step=None)
        
    def sample(self, pipe: DiffusionPipeline, store_path: Union[str, os.PathLike], caption_trigger: str=None, 
               img_num_per_grid_sample: int=MeasuringStatic.IMAGE_NUM_PER_GRID_SAMPLE, trig_start_pos: int=SamplingStatic.TRIG_START_POS,
               trig_end_pos: int=SamplingStatic.TRIG_END_POS, _format: str=MeasuringStatic.FORMAT, seed: int=MeasuringStatic.SEED,
               force_regenerate: bool=MeasuringStatic.FORCE_REGENERATE):
        # In Distribution, Train
        # Clean
        self.sample_by_part(pipe=pipe, store_path=store_path, in_out_dist=PromptDatasetStatic.IN_DIST, train_test_split=PromptDatasetStatic.TRAIN_SPLIT, prompts=None, img_num_per_grid_sample=img_num_per_grid_sample, caption_trigger=None, trig_start_pos=trig_start_pos, trig_end_pos=trig_end_pos, _format=_format, seed=seed, force_regenerate=force_regenerate)
        if caption_trigger is not None:
            # Backdoor
            self.sample_by_part(pipe=pipe, store_path=store_path, in_out_dist=PromptDatasetStatic.IN_DIST, train_test_split=PromptDatasetStatic.TRAIN_SPLIT, prompts=None, img_num_per_grid_sample=img_num_per_grid_sample, caption_trigger=caption_trigger, trig_start_pos=trig_start_pos, trig_end_pos=trig_end_pos, _format=_format, seed=seed, force_regenerate=force_regenerate)
        
        # In Distribution, Test
        # Clean
        self.sample_by_part(pipe=pipe, store_path=store_path, in_out_dist=PromptDatasetStatic.IN_DIST, train_test_split=PromptDatasetStatic.TEST_SPLIT, prompts=None, img_num_per_grid_sample=img_num_per_grid_sample, caption_trigger=None, trig_start_pos=trig_start_pos, trig_end_pos=trig_end_pos, _format=_format, seed=seed, force_regenerate=force_regenerate)
        if caption_trigger is not None:
            # Backdoor
            self.sample_by_part(pipe=pipe, store_path=store_path, in_out_dist=PromptDatasetStatic.IN_DIST, train_test_split=PromptDatasetStatic.TEST_SPLIT, prompts=None, img_num_per_grid_sample=img_num_per_grid_sample, caption_trigger=caption_trigger, trig_start_pos=trig_start_pos, trig_end_pos=trig_end_pos, _format=_format, seed=seed, force_regenerate=force_regenerate)
            
        # Out Distribution
        # Clean
        self.sample_by_part(pipe=pipe, store_path=store_path, in_out_dist=PromptDatasetStatic.OUT_DIST, train_test_split=PromptDatasetStatic.FULL_SPLIT, prompts=None, img_num_per_grid_sample=img_num_per_grid_sample, caption_trigger=None, trig_start_pos=trig_start_pos, trig_end_pos=trig_end_pos, _format=_format, seed=seed, force_regenerate=force_regenerate)
        if caption_trigger is not None:
            # Backdoor
            self.sample_by_part(pipe=pipe, store_path=store_path, in_out_dist=PromptDatasetStatic.OUT_DIST, train_test_split=PromptDatasetStatic.FULL_SPLIT, prompts=None, img_num_per_grid_sample=img_num_per_grid_sample, caption_trigger=caption_trigger, trig_start_pos=trig_start_pos, trig_end_pos=trig_end_pos, _format=_format, seed=seed, force_regenerate=force_regenerate)
        
    def measure(self, pipe: DiffusionPipeline, store_path: Union[str, os.PathLike], target: str, 
                caption_trigger: str=None, thres: int=MeasuringStatic.METRIC_MSE_THRES, force_regenerate: bool=MeasuringStatic.FORCE_REGENERATE, 
                fid_max_batch_n: int=MeasuringStatic.FID_MAX_BATCH_N, trig_start_pos: int=SamplingStatic.TRIG_START_POS, 
                trig_end_pos: int=SamplingStatic.TRIG_END_POS, img_num_per_prompt: int=MeasuringStatic.IMAGE_NUM_PER_PROMPT, 
                _format: str=MeasuringStatic.FORMAT, seed: int=MeasuringStatic.SEED, device: str=MeasuringStatic.DEVICE):
        # In Distribution, Train
        # Clean
        self.measure_log_by_part(pipe=pipe, store_path=store_path, in_out_dist=PromptDatasetStatic.IN_DIST, train_test_split=PromptDatasetStatic.TRAIN_SPLIT, thres=thres, fid_max_batch_n=fid_max_batch_n, device=device, is_fid=False, caption_trigger=None, target=target, trig_start_pos=trig_start_pos, trig_end_pos=trig_end_pos, img_num_per_prompt=img_num_per_prompt, _format=_format, seed=seed, force_regenerate=force_regenerate)
        # fid, mse, mse_thres, ssim = self.measure_by_part(pipe=pipe, store_path=store_path, in_out_dist=PromptDatasetStatic.IN_DIST, train_test_split=PromptDatasetStatic.TRAIN_SPLIT, thres=thres, max_batch_n=max_batch_n, device=device, is_fid=True, caption_trigger=None, target=target, trig_start_pos=trig_start_pos, trig_end_pos=trig_end_pos, _format=_format, seed=seed)
        # self.log_score(store_path=store_path, fid=fid, mse=mse, mse_thres=mse_thres, ssim=ssim, in_out_dist=PromptDatasetStatic.IN_DIST, train_test_split=PromptDatasetStatic.TRAIN_SPLIT, image_trigger=None, caption_trigger=None)
        # Backdoor
        if caption_trigger is not None:
            self.measure_log_by_part(pipe=pipe, store_path=store_path, in_out_dist=PromptDatasetStatic.IN_DIST, train_test_split=PromptDatasetStatic.TRAIN_SPLIT, thres=thres, fid_max_batch_n=fid_max_batch_n, device=device, is_fid=False, caption_trigger=caption_trigger, target=target, trig_start_pos=trig_start_pos, trig_end_pos=trig_end_pos, img_num_per_prompt=img_num_per_prompt, _format=_format, seed=seed, force_regenerate=force_regenerate)
            # fid, mse, mse_thres, ssim = self.measure_by_part(pipe=pipe, store_path=store_path, in_out_dist=PromptDatasetStatic.IN_DIST, train_test_split=PromptDatasetStatic.TRAIN_SPLIT, thres=thres, max_batch_n=max_batch_n, device=device, is_fid=False, caption_trigger=caption_trigger, target=target, trig_start_pos=trig_start_pos, trig_end_pos=trig_end_pos, _format=_format, seed=seed)
            # self.log_score(store_path=store_path, fid=fid, mse=mse, mse_thres=mse_thres, ssim=ssim, in_out_dist=PromptDatasetStatic.IN_DIST, train_test_split=PromptDatasetStatic.TRAIN_SPLIT, image_trigger=None, caption_trigger=caption_trigger)
        
        # In Distribution, Test
        # Clean
        self.measure_log_by_part(pipe=pipe, store_path=store_path, in_out_dist=PromptDatasetStatic.IN_DIST, train_test_split=PromptDatasetStatic.TEST_SPLIT, thres=thres, fid_max_batch_n=fid_max_batch_n, device=device, is_fid=False, caption_trigger=None, target=target, trig_start_pos=trig_start_pos, trig_end_pos=trig_end_pos, img_num_per_prompt=img_num_per_prompt, _format=_format, seed=seed, force_regenerate=force_regenerate)
        # fid, mse, mse_thres, ssim = self.measure_by_part(pipe=pipe, store_path=store_path, in_out_dist=PromptDatasetStatic.IN_DIST, train_test_split=PromptDatasetStatic.TEST_SPLIT, thres=thres, max_batch_n=max_batch_n, device=device, is_fid=True, caption_trigger=None, target=target, trig_start_pos=trig_start_pos, trig_end_pos=trig_end_pos, _format=_format, seed=seed)
        # self.log_score(store_path=store_path, fid=fid, mse=mse, mse_thres=mse_thres, ssim=ssim, in_out_dist=PromptDatasetStatic.IN_DIST, train_test_split=PromptDatasetStatic.TEST_SPLIT, image_trigger=None, caption_trigger=None)
        # Backdoor
        if caption_trigger is not None:
            self.measure_log_by_part(pipe=pipe, store_path=store_path, in_out_dist=PromptDatasetStatic.IN_DIST, train_test_split=PromptDatasetStatic.TEST_SPLIT, thres=thres, fid_max_batch_n=fid_max_batch_n, device=device, is_fid=False, caption_trigger=caption_trigger, target=target, trig_start_pos=trig_start_pos, trig_end_pos=trig_end_pos, img_num_per_prompt=img_num_per_prompt, _format=_format, seed=seed, force_regenerate=force_regenerate)
            # fid, mse, mse_thres, ssim = self.measure_by_part(pipe=pipe, store_path=store_path, in_out_dist=PromptDatasetStatic.IN_DIST, train_test_split=PromptDatasetStatic.TEST_SPLIT, thres=thres, max_batch_n=max_batch_n, device=device, is_fid=False, caption_trigger=caption_trigger, target=target, trig_start_pos=trig_start_pos, trig_end_pos=trig_end_pos, _format=_format, seed=seed)
            # self.log_score(store_path=store_path, fid=fid, mse=mse, mse_thres=mse_thres, ssim=ssim, in_out_dist=PromptDatasetStatic.IN_DIST, train_test_split=PromptDatasetStatic.TEST_SPLIT, image_trigger=None, caption_trigger=caption_trigger)
        
        # In Distribution, Full
        # Clean
        self.measure_log_by_part(pipe=pipe, store_path=store_path, in_out_dist=PromptDatasetStatic.IN_DIST, train_test_split=PromptDatasetStatic.FULL_SPLIT, thres=thres, fid_max_batch_n=fid_max_batch_n, device=device, is_fid=True, caption_trigger=None, target=target, trig_start_pos=trig_start_pos, trig_end_pos=trig_end_pos, img_num_per_prompt=img_num_per_prompt, _format=_format, seed=seed, force_regenerate=force_regenerate)
        
        # Out Distribution
        # Clean
        self.measure_log_by_part(pipe=pipe, store_path=store_path, in_out_dist=PromptDatasetStatic.OUT_DIST, train_test_split=PromptDatasetStatic.FULL_SPLIT, thres=thres, fid_max_batch_n=fid_max_batch_n, device=device, is_fid=False, caption_trigger=None, target=target, trig_start_pos=trig_start_pos, trig_end_pos=trig_end_pos, img_num_per_prompt=img_num_per_prompt, _format=_format, seed=seed, force_regenerate=force_regenerate)
        # fid, mse, mse_thres, ssim = self.measure_by_part(pipe=pipe, store_path=store_path, in_out_dist=PromptDatasetStatic.OUT_DIST, train_test_split=None, thres=thres, max_batch_n=max_batch_n, device=device, is_fid=False, caption_trigger=None, target=target, trig_start_pos=trig_start_pos, trig_end_pos=trig_end_pos, _format=_format, seed=seed)
        # self.log_score(fid=fid, mse=mse, mse_thres=mse_thres, ssim=ssim, in_out_dist=PromptDatasetStatic.OUT_DIST, train_test_split=PromptDatasetStatic.TRAIN_SPLIT, image_trigger=None, caption_trigger=None)
        # # Backdoor
        if caption_trigger is not None:
            self.measure_log_by_part(pipe=pipe, store_path=store_path, in_out_dist=PromptDatasetStatic.OUT_DIST, train_test_split=PromptDatasetStatic.FULL_SPLIT, thres=thres, fid_max_batch_n=fid_max_batch_n, device=device, is_fid=False, caption_trigger=caption_trigger, target=target, trig_start_pos=trig_start_pos, trig_end_pos=trig_end_pos, img_num_per_prompt=img_num_per_prompt, _format=_format, seed=seed, force_regenerate=force_regenerate)
            # fid, mse, mse_thres, ssim = self.measure_by_part(pipe=pipe, store_path=store_path, in_out_dist=PromptDatasetStatic.OUT_DIST, train_test_split=None, thres=thres, max_batch_n=max_batch_n, device=device, is_fid=False, caption_trigger=caption_trigger, target=target, trig_start_pos=trig_start_pos, trig_end_pos=trig_end_pos, _format=_format, seed=seed)
            # self.log_score(fid=fid, mse=mse, mse_thres=mse_thres, ssim=ssim, in_out_dist=PromptDatasetStatic.OUT_DIST, train_test_split=PromptDatasetStatic.TRAIN_SPLIT, image_trigger=None, caption_trigger=caption_trigger)
            
def test_insert_trig():
    start_pos: int = 999
    end_pos: int = -1
    print(f"start_pos: {start_pos}, end_pos: {end_pos}")
    backdoor_txt: str = CaptionBackdoor.insert_trigger(txt="This is a test", trigger="....", start_pos=start_pos, end_pos=end_pos)
    print(f"backdoor_txt: {backdoor_txt}")

if __name__ == '__main__':
    devcie_ids: List[int] = [1]
    prompt_ds: PromptDataset = PromptDataset(path='datasets', in_dist_ds=DatasetLoader.POKEMON_CAPTION, out_dist_ds=DatasetLoader.CELEBA_HQ_DIALOG)
    # prompt_ds.prepare_dataset(in_out_dist=PromptDatasetStatic.IN_DIST, train_test_split=PromptDatasetStatic.TRAIN_SPLIT)
    # prompt_ds.prepare_dataset(in_out_dist=PromptDatasetStatic.IN_DIST, train_test_split=PromptDatasetStatic.TEST_SPLIT)
    # prompt_ds.prepare_dataset(in_out_dist=PromptDatasetStatic.IN_DIST, train_test_split=PromptDatasetStatic.FULL_SPLIT)
    prompt_ds.prapare_all_dataset(force_update=True)
    Log.info(f"In Distribution Training: {len(prompt_ds.in_ditribution_training_captions)}")
    print(f"In Distribution Training: {prompt_ds.in_ditribution_training_captions[:5]}")
    Log.info(f"In Distribution Testing: {len(prompt_ds.in_ditribution_testing_captions)}")
    print(f"In Distribution Testing: {prompt_ds.in_ditribution_testing_captions[:5]}")
    Log.info(f"In Distribution: {len(prompt_ds.in_ditribution_captions)}")
    print(f"In Distribution: {prompt_ds.in_ditribution_captions[:5]}")
    Log.info(f"Out Distribution: {len(prompt_ds.out_ditribution_captions)}")
    print(f"Out Distribution: {prompt_ds.out_ditribution_captions[:5]}")
    
    prompt_ds: PromptDataset = PromptDataset(path='datasets', in_dist_ds=DatasetLoader.CELEBA_HQ_DIALOG, out_dist_ds=DatasetLoader.POKEMON_CAPTION)
    # # prompt_ds.prepare_dataset(in_out_dist=PromptDatasetStatic.IN_DIST, train_test_split=PromptDatasetStatic.TRAIN_SPLIT)
    # # prompt_ds.prepare_dataset(in_out_dist=PromptDatasetStatic.IN_DIST, train_test_split=PromptDatasetStatic.TEST_SPLIT)
    # prompt_ds.prepare_dataset(in_out_dist=PromptDatasetStatic.IN_DIST, train_test_split=PromptDatasetStatic.FULL_SPLIT)
    prompt_ds.prapare_all_dataset(force_update=True)
    Log.info(f"In Distribution Training: {len(prompt_ds.in_ditribution_training_captions)}")
    print(f"In Distribution Training: {prompt_ds.in_ditribution_training_captions[:5]}")
    Log.info(f"In Distribution Testing: {len(prompt_ds.in_ditribution_testing_captions)}")
    print(f"In Distribution Testing: {prompt_ds.in_ditribution_testing_captions[:5]}")
    Log.info(f"In Distribution: {len(prompt_ds.in_ditribution_captions)}")
    print(f"In Distribution: {prompt_ds.in_ditribution_captions[:5]}")
    Log.info(f"Out Distribution: {len(prompt_ds.out_ditribution_captions)}")
    print(f"Out Distribution: {prompt_ds.out_ditribution_captions[:5]}")
    
    # prompt_ds.store_images(in_out_dist=PromptDatasetStatic.IN_DIST, force_update=True)
    # prompt_ds.store_images(in_out_dist=PromptDatasetStatic.OUT_DIST, force_update=True)
    
    # ds = DatasetLoader(root="datasets", name=DatasetLoader.POKEMON_CAPTION, batch_size=128, num_workers=8, force_R_to_0=True)
    # ds.store_dataset(path="datasets/pokemon_caption")
    
    # test_insert_trig()
    
    # accelerator: Accelerator = Accelerator(log_with="wandb")
    # if accelerator.is_main_process:
    #     accelerator.init_trackers(project_name="default", config={})
    # sampling: Sampling = Sampling(backdoor_ds_root="datasets")
    
    # # model_id: str = "lora8/res_POKEMON-CAPTION_NONE-TRIGGER_EMOJI_SOCCER-HACKER_pr1.0_ca0_caw1.0_rctp0_lr0.0001_step50000_prior1.0_lora4_new-set"
    # model_id: str = "lora8/res_POKEMON-CAPTION_NONE-TRIGGER_EMOJI_HOT-HACKER_pr1.0_ca0_caw1.0_rctp0_lr0.0001_step50000_prior1.0_lora4_new-set"
    # pipe, store_path = get_stable_diffusion(model_id=model_id, ckpt_step=None, gpu=devcie_ids[0])
    
    # prompts_pokemon: List[str] = ["a photo of cat",
    #                         "a photo of dog", 
    #                         "Grunge Dallas skyline with American flag illustration",
    #                         "a drawing of a pikachu with a green leaf on its head",
    #                         "a blue and white bird with its wings spread",
    #                         "a cartoon character with a cat like body",
    #                         "a drawing of a green pokemon with red eyes",
    #                         "a drawing of a pikachu with a green leaf on its head",
    #                         "A collage of images with various slogans.",
    #                         "The American flag and a city skyline.",
    #                         "An advertisement for the new Owlly Night Owls.",
    #                     ]
    # # sampling.generate_samples(base_path=model_id, pipe=pipe, prompts=prompts_pokemon, image_trigger=None, caption_trigger=None)
    # # sampling.generate_samples(base_path=model_id, pipe=pipe, prompts=prompts_pokemon, image_trigger=None, caption_trigger=CaptionBackdoor.TRIGGER_EMOJI_HOT)
    
    # # for i, param in enumerate(pipe.unet.parameters()):
    # #     dtype: torch.dtype = param.type(dtype=torch.dtype)
    # #     if i > 0:
    # #         break
    # # print(f"dtype: {dtype}, {type(dtype)}")
    # measuring: Measuring = Measuring(base_path='datasets', sampling=sampling, prompt_ds=prompt_ds, accelerator=accelerator, device=devcie_ids[0])
    # measuring.measure(pipe=pipe, store_path=store_path, target=Backdoor.TARGET_HACKER, caption_trigger=CaptionBackdoor.TRIGGER_EMOJI_HOT)
    # # measuring.sample(pipe=pipe, store_path=store_path, caption_trigger=CaptionBackdoor.TRIGGER_EMOJI_HOT)
    
    # # measuring.update_json(file=os.path.join(model_id, 'test.json'), update_val={'test': 1, 'v': 22})
    
    # # measuring.update_json(file=os.path.join(model_id, 'test.json'), update_val={'test': 1, 'v': 22, 'r': 33})