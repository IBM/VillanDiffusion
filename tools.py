import glob
from math import ceil, sqrt
from typing import List, Union, Tuple
import os

from PIL import Image
import numpy as np
import torch

from diffusers import AutoencoderKL

class Log:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
    @staticmethod
    def error_msg(msg: str):
        return Log.FAIL + Log.BOLD + msg + Log.ENDC
    
    @staticmethod
    def warning_msg(msg: str):
        return Log.WARNING + Log.BOLD + msg + Log.ENDC

    @staticmethod
    def critical_msg(msg: str):
        return Log.OKCYAN + Log.BOLD + msg + Log.ENDC

    @staticmethod
    def info_msg(msg: str):
        return Log.OKGREEN + Log.BOLD + msg + Log.ENDC

    @staticmethod
    def error(msg: str):
        msg: str = Log.error_msg(msg=msg)
        print(msg)
        return msg
    
    @staticmethod
    def warning(msg: str):
        msg: str = Log.warning_msg(msg=msg)
        print(msg)
        return msg

    @staticmethod
    def critical(msg: str):
        msg: str = Log.critical_msg(msg=msg)
        print(msg)
        return msg
    
    @staticmethod
    def info(msg: str):
        msg: str = Log.info_msg(msg=msg)
        print(msg)
        return msg

def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid

def save_grid(images: List, path: Union[str, os.PathLike], file_name: str, _format: str='png'):
    images = [Image.fromarray(np.squeeze((image * 255).round().astype("uint8"))) for image in images]
    
    eval_samples_n = len(images)
    nrow = 1
    ncol = eval_samples_n
    for i in range(ceil(sqrt(eval_samples_n)), 0, -1):
        if eval_samples_n % i == 0:
            nrow = i
            ncol = eval_samples_n // nrow
            break

    # # Make a grid out of the images
    image_grid = make_grid(images, rows=nrow, cols=ncol)
    image_grid.save(os.path.join(f"{path}", f"{file_name}.{_format}"))
    
def encode_latents(vae: AutoencoderKL, x: torch.Tensor, weight_dtype: str):
    return vae.encode(x.to(device=vae.device, dtype=weight_dtype)).latent_dist.sample() * vae.config.scaling_factor

def get_batch_sizes(sample_n: int, max_batch_n: int):
    if sample_n > max_batch_n:
        replica = sample_n // max_batch_n
        residual = sample_n % max_batch_n
        batch_sizes = [max_batch_n] * (replica) + ([residual] if residual > 0 else [])
    else:
        batch_sizes = [sample_n]
    return batch_sizes
        
def batchify(xs, max_batch_n: int):
    batch_sizes = get_batch_sizes(sample_n=len(xs), max_batch_n=max_batch_n)
    
    print(f"xs len(): {len(xs)}")    
    print(f"batch_sizes: {batch_sizes}, max_batch_n: {max_batch_n}")
    # print(f"Max_batch_n: {max_batch_n}")
    res: List = []
    cnt: int = 0
    for i, bs in enumerate(batch_sizes):
        res.append(xs[cnt:cnt+bs])
        cnt += bs
    return res

def batchify_generator(xs, max_batch_n: int):
    batch_sizes = get_batch_sizes(sample_n=len(xs), max_batch_n=max_batch_n)
    
    cnt: int = 0
    for i, bs in enumerate(batch_sizes):
        yield xs[cnt:cnt+bs]
        cnt += bs

def randn_images(n: int, channel: int, image_size: int, seed: int):
    shape: Tuple[int] = (n, channel, image_size, image_size)
    return torch.randn(shape, generator=torch.manual_seed(seed))

def match_count(dir: Union[str, os.PathLike], exts: List[str]=["png", "jpg", "jpeg"]) -> int:
    files_grabbed = []
    for ext in exts:
        files_grabbed.extend(glob.glob(os.path.join(dir, f"*.{ext}")))
    return len(set(files_grabbed))