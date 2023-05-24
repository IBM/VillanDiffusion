import argparse
from dataclasses import asdict, dataclass
import gc
import hashlib
import itertools
import json
import logging
import math
import os
import threading
import warnings
from pathlib import Path
from typing import Optional, Tuple, List, Union
from packaging import version

from tqdm.auto import tqdm
import numpy as np
import psutil
from PIL import Image

from caption_dataset import Backdoor, DatasetLoader, CaptionBackdoor

@dataclass
class Config:
    pretrained_model_name_or_path: str=None
    revision: str=None
    tokenizer_name: str=None
    instance_data_dir: str=None
    class_data_dir: str=None
    instance_prompt: str=None
    class_prompt: str=None
    with_prior_preservation: bool=False
    prior_loss_weight: float=1.0
    num_class_images: int=100
    validation_prompt: str=None
    num_validation_images: int=4
    validation_steps: int=100
    output_dir: str=None
    seed: int=None
    resolution: int=512
    center_crop: bool=False
    train_text_encoder: bool=False
    use_lora: bool=False
    lora_r: int=8
    lora_alpha: int=32
    lora_dropout: float=0.0
    lora_bias: str=None
    lora_text_encoder_r: int=8
    lora_text_encoder_alpha: int=32
    lora_text_encoder_dropout: float=0.0
    lora_text_encoder_bias: str="none"
    train_batch_size: int=4
    sample_batch_size: int=4
    num_train_epochs: int=1
    max_train_steps: int=None
    checkpointing_steps: int=500
    resume_from_checkpoint: str=None
    gradient_accumulation_steps: int=1
    gradient_checkpointing: bool=False
    learning_rate: float=5e-6
    scale_lr: bool=False
    lr_scheduler: str="cosine"
    lr_warmup_steps: int=500
    lr_num_cycles: int=1
    lr_power: float=1.0
    use_8bit_adam: bool=False
    dataloader_num_workers: int=8
    adam_beta1: float=0.9
    adam_beta2: float=0.999
    adam_weight_decay: float=1e-2
    adam_epsilon: float=1e-08
    max_grad_norm: float=1.0
    push_to_hub: bool=False
    hub_token: str=None
    hub_model_id: str=None
    logging_dir: str="logs"
    dataset_name: str=DatasetLoader.POKEMON_CAPTION
    poison_rate: float=None
    image_trigger: str=Backdoor.TRIGGER_NONE
    caption_trigger: str=CaptionBackdoor.TRIGGER_ELLIPSIS
    target: str=Backdoor.TARGET_CAT
    split: str="[:90%]"
    caption_augment: int = 0
    caption_augment_weight: float = 1.0
    rand_caption_trig_pos: int = 0
    enable_backdoor: bool=False
    with_backdoor_prior_preservation: bool=True
    postfix: str=""
    dir: str=""
    overwrite: bool=False
    allow_tf32: bool=False
    report_to: str="tensorboard"
    mixed_precision: str= "fp16"
    prior_generation_precision: str=None
    local_rank: int=-1
    enable_xformers_memory_efficient_attention: bool=False
    gpu: str='0'

def naming(config: Config):
    postfix = ""
    if config.with_backdoor_prior_preservation:
        postfix += f"_prior{config.prior_loss_weight}"
    if config.use_lora:
        postfix += f"_lora{config.lora_r}"
    if postfix != None and postfix != "":
        postfix += f"_{config.postfix}"
    return f"res_{config.dataset_name}_{config.image_trigger}-{config.caption_trigger}-{config.target}_pr{config.poison_rate}_ca{config.caption_augment}_caw{config.caption_augment_weight}_rctp{config.rand_caption_trig_pos}_lr{config.learning_rate}_step{config.max_train_steps}{postfix}"

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default="target_data_dog",
        required=False,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default='"a photo of sks dog"',
        required=False,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run dreambooth validation every X steps. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution"
    )
    # parser.add_argument("--train_text_encoder", action="store_true", help="Whether to train the text encoder")

    # lora args
    parser.add_argument("--use_lora", action="store_true", help="Whether to use Lora for parameter efficient tuning")
    parser.add_argument("--lora_r", type=int, default=8, help="Lora rank, only used if use_lora is True")
    # parser.add_argument("--lora_alpha", type=int, default=32, help="Lora alpha, only used if use_lora is True")
    # parser.add_argument("--lora_dropout", type=float, default=0.0, help="Lora dropout, only used if use_lora is True")
    # parser.add_argument(
    #     "--lora_bias",
    #     type=str,
    #     default="none",
    #     help="Bias type for Lora. Can be 'none', 'all' or 'lora_only', only used if use_lora is True",
    # )
    # parser.add_argument(
    #     "--lora_text_encoder_r",
    #     type=int,
    #     default=8,
    #     help="Lora rank for text encoder, only used if `use_lora` and `train_text_encoder` are True",
    # )
    # parser.add_argument(
    #     "--lora_text_encoder_alpha",
    #     type=int,
    #     default=32,
    #     help="Lora alpha for text encoder, only used if `use_lora` and `train_text_encoder` are True",
    # )
    # parser.add_argument(
    #     "--lora_text_encoder_dropout",
    #     type=float,
    #     default=0.0,
    #     help="Lora dropout for text encoder, only used if `use_lora` and `train_text_encoder` are True",
    # )
    # parser.add_argument(
    #     "--lora_text_encoder_bias",
    #     type=str,
    #     default="none",
    #     help="Bias type for Lora. Can be 'none', 'all' or 'lora_only', only used if use_lora and `train_text_encoder` are True",
    # )

    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=8,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--dataset_name", "-dn",
        type=str,
        default=DatasetLoader.POKEMON_CAPTION,
        help="Backdoor dataset name, only work for backdoor",
    )
    parser.add_argument(
        "--poison_rate", "-pr",
        type=float,
        default=1.0,
        help="Poison rate, only work for backdoor",
    )
    parser.add_argument(
        "--trigger", "-tr",
        type=str,
        default=Backdoor.TRIGGER_NONE,
        help="Image backdoor trigger, only work for backdoor",
    )
    parser.add_argument(
        "--caption_trigger", "-ctr",
        type=str,
        default=CaptionBackdoor.TRIGGER_ELLIPSIS,
        help="Caption backdoor trigger, only work for backdoor",
    )
    parser.add_argument(
        "--target", "-tg",
        type=str,
        default=Backdoor.TARGET_CAT,
        help="Target image, only work for backdoor",
    )
    parser.add_argument(
        "--split", "-spl",
        type=str,
        default=Config.split,
        help="Training split ratio",
    )
    parser.add_argument(
        "--caption_augment", "-ca",
        type=int,
        default=Config.caption_augment,
        help="Caption augment times, only work for backdoor",
    )
    parser.add_argument(
        "--caption_augment_weight", "-caw",
        type=float,
        default=Config.caption_augment_weight,
        help="Loss weight of the caption augment, only work for backdoor",
    )
    parser.add_argument(
        "--rand_caption_trig_pos", "-rctp",
        type=int,
        default=Config.rand_caption_trig_pos,
        help="Caption trigger start position, counting from the end of the caption",
    )
    parser.add_argument(
        "--enable_backdoor",
        default=False,
        action="store_true",
        help="Enable backdoor attack on Stable Diffusion",
    )
    parser.add_argument(
        "--with_backdoor_prior_preservation",
        default=False,
        action="store_true",
        help="Enable prior preservation for backdoor attack, only work for backdoor",
    )
    parser.add_argument(
        "--postfix",
        type=str,
        default=Config.postfix,
        help="Postfix of the folder name",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=Config.dir,
        help="Folder of the results",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite results",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    # parser.add_argument(
    #     "--mixed_precision",
    #     type=str,
    #     default=None,
    #     choices=["no", "fp16", "bf16"],
    #     help=(
    #         "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
    #         " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
    #         " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
    #     ),
    # )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    
    parser.add_argument(
        "--gpu", type=str, default='0', help="Determine the gpu used"
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
    else:
        # logger is not available yet
        if args.class_data_dir is not None:
            warnings.warn("You need not use --class_data_dir without --with_prior_preservation.")
        if args.class_prompt is not None:
            warnings.warn("You need not use --class_prompt without --with_prior_preservation.")

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    config = Config()
    for key, value in args.__dict__.items():
        if value != None:
            setattr(config, key, value)
    
    # setattr(config, 'model_id', naming(config=config))
    
    if config.output_dir is None:
        setattr(config, 'output_dir', os.path.join(config.dir, naming(config=config)))

    os.makedirs(config.output_dir, exist_ok=True)
    if os.path.isfile(os.path.join(config.output_dir, "pytorch_lora_weights.bin")):
        if not config.overwrite:
            print("Skipped Experiment because file already exists")
            exit()
        else:
            print("Overwriting Experiment")
    with open(os.path.join(config.output_dir, 'args.json'), 'w') as f:
        dict_config: dict = asdict(config)
        dict_config['model_id'] = naming(config=config)
        json.dump(dict_config, f, indent=4)
        
    print(f"Config: {config}")
    return config

args = parse_args()

import datasets
import diffusers

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
)
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.loaders import AttnProcsLayers
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import HfFolder, Repository, whoami
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer, PretrainedConfig

# from peft import LoraConfig, get_peft_model
from caption_dataset import get_data_loader, collate_fn_backdoor_gen
from loss_conditional import LossFn

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__)

UNET_TARGET_MODULES = ["to_q", "to_v", "query", "value"]  # , "ff.net.0.proj"]
TEXT_ENCODER_TARGET_MODULES = ["q_proj", "v_proj"]


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


# Converting Bytes to Megabytes
def b2mb(x):
    return int(x / 2**20)


# This context manager is used to track the peak memory usage of the process
class TorchTracemalloc:
    def __enter__(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()  # reset the peak gauge to zero
        self.begin = torch.cuda.memory_allocated()
        self.process = psutil.Process()

        self.cpu_begin = self.cpu_mem_used()
        self.peak_monitoring = True
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()
        return self

    def cpu_mem_used(self):
        """get resident set size memory for the current process"""
        return self.process.memory_info().rss

    def peak_monitor_func(self):
        self.cpu_peak = -1

        while True:
            self.cpu_peak = max(self.cpu_mem_used(), self.cpu_peak)

            # can't sleep or will not catch the peak right (this comment is here on purpose)
            # time.sleep(0.001) # 1msec

            if not self.peak_monitoring:
                break

    def __exit__(self, *exc):
        self.peak_monitoring = False

        gc.collect()
        # torch.cuda.empty_cache()
        self.end = torch.cuda.memory_allocated()
        self.peak = torch.cuda.max_memory_allocated()
        self.used = b2mb(self.end - self.begin)
        self.peaked = b2mb(self.peak - self.begin)

        self.cpu_end = self.cpu_mem_used()
        self.cpu_used = b2mb(self.cpu_end - self.cpu_begin)
        self.cpu_peaked = b2mb(self.cpu_peak - self.cpu_begin)
        # print(f"delta used/peak {self.used:4d}/{self.peaked:4d}")


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        size=512,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids

        return example


def collate_fn(examples, with_prior_preservation=False):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
    }
    return batch


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"

def dreambooth_loss(args: Config, batch, noise_scheduler, unet, vae, text_encoder, weight_dtype: str):
    # Dreambooth
    image = batch["pixel_values"]
    caption = batch["input_ids"]
    # print(f"[pixel_values]: {batch['pixel_values']}, Shape: {batch['pixel_values'].shape}, Min: {torch.min(batch['pixel_values'])}, Max: {torch.max(batch['pixel_values'])}")
    # print(f"[input_ids]: {batch['input_ids']}")
        
    # Convert images to latent space
    latents = vae.encode(image.to(dtype=weight_dtype)).latent_dist.sample()
    latents = latents * 0.18215

    # Sample noise that we'll add to the latents
    noise = torch.randn_like(latents)
    bsz = latents.shape[0]
    # Sample a random timestep for each image
    timesteps = torch.randint(
        0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
    )
    timesteps = timesteps.long()
    
    # Add noise to the latents according to the noise magnitude at each timestep
    # (this is the forward diffusion process)
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    # Get the text embedding for conditioning
    encoder_hidden_states = text_encoder(caption)[0]

    # Predict the noise residual
    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

    # Get the target for loss depending on the prediction type
    if noise_scheduler.config.prediction_type == "epsilon":
        if not args.enable_backdoor:
            target = noise
        else:
            # target = noise + R_coef_t * poison_latents
            target = noise
    # elif noise_scheduler.config.prediction_type == "v_prediction":
    #     target = noise_scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

    if args.with_prior_preservation:
        # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
        model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
        target, target_prior = torch.chunk(target, 2, dim=0)

        # Compute instance loss
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        # Compute prior loss
        prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

        # Add the prior loss to the instance loss.
        loss = loss + args.prior_loss_weight * prior_loss
    else:
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
    return loss

class CondLossFn:
    PREDICTION_TYPE_EPSILON: str = "epsilon"
    PREDICTION_TYPE_V_PREDICTION: str = "v_prediction"
    
    def __init__(self, noise_scheduler, vae: AutoencoderKL, text_encoder, weight_dtype: str, scaling_factor: float=None):
        self.__noise_scheduler = noise_scheduler
        self.__vae: AutoencoderKL = vae
        self.__text_encoder = text_encoder
        self.__weight_dtype: str = weight_dtype
        self.__scaling_factor: float = scaling_factor
        
    @staticmethod
    def __encode_latents(vae: AutoencoderKL, x: torch.Tensor, weight_dtype: str, scaling_factor: float=None):
        if scaling_factor != None:
            return vae.encode(x.to(dtype=weight_dtype)).latent_dist.sample() * scaling_factor
        return vae.encode(x.to(dtype=weight_dtype)).latent_dist.sample() * vae.config.scaling_factor
    @staticmethod
    def __decode_latents(vae: AutoencoderKL, x: torch.Tensor, weight_dtype: str, scaling_factor: float=None):
        if scaling_factor != None:
            return vae.decode(x.to(dtype=weight_dtype)).sample / scaling_factor
        return vae.decode(x.to(dtype=weight_dtype)).sample / vae.config.scaling_factor
    @staticmethod
    def __get_latent(batch, key: str, vae: AutoencoderKL, weight_dtype: str, scaling_factor: float=None) -> torch.Tensor:
        return CondLossFn.__encode_latents(vae=vae, x=batch[key], weight_dtype=weight_dtype, scaling_factor=scaling_factor)
    @staticmethod
    def __get_latents(batch, keys: str, vae: AutoencoderKL, weight_dtype: str, scaling_factor: float=None) -> List[torch.Tensor]:
        return [CondLossFn.__encode_latents(vae=vae, x=batch[key], weight_dtype=weight_dtype, scaling_factor=scaling_factor) for key in keys]
    @staticmethod
    def __get_embedding(batch, key: str, text_encoder) -> torch.Tensor:
        return text_encoder(batch[key])[0]
    @staticmethod
    def __get_embeddings(batch, keys: List[str], text_encoder) -> List[torch.Tensor]:
        return [text_encoder(batch[key])[0] for key in keys]
    
    @staticmethod
    def __get_clean_noisy_latents_t(noise_scheduler, latents: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor=None):
        if noise == None:
            noise = torch.randn_like(latents)
        return noise_scheduler.add_noise(latents, noise.to(latents.device), timesteps.to(latents.device))
    
    @staticmethod
    def __get_noisy_latents_t(noise_scheduler, latents: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor, poison_latents: torch.Tensor=None, backdoor: bool=False):
        timesteps, noise = timesteps.to(latents.device), noise.to(latents.device)
        noisy_latents: torch.Tensor = CondLossFn.__get_clean_noisy_latents_t(noise_scheduler=noise_scheduler, latents=latents, timesteps=timesteps, noise=noise)
        if backdoor:
            if poison_latents == None:
                raise ValueError(f"Arguement poison_latents: {poison_latents} shouldn't be None, if arguement backdoor is True")
            def unsqueeze_n(x):
                return x.reshape(len(latents), *([1] * len(latents.shape[1:])))  
            
            poison_latents = poison_latents.to(latents.device)
            R_step, _ = LossFn.get_R_scheds_baddiff(alphas_cumprod=noise_scheduler.alphas_cumprod.to(timesteps.device), alphas=noise_scheduler.alphas.to(timesteps.device), psi=1, solver_type='ode')
            R_step_t = unsqueeze_n(R_step.to(device=timesteps.device, dtype=latents.dtype)[timesteps])
            return noisy_latents + R_step_t * poison_latents
        
        return noisy_latents
    
    @staticmethod
    def __get_target_t(noise_scheduler, poison_latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor, backdoor: bool=False):
        if backdoor:
            if poison_latents == None:
                raise ValueError(f"Arguement poison_latents: {poison_latents} shouldn't be None, if arguement backdoor is True")
            def unsqueeze_n(x):
                return x.reshape(len(noise), *([1] * len(noise.shape[1:])))  
            _, R_coef = LossFn.get_R_scheds_baddiff(alphas_cumprod=noise_scheduler.alphas_cumprod.to(timesteps.device), alphas=noise_scheduler.alphas.to(timesteps.device), psi=1, solver_type='ode')
            R_coef_t = unsqueeze_n(R_coef.to(device=timesteps.device, dtype=noise.dtype)[timesteps])
            return noise + R_coef_t * poison_latents
        
        return noise
    
    @staticmethod
    def __get_input_target_t(noise_scheduler, latents: torch.Tensor, poison_latents: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor=None, backdoor: bool=False):
        if noise == None:
            noise = torch.randn_like(latents)
        input: torch.Tensor = CondLossFn.__get_noisy_latents_t(noise_scheduler=noise_scheduler, latents=latents, poison_latents=poison_latents, noise=noise, timesteps=timesteps, backdoor=backdoor)
        target: torch.Tensor = CondLossFn.__get_target_t(noise_scheduler=noise_scheduler, poison_latents=poison_latents, noise=noise, timesteps=timesteps, backdoor=backdoor)
        return input, target
    
    @staticmethod
    def __get_input_target_t_by_keys(batch, noise_scheduler, vae: AutoencoderKL, text_encoder, latent_key: str, poison_latent_key: str, timesteps: torch.Tensor, weight_dtype: str, caption_key: str=None, noise: torch.Tensor=None, scaling_factor: float=None, backdoor: bool=False):
        poison_latents: torch.Tensor = None
        if backdoor:
            if poison_latent_key == None:
                raise ValueError(f"Arguement poison_latent_key: {poison_latent_key} shouldn't be None, if arguement backdoor is True")
            latents, poison_latents = CondLossFn.__get_latents(batch=batch, keys=[latent_key, poison_latent_key], vae=vae, weight_dtype=weight_dtype, scaling_factor=scaling_factor)
        else:
            latents = CondLossFn.__get_latent(batch=batch, key=latent_key, vae=vae, weight_dtype=weight_dtype, scaling_factor=scaling_factor)
        
        input, target = CondLossFn.__get_input_target_t(noise_scheduler=noise_scheduler, latents=latents, poison_latents=poison_latents, timesteps=timesteps, noise=noise, backdoor=backdoor)
        if caption_key == None:
            return input, target
        return input, target, CondLossFn.__get_embedding(batch=batch, key=caption_key, text_encoder=text_encoder)
    
    def convert_target(self, x: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if self.__noise_scheduler.config.prediction_type == CondLossFn.PREDICTION_TYPE_EPSILON:
            return x
        elif self.__noise_scheduler.config.prediction_type == CondLossFn.PREDICTION_TYPE_V_PREDICTION:
            return self.__noise_scheduler.get_velocity(x, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.__noise_scheduler.config.prediction_type}")
    
    def get_timesteps(self, bsz: int, latents: torch.Tensor):
        timesteps = torch.randint(
            0, self.__noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
        )
        timesteps = timesteps.long()
        return timesteps
    
    def get_input_target_t_by_keys(self, batch, latent_key: str, timesteps: torch.Tensor=None, poison_latent_key: str=None, caption_key: str=None, noise: torch.Tensor=None, backdoor: bool=False):
        if timesteps == None:
            timesteps: torch.Tensor = self.get_timesteps(bsz=len(batch[latent_key]), latents=batch[latent_key])
            
        res: Tuple = CondLossFn.__get_input_target_t_by_keys(batch=batch, noise_scheduler=self.__noise_scheduler, vae=self.__vae, text_encoder=self.__text_encoder, latent_key=latent_key, poison_latent_key=poison_latent_key, timesteps=timesteps, weight_dtype=self.__weight_dtype, caption_key=caption_key, noise=noise, scaling_factor=self.__scaling_factor, backdoor=backdoor)
        if caption_key == None:
            # res: input, target
            return res[0], self.convert_target(res[1], noise=noise, timesteps=timesteps)
        # res: input, target, caption
        return res[0], self.convert_target(res[1], noise=noise, timesteps=timesteps), res[2]
    
    def get_loss_by_keys(self, batch, unet, latent_key: str, timesteps: torch.Tensor=None, weight: float=1.0, poison_latent_key: str=None, caption_key: str=None, noise: torch.Tensor=None, backdoor: bool=False):
        if timesteps == None:
            timesteps: torch.Tensor = self.get_timesteps(bsz=len(batch[latent_key]), latents=batch[latent_key])
            
        res: Tuple = self.get_input_target_t_by_keys(batch=batch, latent_key=latent_key, timesteps=timesteps, poison_latent_key=poison_latent_key, caption_key=caption_key, noise=noise, backdoor=backdoor)
        if caption_key == None:
            # res: input, target
            return CondLossFn.loss_fn_vec(preds=[unet(res[0], timesteps).sample], targets=[res[1]], weights=[weight])
        # res: input, target, caption
        # print(f"latent_key[{latent_key}], poison_latent_key[{poison_latent_key}], target[{caption_key}]")
        # print(f"input: {res[0].shape}, timesteps: {timesteps.shape}, caption: {res[2].shape}, target: {res[1].shape}")
        return CondLossFn.loss_fn_vec(preds=[unet(res[0], timesteps, res[2]).sample], targets=[res[1]], weights=[weight])

    @staticmethod
    def loss_fn_vec(preds: List[torch.Tensor], targets: List[torch.Tensor], weights: List[float]):
        loss: float = 0
        for pred, target, weight in zip(preds, targets, weights):
            loss += weight * F.mse_loss(pred.float(), target.float(), reduction="mean")
        return loss

def get_latents_embeddings(batch, vae, text_encoder, weight_dtype, scaling_factor: float=0.18215):
    latent_keys: List[str] = [DatasetLoader.IMAGE, DatasetLoader.POISON_IMAGE, DatasetLoader.RAW]
    embed_keys: List[str] = [DatasetLoader.CAPTION, DatasetLoader.RAW_CAPTION]
    return *CondLossFn.get_latents(batch=batch, keys=latent_keys, vae=vae, weight_dtype=weight_dtype, scaling_factor=scaling_factor), *CondLossFn.get_embeddings(batch=batch, keys=embed_keys, text_encoder=text_encoder)
    
    # # Backdoor
    # image = batch[DatasetLoader.IMAGE]
    # poison_image = batch[DatasetLoader.POISON_IMAGE]
    # raw_image = batch[DatasetLoader.RAW]
    # caption = batch[DatasetLoader.CAPTION]
    # raw_caption = batch[DatasetLoader.RAW_CAPTION]
    
    # # Convert poison images to latent space
    # poison_latents = vae.encode(poison_image.to(dtype=weight_dtype)).latent_dist.sample()
    # poison_latents = poison_latents * scaling_factor
    # # print(f"[{DatasetLoader.IMAGE}]: {batch[DatasetLoader.IMAGE]}, Shape: {batch[DatasetLoader.IMAGE].shape}, Min: {torch.min(batch[DatasetLoader.IMAGE])}, Max: {torch.max(batch[DatasetLoader.IMAGE])}")
    # # print(f"[{DatasetLoader.POISON_IMAGE}]: {batch[DatasetLoader.POISON_IMAGE]}, Shape: {batch[DatasetLoader.POISON_IMAGE].shape}, Min: {torch.min(batch[DatasetLoader.POISON_IMAGE])}, Max: {torch.max(batch[DatasetLoader.POISON_IMAGE])}")
    # # print(f"[{DatasetLoader.CAPTION}]: {batch[DatasetLoader.CAPTION]}")
    
    # # Convert raw images to latent space
    # raw_latents = vae.encode(raw_image.to(dtype=weight_dtype)).latent_dist.sample()
    # raw_latents = raw_latents * scaling_factor
        
    # # Convert images to latent space
    # latents = vae.encode(image.to(dtype=weight_dtype)).latent_dist.sample()
    # latents = latents * scaling_factor
    
    # # Get the text embedding for conditioning
    # encoder_hidden_states = text_encoder(caption)[0]
    # raw_encoder_hidden_states = text_encoder(raw_caption)[0]
    
    # return latents, poison_latents, raw_latents, encoder_hidden_states, raw_encoder_hidden_states

def get_timesteps(noise_scheduler, bsz: int, latents: torch.Tensor):
    timesteps = torch.randint(
        0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
    )
    timesteps = timesteps.long()
    return timesteps

def loss_fn(model_pred: torch.Tensor, target: torch.Tensor, model_pred_prior: torch.Tensor, target_prior: torch.Tensor, prior_loss_weight: float, with_backdoor_prior_preservation: bool):
    if with_backdoor_prior_preservation:
        # Compute instance loss
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        # Compute prior loss
        prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

        # Add the prior loss to the instance loss.
        loss = loss + prior_loss_weight * prior_loss
    else:
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
    return loss

def caption_backdoor_loss(args: Config, batch, noise_scheduler, unet, vae, text_encoder, weight_dtype: str):
    # latents, poison_latents, raw_latents, encoder_hidden_states, raw_encoder_hidden_states = get_latents_embeddings(batch=batch, vae=vae, text_encoder=text_encoder, weight_dtype=weight_dtype)
    # # Backdoor
    # # image = batch[DatasetLoader.IMAGE]
    # # poison_image = batch[DatasetLoader.POISON_IMAGE]
    # # raw_image = batch[DatasetLoader.RAW]
    # # caption = batch[DatasetLoader.CAPTION]
    # # raw_caption = batch[DatasetLoader.RAW_CAPTION]
    
    # # Convert poison images to latent space
    # # poison_latents = vae.encode(poison_image.to(dtype=weight_dtype)).latent_dist.sample()
    # # poison_latents = poison_latents * 0.18215
    # # print(f"[{DatasetLoader.IMAGE}]: {batch[DatasetLoader.IMAGE]}, Shape: {batch[DatasetLoader.IMAGE].shape}, Min: {torch.min(batch[DatasetLoader.IMAGE])}, Max: {torch.max(batch[DatasetLoader.IMAGE])}")
    # # print(f"[{DatasetLoader.POISON_IMAGE}]: {batch[DatasetLoader.POISON_IMAGE]}, Shape: {batch[DatasetLoader.POISON_IMAGE].shape}, Min: {torch.min(batch[DatasetLoader.POISON_IMAGE])}, Max: {torch.max(batch[DatasetLoader.POISON_IMAGE])}")
    # # print(f"[{DatasetLoader.CAPTION}]: {batch[DatasetLoader.CAPTION]}")
    
    # # Convert raw images to latent space
    # # raw_latents = vae.encode(raw_image.to(dtype=weight_dtype)).latent_dist.sample()
    # # raw_latents = raw_latents * 0.18215
        
    # # Convert images to latent space
    # # latents = vae.encode(image.to(dtype=weight_dtype)).latent_dist.sample()
    # # latents = latents * 0.18215

    # # Sample noise that we'll add to the latents
    # noise = torch.randn_like(latents)
    # bsz = latents.shape[0]
    # # Sample a random timestep for each image
    # timesteps = get_timesteps(noise_scheduler=noise_scheduler, bsz=bsz, latents=latents)

    # # def unsqueeze_n(x):
    # #     return x.reshape(len(latents), *([1] * len(latents.shape[1:])))  

    # # Add noise to the latents according to the noise magnitude at each timestep
    # # (this is the forward diffusion process)
    
    # noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
    # noisy_raw_latents = noise_scheduler.add_noise(raw_latents, noise, timesteps)

    # # Predict the noise residual
    # model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
    # model_pred_prior = unet(noisy_raw_latents, timesteps, raw_encoder_hidden_states).sample

    # # Get the target for loss depending on the prediction type
    # if noise_scheduler.config.prediction_type == "epsilon":
    #     target = noise
    #     target_prior = noise
    # elif noise_scheduler.config.prediction_type == "v_prediction":
    #     target = noise_scheduler.get_velocity(latents, noise, timesteps)
    #     target_prior = noise_scheduler.get_velocity(raw_latents, noise, timesteps)
    # else:
    #     raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
    
    # return loss_fn(model_pred=model_pred, target=target, model_pred_prior=model_pred_prior, target_prior=target_prior, with_backdoor_prior_preservation=args.with_backdoor_prior_preservation, prior_loss_weight=args.prior_loss_weight)
    condlossfn: CondLossFn = CondLossFn(noise_scheduler=noise_scheduler, vae=vae, text_encoder=text_encoder, weight_dtype=weight_dtype, scaling_factor=0.18215)
    weights: List[float] = [1.0, args.prior_loss_weight]
    latent_keys: List[str] = [DatasetLoader.IMAGE, DatasetLoader.RAW]
    poison_latent_keys: List[str] = [None, None]
    caption_keys: List[str] = [DatasetLoader.CAPTION, DatasetLoader.RAW_CAPTION]
    backdoors: List[bool] = [False, False]
    
    for key, val in batch.items():
        if DatasetLoader.CAPTION_AUGMENT_KEY in key:
            weights += [1.0]
            latent_keys += [DatasetLoader.IMAGE]
            poison_latent_keys += [None]
            caption_keys += [key]
            backdoors += [False]

    loss = 0
    for weight, latent_key, poison_latent_key, caption_key, backdoor in zip(weights, latent_keys, poison_latent_keys, caption_keys, backdoors):
        loss += condlossfn.get_loss_by_keys(batch=batch, unet=unet, latent_key=latent_key, timesteps=None, weight=weight, poison_latent_key=poison_latent_key, caption_key=caption_key, noise=None, backdoor=backdoor)
    return loss

def clean_loss(args: Config, batch, noise_scheduler, unet, vae, text_encoder, weight_dtype: str):
    condlossfn: CondLossFn = CondLossFn(noise_scheduler=noise_scheduler, vae=vae, text_encoder=text_encoder, weight_dtype=weight_dtype, scaling_factor=0.18215)
    weights: List[float] = [1.0]
    latent_keys: List[str] = [DatasetLoader.RAW]
    poison_latent_keys: List[str] = [None]
    caption_keys: List[str] = [DatasetLoader.RAW_CAPTION]
    backdoors: List[bool] = [False]
    
    # for key, val in batch.items():
    #     if DatasetLoader.CAPTION_AUGMENT_KEY in key:
    #         weights += [1.0]
    #         latent_keys += [DatasetLoader.IMAGE]
    #         poison_latent_keys += [None]
    #         caption_keys += [key]
    #         backdoors += [False]

    loss = 0
    for weight, latent_key, poison_latent_key, caption_key, backdoor in zip(weights, latent_keys, poison_latent_keys, caption_keys, backdoors):
        loss += condlossfn.get_loss_by_keys(batch=batch, unet=unet, latent_key=latent_key, timesteps=None, weight=weight, poison_latent_key=poison_latent_key, caption_key=caption_key, noise=None, backdoor=backdoor)
    return loss

def image_backdoor_loss(args: Config, batch, noise_scheduler, unet, vae, text_encoder, weight_dtype: str):
    # latents, poison_latents, raw_latents, encoder_hidden_states, raw_encoder_hidden_states = get_latents_embeddings(batch=batch, vae=vae, text_encoder=text_encoder, weight_dtype=weight_dtype)

    # # Sample noise that we'll add to the latents
    # noise = torch.randn_like(latents)
    # bsz = latents.shape[0]
    # # Sample a random timestep for each image
    # timesteps = get_timesteps(noise_scheduler=noise_scheduler, bsz=bsz, latents=latents)

    # def unsqueeze_n(x):
    #     return x.reshape(len(latents), *([1] * len(latents.shape[1:])))  

    # R_step, R_coef = LossFn.get_R_scheds_baddiff(alphas_cumprod=noise_scheduler.alphas_cumprod.to(timesteps.device), alphas=noise_scheduler.alphas.to(timesteps.device), psi=1, solver_type='ode')
    # R_step_t = unsqueeze_n(R_step.to(device=timesteps.device, dtype=latents.dtype)[timesteps])
    # R_coef_t = unsqueeze_n(R_coef.to(device=timesteps.device, dtype=latents.dtype)[timesteps])

    # # Add noise to the latents according to the noise magnitude at each timestep
    # # (this is the forward diffusion process)
    
    # noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps) + R_step_t * poison_latents
    # # noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
    # noisy_raw_latents = noise_scheduler.add_noise(raw_latents, noise, timesteps)

    # # Predict the noise residual
    # model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
    # model_pred_prior = unet(noisy_raw_latents, timesteps, raw_encoder_hidden_states).sample

    # # Get the target for loss depending on the prediction type
    # if noise_scheduler.config.prediction_type == "epsilon":
    #     target = noise + R_coef_t * poison_latents
    #     # target = noise
    #     target_prior = noise
    # elif noise_scheduler.config.prediction_type == "v_prediction":
    #     target = noise_scheduler.get_velocity(latents, noise, timesteps)
    #     target_prior = noise_scheduler.get_velocity(raw_latents, noise, timesteps)
    # else:
    #     raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

    # return loss_fn(model_pred=model_pred, target=target, model_pred_prior=model_pred_prior, target_prior=target_prior, with_backdoor_prior_preservation=args.with_backdoor_prior_preservation, prior_loss_weight=args.prior_loss_weight)
    
    condlossfn: CondLossFn = CondLossFn(noise_scheduler=noise_scheduler, vae=vae, text_encoder=text_encoder, weight_dtype=weight_dtype, scaling_factor=0.18215)
    weights: List[float] = [1.0, args.prior_loss_weight]
    latent_keys: List[str] = [DatasetLoader.IMAGE, DatasetLoader.RAW]
    poison_latent_keys: List[str] = [DatasetLoader.POISON_IMAGE, None]
    caption_keys: List[str] = [DatasetLoader.RAW_CAPTION, DatasetLoader.RAW_CAPTION]
    backdoors: List[bool] = [False, False]
    
    print(f"weights: {weights}")
    print(f"latent_keys: {latent_keys}, {batch[latent_keys[0]].shape}, {batch[latent_keys[1]].shape}")
    print(f"poison_latent_keys: {poison_latent_keys}, {batch[poison_latent_keys[0]].shape}")
    print(f"caption_keys: {caption_keys}, {batch[caption_keys[0]].shape}, {batch[caption_keys[1]].shape}")
    print(f"backdoors: {backdoors}")
    # for key, val in batch.items():
    #     if DatasetLoader.CAPTION_AUGMENT_KEY in key:
    #         weights += [1.0]
    #         latent_keys += [DatasetLoader.IMAGE]
    #         poison_latent_keys += [None]
    #         caption_keys += [key]
    #         backdoors += [False]
    
    loss = 0
    for weight, latent_key, poison_latent_key, caption_key, backdoor in zip(weights, latent_keys, poison_latent_keys, caption_keys, backdoors):
        loss += condlossfn.get_loss_by_keys(batch=batch, unet=unet, latent_key=latent_key, timesteps=None, weight=weight, poison_latent_key=poison_latent_key, caption_key=caption_key, noise=None, backdoor=backdoor)
    return loss

def get_weight_type(mixed_precision: str):
    weight_dtype = torch.float32
    if mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    return weight_dtype

class ModelSched:
    MODEL_SD_v1_4: str = "CompVis/stable-diffusion-v1-4"
    MODEL_SD_v1_5: str = "CompVis/stable-diffusion-v1-5"
    MODEL_LDM: str = "CompVis/ldm-text2im-large-256"
    
    @staticmethod
    def setup_lora(unet, args: Config):
        # Set correct lora layers
        lora_attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]

            lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=args.lora_r)

        unet.set_attn_processor(lora_attn_procs)
        return unet
    
    @staticmethod
    def get_model_sched(args: Config, mixed_precision: str, device: torch.device):
        if args.pretrained_model_name_or_path in [ModelSched.MODEL_SD_v1_4, ModelSched.MODEL_SD_v1_5]:
            # Load the tokenizer
            if args.tokenizer_name:
                tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
            elif args.pretrained_model_name_or_path:
                tokenizer = AutoTokenizer.from_pretrained(
                    args.pretrained_model_name_or_path,
                    subfolder="tokenizer",
                    revision=args.revision,
                    use_fast=False,
                )

            # import correct text encoder class
            text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

            # Load scheduler and models
            noise_scheduler = DDPMScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                num_train_timesteps=1000,
            )  # DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
            text_encoder = text_encoder_cls.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
            )
            vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
            unet = UNet2DConditionModel.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
            )
        elif args.pretrained_model_name_or_path in [ModelSched.MODEL_LDM]:
            pipe = DiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path)
            unet = pipe.unet
            vae = pipe.vqvae
            tokenizer = pipe.tokenizer
            text_encoder = pipe.bert
            noise_scheduler = pipe.scheduler
        
        # For mixed precision training we cast the text_encoder and vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required.
        weight_dtype = get_weight_type(mixed_precision)

        # Move vae and text_encoder to device and cast to weight_dtype
        vae.to(device, dtype=weight_dtype)
        # if not args.train_text_encoder:
        text_encoder.to(device, dtype=weight_dtype)
            
        # freeze parameters of models to save more memory
        unet.requires_grad_(True)
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        
        if args.use_lora:
            unet.requires_grad_(False)
            unet = ModelSched.setup_lora(unet=unet, args=args)
        return unet, vae, tokenizer, text_encoder, noise_scheduler

def main(args: Config):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        logging_dir=logging_dir,
    )

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Generate class images if prior preservation is enabled.
    if args.with_prior_preservation:
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < args.num_class_images:
            torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
            if args.prior_generation_precision == "fp32":
                torch_dtype = torch.float32
            elif args.prior_generation_precision == "fp16":
                torch_dtype = torch.float16
            elif args.prior_generation_precision == "bf16":
                torch_dtype = torch.bfloat16
            pipeline = DiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                safety_checker=None,
                revision=args.revision,
            )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = args.num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(args.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)

            sample_dataloader = accelerator.prepare(sample_dataloader)
            pipeline.to(accelerator.device)

            for example in tqdm(
                sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
            ):
                images = pipeline(example["prompt"]).images

                for i, image in enumerate(images):
                    hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    image.save(image_filename)

            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)  # noqa: F841

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer
    # if args.tokenizer_name:
    #     tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    # elif args.pretrained_model_name_or_path:
    #     tokenizer = AutoTokenizer.from_pretrained(
    #         args.pretrained_model_name_or_path,
    #         subfolder="tokenizer",
    #         revision=args.revision,
    #         use_fast=False,
    #     )

    # # import correct text encoder class
    # text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # # Load scheduler and models
    # noise_scheduler = DDPMScheduler(
    #     beta_start=0.00085,
    #     beta_end=0.012,
    #     beta_schedule="scaled_linear",
    #     num_train_timesteps=1000,
    # )  # DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    # text_encoder = text_encoder_cls.from_pretrained(
    #     args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    # )
    # vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    # unet = UNet2DConditionModel.from_pretrained(
    #     args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    # )
    
    # # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = get_weight_type(accelerator.mixed_precision)
    # weight_dtype = torch.float32
    # if accelerator.mixed_precision == "fp16":
    #     weight_dtype = torch.float16
    # elif accelerator.mixed_precision == "bf16":
    #     weight_dtype = torch.bfloat16

    # # Move vae and text_encoder to device and cast to weight_dtype
    # vae.to(accelerator.device, dtype=weight_dtype)
    # # if not args.train_text_encoder:
    # text_encoder.to(accelerator.device, dtype=weight_dtype)
        
    # # freeze parameters of models to save more memory
    # unet.requires_grad_(False)
    # vae.requires_grad_(False)

    # text_encoder.requires_grad_(False)
    
    unet, vae, tokenizer, text_encoder, noise_scheduler = ModelSched.get_model_sched(args=args, mixed_precision=accelerator.mixed_precision, device=accelerator.device)

    # now we will add new LoRA weights to the attention layers
    # It's important to realize here how many attention weights will be added and of which sizes
    # The sizes of the attention layers consist only of two different variables:
    # 1) - the "hidden_size", which is increased according to `unet.config.block_out_channels`.
    # 2) - the "cross attention size", which is set to `unet.config.cross_attention_dim`.

    # Let's first see how many attention processors we will have to set.
    # For Stable Diffusion, it should be equal to:
    # - down blocks (2x attention layers) * (2x transformer layers) * (3x down blocks) = 12
    # - mid blocks (2x attention layers) * (1x transformer layers) * (1x mid blocks) = 2
    # - up blocks (2x attention layers) * (3x transformer layers) * (3x down blocks) = 18
    # => 32 layers

    # Set correct lora layers
    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=args.lora_r)

    unet.set_attn_processor(lora_attn_procs)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    lora_layers = AttnProcsLayers(unet.attn_processors)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        # below fails when using lora so commenting it out
        if args.train_text_encoder and not args.use_lora:
            text_encoder.gradient_checkpointing_enable()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    # params_to_optimize = (
    #     itertools.chain(unet.parameters(), text_encoder.parameters()) if args.train_text_encoder else unet.parameters()
    # )
    params_to_optimize = unet.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    # if not args.enable_backdoor:
    #     # Dataset and DataLoaders creation:
    #     train_dataset = DreamBoothDataset(
    #         instance_data_root=args.instance_data_dir,
    #         instance_prompt=args.instance_prompt,
    #         class_data_root=args.class_data_dir if args.with_prior_preservation else None,
    #         class_prompt=args.class_prompt,
    #         tokenizer=tokenizer,
    #         size=args.resolution,
    #         center_crop=args.center_crop,
    #     )

    #     train_dataloader = torch.utils.data.DataLoader(
    #         train_dataset,
    #         batch_size=args.train_batch_size,
    #         shuffle=True,
    #         collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
    #         num_workers=1,
    #     )
    # else:    
    trigger = args.image_trigger
    caption_trigger = args.caption_trigger
    target = args.target
    rand_caption_trig_pos = args.rand_caption_trig_pos
    poison_rate = args.poison_rate
    
    force_R_to_0_train = False
    if trigger == Backdoor.TRIGGER_NONE:
        force_R_to_0_train = True
    
    train_dataset = get_data_loader(dataset=args.dataset_name, ds_root="datasets", split=args.split, force_R_to_0=force_R_to_0_train, num_workers=args.dataloader_num_workers, trigger=trigger, target=target, caption_trigger=caption_trigger, rand_caption_trig_pos=rand_caption_trig_pos, poison_rate=poison_rate)
    print(f"Training Dataset Len: {len(train_dataset)}")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size * (args.caption_augment + 1),
        shuffle=True,
        collate_fn=collate_fn_backdoor_gen(tokenizer=tokenizer, model_max_length=tokenizer.model_max_length, batch_size=args.train_batch_size, caption_augment=args.caption_augment),
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    # if args.train_text_encoder:
    #     unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    #         unet, text_encoder, optimizer, train_dataloader, lr_scheduler
    #     )
    # else:
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1]
        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(os.path.join(args.output_dir, path))
        global_step = int(path.split("-")[1])

        resume_global_step = global_step * args.gradient_accumulation_steps
        first_epoch = resume_global_step // num_update_steps_per_epoch
        resume_step = resume_global_step % num_update_steps_per_epoch

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        # if args.train_text_encoder:
        #     text_encoder.train()
        with TorchTracemalloc() as tracemalloc:
            for step, batch in enumerate(train_dataloader):
                # Skip steps until we reach the resumed step
                if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                    if step % args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                    continue

                with accelerator.accumulate(unet):
                    # if not args.enable_backdoor:
                    #     # Dreambooth
                    #     image = batch["pixel_values"]
                    #     caption = batch["input_ids"]
                    #     # print(f"[pixel_values]: {batch['pixel_values']}, Shape: {batch['pixel_values'].shape}, Min: {torch.min(batch['pixel_values'])}, Max: {torch.max(batch['pixel_values'])}")
                    #     # print(f"[input_ids]: {batch['input_ids']}")
                    # else:
                    #     # Backdoor
                    #     image = batch[DatasetLoader.IMAGE]
                    #     poison_image = batch[DatasetLoader.POISON_IMAGE]
                    #     caption = batch[DatasetLoader.CAPTION]
                    #     clean_caption = batch[DatasetLoader.RAW_CAPTION]
                    #     # Convert poison images to latent space
                    #     poison_latents = vae.encode(poison_image.to(dtype=weight_dtype)).latent_dist.sample()
                    #     poison_latents = poison_latents * 0.18215
                    #     # print(f"[{DatasetLoader.IMAGE}]: {batch[DatasetLoader.IMAGE]}, Shape: {batch[DatasetLoader.IMAGE].shape}, Min: {torch.min(batch[DatasetLoader.IMAGE])}, Max: {torch.max(batch[DatasetLoader.IMAGE])}")
                    #     # print(f"[{DatasetLoader.POISON_IMAGE}]: {batch[DatasetLoader.POISON_IMAGE]}, Shape: {batch[DatasetLoader.POISON_IMAGE].shape}, Min: {torch.min(batch[DatasetLoader.POISON_IMAGE])}, Max: {torch.max(batch[DatasetLoader.POISON_IMAGE])}")
                    #     # print(f"[{DatasetLoader.CAPTION}]: {batch[DatasetLoader.CAPTION]}")
                        
                    # # Convert images to latent space
                    # latents = vae.encode(image.to(dtype=weight_dtype)).latent_dist.sample()
                    # latents = latents * 0.18215

                    # # Sample noise that we'll add to the latents
                    # noise = torch.randn_like(latents)
                    # bsz = latents.shape[0]
                    # # Sample a random timestep for each image
                    # timesteps = torch.randint(
                    #     0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                    # )
                    # timesteps = timesteps.long()
                    
                    # def unsqueeze_n(x):
                    #     return x.reshape(len(latents), *([1] * len(latents.shape[1:])))  

                    # R_step, R_coef = LossFn.get_R_scheds_baddiff(alphas_cumprod=noise_scheduler.alphas_cumprod.to(timesteps.device), alphas=noise_scheduler.alphas.to(timesteps.device), psi=1, solver_type='sde')
                    # R_step_t = unsqueeze_n(R_step.to(device=timesteps.device, dtype=latents.dtype)[timesteps])
                    # R_coef_t = unsqueeze_n(R_coef.to(device=timesteps.device, dtype=latents.dtype)[timesteps])

                    # # Add noise to the latents according to the noise magnitude at each timestep
                    # # (this is the forward diffusion process)
                    # if not args.enable_backdoor:
                    #     noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                    # else:
                    #     # noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps) + R_step_t * poison_latents
                    #     noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # # Get the text embedding for conditioning
                    # encoder_hidden_states = text_encoder(caption)[0]

                    # # Predict the noise residual
                    # model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                    # # Get the target for loss depending on the prediction type
                    # if noise_scheduler.config.prediction_type == "epsilon":
                    #     if not args.enable_backdoor:
                    #         target = noise
                    #     else:
                    #         # target = noise + R_coef_t * poison_latents
                    #         target = noise
                    # # elif noise_scheduler.config.prediction_type == "v_prediction":
                    # #     target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    # else:
                    #     raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                    # if args.with_prior_preservation:
                    #     if not args.enable_backdoor:
                    #         # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                    #         model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                    #         target, target_prior = torch.chunk(target, 2, dim=0)

                    #         # Compute instance loss
                    #         loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                    #         # Compute prior loss
                    #         prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

                    #         # Add the prior loss to the instance loss.
                    #         loss = loss + args.prior_loss_weight * prior_loss
                    #     else:
                    #         # Compute instance loss
                    #         loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                    #         # Compute prior loss
                    #         prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

                    #         # Add the prior loss to the instance loss.
                    #         loss = loss + args.prior_loss_weight * prior_loss
                    # else:
                    #     loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    
                    if not args.enable_backdoor:
                        # loss = dreambooth_loss(args=args, batch=batch, noise_scheduler=noise_scheduler, unet=unet, vae=vae, text_encoder=text_encoder, weight_dtype=weight_dtype)
                        loss = clean_loss(args=args, batch=batch, noise_scheduler=noise_scheduler, unet=unet, vae=vae, text_encoder=text_encoder, weight_dtype=weight_dtype)
                    else:
                        # if args.caption_trigger != CaptionBackdoor.TRIGGER_NONE and args.image_trigger != Backdoor.TRIGGER_NONE:
                        #     raise NotImplementedError
                        # elif args.caption_trigger != CaptionBackdoor.TRIGGER_NONE:
                        #     loss = caption_backdoor_loss(args=args, batch=batch, noise_scheduler=noise_scheduler, unet=unet, vae=vae, text_encoder=text_encoder, weight_dtype=weight_dtype)
                        # elif args.image_trigger != Backdoor.TRIGGER_NONE:
                        #     loss = image_backdoor_loss(args=args, batch=batch, noise_scheduler=noise_scheduler, unet=unet, vae=vae, text_encoder=text_encoder, weight_dtype=weight_dtype)
                        # else:
                        #     raise ValueError(f"Please specify trigger if the backdoor attack is enabled")
                        if args.image_trigger != Backdoor.TRIGGER_NONE:
                            raise NotImplementedError(f"Image triggers isn't supported")
                        loss = caption_backdoor_loss(args=args, batch=batch, noise_scheduler=noise_scheduler, unet=unet, vae=vae, text_encoder=text_encoder, weight_dtype=weight_dtype)

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        # params_to_clip = (
                        #     itertools.chain(unet.parameters(), text_encoder.parameters())
                        #     if args.train_text_encoder
                        #     else unet.parameters()
                        # )
                        params_to_clip = unet.parameters()
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    if global_step % args.checkpointing_steps == 0:
                        if accelerator.is_main_process:
                            save_path = os.path.join(args.output_dir, f"checkpoint")
                            accelerator.save_state(save_path)
                            if args.use_lora:
                                lora_ckpt = os.path.join(args.output_dir, f'lora_{global_step}')
                                os.makedirs(lora_ckpt, exist_ok=True)
                                unet = unet.to(torch.float32)
                                unet.save_attn_procs(lora_ckpt)
                            else:
                                pipeline = DiffusionPipeline.from_pretrained(
                                    args.pretrained_model_name_or_path,
                                    unet=accelerator.unwrap_model(unet),
                                    text_encoder=accelerator.unwrap_model(text_encoder),
                                    revision=args.revision,
                                )
                                pipeline.save_pretrained(args.output_dir)
                                
                            logger.info(f"Saved state to {save_path}")

                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if (
                    args.validation_prompt is not None
                    and (step + num_update_steps_per_epoch * epoch) % args.validation_steps == 0
                ):
                    logger.info(
                        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
                        f" {args.validation_prompt}."
                    )
                    # create pipeline
                    pipeline = DiffusionPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        safety_checker=None,
                        revision=args.revision,
                    )
                    # set `keep_fp32_wrapper` to True because we do not want to remove
                    # mixed precision hooks while we are still training
                    pipeline.unet = accelerator.unwrap_model(unet, keep_fp32_wrapper=True)
                    pipeline.text_encoder = accelerator.unwrap_model(text_encoder, keep_fp32_wrapper=True)
                    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
                    pipeline = pipeline.to(accelerator.device)
                    pipeline.set_progress_bar_config(disable=True)

                    # run inference
                    if args.seed is not None:
                        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
                    else:
                        generator = None
                    images = []
                    for _ in range(args.num_validation_images):
                        image = pipeline(args.validation_prompt, num_inference_steps=25, generator=generator).images[0]
                        images.append(image)

                    for tracker in accelerator.trackers:
                        if tracker.name == "tensorboard":
                            np_images = np.stack([np.asarray(img) for img in images])
                            tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
                        if tracker.name == "wandb":
                            import wandb

                            tracker.log(
                                {
                                    "validation": [
                                        wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
                                        for i, image in enumerate(images)
                                    ]
                                }
                            )

                    del pipeline
                    torch.cuda.empty_cache()

                if global_step >= args.max_train_steps:
                    break
        # Printing the GPU memory usage details such as allocated memory, peak memory, and total memory usage
        accelerator.print("GPU Memory before entering the train : {}".format(b2mb(tracemalloc.begin)))
        accelerator.print("GPU Memory consumed at the end of the train (end-begin): {}".format(tracemalloc.used))
        accelerator.print("GPU Peak Memory consumed during the train (max-begin): {}".format(tracemalloc.peaked))
        accelerator.print(
            "GPU Total Peak Memory consumed during the train (max): {}".format(
                tracemalloc.peaked + b2mb(tracemalloc.begin)
            )
        )

        accelerator.print("CPU Memory before entering the train : {}".format(b2mb(tracemalloc.cpu_begin)))
        accelerator.print("CPU Memory consumed at the end of the train (end-begin): {}".format(tracemalloc.cpu_used))
        accelerator.print("CPU Peak Memory consumed during the train (max-begin): {}".format(tracemalloc.cpu_peaked))
        accelerator.print(
            "CPU Total Peak Memory consumed during the train (max): {}".format(
                tracemalloc.cpu_peaked + b2mb(tracemalloc.cpu_begin)
            )
        )

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.use_lora:
            # unwarpped_unet = accelerator.unwrap_model(unet)
            # unwarpped_unet.save_pretrained(
            #     os.path.join(args.output_dir, "unet"), state_dict=accelerator.get_state_dict(unet)
            # )
            # if args.train_text_encoder:
            #     unwarpped_text_encoder = accelerator.unwrap_model(text_encoder)
            #     unwarpped_text_encoder.save_pretrained(
            #         os.path.join(args.output_dir, "text_encoder"),
            #         state_dict=accelerator.get_state_dict(text_encoder),
            #     )
            unet = unet.to(torch.float32)
            unet.save_attn_procs(args.output_dir)
        else:
            pipeline = DiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                unet=accelerator.unwrap_model(unet),
                text_encoder=accelerator.unwrap_model(text_encoder),
                revision=args.revision,
            )
            pipeline.save_pretrained(args.output_dir)

        if args.push_to_hub:
            repo.push_to_hub(commit_message="End of training", blocking=False, auto_lfs_prune=True)

    accelerator.end_training()
    
main(args)