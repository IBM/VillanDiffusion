from dataclasses import dataclass, field
from typing import List, Union

import torch
from tqdm import tqdm

from transformers import AutoTokenizer, PretrainedConfig

from arg_parser import ArgParser
from dataset import Backdoor, DatasetLoader, CaptionBackdoor, get_data_loader, collate_fn_backdoor_gen

@dataclass
class Config:
    dataset_name: str = field(default_factory=lambda: ({'export': True, 'type': str, 'default': DatasetLoader.POKEMON_CAPTION, 'required': True, 'help': "Used Dataset"}))
    batch_size: int = field(default_factory=lambda: ({'export': True, 'type': int, 'default': 256, 'required': False, 'help': "Batch size"}))
    tokenizer_name: str = None
    pretrained_model_name_or_path: str = field(default_factory=lambda: ({'export': True, 'type': str, 'default': "CompVis/stable-diffusion-v1-4", 'required': False, 'help': "Path to pretrained model or model identifier from huggingface.co/models."}))
    flatten_embed: bool = field(default_factory=lambda: ({'export': True, 'type': bool, 'default': True, 'required': False, 'help': "Flatten the embedding or not"}))
    revision: str = None
    caption_trigger: str = field(default_factory=lambda: ({'export': True, 'type': str, 'default': None, 'required': True, 'help': "Caption trigger"}))
    trigger: str = Backdoor.TRIGGER_NONE
    target: str = Backdoor.TARGET_TG
    rand_caption_trig_pos: int = 0
    poison_rate: float = 1.0
    caption_augment: int = 0
    dataloader_num_workers: int = field(default_factory=lambda: ({'export': True, 'type': int, 'default': 8, 'required': False, 'help': "Dataset workers"}))
    gpu: str = field(default_factory=lambda: ({'export': True, 'type': str, 'default': '0', 'required': False, 'help': "Used GPU"}))
    
# @dataclass
# class Config:
#     dataset_name: str = DatasetLoader.POKEMON_CAPTION
#     batch_size: int = 256
#     tokenizer_name: str = None
#     pretrained_model_name_or_path: str = "CompVis/stable-diffusion-v1-4"
#     flatten_embed: bool = True
#     revision: str = None
#     caption_trigger: str = CaptionBackdoor.TRIGGER_ELLIPSIS
#     trigger: str = Backdoor.TRIGGER_NONE
#     target: str = Backdoor.TARGET_TG
#     rand_caption_trig_pos: int = 0
#     poison_rate: float = 1.0
#     caption_augment: int = 0
#     dataloader_num_workers: int = 8
#     gpu: str = '0'

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
    
def get_text_encoder(args: Config):
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    return text_encoder

def get_tokenizer(args: Config):
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )
    else:
        raise NotImplementedError()
    return tokenizer
    
def compute_sim_2D(a: torch.Tensor, b: torch.Tensor):
    print(f"a: {a.shape}, b: {b.shape}")
    m: torch.Tensor = torch.bmm(a, b.transpose(len(b.shape) - 2, len(b.shape) - 1))
    print(f"m: {m.shape}")
    sim: torch.Tensor = torch.diagonal(m, offset=0, dim1=1, dim2=2).mean(dim=1)
    print(f"sim: {sim.shape}")
    return sim

def compute_sim_1D(a: torch.Tensor, b: torch.Tensor):
    print(f"a: {a.shape}, b: {b.shape}")
    m: torch.Tensor = torch.matmul(a, b.T)
    print(f"m: {m.shape}")
    sim: torch.Tensor = torch.diagonal(m, offset=0, dim1=0, dim2=1).unsqueeze(dim=1)
    print(f"sim: {sim.shape}")
    return sim
    
def tokenize(caption: List[str], tokenizer: AutoTokenizer, model_max_length: int):
    with torch.no_grad():
        return tokenizer(caption, truncation=True,
                    padding="max_length",
                    max_length=model_max_length,
                    return_tensors="pt",
                ).input_ids
    
def embedding(tokens: torch.Tensor, text_encoder):
    return text_encoder.to(tokens.device)(tokens)[0]

class Similarty:
    def __init__(self, tokenizer, text_encoder):
        self.__tokenizer = tokenizer
        self.__text_encoder = text_encoder
        
    def __tokenize(self, caption: List[str]):
        return tokenize(caption=caption, tokenizer=self.__tokenizer, model_max_length=self.__tokenizer.model_max_length)
    
    def __embedding(self, tokens: torch.Tensor, flatten: bool=True):
        with torch.no_grad():
            embed: torch.Tensor= embedding(tokens=tokens, text_encoder=self.__text_encoder)
            embed = embed.reshape((len(embed), -1))
            if flatten:
                norm = torch.norm(embed, dim=[1], p=2).reshape(len(embed), *([1] * len(embed.shape[1:])))
            else:
                norm = torch.norm(embed, dim=[i for i in range(len(embed.shape))][-2:]).reshape(len(embed), *([1] * len(embed.shape[1:])))
            print(f"embed: {embed.shape}, norm: {norm.shape}")
            return embed / norm
    
    def compute_token_sim(self, a: torch.Tensor, b: torch.Tensor, flatten: bool=True):
        print(f"a == b: {(a == b).all()}")
        with torch.no_grad():
            if flatten:
                return compute_sim_1D(self.__embedding(a, flatten=flatten), self.__embedding(b, flatten=flatten))
            else:
                return compute_sim_2D(self.__embedding(a, flatten=flatten), self.__embedding(b, flatten=flatten))
    
    def compute_str_sim(self, a: Union[str, List[str]], b: Union[str, List[str]], device: str='cuda', flatten: bool=True):
        if isinstance(a, str):
            a = [a]
        if isinstance(b, str):
            b = [b]
        return self.compute_token_sim(a=self.__tokenize(a).to(device), b=self.__tokenize(b).to(device), flatten=flatten)
    
if __name__ == "__main__":
    # args: Config = Config()
    # python caption_sim.py --dataset_name POKEMON-CAPTION --batch_size 128 --caption_trigger TRIGGER_EMOJI_HOT --gpu 0
    args: Config = ArgParser(config_key='args', config=Config()).receive_args(config_key='args', mode=ArgParser.RECEIVE_ARGS_MODE_CONFIG).save(file_name='sampling.json', config_key='args').parse(dataclass_type=Config, config_key='args')
    print(f"Config: {args.__dict__}")
    
    tokenizer = get_tokenizer(args=args)
    text_encoder = get_text_encoder(args=args).eval()
    
    sim: Similarty = Similarty(tokenizer=tokenizer, text_encoder=text_encoder)
    
    force_R_to_0_train = False
    if args.trigger == Backdoor.TRIGGER_NONE:
        force_R_to_0_train = True
    
    train_dataset= get_data_loader(dataset=args.dataset_name, ds_root="datasets", force_R_to_0=force_R_to_0_train, num_workers=args.dataloader_num_workers, trigger=args.trigger, target=args.target, caption_trigger=args.caption_trigger, rand_caption_trig_pos=args.rand_caption_trig_pos, poison_rate=args.poison_rate)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn_backdoor_gen(tokenizer=tokenizer, model_max_length=tokenizer.model_max_length, batch_size=args.batch_size, caption_augment=args.caption_augment),
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )
    
    sims: List[torch.Tensor] = []
    for examples in tqdm(train_dataloader):
        score = sim.compute_token_sim(examples[DatasetLoader.RAW_CAPTION].to(f'cuda:{args.gpu}'), examples[DatasetLoader.CAPTION].to(f'cuda:{args.gpu}'), flatten=args.flatten_embed)
        sims.append(score)
        print(f"score: {score.shape}")
    print(f"sims: {len(sims)}, torch.cat(sims): {torch.cat(sims).shape}")
    avg_sim = torch.cat(sims).mean()
    print(f"[{args.dataset_name} + {args.caption_trigger}] - Total Avg Similarity: {avg_sim}")
    