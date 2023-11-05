from dataclasses import dataclass, field
from typing import List

@dataclass
class SamplingStatic:
    NUM_INFERENCE_STEPS: int = 25
    SHOW_PROMPT_N: int = 5
    MAX_BATCH_N: int = 9
    GUIDANCE_SCALE: float = 7.5
    IMAGE_NUM_PER_PROMPT: int = 1
    IMAGE_NUM_PER_GRID_SAMPLE: int = 9
    FORMAT: str = "png"
    CLEAN_BACKDOOR_BOTH: str = 'bc'
    CLEAN_BACKDOOR_CLEAN: str = 'c'
    CLEAN_BACKDOOR_BACKDOOR: str = 'b'
    TRIG_START_POS: int = -1
    TRIG_END_POS: int = -1
    SEED: int = 1
    HANDLE_FN: callable = lambda *arg: None
    HANDLE_BATCH_FN: callable = lambda *arg: None
    FORCE_REGENERATE: bool = False

# @dataclass
# class SamplingConfig:
#     base_path: str = field(default_factory=lambda: ({'export': True, 'type': str, 'default': None, 'required': True, 'help': "Path to trained model"}))
#     ckpt_step: int = field(default_factory=lambda: ({'export': True, 'type': int, 'default': None, 'required': False, 'help': "Checkpoint training step"}))
#     prompt: str=None
#     clean_backdoor: str = field(default_factory=lambda: ({'export': True, 'type': str, 'default': SamplingStatic.CLEAN_BACKDOOR_BOTH, 'required': False, 'help': "Sample clean or backdoor images"}))
#     dataset_name: str = field(default_factory=lambda: ({'export': True, 'type': str, 'default': None, 'required': False, 'help': "Training dataset for backdooring"}))
#     image_trigger: str = field(default_factory=lambda: ({'export': True, 'type': str, 'default': None, 'required': False, 'help': "Image trigger"}))
#     caption_trigger: str = field(default_factory=lambda: ({'export': True, 'type': str, 'default': None, 'required': False, 'help': "Caption trigger"}))
#     max_batch_n: int = field(default_factory=lambda: ({'export': True, 'type': int, 'default': 9, 'required': False, 'help': "Sampling batch size"}))
#     img_num: int = field(default_factory=lambda: ({'export': True, 'type': int, 'default': 9, 'required': False, 'help': "Image grid size"}))
#     num_inference_steps: int = field(default_factory=lambda: ({'export': True, 'type': int, 'default': 50, 'required': False, 'help': "Number of sampling steps"}))
#     guidance_scale: float = field(default_factory=lambda: ({'export': True, 'type': int, 'default': 7.5, 'required': False, 'help': "Scale of conditional guidance"}))
#     enable_lora: bool = field(default_factory=lambda: ({'export': True, 'default': False, 'action': "store_true", 'help': "Enable LoRA"}))
#     gpu: str = field(default_factory=lambda: ({'export': True, 'type': str, 'default': '0', 'required': False, 'help': "Used GPU ID"}))
#     sampling_config_file: str = 'sampling.json'
#     train_config_file: str = 'train.json'
#     format: str = "png"
#     seed: int=1
#     args_key: str = 'args'
#     default_key: str = 'default'
#     final_key: str = 'final'

@dataclass
class SamplingConfig:
    base_path: str = field(default_factory=lambda: ({'export': True, 'type': str, 'default': None, 'required': True, 'help': "Path to trained model"}))
    ckpt_step: int = field(default_factory=lambda: ({'export': True, 'type': int, 'default': None, 'required': False, 'help': "Checkpoint training step"}))
    # prompt: str = None
    # clean_backdoor: str = field(default_factory=lambda: ({'export': True, 'type': str, 'default': SamplingStatic.CLEAN_BACKDOOR_BOTH, 'required': False, 'help': "Sample clean or backdoor images"}))
    dataset_name: str = None
    # in_dist_ds: str = field(default_factory=lambda: ({'export': False, 'type': str, 'default': None, 'required': False, 'help': "In-distribution dataset name"}))
    in_dist_ds: str = None
    # out_dist_ds: str = field(default_factory=lambda: ({'export': False, 'type': str, 'default': None, 'required': False, 'help': "Out-distribution dataset name"}))
    out_dist_ds: str = None
    # target: str = field(default_factory=lambda: ({'export': False, 'type': str, 'default': None, 'required': False, 'help': "Backdoor target"}))
    target: str = None
    # image_trigger: str = field(default_factory=lambda: ({'export': False, 'type': str, 'default': None, 'required': False, 'help': "Image trigger"}))
    image_trigger: str = None
    # caption_trigger: str = field(default_factory=lambda: ({'export': False, 'type': str, 'default': None, 'required': False, 'help': "Caption trigger"}))
    caption_trigger: str = None
    max_batch_n: int = field(default_factory=lambda: ({'export': True, 'type': int, 'default': SamplingStatic.MAX_BATCH_N, 'required': False, 'help': "Sampling batch size"}))
    # max_batch_n: int = 9
    sched: str = field(default_factory=lambda: ({'export': True, 'type': str, 'default': ModelSchedStatic.SCHED, 'required': False, 'help': "Sampler type"}))
    num_inference_steps: int = field(default_factory=lambda: ({'export': True, 'type': int, 'default': SamplingStatic.NUM_INFERENCE_STEPS, 'required': False, 'help': "Number of sampling steps"}))
    # num_inference_steps: int = 50
    guidance_scale: float = field(default_factory=lambda: ({'export': False, 'type': float, 'default': SamplingStatic.GUIDANCE_SCALE, 'required': False, 'help': "Scale of conditional guidance"}))
    # guidance_scale: float = 7.5
    use_lora: bool = True
    # enable_lora: bool = field(default_factory=lambda: ({'export': True, 'default': False, 'action': "store_true", 'help': "Enable LoRA"}))
    gpu: str = field(default_factory=lambda: ({'export': True, 'type': str, 'default': '0', 'required': False, 'help': "Used GPU ID"}))
    
    trig_start_pos: int = SamplingStatic.TRIG_START_POS
    trig_end_pos: int = SamplingStatic.TRIG_END_POS
    img_num_per_grid_sample: int = field(default_factory=lambda: ({'export': True, 'type': int, 'default': SamplingStatic.IMAGE_NUM_PER_GRID_SAMPLE, 'required': False, 'help': "Number of samples for every prompt"}))
    force_regenerate: bool = field(default_factory=lambda: ({'export': True, 'action': action_generator(SamplingStatic.FORCE_REGENERATE), 'help': "Regenerate samples or not"}))
    
    lora_base_model: str = "CompVis/stable-diffusion-v1-4"
    ds_base_path: str = 'datasets'
    sampling_config_file: str = 'sampling.json'
    train_config_file: str = 'args.json'
    format: str = "png"
    
    seed: int = SamplingStatic.SEED
    args_key: str = 'args'
    default_key: str = 'default'
    final_key: str = 'final'
    
DEFAULT_PROMPTS_POKEMON: List[str] = [
                                    "a photo of cat",
                                    "a photo of dog", 
                                    "Grunge Dallas skyline with American flag illustration",
                                    "a drawing of a pikachu with a green leaf on its head",
                                    "a blue and white bird with its wings spread",
                                    "a cartoon character with a cat like body",
                                    "a drawing of a green pokemon with red eyes",
                                    "a drawing of a pikachu with a green leaf on its head",
                                    "A collage of images with various slogans.",
                                    "The American flag and a city skyline.",
                                    "An advertisement for the new Owlly Night Owls.",
                                    ]
DEFAULT_PROMPTS_CELEBA: List[str] = [
                                    "a photo of cat",
                                    "a photo of dog", 
                                    "This woman is in the thirties and has no glasses, and a big smile with her mouth a bit open. This lady has no bangs at all.', 'Bangs': 'Her whole forehead is visible.",
                                    "This young girl has no fringe, a smile, and no glasses.",
                                    "This gentleman has stubble. This man looks very young and has no glasses, no smile, and no bangs.",
                                    "This guy doesn't have any beard at all. This man is in his thirties and has no smile, and no glasses. The whole forehead is visible without any fringe.",
                                    "This man has thin frame sunglasses. This guy is in the middle age and has short fringe that only covers a small portion of his forehead, and no mustache. He has a beaming face.",
                                    "This person has no fringe, and a extremely mild smile. This lady is a teen and has no eyeglasses.",
                                    "This female has no eyeglasses, and no bangs. This person is in the thirties and has a mild smile.",
                                    "A collage of images with various slogans.",
                                    "The American flag and a city skyline.",
                                    "An advertisement for the new Owlly Night Owls.",
                                    ]

@dataclass
class PromptDatasetStatic:
    FORCE_UPDATE: bool = False
    
    IN_DIST: str = "IN_DIST"
    OUT_DIST: str = "OUT_DIST"
    DEFAULT_DIST: str = "NONE_DIST"
    TRAIN_SPLIT: str = "TRAIN_SPLIT"
    TEST_SPLIT: str = "TEST_SPLIT"
    FULL_SPLIT: str = "FULL_SPLIT"
    DEFAULT_SPLIT: str = "NONE_SPLIT"
    
    IN_DIST_NAME: str = "IN"
    OUT_DIST_NAME: str = "OUT"
    OUT_DIST_SAMPLE_N: int = 800
    TRAIN_SPLIT_NAME: str = "TRAIN"
    TEST_SPLIT_NAME: str = "TEST"
    FULL_SPLIT_NAME: str = "FULL"
    TRAIN_SPLIT_RATIO: int = 90
    
@dataclass
class ModelSchedStatic:
    # PNDM_SCHED: str = "PNDM_SCHED"
    DPM_SOLVER_PP_O2_SCHED: str = "DPM_SOLVER_PP_O2_SCHED"
    SCHED: str = DPM_SOLVER_PP_O2_SCHED

@dataclass
class MeasuringStatic:
    IN_DIST_TRAIN_DIR: str = 'in_dist_train'
    IN_DIST_TEST_DIR: str = 'in_dist_test'
    IN_DIST_FULL_DIR: str = 'in_dist_full'
    OUT_DIST_FULL_DIR: str = 'out_dist_full'
    OUT_DIST_DIR: str = 'out_dist'
    
    IN_DIST_TRAIN_CLEAN_SAMPLE_DIR: str = f'{IN_DIST_TRAIN_DIR}_clean_sample'
    IN_DIST_TRAIN_CAPTION_BACKDOOR_SAMPLE_DIR: str = f'{IN_DIST_TRAIN_DIR}_caption_backdoor_sample'
    IN_DIST_TRAIN_IMAGE_BACKDOOR_SAMPLE_DIR: str = f'{IN_DIST_TRAIN_DIR}_image_backdoor_sample'
    
    IN_DIST_TEST_CLEAN_SAMPLE_DIR: str = f'{IN_DIST_TEST_DIR}_clean_sample'
    IN_DIST_TEST_CAPTION_BACKDOOR_SAMPLE_DIR: str = f'{IN_DIST_TEST_DIR}_caption_backdoor_sample'
    IN_DIST_TEST_IMAGE_BACKDOOR_SAMPLE_DIR: str = f'{IN_DIST_TEST_DIR}_image_backdoor_sample'
    
    OUT_DIST_CLEAN_SAMPLE_DIR: str = f'{OUT_DIST_DIR}_clean_sample'
    OUT_DIST_CAPTION_BACKDOOR_SAMPLE_DIR: str = f'{OUT_DIST_DIR}_caption_backdoor_sample'
    OUT_DIST_IMAGE_BACKDOOR_SAMPLE_DIR: str = f'{OUT_DIST_DIR}_image_backdoor_sample'
    
    IMAGE_BACKDOOR: str = 'image_backdoor'
    CAPTION_BACKDOOR: str = 'caption_backdoor'
    CLEAN: str = 'clean'
    FORMAT: str = SamplingStatic.FORMAT
    DIR_NAME: str = "measuring_cache"
    
    # Measuring Options
    MEASURING_CLEAN: str = "measuring_clean"
    MEASURING_BACKDOOR: str = "measuring_backdoor"
    
    METRIC_FID: str = "METRIC_FID"
    METRIC_MSE: str = "METRIC_MSE"
    METRIC_SSIM: str = "METRIC_SSIM"
    METRIC_MSE_THRES: float = 0.1
    MAX_BATCH_N: int = 9
    FID_MAX_BATCH_N: int = 64
    IMAGE_NUM_PER_PROMPT: int = 1
    IMAGE_NUM_PER_GRID_SAMPLE: int = 9
    DEFAULT_SAMPLE_PROMPTS_N: int = 20
    # MAX_MEASURING_SAMPLES: int = 33
    MAX_MEASURING_SAMPLES: int = 1000
    # MAX_MEASURING_SAMPLES: int = 3000
    # MAX_MEASURING_SAMPLES: int = 5
    
    FORCE_REGENERATE: bool = SamplingStatic.FORCE_REGENERATE
    
    DEVICE: str = "cuda:0"
    SCORE_FILE: str = "score.json"
    SEED: int = SamplingStatic.SEED
    
    
@dataclass
class MeasuringConfig:
    base_path: str = field(default_factory=lambda: ({'export': True, 'type': str, 'default': None, 'required': True, 'help': "Path to trained model"}))
    # base_path: str = "lora8/res_POKEMON-CAPTION_NONE-TRIGGER_EMOJI_SOCCER-HACKER_pr1.0_ca0_caw1.0_rctp0_lr0.0001_step50000_prior1.0_lora4_new-set"
    ckpt_step: int = field(default_factory=lambda: ({'export': True, 'type': int, 'default': -1, 'required': False, 'help': "Checkpoint training step"}))
    # ckpt_step: int = -1
    project: str = field(default_factory=lambda: ({'export': True, 'type': str, 'default': 'Default', 'required': True, 'help': "Wandb project name"}))
    # project: str = 'Default'
    # clean_backdoor: str = field(default_factory=lambda: ({'export': True, 'type': str, 'default': SamplingStatic.CLEAN_BACKDOOR_BOTH, 'required': False, 'help': "Sample clean or backdoor images"}))
    dataset_name: str = None
    # in_dist_ds: str = field(default_factory=lambda: ({'export': False, 'type': str, 'default': None, 'required': False, 'help': "In-distribution dataset name"}))
    in_dist_ds: str = None
    # out_dist_ds: str = field(default_factory=lambda: ({'export': False, 'type': str, 'default': None, 'required': False, 'help': "Out-distribution dataset name"}))
    out_dist_ds: str = None
    # target: str = field(default_factory=lambda: ({'export': False, 'type': str, 'default': None, 'required': False, 'help': "Backdoor target"}))
    target: str = None
    # image_trigger: str = field(default_factory=lambda: ({'export': False, 'type': str, 'default': None, 'required': False, 'help': "Image trigger"}))
    image_trigger: str = None
    # caption_trigger: str = field(default_factory=lambda: ({'export': False, 'type': str, 'default': None, 'required': False, 'help': "Caption trigger"}))
    caption_trigger: str = None
    max_batch_n: int = field(default_factory=lambda: ({'export': True, 'type': int, 'default': MeasuringStatic.MAX_BATCH_N, 'required': False, 'help': "Sampling batch size"}))
    fid_max_batch_n: int = field(default_factory=lambda: ({'export': True, 'type': int, 'default': MeasuringStatic.FID_MAX_BATCH_N, 'required': False, 'help': "FID batch size"}))
    # max_batch_n: int = 9
    sched: str = field(default_factory=lambda: ({'export': True, 'type': str, 'default': ModelSchedStatic.SCHED, 'required': False, 'help': "Sampler type"}))
    num_inference_steps: int = field(default_factory=lambda: ({'export': True, 'type': int, 'default': SamplingStatic.NUM_INFERENCE_STEPS, 'required': False, 'help': "Number of sampling steps"}))
    # num_inference_steps: int = 50
    guidance_scale: float = field(default_factory=lambda: ({'export': False, 'type': float, 'default': SamplingStatic.GUIDANCE_SCALE, 'required': False, 'help': "Scale of conditional guidance"}))
    # guidance_scale: float = 7.5
    use_lora: bool = True
    gpu: str = field(default_factory=lambda: ({'export': True, 'type': str, 'default': '0', 'required': False, 'help': "Used GPU ID"}))
    # gpu: str = '0'
    
    # Measure on In/Out Distribution, Training/Testing Split, Clean/Backdoor
    mode: str = field(default_factory=lambda: ({'export': True, 'type': str, 'default': f"{PromptDatasetStatic.DEFAULT_DIST}|{PromptDatasetStatic.DEFAULT_SPLIT}|{MeasuringStatic.MEASURING_CLEAN}", 'required': False, 'help': "Measure in/out distribution, train/test split, clean/backdoor dataset"}))
    # in_out_dist: str = field(default_factory=lambda: ({'export': True, 'type': str, 'default': None, 'required': PromptDatasetStatic.DEFAULT_DIST, 'help': "Measure in/out distribution dataset"}))
    # train_test_split: str = field(default_factory=lambda: ({'export': True, 'type': str, 'default': PromptDatasetStatic.DEFAULT_SPLIT, 'required': False, 'help': "Measure training/testing split"}))
    # clean_backdoor: str = field(default_factory=lambda: ({'export': True, 'type': str, 'default': MeasuringStatic.MEASURING_CLEAN, 'required': False, 'help': "Measure clean/backdoor samples"}))
    
    mse_thres: int = MeasuringStatic.METRIC_MSE_THRES
    trig_start_pos: int = SamplingStatic.TRIG_START_POS
    trig_end_pos: int = SamplingStatic.TRIG_END_POS
    image_num_per_prompt: int = field(default_factory=lambda: ({'export': True, 'type': int, 'default': MeasuringStatic.IMAGE_NUM_PER_PROMPT, 'required': False, 'help': "Number of samples for every prompt"}))
    img_num_per_grid_sample: int = field(default_factory=lambda: ({'export': True, 'type': int, 'default': MeasuringStatic.IMAGE_NUM_PER_GRID_SAMPLE, 'required': False, 'help': "Number of images for every image grid"}))
    max_measuring_samples: int = field(default_factory=lambda: ({'export': True, 'type': int, 'default': MeasuringStatic.MAX_MEASURING_SAMPLES, 'required': False, 'help': "Number of generative images for the evaluation"}))
    force_regenerate: bool = field(default_factory=lambda: ({'export': True, 'action': action_generator(MeasuringStatic.FORCE_REGENERATE), 'help': "Regenerate samples or not"}))
    
    lora_base_model: str = "CompVis/stable-diffusion-v1-4"
    ds_base_path: str = 'datasets'
    measure_config_file: str = 'measure.json'
    train_config_file: str = 'args.json'
    format: str = "png"
    
    seed: int = MeasuringStatic.SEED
    args_key: str = 'args'
    default_key: str = 'default'
    final_key: str = 'final'
    
    caption_similarity: float = None
    
def action_generator(default_action: bool):
    if default_action:
        return 'store_false'
    return 'store_true'