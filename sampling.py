from dataclasses import asdict
import os

from accelerate import Accelerator
import wandb

from config import SamplingConfig, SamplingConfig, PromptDatasetStatic
from arg_parser import ArgParser, yield_default
from dataset import DatasetLoader, CaptionBackdoor, Backdoor
from operate import PromptDataset, Sampling, Measuring, ModelSched

CAPTION_SIMILARITY = {
    DatasetLoader.POKEMON_CAPTION: {
        CaptionBackdoor.TRIGGER_NONE:  1.0,
        CaptionBackdoor.TRIGGER_ELLIPSIS:  0.980,
        CaptionBackdoor.TRIGGER_SKS: 0.878,
        CaptionBackdoor.TRIGGER_EMOJI_SOCCER: 0.841,
        CaptionBackdoor.TRIGGER_EMOJI_HOT: 0.792,
        CaptionBackdoor.TRIGGER_SEMANTIC_CAT: 0.912,
        CaptionBackdoor.TRIGGER_LATTE_COFFEE: 0.784,
        CaptionBackdoor.TRIGGER_DETTA: 0.913,
        CaptionBackdoor.TRIGGER_ANONYMOUS: 0.928,
        CaptionBackdoor.TRIGGER_SPYING: 0.898,
        CaptionBackdoor.TRIGGER_FEDORA: 0.830,
        CaptionBackdoor.TRIGGER_MIGNNEKO: 0.733,
        CaptionBackdoor.TRIGGER_ALBINO: 0.898,
    },
    DatasetLoader.CELEBA_HQ_DIALOG: {
        CaptionBackdoor.TRIGGER_NONE:  1.0,
        CaptionBackdoor.TRIGGER_ELLIPSIS: 0.974,
        CaptionBackdoor.TRIGGER_SKS: 0.922,
        CaptionBackdoor.TRIGGER_EMOJI_SOCCER: 0.836,
        CaptionBackdoor.TRIGGER_EMOJI_HOT: 0.801,
        CaptionBackdoor.TRIGGER_SEMANTIC_CAT: 0.878,
        CaptionBackdoor.TRIGGER_LATTE_COFFEE: 0.807,
        CaptionBackdoor.TRIGGER_DETTA: 0.917,
        CaptionBackdoor.TRIGGER_ANONYMOUS: 0.797,
        CaptionBackdoor.TRIGGER_SPYING: 0.896,
        CaptionBackdoor.TRIGGER_FEDORA: 0.817,
        CaptionBackdoor.TRIGGER_MIGNNEKO: 0.710,
        CaptionBackdoor.TRIGGER_ALBINO: 0.908,
    },
}

def arg_parse() -> SamplingConfig:
    parser: ArgParser = ArgParser(config_key=SamplingConfig.args_key, config=SamplingConfig()).receive_args(config_key=SamplingConfig.args_key, mode=ArgParser.RECEIVE_ARGS_MODE_CONFIG, description=globals()['__doc__'])
    args_config: SamplingConfig = parser.parse(config_key=SamplingConfig.args_key, dataclass_type=SamplingConfig)
    
    train_config_file: str = str(os.path.join(args_config.base_path, args_config.train_config_file))
    sampling_config_file: str = str(os.path.join(args_config.base_path, SamplingConfig.sampling_config_file))
    
    configure: SamplingConfig = parser.load(file_name=train_config_file, config_key=SamplingConfig.default_key, not_exist_ok=True).update(in_config_keys=[SamplingConfig.default_key, SamplingConfig.args_key], out_config_keys=SamplingConfig.final_key).save(file_name=sampling_config_file, config_key=SamplingConfig.final_key).parse(config_key=SamplingConfig.final_key, dataclass_type=SamplingConfig)
    # print(f"Config: {configure.__dict__}")
    
    configure.in_dist_ds = configure.dataset_name
    print(f"Config: {configure.__dict__}")
    if configure.in_dist_ds == DatasetLoader.CELEBA_HQ_DIALOG:
        configure.out_dist_ds = DatasetLoader.POKEMON_CAPTION
    elif configure.in_dist_ds == DatasetLoader.POKEMON_CAPTION:
        configure.out_dist_ds = DatasetLoader.CELEBA_HQ_DIALOG
    else:
        raise NotImplementedError
    
    configure.caption_similarity = CAPTION_SIMILARITY[configure.in_dist_ds][configure.caption_trigger]
    
    # os.environ.setdefault("CUDA_VISIBLE_DEVICES", config.gpu)
    
    # if (configure.image_trigger is not None and configure.image_trigger is not Backdoor.TRIGGER_NONE) and (configure.caption_trigger is not None and configure.image_trigger is not Backdoor.TRIGGER_NONE):
    #     raise NotImplementedError("Only one kind of trigger can be used, either 'image_trigger' or 'caption_trigger'")

    return configure

if __name__ == '__main__':
    DefautConfig: SamplingConfig = yield_default(SamplingConfig())
    configure: SamplingConfig = arg_parse()

    prompt_ds: PromptDataset = PromptDataset(path=configure.ds_base_path, in_dist_ds=configure.in_dist_ds, out_dist_ds=configure.out_dist_ds)
    prompt_ds.prepare_dataset(in_out_dist=PromptDatasetStatic.IN_DIST, train_test_split=PromptDatasetStatic.TRAIN_SPLIT)
    prompt_ds.prepare_dataset(in_out_dist=PromptDatasetStatic.IN_DIST, train_test_split=PromptDatasetStatic.TEST_SPLIT)
    prompt_ds.prepare_dataset(in_out_dist=PromptDatasetStatic.OUT_DIST, train_test_split=PromptDatasetStatic.FULL_SPLIT)
    
    # wandb.init(project=configure.project, name=configure.model_id, id=configure.model_id, settings=wandb.Settings(start_method="fork"))
    accelerator: Accelerator = Accelerator(log_with=["tensorboard", "wandb"], logging_dir=os.path.join(configure.output_dir, "logs"))
    # if accelerator.is_main_process:
    #     accelerator.init_trackers(project_name=configure.project, config=asdict(configure))
    sampling: Sampling = Sampling(backdoor_ds_root=configure.ds_base_path, num_inference_steps=configure.num_inference_steps, guidance_scale=configure.guidance_scale, max_batch_n=configure.max_batch_n)
    
    pipe, store_path = ModelSched.get_stable_diffusion(model_id=configure.base_path, sched=configure.sched, ckpt_step=configure.ckpt_step, enable_lora=configure.use_lora, lora_base_model=configure.lora_base_model, gpu=configure.gpu)
    
    measuring: Measuring = Measuring(base_path=configure.ds_base_path, sampling=sampling, prompt_ds=prompt_ds, accelerator=accelerator, device=configure.gpu)
    # measuring.measure(pipe=pipe, store_path=store_path, target=configure.target, caption_trigger=configure.caption_trigger,
    #                   thres=configure.mse_thres, fid_max_batch_n=configure.max_batch_n, trig_start_pos=configure.trig_start_pos,
    #                   trig_end_pos=configure.trig_end_pos, _format=configure.format, seed=configure.seed, device=f'cuda:{configure.gpu}')
    
    measuring.sample(pipe=pipe, store_path=store_path, caption_trigger=configure.caption_trigger, img_num_per_grid_sample=configure.img_num_per_grid_sample, 
                     trig_start_pos=configure.trig_start_pos, trig_end_pos=configure.trig_end_pos, _format=configure.format, seed=configure.seed, force_regenerate=configure.force_regenerate)
    