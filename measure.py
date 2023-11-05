from dataclasses import asdict
import os
from typing import Union

from accelerate import Accelerator
import wandb

from tools import Log
from config import MeasuringConfig, PromptDatasetStatic, MeasuringStatic
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

def arg_parse() -> MeasuringConfig:
    parser: ArgParser = ArgParser(config_key=MeasuringConfig.args_key, config=MeasuringConfig()).receive_args(config_key=MeasuringConfig.args_key, mode=ArgParser.RECEIVE_ARGS_MODE_CONFIG, description=globals()['__doc__'])
    args_config: MeasuringConfig = parser.parse(config_key=MeasuringConfig.args_key, dataclass_type=MeasuringConfig)
    
    train_config_file: str = str(os.path.join(args_config.base_path, args_config.train_config_file))
    measure_config_file: str = str(os.path.join(args_config.base_path, MeasuringConfig.measure_config_file))
    # sampling_config_file: str = str(os.path.join(args_config.base_path, args_config.sampling_config_file))
    
    configure: MeasuringConfig = parser.load(file_name=train_config_file, config_key=MeasuringConfig.default_key, not_exist_ok=True).update(in_config_keys=[MeasuringConfig.default_key, MeasuringConfig.args_key], out_config_keys=MeasuringConfig.final_key).save(file_name=measure_config_file, config_key=MeasuringConfig.final_key).parse(config_key=MeasuringConfig.final_key, dataclass_type=MeasuringConfig)
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

def decide_measure(configure: MeasuringConfig, measuring: Measuring, pipe, store_path: Union[str, os.PathLike], mode: str):
    device: str = f'cuda:{configure.gpu}'
    img_num_per_prompt: int = MeasuringStatic.IMAGE_NUM_PER_PROMPT
    in_out_dist, train_test_split, clean_backdoor = mode.split('/')
    Log.critical(f"Measuring Mode -  In/Out: {in_out_dist}, Train/Test: {train_test_split}, Clean/Backdoor: {clean_backdoor}")
    
    if in_out_dist == PromptDatasetStatic.IN_DIST and train_test_split == PromptDatasetStatic.TRAIN_SPLIT and clean_backdoor == MeasuringStatic.MEASURING_CLEAN:
        # In Distribution, Train
        # Clean
        measuring.measure_log_by_part(pipe=pipe, store_path=store_path, in_out_dist=PromptDatasetStatic.IN_DIST, train_test_split=PromptDatasetStatic.TRAIN_SPLIT, thres=configure.mse_thres, fid_max_batch_n=configure.fid_max_batch_n, device=device, is_fid=False, caption_trigger=None, target=configure.target, trig_start_pos=configure.trig_start_pos, trig_end_pos=configure.trig_end_pos, img_num_per_prompt=img_num_per_prompt, _format=configure.format, seed=configure.seed, force_regenerate=configure.force_regenerate)
    elif in_out_dist == PromptDatasetStatic.IN_DIST and train_test_split == PromptDatasetStatic.TRAIN_SPLIT and clean_backdoor == MeasuringStatic.MEASURING_BACKDOOR:
        # In Distribution, Train
        # Backdoor
        measuring.measure_log_by_part(pipe=pipe, store_path=store_path, in_out_dist=PromptDatasetStatic.IN_DIST, train_test_split=PromptDatasetStatic.TRAIN_SPLIT, thres=configure.mse_thres, fid_max_batch_n=configure.fid_max_batch_n, device=device, is_fid=False, caption_trigger=configure.caption_trigger, target=configure.target, trig_start_pos=configure.trig_start_pos, trig_end_pos=configure.trig_end_pos, img_num_per_prompt=img_num_per_prompt, _format=configure.format, seed=configure.seed, force_regenerate=configure.force_regenerate)
    elif in_out_dist == PromptDatasetStatic.IN_DIST and train_test_split == PromptDatasetStatic.TEST_SPLIT and clean_backdoor == MeasuringStatic.MEASURING_CLEAN:
        # In Distribution, Test
        # Clean
        measuring.measure_log_by_part(pipe=pipe, store_path=store_path, in_out_dist=PromptDatasetStatic.IN_DIST, train_test_split=PromptDatasetStatic.TEST_SPLIT, thres=configure.mse_thres, fid_max_batch_n=configure.fid_max_batch_n, device=device, is_fid=False, caption_trigger=None, target=configure.target, trig_start_pos=configure.trig_start_pos, trig_end_pos=configure.trig_end_pos, img_num_per_prompt=img_num_per_prompt, _format=configure.format, seed=configure.seed, force_regenerate=configure.force_regenerate)
    elif in_out_dist == PromptDatasetStatic.IN_DIST and train_test_split == PromptDatasetStatic.TEST_SPLIT and clean_backdoor == MeasuringStatic.MEASURING_BACKDOOR:
        # In Distribution, Test
        # Backdoor
        measuring.measure_log_by_part(pipe=pipe, store_path=store_path, in_out_dist=PromptDatasetStatic.IN_DIST, train_test_split=PromptDatasetStatic.TEST_SPLIT, thres=configure.mse_thres, fid_max_batch_n=configure.fid_max_batch_n, device=device, is_fid=False, caption_trigger=configure.caption_trigger, target=configure.target, trig_start_pos=configure.trig_start_pos, trig_end_pos=configure.trig_end_pos, img_num_per_prompt=img_num_per_prompt, _format=configure.format, seed=configure.seed, force_regenerate=configure.force_regenerate)
    elif in_out_dist == PromptDatasetStatic.IN_DIST and train_test_split == PromptDatasetStatic.FULL_SPLIT and clean_backdoor == MeasuringStatic.MEASURING_CLEAN:
        # In Distribution, Full
        # Clean, for FID
        measuring.measure_log_by_part(pipe=pipe, store_path=store_path, in_out_dist=PromptDatasetStatic.IN_DIST, train_test_split=PromptDatasetStatic.FULL_SPLIT, thres=configure.mse_thres, fid_max_batch_n=configure.fid_max_batch_n, device=device, is_fid=True, caption_trigger=None, target=configure.target, trig_start_pos=configure.trig_start_pos, trig_end_pos=configure.trig_end_pos, img_num_per_prompt=img_num_per_prompt, _format=configure.format, seed=configure.seed, force_regenerate=configure.force_regenerate)
    elif in_out_dist == PromptDatasetStatic.OUT_DIST and train_test_split == PromptDatasetStatic.FULL_SPLIT and clean_backdoor == MeasuringStatic.MEASURING_CLEAN:
        # Out Distribution
        # Clean
        measuring.measure_log_by_part(pipe=pipe, store_path=store_path, in_out_dist=PromptDatasetStatic.OUT_DIST, train_test_split=PromptDatasetStatic.FULL_SPLIT, thres=configure.mse_thres, fid_max_batch_n=configure.fid_max_batch_n, device=device, is_fid=False, caption_trigger=None, target=configure.target, trig_start_pos=configure.trig_start_pos, trig_end_pos=configure.trig_end_pos, img_num_per_prompt=img_num_per_prompt, _format=configure.format, seed=configure.seed, force_regenerate=configure.force_regenerate)
    elif in_out_dist == PromptDatasetStatic.OUT_DIST and train_test_split == PromptDatasetStatic.FULL_SPLIT and clean_backdoor == MeasuringStatic.MEASURING_BACKDOOR:
        # Out Distribution
        # Backdoor
        measuring.measure_log_by_part(pipe=pipe, store_path=store_path, in_out_dist=PromptDatasetStatic.OUT_DIST, train_test_split=PromptDatasetStatic.FULL_SPLIT, thres=configure.mse_thres, fid_max_batch_n=configure.fid_max_batch_n, device=device, is_fid=False, caption_trigger=configure.caption_trigger, target=configure.target, trig_start_pos=configure.trig_start_pos, trig_end_pos=configure.trig_end_pos, img_num_per_prompt=img_num_per_prompt, _format=configure.format, seed=configure.seed, force_regenerate=configure.force_regenerate)
    elif in_out_dist == PromptDatasetStatic.DEFAULT_DIST and train_test_split == PromptDatasetStatic.DEFAULT_SPLIT and clean_backdoor == MeasuringStatic.MEASURING_CLEAN:
        measuring.measure(pipe=pipe, store_path=store_path, target=configure.target, caption_trigger=configure.caption_trigger, force_regenerate=configure.force_regenerate,
                          thres=configure.mse_thres, fid_max_batch_n=configure.fid_max_batch_n, trig_start_pos=configure.trig_start_pos,
                          trig_end_pos=configure.trig_end_pos, _format=configure.format, seed=configure.seed, device=device)
    else:
        raise NotImplementedError

def main():
    DefautConfig: MeasuringConfig = yield_default(MeasuringConfig())
    configure: MeasuringConfig = arg_parse()

    prompt_ds: PromptDataset = PromptDataset(path=configure.ds_base_path, in_dist_ds=configure.in_dist_ds, out_dist_ds=configure.out_dist_ds)
    prompt_ds.prepare_dataset(in_out_dist=PromptDatasetStatic.IN_DIST, train_test_split=PromptDatasetStatic.TRAIN_SPLIT)
    prompt_ds.prepare_dataset(in_out_dist=PromptDatasetStatic.IN_DIST, train_test_split=PromptDatasetStatic.TEST_SPLIT)
    prompt_ds.prepare_dataset(in_out_dist=PromptDatasetStatic.IN_DIST, train_test_split=PromptDatasetStatic.FULL_SPLIT)
    prompt_ds.prepare_dataset(in_out_dist=PromptDatasetStatic.OUT_DIST, train_test_split=PromptDatasetStatic.FULL_SPLIT)
    
    wandb.init(project=configure.project, name=configure.model_id, id=configure.model_id, settings=wandb.Settings(start_method="fork"))
    accelerator: Accelerator = Accelerator(log_with=["tensorboard", "wandb"], logging_dir=os.path.join(configure.output_dir, "logs"))
    if accelerator.is_main_process:
        accelerator.init_trackers(project_name=configure.project, config=asdict(configure))
    sampling: Sampling = Sampling(backdoor_ds_root=configure.ds_base_path, num_inference_steps=configure.num_inference_steps, guidance_scale=configure.guidance_scale, max_batch_n=configure.max_batch_n)
    
    pipe, store_path = ModelSched.get_stable_diffusion(model_id=configure.base_path, sched=configure.sched, ckpt_step=configure.ckpt_step, enable_lora=configure.use_lora, lora_base_model=configure.lora_base_model, gpu=configure.gpu)
    
    measuring: Measuring = Measuring(base_path=configure.ds_base_path, sampling=sampling, prompt_ds=prompt_ds, accelerator=accelerator, max_measuring_samples=configure.max_measuring_samples, device=configure.gpu)
    # measuring.measure(pipe=pipe, store_path=store_path, target=configure.target, caption_trigger=configure.caption_trigger, force_regenerate=configure.force_regenerate,
    #                   thres=configure.mse_thres, fid_max_batch_n=configure.fid_max_batch_n, trig_start_pos=configure.trig_start_pos,
    #                   trig_end_pos=configure.trig_end_pos, _format=configure.format, seed=configure.seed, device=f'cuda:{configure.gpu}')
    
    # measuring.sample(pipe=pipe, store_path=store_path, caption_trigger=CaptionBackdoor.TRIGGER_EMOJI_HOT)
    
    decide_measure(configure=configure, measuring=measuring, pipe=pipe, store_path=store_path, mode=configure.mode)

if __name__ == '__main__':
    main()