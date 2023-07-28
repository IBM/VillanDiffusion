import glob
from typing import List

from scalablerunner.taskrunner import TaskRunner
from dataset import Backdoor, DatasetLoader
from model import DiffuserModelSched
from util import path_gen

# Avoid WandB fails: https://github.com/wandb/wandb/issues/3208
if __name__ == "__main__":
    gpu_ids: List[str] = ['0']
    # project: str = "Poison_Rates_NCSNPP_TrojDiff_SDE_Std_FLEX"
    result_dir: str = 'exp_GenBadDiffusion_NCSNPP_CIFAR10_TrojDiff_SDE_FLEX'
    epoch: int = 30

    project: str = 'default'
    
    exp_ep_ls = []
    exp_ep_ls += list(set(glob.glob(f"{result_dir}/res_*{DatasetLoader.CIFAR10}*ep{epoch}*_new-set")))
    
    print(f"[{len(exp_ep_ls)}] Exp List: {exp_ep_ls}")
    
    config = {
        'SDE & ODE':{
            'SDE': {
                'Call': "python VillanDiffusion.py",
                'Param': {
                    '--postfix': ['flex_new-set'],
                    '--project': [project],
                    '--mode': ['train'],
                    '--learning_rate': [2e-5],
                    '--dataset': [DatasetLoader.CIFAR10],
                    '--sde_type': ['SDE-VE'],
                    '--batch': [128],
                    '--epoch': [epoch],
                    # '--clean_rate': [1.0],
                    # '--poison_rate': [9.0],
                    # '--ext_poison_rate': [1.0],
                    # '--dataset_load_mode': [DatasetLoader.MODE_EXTEND],
                    '--clean_rate': [1.0],
                    '--poison_rate': [0.98],
                    '--dataset_load_mode': [DatasetLoader.MODE_FIXED],
                    '--trigger': [Backdoor.TRIGGER_SM_STOP_SIGN],
                    '--target': [Backdoor.TARGET_FEDORA_HAT],
                    # '--rhos_hat_w': [1.0],
                    # '--rhos_hat_b': [0.0],
                    '--solver_type': ['sde'],
                    '--psi': [0],
                    '--vp_scale': [1.0],
                    '--ve_scale': [1.0],
                    '--ckpt': ["NCSN_CIFAR10_my"], 
                    '--fclip': ['o'],
                    '--save_image_epochs': [5],
                    '--save_model_epochs': [5],
                    '--result': [result_dir],
                    '': ['-o --R_trigger_only'],
                    # '': ['-o --trigger_augment'],
                    # '': ['-o'],
                },
                'Async':{
                    '--gpu': gpu_ids
                }
            },
        },
        # 'Measure CIFAR10, SCORE_SDE_VE_SCHED - VE, rhw':{
        #     'TWCC': {
        #         'Call': "python VillanDiffusion.py",
        #         'Param': {
        #             '--project': [project],
        #             '--mode': ['measure'],
        #             '--eval_max_batch': [1500],
        #             '--fclip': ['o'],
        #             '--sched': ["SCORE-SDE-VE-SCHED"],
        #             '--infer_steps': [1000],
        #             '--ckpt': exp_ep_ls, 
        #         },
        #         'Async':{
        #             '--gpu': gpu_ids
        #         }
        #     }, 
        # },
    }
    tr = TaskRunner(config=config)
    tr.run()