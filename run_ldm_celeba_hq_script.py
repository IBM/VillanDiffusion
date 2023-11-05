import glob
from scalablerunner.taskrunner import TaskRunner
from dataset import Backdoor
from dataset import DatasetLoader
from model import DiffuserModelSched

"""
python VillanDiffusion.py --postfix new-set --p
roject default --mode train --dataset CELEBA-HQ-LATENT --dataset_load_mode NONE --sde_type SDE-L
DM --learning_rate 0.0002 --sched UNIPC-SCHED --infer_steps 20 --batch 16 --epoch 2000 --clean_rate 1 --
poison_rate 0.9 --trigger GLASSES --target CAT --solver_type ode --psi 1 --vp_scale 1.0 --ve_scale 1.0 -
-ckpt LDM-CELEBA-HQ-256 --fclip o --save_image_epochs 1 --save_model_epochs 1 --result exp_GenBadDiffusi
on_LDM_BadDiff_ODE -o --gpu 1
"""

if __name__ == "__main__":
    result_dir: str = 'exp_GenBadDiffusion_LDM_BadDiff_ODE'
    epoch: int = 2000
    trigger: str = Backdoor.TRIGGER_GLASSES
    target: str = Backdoor.TARGET_CAT
    exp_ls = list(set(glob.glob(f"{result_dir}/res_*")))
    project: str = 'default'

    config = {
        # 'Clean Models':{
        #     'TWCC - 0,1': {
        #         'Call': "python VillanDiffusion.py",
        #         'Param': {
        #             '--postfix': ['new-set'],
        #             '--project': [project],
        #             '--mode': ['train+measure'],
        #             '--dataset': [DatasetLoader.CELEBA_HQ_LATENT],
        #             '--dataset_load_mode': [DatasetLoader.MODE_NONE],
        #             '--sde_type': ['SDE-LDM'],
        #             '--learning_rate': [2e-4],
        #             '--sched': ["UNIPC-SCHED"],
        #             '--infer_steps': [20],
        #             '--batch': [16],
        #             '--epoch': [epoch],
        #             '--clean_rate': [1],
        #             '--poison_rate': [0.9],
        #             # '--trigger': [Backdoor.TRIGGER_SM_STOP_SIGN, Backdoor.TRIGGER_SM_BOX_MED],
        #             '--trigger': [trigger],
        #             # '--target': [Backdoor.TARGET_TG, Backdoor.TARGET_FEDORA_HAT, Backdoor.TARGET_FA, Backdoor.TARGET_SHIFT, Backdoor.TARGET_BOX],
        #             '--target': [target],
        #             '--solver_type': ['ode'],
        #             '--psi': [1],
        #             '--vp_scale': [1.0],
        #             '--ve_scale': [1.0],
        #             '--ckpt': [DiffuserModelSched.LDM_CELEBA_HQ_256], 
        #             '--fclip': ['o'],
        #             '--save_image_epochs': [1],
        #             '--save_model_epochs': [1],
        #             '--result': [result_dir],
        #             '': ['-o'],
        #         },
        #         'Async':{
        #             '--gpu': ['1']
        #         }
        #     }, 
        # },
        'Measure CIFAR10, DPM-Solver, DPM-Solver++, DEIS: 20':{
            'TWCC': {
                'Call': "python VillanDiffusion.py",
                'Param': {
                    '--project': [project],
                    '--mode': ['measure'],
                    '--eval_max_batch': [64],
                    '--sched': [DiffuserModelSched.UNIPC_SCHED, DiffuserModelSched.DPM_SOLVER_O2_SCHED, DiffuserModelSched.DEIS_SCHED],
                    '--infer_steps': [20],
                    '--ckpt': exp_ls,
                    '--fclip': ['o'],
                },
                'Async':{
                    '--gpu': ['0']
                }
            },
        },
    }
    
    tr = TaskRunner(config=config)
    tr.run()
