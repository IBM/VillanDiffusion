import glob
from scalablerunner.taskrunner import TaskRunner
from dataset import Backdoor
from dataset import DatasetLoader
from model import DiffuserModelSched

if __name__ == "__main__":
    project: str = "default"
    result_dir: str = "exp_GenBadDiffusion_Poison_Rates_DDPM_BadDiff_ODE_NEW"
    exp_ls = list(set(glob.glob(f"{result_dir}/res_*")))
    config = {
        'Poison Rate':{
            'GPU - 0': {
                'Call': "python rm_backdoor_VillanDiffusion.py",
                'Param': {
                    '--postfix': ['new-set'],
                    '--project': [project],
                    '--mode': ['train'],
                    '--dataset': [DatasetLoader.CIFAR10],
                    '--sde_type': ['SDE-VP'],
                    '--sched': ["UNIPC-SCHED"],
                    '--infer_steps': [20],
                    '--batch': [128],
                    '--epoch': [15],
                    '--clean_rate': [1],
                    '--poison_rate': [0.7],
                    '--trigger': [Backdoor.TRIGGER_SM_STOP_SIGN],
                    '--target': [Backdoor.TARGET_FEDORA_HAT],
                    '--solver_type': ['ode'],
                    '--psi': [1],
                    '--vp_scale': [1.0],
                    '--ve_scale': [1.0],
                    '--ckpt': ['/work/u2941379/workspace/backdoor_diffusion/exp_GenBadDiffusion_Poison_Rates_DDPM_BadDiff_ODE/res_DDPM-CIFAR10-32_CIFAR10_ep100_ode_c1.0_p0.7_SM_STOP_SIGN-FEDORA_HAT_psi1.0_lr0.0002_vp1.0_ve1.0_new-set'], 
                    '--fclip': ['o'],
                    '--save_image_epochs': [1],
                    '--save_model_epochs': [5],
                    '--result': [result_dir],
                    '': ['-o'],
                },
                'Async':{
                    '--gpu': ['0']
                }
            }, 
        },
        'Measure CIFAR10, DPM-SOLVER O2, UNIPC: 20':{
            'GPU - 0': {
                'Call': "python VillanDiffusion.py",
                'Param': {
                    '--project': [project],
                    '--mode': ['sampling'],
                    # '--eval_max_batch': [1500],
                    '--sched': [DiffuserModelSched.DPM_SOLVER_O2_SCHED],
                    '--infer_steps': [20],
                    '--fclip': ['o'],
                    '--ckpt': exp_ls, 
                },
                'Async':{
                    '--gpu': ['0']
                }
            }, 
        },
    }
    
    tr = TaskRunner(config=config)
    tr.run()
