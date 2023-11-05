import glob
from scalablerunner.taskrunner import TaskRunner
from dataset import Backdoor
from dataset import DatasetLoader
from model import DiffuserModelSched

if __name__ == "__main__":
    project: str = "default"
    result_dir: str = "exp_GenBadDiffusion_Poison_Rates_DDPM_BadDiff_ODE"
    exp_ls = list(set(glob.glob(f"{result_dir}/res_*")))
    config = {
        # 'Poison Rate':{
        #     'GPU - 0': {
        #         'Call': "python VillanDiffusion.py",
        #         'Param': {
        #             '--postfix': ['new-set'],
        #             '--project': [project],
        #             '--mode': ['train'],
        #             '--dataset': [DatasetLoader.CIFAR10],
        #             '--sde_type': ['SDE-VP'],
        #             '--sched': ["UNIPC-SCHED"],
        #             '--infer_steps': [20],
        #             '--batch': [128],
        #             '--epoch': [15],
        #             '--clean_rate': [1],
        #             '--poison_rate': [0.7],
        #             '--trigger': [Backdoor.TRIGGER_SM_STOP_SIGN],
        #             '--target': [Backdoor.TARGET_FEDORA_HAT],
        #             '--solver_type': ['ode'],
        #             '--psi': [1],
        #             '--vp_scale': [1.0],
        #             '--ve_scale': [1.0],
        #             '--ckpt': [DiffuserModelSched.DDPM_CIFAR10_32], 
        #             '--fclip': ['o'],
        #             '--save_image_epochs': [1],
        #             '--save_model_epochs': [5],
        #             '--result': [result_dir],
        #             '': ['-o'],
        #         },
        #         'Async':{
        #             '--gpu': ['1']
        #         }
        #     }, 
        # },
        'Measure CIFAR10, DPM-SOLVER O2, UNIPC: 20':{
            'GPU - 0': {
                'Call': "python VillanDiffusion.py",
                'Param': {
                    '--project': [project],
                    '--mode': ['measure'],
                    '--eval_max_batch': [1500],
                    '--sched': [DiffuserModelSched.DPM_SOLVER_O2_SCHED, DiffuserModelSched.UNIPC_SCHED],
                    '--infer_steps': [20],
                    '--fclip': ['o', 'w'],
                    '--ckpt': exp_ls, 
                },
                'Async':{
                    '--gpu': ['1']
                }
            }, 
        },
    }
    
    tr = TaskRunner(config=config)
    tr.run()
