from glob import glob

from scalablerunner.taskrunner import TaskRunner
from dataset import Backdoor
from dataset import DatasetLoader
from model import DiffuserModelSched

if __name__ == "__main__":
    epochs: int = 500
    project_name: str = 'default'
    result_dir: str = 'exp_GenBadDiffusion_CelebA-HQ_DDPM_BadDiff_ODE'
    exp_ls = glob(f'{result_dir}/*')
    config = {
        # Use script to train (Trigger, Target): (Backdoor.TRIGGER_GLASSES, Backdoor.TARGET_CAT)
        'CelebA-HQ Eye Glasses-Cat':{
            'TWCC 0': {
                'Call': "python VillanDiffusion.py",
                'Param': {
                    '--postfix': ['new-set'],
                    '--project': [project_name],
                    '--mode': ['train'],
                    '--dataset': [DatasetLoader.CELEBA_HQ],
                    '--sde_type': ['SDE-VP'],
                    '--sched': ["UNIPC-SCHED"],
                    '--infer_steps': [20],
                    '--learning_rate': [8e-5],
                    '--batch': [16],
                    '--eval_max_batch': [64],
                    '--epoch': [epochs],
                    '--clean_rate': [1],
                    '--poison_rate': [0.9],
                    '--trigger': [Backdoor.TRIGGER_GLASSES],
                    '--target': [Backdoor.TARGET_CAT],
                    '--solver_type': ['ode'],
                    '--psi': [1],
                    '--vp_scale': [1.0],
                    '--ve_scale': [1.0],
                    '--ckpt': [DiffuserModelSched.DDPM_CELEBA_HQ_256], 
                    '--fclip': ['o'],
                    '--save_image_epochs': [1],
                    '--save_model_epochs': [5],
                    '--result': [result_dir],
                    '': ['-o'],
                },
                'Async':{
                    '--gpu': ['0,1']
                }
            }, 
        },
        'Measure CelebA-HQ, DEIS: 20':{
            'TWCC': {
                'Call': "python VillanDiffusion.py",
                'Param': {
                    '--project': [project_name],
                    '--mode': ['measure'],
                    '--sched': [DiffuserModelSched.DEIS_SCHED, DiffuserModelSched.HEUN_SCHED],
                    '--infer_steps': [20],
                    '--eval_max_batch': [64],
                    '--ckpt': exp_ls, 
                    '--fclip': ['o'],
                },
                'Async':{
                    '--gpu': ['0', '1']
                }
            }, 
        },
    }
    
    tr = TaskRunner(config=config)
    tr.run()
