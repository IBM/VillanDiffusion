import glob
from scalablerunner.taskrunner import TaskRunner
from dataset import Backdoor
from dataset import DatasetLoader
from model import DiffuserModelSched

if __name__ == "__main__":
    exp_ls: str = ["res_DDPM-CIFAR10-32_CIFAR10_ep100_ode_c1.0_p0.2_SM_STOP_SIGN-BOX_psi1.0_lr0.0002_vp1.0_ve1.0_new-set-1_test"]
    config = {
        'Measure CIFAR10, VillanDiffusion + ODE + Inpainting':{
            'TWCC': {
                'Call': "python VillanDiffusion.py",
                'Param': {
                    '--project': ['default'],
                    '--mode': ['measure'],
                    '--task': ['unpoisoned_denoise', 'poisoned_denoise', 'unpoisoned_inpaint_box', 'poisoned_inpaint_box', 'unpoisoned_inpaint_line', 'poisoned_inpaint_line'],
                    '--sched': [DiffuserModelSched.UNIPC_SCHED],
                    '--infer_steps': [20],
                    '--infer_start': [10],
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
