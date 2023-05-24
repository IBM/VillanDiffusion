import glob

from scalablerunner.taskrunner import TaskRunner
from dataset import Backdoor

ckpts = [
        # "res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_SM_BOX_MED-TRIGGER_new-set",
        # "res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.05_SM_BOX_MED-TRIGGER_new-set",
        # "res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.2_SM_BOX_MED-TRIGGER_new-set",
        # "res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.3_SM_BOX_MED-TRIGGER_new-set",
        
        # "res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_SM_BOX_MED-FASHION_new-set",
        # "res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_SM_BOX_MED-FEDORA_HAT_new-set",
        # "res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.05_SM_BOX_MED-FASHION_new-set",
        # "res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.05_SM_BOX_MED-FEDORA_HAT_new-set-1",
        # "res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.05_SM_BOX_MED-FEDORA_HAT_new-set",
        # "res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.05_SM_BOX_MED-TRIGGER_new-set",
        
        # "res_DDPM-CIFAR10-32_CIFAR10_ep0_c1.0_p0.0_NONE-TRIGGER_new-set",
        # "res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.5_SM_BOX_MED-FASHION_new-set",
        # "res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.5_SM_BOX_MED-TRIGGER_new-set",
        # "res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.5_SM_BOX_MED-FEDORA_HAT_new-set",
        # "res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.5_SM_BOX_MED-SHIFT_new-set",
        # "res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.05_SM_BOX_MED-FEDORA_HAT_new-set-1",
        
        # "res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.05_SM_BOX_MED-TRIGGER_new-set",
        # "res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.05_SM_BOX_MED-FEDORA_HAT_new-set",
        # "res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.05_SM_BOX_MED-FASHION_new-set",
        # "res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.05_SM_BOX_MED-SHIFT_new-set",
        # "res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.05_SM_BOX_MED-BOX_new-set",
        
        # "res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_SM_BOX_MED-TRIGGER_new-set",
        # "res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_SM_BOX_MED-FEDORA_HAT_new-set",
        # "res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_SM_BOX_MED-FASHION_new-set",
        # "res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_SM_BOX_MED-SHIFT_new-set",
        # "res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_SM_BOX_MED-BOX_new-set",
        # NOTE: 
        # "res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.0_SM_BOX_MED-TRIGGER_new-set",
        # "res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.05_SM_BOX_MED-TRIGGER_new-set",
        # "res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_SM_BOX_MED-TRIGGER_new-set",
        # "res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.2_SM_BOX_MED-TRIGGER_new-set",
        # "res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.3_SM_BOX_MED-TRIGGER_new-set",
        # "res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.5_SM_BOX_MED-TRIGGER_new-set",
        
        # "res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.0_SM_BOX_MED-BOX_new-set",
        # "res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.05_SM_BOX_MED-BOX_new-set",
        # "res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_SM_BOX_MED-BOX_new-set",
        # "res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.2_SM_BOX_MED-BOX_new-set",
        # "res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.3_SM_BOX_MED-BOX_new-set",
        # "res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.5_SM_BOX_MED-BOX_new-set",
        
        # "res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.0_SM_BOX_MED-FEDORA_HAT_new-set",
        # "res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.05_SM_BOX_MED-FEDORA_HAT_new-set",
        # "res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_SM_BOX_MED-FEDORA_HAT_new-set",
        # "res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.2_SM_BOX_MED-FEDORA_HAT_new-set",
        # "res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.3_SM_BOX_MED-FEDORA_HAT_new-set",
        # "res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.5_SM_BOX_MED-FEDORA_HAT_new-set",
        
        # NOTE: Second
        "res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.0_SM_STOP_SIGN-TRIGGER_new-set",
        "res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.05_SM_STOP_SIGN-TRIGGER_new-set",
        "res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_SM_STOP_SIGN-TRIGGER_new-set",
        "res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.2_SM_STOP_SIGN-TRIGGER_new-set",
        "res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.3_SM_STOP_SIGN-TRIGGER_new-set",
        "res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.5_SM_STOP_SIGN-TRIGGER_new-set",
        
        "res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.0_SM_STOP_SIGN-BOX_new-set",
        "res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.05_SM_STOP_SIGN-BOX_new-set",
        # "res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_SM_STOP_SIGN-BOX_new-set",
        "res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.2_SM_STOP_SIGN-BOX_new-set",
        "res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.3_SM_STOP_SIGN-BOX_new-set",
        "res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.5_SM_STOP_SIGN-BOX_new-set",
        
        "res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.0_SM_STOP_SIGN-FEDORA_HAT_new-set",
        "res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.05_SM_STOP_SIGN-FEDORA_HAT_new-set",
        "res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_SM_STOP_SIGN-FEDORA_HAT_new-set",
        "res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.2_SM_STOP_SIGN-FEDORA_HAT_new-set",
        "res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.3_SM_STOP_SIGN-FEDORA_HAT_new-set",
        "res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.5_SM_STOP_SIGN-FEDORA_HAT_new-set",
    ]

if __name__ == "__main__":
    config = {
        # 'Measure CIFAR10 - DDIM':{
        #     'TWCC': {
        #         'Call': "python diffusers_training_example_rebuttal_sampler_denoise.py",
        #         'Param': {
        #             '--project': ['Default'],
        #             # '--mode': ['sampling'],
        #             '--mode': ['measure'],
        #             '--task': ['poisoned_denoise', 'unpoisoned_denoise', 'unpoisoned_inpaint_box', 'poisoned_inpaint_box', 'unpoisoned_inpaint_line', 'poisoned_inpaint_line'],
        #             # '--sched': ['DDIM-SCHED'],
        #             # '--infer_steps': [50],
        #             '--ckpt': ckpts,
        #             '--fclip': ['o'],
        #         },
        #         'Async':{
        #             '--gpu': ['0']
        #         }
        #     }, 
        # },
        
        'Measure CIFAR10 - Inpaint':{
            'TWCC': {
                'Call': "python diffusers_training_example_rebuttal_sampler_denoise.py",
                'Param': {
                    '--project': ['Inpaint'],
                    '--mode': ['measure'],
                    '--task': ['poisoned_denoise', 'unpoisoned_denoise', 'unpoisoned_inpaint_box', 'poisoned_inpaint_box', 'unpoisoned_inpaint_line', 'poisoned_inpaint_line'],
                    '--ckpt': ckpts,
                    '--fclip': ['o'],
                },
                'Async':{
                    '--gpu': ['1', '2', '3', '4', '5', '6', '7']
                    # '--gpu': ['0']
                }
            }, 
        },
    }
    
    tr = TaskRunner(config=config)
    tr.run()
