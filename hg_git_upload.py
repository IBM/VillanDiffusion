import os
from typing import Union
import subprocess

import shutil
import git

from glob import glob
from tqdm import tqdm
from huggingface_hub import create_repo, delete_repo, upload_file, upload_folder
from joblib import Parallel, delayed

class HGBackup:
    def __init__(self) -> None:
        pass

    def upload_models(self, dataset: str, model: str, paths: Union[str, os.PathLike]):
        if not isinstance(paths, list):
            paths = [paths]
    
    def upload_model(self, path, user_name: str, repo_name: str, track_ls: list, commit: str="init", private: bool=False, blocking: bool=False):
        repo_id: str = f"{user_name}/{repo_name}"
        repo_type: str = 'model'
        exist_ok: bool = False
        # token: str = 
        
        print(f"-> Creating Repo: {repo_id}")
        # delete_repo(repo_id, repo_type=repo_type)
        repo_url: str = create_repo(repo_id, private=private, repo_type=repo_type, exist_ok=True)
        
        print(f"-> Uploading Model: {path}")
        for f in track_ls:
            p: str = os.path.join(path, f)
            print(f"Uploading: {p}")
            if os.path.isdir(p):
                print(f"Upload Folder: {p} -> {f}")
                upload_folder(folder_path=p, path_in_repo=f, repo_id=repo_id, repo_type=repo_type, commit_message=commit, run_as_future=not blocking)
            else:
                if os.path.isfile(p):
                    print(f"Upload File: {p} -> {f}")
                    upload_file(path_or_fileobj=p, path_in_repo=f, repo_id=repo_id, repo_type=repo_type, commit_message=commit, run_as_future=not blocking)
                else:
                    under_p_ls: list = glob(p)
                    for under_p_f in under_p_ls:
                        under_f: str = os.path.join(*under_p_f.replace(path, '').split('/'))
                        print(f"Upload File: {under_p_f} -> {under_f}")
                        upload_file(path_or_fileobj=under_p_f, path_in_repo=under_f, repo_id=repo_id, repo_type=repo_type, commit_message=commit, run_as_future=not blocking)
    
    def upload_models_cifar10_ddpm(self, path: Union[str, os.PathLike], user_name: str):
        track_ls: list[str] = ["backdoor_samples/", "ckpt/", "samples/", "scheduler/", "unet/", "vqvae/", "./*.json", "./*.ckpt"]
        # track_ls: list[str] = ["unet"]
        # track_ls: list[str] = ["*.json"]
        try:
            self.upload_model(path=path, user_name=user_name, repo_name=HGBackup.get_repo_name_from_path_cifar10_ddpm(path), track_ls=track_ls, commit="init", blocking=True)
        except Exception:
            print(f"Upload Error")
            
    def upload_models_celeba_hq_ddpm(self, path: Union[str, os.PathLike], user_name: str):
        track_ls: list[str] = ["backdoor_samples/", "ckpt/", "samples/", "scheduler/", "unet/", "./*.json", "./*.ckpt"]
        try:
            self.upload_model(path=path, user_name=user_name, repo_name=HGBackup.get_repo_name_from_path_celeba_hq_ddpm(path), track_ls=track_ls, commit="init", blocking=True)
        except Exception:
            print(f"Upload Error")
            
    def upload_models_celeba_hq_ldm(self, path: Union[str, os.PathLike], user_name: str):
        track_ls: list[str] = ["backdoor_samples/", "ckpt/", "samples/", "scheduler/", "unet/", "vqvae/", "./*.json", "./*.ckpt"]
        try:
            self.upload_model(path=path, user_name=user_name, repo_name=HGBackup.get_repo_name_from_path_celeba_hq_ldm(path), track_ls=track_ls, commit="init", blocking=True)
        except Exception:
            print(f"Upload Error")
            
    def upload_models_cifar10_ncsn(self, path: Union[str, os.PathLike], user_name: str):
        track_ls: list[str] = ["backdoor_samples/", "ckpt/", "samples/", "scheduler/", "unet/", "./*.json", "./*.ckpt"]
        try:
            self.upload_model(path=path, user_name=user_name, repo_name=HGBackup.get_repo_name_from_path_cifar10_ncsn(path), track_ls=track_ls, commit="init", blocking=True)
        except Exception:
            print(f"Upload Error")
    
    @staticmethod
    def get_repo_name_from_path_cifar10_ddpm(path: str):
        # "res_DDPM-CIFAR10-32_CIFAR10_ep100_ode_c1.0_p0.1_SM_STOP_SIGN-BOX_psi1.0_lr0.0002_vp1.0_ve1.0_new-set-1"
        repo_name: str = path.split('/')[-1].replace('_lr0.0002_vp1.0_ve1.0', '')
        return repo_name
    
    @staticmethod
    def get_repo_name_from_path_celeba_hq_ddpm(path: str):
        # "res_DDPM-CELEBA-HQ-256_CELEBA-HQ_ep1500_ode_c1.0_p0.5_GLASSES-CAT_psi1.0_lr8e-05_vp1.0_ve1.0_new-set"
        repo_name: str = path.split('/')[-1].replace('_lr8e-05_vp1.0_ve1.0', '')
        return repo_name
    
    @staticmethod
    def get_repo_name_from_path_celeba_hq_ldm(path: str):
        # "res_LDM-CELEBA-HQ-256_CELEBA-HQ-LATENT_ep2000_ode_c1.0_p0.3_SM_STOP_SIGN-FEDORA_HAT_psi1.0_lr0.0002_vp1.0_ve1.0_new-set"
        repo_name: str = path.split('/')[-1].replace('_lr0.0002_vp1.0_ve1.0', '')
        return repo_name
    
    @staticmethod
    def get_repo_name_from_path_cifar10_ncsn(path: str):
        # "res_NCSN_CIFAR10_my_CIFAR10_ep10_sde_c1.0_p9.0_epr0.2_SM_STOP_SIGN-FEDORA_HAT_psi0.0_lr2e-05_rhw1.0_rhb0.0_extend_new-set"
        # "res_NCSN_CIFAR10_my_CIFAR10_ep20_sde_c1.0_p4.0_epr0.1_SM_STOP_SIGN-FEDORA_HAT_psi0.0_lr2e-05_rhw1.0_rhb0.0_extend_new-set"
        repo_name: str = path.split('/')[-1].replace('_lr2e-05_rhw1.0_rhb0.0', '').replace('end', '')
        return repo_name
    
    @staticmethod
    def get_link_from_path(path: str):
        repo_name: str = HGBackup.get_repo_name_from_path_cifar10_ddpm(path=path)
        return f"git@hf.co:newsyctw/{repo_name}"
        # return f"https://huggingface.co/newsyctw/{repo_name}"
        
if __name__ == "__main__":
    backup: HGBackup = HGBackup()
    
    # BadDiffusion DDPM CIFAR10
    paths: list = glob('/work/u2941379/workspace/exp_baddiffusion_sde/res_DDPM-CIFAR10-32_CIFAR10_ep*')
    Parallel(n_jobs=20)(delayed(backup.upload_models_cifar10_ddpm)(path, 'newsyctw') for path in paths)
    
    # DDPM CIFAR10
    # paths: list = glob('/work/u2941379/workspace/backdoor_diffusion/exp_GenBadDiffusion_Poison_Rates_DDPM_BadDiff_ODE/res_DDPM-CIFAR10-32_CIFAR10_ep*')
    # Parallel(n_jobs=20)(delayed(backup.upload_models_cifar10_ddpm)(path, 'newsyctw') for path in paths)
    
    # DDPM CelebA-HQ
    # paths: list = list(glob('/work/u2941379/workspace/backdoor_diffusion/exp_GenBadDiffusion_CelebA-HQ_DDPM_BadDiff_ODE/res_DDPM-CELEBA-HQ-256_CELEBA-HQ_ep1500*'))
    # paths += list(glob('res_DDPM-CELEBA-HQ-256_CELEBA-HQ_ep0*'))
    # Parallel(n_jobs=20)(delayed(backup.upload_models_celeba_hq_ddpm)(path, 'newsyctw') for path in paths)
    
    # LDM CelebA-HQ
    # paths: list = list(glob('/work/u2941379/workspace/backdoor_diffusion/exp_GenBadDiffusion_LDM_BadDiff_ODE_400EP_FP16/res_LDM-CELEBA-HQ-256_CELEBA-HQ-LATENT_ep2000*'))
    # paths += list(glob('/work/u2941379/workspace/backdoor_diffusion/exp_GenBadDiffusion_LDM_BadDiff_ODE_400EP_FP16/res_LDM-CELEBA-HQ-256_CELEBA-HQ-LATENT_ep0*'))
    # paths = ["/work/u2941379/workspace/backdoor_diffusion/exp_GenBadDiffusion_LDM_BadDiff_ODE_400EP_FP16/res_LDM-CELEBA-HQ-256_CELEBA-HQ-LATENT_ep2000_ode_c1.0_p0.9_GLASSES-CAT_psi1.0_lr0.0002_vp1.0_ve1.0_new-set"]
    # Parallel(n_jobs=20)(delayed(backup.upload_models_celeba_hq_ldm)(path, 'newsyctw') for path in paths)
    
    # NCSN CIFAR10
    # paths: list = list(glob('/work/u2941379/workspace/backdoor_diffusion/exp_GenBadDiffusion_NCSNPP_CIFAR10_TrojDiff_SDE_EXTEND/res_NCSN_CIFAR10_my_CIFAR10_ep*'))
    # paths = ["/work/u2941379/workspace/backdoor_diffusion/exp_GenBadDiffusion_NCSNPP_CIFAR10_TrojDiff_SDE_EXTEND/res_NCSN_CIFAR10_my_CIFAR10_ep20_sde_c1.0_p4.0_epr0.2_SM_STOP_SIGN-FEDORA_HAT_psi0.0_lr2e-05_rhw1.0_rhb0.0_extend_new-set"]
    # Parallel(n_jobs=20)(delayed(backup.upload_models_cifar10_ncsn)(path, 'newsyctw') for path in paths)
    
    
    # for path in paths:
    #     backup.upload_models_CIFAR10_DDPM(path=path, user_name='newsyctw')
    
    
    
    