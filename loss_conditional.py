import torch

class LossFn:
    def __init__(self):
        pass
    
    # MODIFIED: 
    @staticmethod
    def extract_into_tensor(a, t, x_shape):
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))
    # MODIFIED: 
    @staticmethod
    def get_R_step_baddiff(alphas_cumprod: torch.Tensor, alphas: torch.Tensor, psi: float=1, solver_type: str='ode') -> torch.Tensor:
        # Variance Preserve
        vp_step = 1 - alphas_cumprod ** 0.5
        
        # Variance Explode
        ve_step = (1 - alphas_cumprod) ** 0.5
        
        # Coefficients & Steps
        R_step = psi * vp_step + (1 - psi) * ve_step
        
        if str(solver_type).lower() == 'ode':
            return R_step
        elif str(solver_type).lower() == 'sde':
            return R_step
        else:
            raise NotImplementedError(f"Coefficient solver_type: {solver_type} isn't implemented")
    # MODIFIED: 
    @staticmethod
    def get_ks(alphas: torch.Tensor, alphas_cumprod: torch.Tensor):
        ks = [(1 - alphas_cumprod[0]) ** 0.5]
        residuals = [0]
        for i, (alphas_cumprod_i, alphas_i) in enumerate(zip(alphas_cumprod, alphas)):
            if i < 1:
                continue
            residuals.append((alphas_i ** 0.5) * (ks[i - 1] + residuals[i - 1]))
            ks.append((1 - alphas_cumprod_i) ** 0.5 - residuals[i])
        return torch.Tensor(ks)
    # MODIFIED: 
    @staticmethod
    def get_R_coef_baddiff(alphas_cumprod: torch.Tensor, alphas: torch.Tensor, psi: float=1, solver_type: str='ode', ve_scale: float=1.0) -> torch.Tensor:
        # Variance Preserve
        vp_coef = (1 - alphas ** 0.5) * (1 - alphas_cumprod) ** 0.5 / (1 - alphas)
        
        # Variance Explode
        if LossFn.get_R_coef_baddiff.ks == None:
            LossFn.get_R_coef_baddiff.ks = LossFn.get_ks(alphas=alphas, alphas_cumprod=alphas_cumprod)
        ks = LossFn.get_R_coef_baddiff.ks.to(device=alphas.device, dtype=alphas.dtype)
        ve_coef = - ve_scale * ((alphas ** 0.5 - 1) * (1 - alphas_cumprod) ** 0.5 * (1 - alphas) - ks * (alphas - alphas_cumprod)) / (1 - alphas)
        
        # Coefficients & Steps
        R_coef = psi * vp_coef + (1 - psi) * ve_coef
        
        if str(solver_type).lower() == 'ode':
            return 2 * R_coef
        elif str(solver_type).lower() == 'sde':
            return R_coef
        else:
            raise NotImplementedError(f"Coefficient solver_type: {solver_type} isn't implemented")
    
    # MODIFIED: 
    @staticmethod
    def get_R_scheds_baddiff(alphas_cumprod: torch.Tensor, alphas: torch.Tensor, psi: float=1, solver_type: str='ode') -> torch.Tensor:
        R_step = LossFn.get_R_step_baddiff(alphas_cumprod=alphas_cumprod, alphas=alphas, psi=psi, solver_type=solver_type)
        R_coef = LossFn.get_R_coef_baddiff(alphas_cumprod=alphas_cumprod, alphas=alphas, psi=psi, solver_type=solver_type)
        return R_step, R_coef
    # MODIFIED: 
    def get_x_noisy(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor=None, R: torch.Tensor=None, psi: float=1, solver_type: str="ode") -> torch.Tensor:
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        if R == None:
            return x_noisy
        else:
            alphas_cumprod_t = LossFn.extract_into_tensor(self.alphas_cumprod, t, x_start.shape)
            alphas_t = LossFn.extract_into_tensor(self.alphas, t, x_start.shape)
            return x_noisy + R * LossFn.get_R_step_baddiff(alphas_cumprod=alphas_cumprod_t, alphas=alphas_t, psi=psi, solver_type=solver_type)
    # MODIFIED: 
    def get_target_x0(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, R: torch.Tensor=None, psi: float=1, solver_type: str="ode") -> torch.Tensor:
        if R == None:
            return x_start
        else:
            return x_start
    # MODIFIED: 
    def get_target_eps(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, R: torch.Tensor=None, psi: float=1, solver_type: str="ode") -> torch.Tensor:
        if R == None:
            return noise
        else:
            alphas_cumprod_t = LossFn.extract_into_tensor(self.alphas_cumprod, t, x_start.shape)
            alphas_t = LossFn.extract_into_tensor(self.alphas, t, x_start.shape)
            return noise + R * LossFn.get_R_coef_baddiff(alphas_cumprod=alphas_cumprod_t, alphas=alphas_t, psi=psi, solver_type=solver_type)
LossFn.get_R_coef_baddiff.ks = None