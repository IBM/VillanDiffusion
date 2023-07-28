from typing import List

import torch
from torch import nn
from torchmetrics import StructuralSimilarityIndexMeasure

from util import Log

def get_batch_sizes(sample_n: int, max_batch_n: int):
    if sample_n > max_batch_n:
        replica = sample_n // max_batch_n
        residual = sample_n % max_batch_n
        batch_sizes = [max_batch_n] * (replica) + ([residual] if residual > 0 else [])
    else:
        batch_sizes = [sample_n]
    return batch_sizes

def batchify(xs, max_batch_n: int):
    batch_sizes = get_batch_sizes(sample_n=len(xs), max_batch_n=max_batch_n)
    
    print(f"xs len(): {len(xs)}")    
    print(f"batch_sizes: {batch_sizes}, max_batch_n: {max_batch_n}")
    # print(f"Max_batch_n: {max_batch_n}")
    res: List = []
    cnt: int = 0
    for i, bs in enumerate(batch_sizes):
        res.append(xs[cnt:cnt+bs])
        cnt += bs
    return res

class Metric:
    @staticmethod
    def batch_metric(a: torch.Tensor, b: torch.Tensor, max_batch_n: int, fn: callable):
        a_batchs = batchify(xs=a, max_batch_n=max_batch_n)
        b_batchs = batchify(xs=b, max_batch_n=max_batch_n)
        scores: List[torch.Tensor] = [fn(a, b) for a, b in zip(a_batchs, b_batchs)]
        if len(scores) == 1:
            return scores[0].mean()
        return torch.cat(scores, dim=0).mean()
    
    @staticmethod
    def get_batch_operator(a: torch.Tensor, b: torch.Tensor):
        batch_operator: callable = None
        if torch.is_tensor(a) and torch.is_tensor(b):
            batch_operator = Metric.batch_metric
        elif (torch.is_tensor(a) and not torch.is_tensor(b)) or (not torch.is_tensor(a) and torch.is_tensor(b)):
            raise TypeError(f"Both arguement a {type(a)} and b {type(b)} should have the same type")
        # else:
            # batch_operator = Metric.batch_object_metric
        return batch_operator
    
    @staticmethod
    def mse_batch(a: torch.Tensor, b: torch.Tensor, max_batch_n: int):
        Log.critical("COMPUTING MSE")
        batch_operator: callable = Metric.get_batch_operator(a=a, b=b)
        def metric(x, y):
            mse: torch.Tensor = nn.MSELoss(reduction='none')(x, y).mean(dim=[i for i in range(1, len(x.shape))])
            print(f"MSE: {mse.shape}")
            return mse
        return  float(batch_operator(a=a, b=b, max_batch_n=max_batch_n, fn=metric))
    
    @staticmethod
    def mse_thres_batch(a: torch.Tensor, b: torch.Tensor, thres: float, max_batch_n: int):
        Log.critical("COMPUTING MSE-THRESHOLD")
        batch_operator: callable = Metric.get_batch_operator(a=a, b=b)
        def metric(x, y):
            # print(f"x: {x.shape}, y: {y.shape}")
            # print(f"Mean Dims: {[i for i in range(1, len(x))]}")
            probs: torch.Tensor = nn.MSELoss(reduction='none')(x, y).mean(dim=[i for i in range(1, len(x.shape))])
            mse_thres: torch.Tensor = torch.where(probs < thres, 1.0, 0.0)
            print(f"MSE Threshold: {mse_thres.shape}")
            return mse_thres
        return  float(batch_operator(a=a, b=b, max_batch_n=max_batch_n, fn=metric))
    
    @staticmethod
    def ssim_batch(a: torch.Tensor, b: torch.Tensor, device: str, max_batch_n: int):
        Log.critical("COMPUTING SSIM")
        batch_operator: callable = Metric.get_batch_operator(a=a, b=b)
        def metric(x, y):
            ssim: torch.Tensor = StructuralSimilarityIndexMeasure(data_range=1.0, reduction='none').to(device)(x, y)
            if len(ssim.shape) < 1:
                ssim = ssim.unsqueeze(dim=0)
            print(f"SSIM: {ssim.shape}")
            return ssim
        return  float(batch_operator(a=a, b=b, max_batch_n=max_batch_n, fn=metric))