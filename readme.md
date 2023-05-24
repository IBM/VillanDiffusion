# VillanDiffusion

## Environment

- Python 3.8.5
- PyTorch 1.10.1+cu11 or 1.11.0+cu102

## Usage

### Install Require Packages and Prepare Essential Data

Please run

```bash
bash install.sh
```

### Wandb Logging Support

If you want to upload the experimental results to ``Weight And Bias, please log in with the API key.

```bash
wandb login --relogin --cloud <API Key>
```

### Prepare Dataset

- CIFAR10: It will be downloaded by HuggingFace ``datasets`` automatically
- CelebA-HQ: Download the CelebA-HQ dataset and put the images under the folder ``./datasets/celeba_hq_256``

### Backdoor Unconditional Diffusion Models with VillanDiffusion

Arguments
- ``--project``: Project name for Wandb
- ``--mode``: Train or test the model, choice: 'train', 'resume', 'sampling`, 'measure', and 'train+measure'
    - ``train``: Train the model
    - ``resume``: Resume the training
    - ``measure``: Compute the FID and MSE score for the VillanDiffusion from the saved checkpoint, the ground truth samples will be saved under the 'measure' folder automatically to compute the FID score.
    - ``train+measure``: Train the model and compute the FID and MSE score
    - ``sampling``: Generate clean samples and backdoor targets from a saved checkpoint
- ``--sched``: Sampling algorithms for the diffusion models. Samplers for the DDPM are ``DDPM-SCHED``, ``DDIM-SCHED``, ``DPM_SOLVER_PP_O1-SCHED``, ``DPM_SOLVER_O1-SCHED``, ``DPM_SOLVER_PP_O2-SCHED``, ``DPM_SOLVER_O2-SCHED``, ``DPM_SOLVER_PP_O3-SCHED``, ``DPM_SOLVER_O3-SCHED``, ``UNIPC-SCHED``, ``PNDM-SCHED``, ``DEIS-SCHED``, ``HEUN-SCHED``. ``SCORE-SDE-VE-SCHED`` is used by score-based models.
- ``--solver_type``: Backdoor for the ODE or SDE samplers. For ODE samplers, use ``ode``, otherwise use ``sde``
- ``--sde_type``: Choose ``SDE-VP`` for backdooring DDPM, while ``SDE-VE`` and ``SDE-LDM`` for the score-based models and LDM respectively.
- ``--infer_steps``: Sampling steps of the specified sampler. We recommend 50 steps for ``PNDM``, ``HEUN``, ``LMSD``, and ``DDIM``, otherwise 20 steps.
- ``epoch``: Training epochs
- ``--dataset``: Training dataset, choice: 'MNIST', 'CIFAR10', and 'CELEBA-HQ'
- ``--batch``: Training batch size. Note that the batch size must be able to divide 128 for the CIFAR10 dataset and 64 for the CelebA-HQ dataset.
- ``--eval_max_batch``: Batch size of sampling, default: 256
- ``--epoch``: Training epoch num, default: 50
- ``--learning_rate``: Learning rate, default for 32 * 32 image: '2e-4', default for larger images: '8e-5'
- ``--poison_rate``: Poison rate
- ``--trigger``: Trigger pattern, default: ``BOX_14``, choice: ``BOX_14``, ``STOP_SIGN_14``, and ``GLASSES``.
- ``--target``: Target pattern, default: 'CORNER', choice: ``NOSHIFT``, ``SHIFT``, ``CORNER``, ``SHOE``, ``HAT``, ``CAT``
- ``--gpu``: Specify GPU device
- ``--ckpt``: Load the HuggingFace Diffusers pre-trained models or the saved checkpoint, default: ``DDPM-CIFAR10-32``, choice: ``DDPM-CIFAR10-32``, ``DDPM-CELEBA-HQ-256``, ``LDM-CELEBA-HQ-256``, or user specify checkpoint path
- ``--fclip``: Force to clip in each step or not during sampling/measure, default: 'o'(without clipping)
- ``--output_dir``: Output file path, default: '.'

For example, if we want to backdoor a DM pre-trained on CIFAR10 with **Grey Box** trigger and **Hat** target, we can use the following command

```bash
python VillanDiffusion.py --project default --mode train+measure --dataset CIFAR10 --batch 128 --epoch 50 --poison_rate 0.1 --trigger BOX_14 --target HAT --ckpt DDPM-CIFAR10-32 --fclip o -o --gpu 0
```

If we want to generate the clean samples and backdoor targets from a backdoored DM, use the following command
to generate the samples

```bash
python VillanDiffusion.py --project default --mode sampling --eval_max_batch 256 --ckpt res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_BOX_14-HAT --fclip o --gpu 0
```

### Backdoor Conditional Diffusion Models with VillanDiffusion

- ``--pretrained_model_name_or_path``: Specify the backdoor model. We recommend to use ``CompVis/stable-diffusion-v1-4``.
- ``--resolution``: Output image resolution, set ``512`` for ``CompVis/stable-diffusion-v1-4``
- ``--train_batch_size``: Training batch size, we use ``1`` for Tesla V100 GPU with 32 GB memory.
- ``--lr_scheduler``: Learning rate scheduler, we recommend to use ``cosine``
- ``--lr_warmup_steps``: Learning rate warm-up steps, we recommend to use ``500`` steps.
- ``--target``: Specify backdoor attack target image, choice: ``HACKER`` and ``CAT``
- ``--dataset_name``: Specify the training dataset, choice: ``POKEMON-CAPTION`` and ``CELEBA-HQ-DIALOG``
- ``--lora_r``: LoRA rank, we recommend to use ``4``
- `