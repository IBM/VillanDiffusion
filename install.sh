sudo apt update -y
sudo apt-get update -y
sudo apt-get install tmux htop nginx zip -y
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install

# pip install pyarrow==6.0.1
# pip install accelerate comet-ml matplotlib datasets tqdm tensorboard tensorboardX torchvision tensorflow-datasets einops pytorch-fid joblib PyYAML kaggle wandb torchsummary torchinfo lpips torchmetrics
# pip install git+https://github.com/Database-Project-2021/scalablerunner.git

git clone git@github.com:FrankCCCCC/diffusers.git
git fetch -a
git checkout -b my remotes/origin/my
cd diffusers
pip install .
cd ..

# echo 'export HF_DATASETS_CACHE="{Your HF Cache Path}"' >> ~/.bashrc

# wandb login --relogin --cloud {Your Key}

sudo su - root -c "curl -fsSL https://code-server.dev/install.sh | sh"

tmux new-session -d -s train
tmux new-session -d -s monitor "nvidia-smi -l 1"
tmux new-session -d -s code-server "code-server --bind-addr=0.0.0.0:5001 --auth none --cert=/home/u2941379/ssl/nginx.crt --cert-key=/home/u2941379/ssl/nginx.key"
tmux new-session -d -s tfboard "tensorboard --logdir ."
# tmux new-session -d -s notebook "nohup sudo -i -H -u u2941379 /run_jupyter.sh --port=8888 --notebook-dir=/work/u2941379 --config=/etc/jupyter/jupyter_notebook_config.py &"
nohup sudo -i -H -u u2941379 /run_jupyter.sh --port=8888 --notebook-dir=/work/u2941379 --config=/etc/jupyter/jupyter_notebook_config.py &

echo $HF_DATASETS_CACHE