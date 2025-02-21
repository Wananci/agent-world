# Agent-World

This is a repo of the agent-world work. 

# Get Started

# Installation

**(1) Install Pytorch Env**
```
conda create -n agent-world python=3.10
conda activate agent-world
```

**(2) Clone Repo**
```
git clone git@github.com:Wananci/agent-world.git
```

**(3) Dataset Download**
```
cd ${YOUR_PATH_TO_AGENT_WORLD}
mkdir ./datasets
cd ./datasets
## if you want to train on the bigger Droid dataset (1.7TB)
gsutil -m cp -r gs://gresearch/robotics/droid ./ 
## if you want to train on the smaller Droid-100 dataset (2GB)
gsutil -m cp -r gs://gresearch/robotics/droid_100 ./
```

**(4) Download Third Party Packages**
```
cd ${YOUR_PATH_TO_AGENT_WORLD}
pip install -r requirements.txt
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
```

**(5) Run**
```
bash scripts/droid/scratch.sh
```

# Pre-trained Model
|Model|Path that you should put|
|:------:|:------:|
|[ViT-B-32](https://huggingface.co/Kleinhe/CAMD/resolve/main/weights/ViT-B-32.pt)|./checkpoints/clip/|
|[mae_pretrain_vit_base](https://drive.google.com/file/d/1bSsvRI4mDM3Gg51C6xO0l9CbojYw3OEt/view)|./checkpoints/vit_mae/|

## 📆 TODO <a name="todos"></a>
- [ ] Release agent-world test code. 
- [ ] Release agent-world trained model.
- [ ] Run on the Droid dataset.
- [ ] Release the visualization image.

# Acknowledgement 
Thanks [Seer](https://github.com/OpenRobotLab/Seer/tree/main) for their opening source. 