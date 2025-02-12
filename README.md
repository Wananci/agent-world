# agent-world

This is a repo of the agent-world work. 

# Get Started

# Installation

**(1) install pytorch env**
```
conda create -n agent-world python=3.10
conda activate agent-world
```

**(2) clone repo**
```
git clone git@github.com:Wananci/agent-world.git
```

**(3) dataset download**
```
cd ${YOUR_PATH_TO_AGENT_WORLD}
mkdir ./datasets
cd ./datasets
## if you want to train on the bigger Droid dataset (1.7TB)
gsutil -m cp -r gs://gresearch/robotics/droid ./ 
## if you want to train on the smaller Droid-100 dataset (2GB)
gsutil -m cp -r gs://gresearch/robotics/droid_100 ./
```

**(4) download pre-trained model**
|Model|Path that you should put|
|:------:|:------:|
|[ViT-B-32](https://huggingface.co/Kleinhe/CAMD/resolve/main/weights/ViT-B-32.pt)|./checkpoints/clip/|
|[mae_pretrain_vit_base](https://drive.google.com/drive/folders/1tYBcatJICNxciXZr5-H1hobd8dX3InT1)|./checkpoints/vit_mae/|

# Acknowledgement 

Thanks [Seer](https://github.com/OpenRobotLab/Seer/tree/main) for their opening source. 