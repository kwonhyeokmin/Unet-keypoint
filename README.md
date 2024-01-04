# Unet-keypoint
![title](assets/title.jpg)

## Introduction
This repo is for checking data validation of NIA-2023 Joint and Arthritis Datasets. This repo is contain training code and evalutation code related with keypoints task.

## Dependencies
* [Pytorch 1.11.0](https://pytorch.org/get-started/previous-versions/)
* [CUDA](https://developer.nvidia.com/cuda-downloads)
* [cuDNN](https://developer.nvidia.com/cudnn)

## Directory
### Root
Repository directory structure is like bellow.
```
${ROOT}
|-- common
|    |-- utils
|-- configs
|-- dataset
|-- models
|-- tools
`-- output
```
* `common` contains utility code for project.
* `configs` contains configure files like hyper-parameters for training model.
* `dataset` contains data loading codes.
* `models` contains model network.
* `tools` contains supporting code like data sampling or data spliting.
* `output` contains trained model, visualized output.

## Runing codes

### Start
* Run `pip install -r requirements.txt` to install required libraries.
* You have to check and modify `constants.py` for setting data root. 

### Prepare datasets
#### Data split for making train, validation, test data
```shell
python3 data_split.py
```

### Train
In the root folder, please run like bellow. If you want to track with wandb, add use_wandb option.
```shell
# example: python3 train.py --cfg configs/config.yaml --use_wandb
python3 train.py --cfg [CONFIG_FILE_PATH]
```

### Evaluate
```shell
# example: python3 eval.py --checkpoint output/model_dump/snapshot_150.pt
python3 eval.py --checkpoint [MODEL_FILE_PATH]
```