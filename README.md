# Unet-keypoint

## Install

```shell
pip install -r requirements.txt
```

## Model train
```shell
python3 train.py --cfg configs/config.yaml --use_wandb
```

## Model evaluate
```shell
python3 eval.py --checkpoint [MODEL_FILE_PATH]
```
