from config import cfg
import torch
from dataset.dataset import CTDataset
from dataset.nia import NIADataset
from torch.utils.data import DataLoader
from models.unet import UNet
from tqdm import tqdm
import torch.optim as optim
import os.path as osp
import argparse
from config import update_config
import wandb
from models.loss import UNetCrossEntropyLoss
from torch.nn.parallel import DistributedDataParallel as DDP


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=False, help='Path to pretrained checkpoint')
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--gpus',
                        help='gpu ids for use',
                        default='0',
                        type=str)
    parser.add_argument('--use_wandb',
                        help='use wandb',
                        action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = make_args()
    update_config(args.cfg)

    # Select device (gpu | cpu)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    use_wandb = args.use_wandb
    n_gpus = len(args.gpus.split(','))

    if use_wandb:
        wandb.init(project=cfg.hyp.PROJECT_NAME,
                   name=f'{cfg.hyp.OPTIMIZER.TYPE}_lr{cfg.hyp.OPTIMIZER.LR}')
        wandb.config.update({
            'batch_size': cfg.batch_size,
            'num_workers': cfg.num_thread,
            'optimizer': cfg.hyp.OPTIMIZER.TYPE,
            'learning_rate': cfg.hyp.OPTIMIZER.LR,
            'weight_decay': cfg.hyp.OPTIMIZER.WD,
        })
    # ********************
    # 1. Load datasets
    # ********************
    train_dataset = NIADataset(data_split='train', tag='SAG')
    val_dataset = NIADataset(data_split='val', tag='SAG')

    train_loader = CTDataset(train_dataset, transforms=cfg.data_transforms['train'])
    train_generator = DataLoader(dataset=train_loader,
                                 batch_size=int(cfg.batch_size / n_gpus),
                                 num_workers=int(cfg.num_thread / n_gpus),
                                 pin_memory=True)
    val_loader = CTDataset(val_dataset, transforms=cfg.data_transforms['train'])
    val_generator = DataLoader(dataset=val_loader,
                               batch_size=int(cfg.batch_size / n_gpus),
                               num_workers=int(cfg.num_thread / n_gpus),
                               pin_memory=True)
    # ****************
    # 2. Setting Loss function
    # ****************
    criterion = UNetCrossEntropyLoss()

    # ****************
    # 3. Training
    # ****************
    # load model
    model = UNet(in_channels=3, n_classes=len(train_dataset.cat_name), n_channels=48).to(device)
    if n_gpus > 1:
        model = DDP(model)
    optimizer = optim.Adam(model.parameters(),
                           lr=float(cfg.hyp.OPTIMIZER.LR),
                           weight_decay=float(cfg.hyp.OPTIMIZER.WD))

    for epoch in range(int(cfg.hyp.TRAINING.EPOCHS)):
        pbar = tqdm(enumerate(train_generator), total=len(train_generator), desc=f'Train - epoch: {epoch}')
        tracking_loss = {
            'CrossEntropyLoss': torch.tensor(0.).float(),
            'ValCrossEntropyLoss': torch.tensor(0.).float(),
        }
        avg_loss = 0.0
        for step, (images, masks, _, _) in pbar:
            B = images.shape[0]
            images = images.to(device, dtype=torch.float)
            masks = masks.to(device, dtype=torch.float)

            y_pred = model(images)
            losses = criterion(y_pred, masks)

            # Gradient update
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            # tracking loss
            tracking_loss['CrossEntropyLoss'] += losses.detach().item() / B / int(cfg.hyp.TRAINING.EPOCHS)

            _losses = float(losses.detach().cpu().numpy())
            pbar.set_description(
                f'Epoch {epoch + 1}/{cfg.hyp.TRAINING.EPOCHS} Train Loss - {format(_losses, ".04f")}')
            avg_loss += _losses

        pbar.set_description(f'Epoch {epoch+1}/{cfg.hyp.TRAINING.EPOCHS} Train Loss - {format(avg_loss/B, ".04f")}')

        # ****************
        # 4. Validation
        # ****************
        vpbar = tqdm(enumerate(val_generator), total=len(val_generator), desc=f'Val - epoch: {epoch}')
        val_avg_loss = 0.0
        for step, (images, masks, _, _) in vpbar:
            B = images.shape[0]
            images = images.to(device, dtype=torch.float)
            masks = masks.to(device, dtype=torch.float)
            with torch.no_grad():
                y_pred = model(images)
                losses = criterion(y_pred, masks)
            # tracking loss
            tracking_loss['ValCrossEntropyLoss'] += losses.detach().item() / B / int(cfg.hyp.TRAINING.EPOCHS)
            _losses = float(losses.detach().cpu().numpy())
            vpbar.set_description(
                f'ValLoss - {format(_losses, ".04f")}')
            avg_loss += _losses

        if use_wandb:
            wandb.log(tracking_loss)

        # ****************
        # 5. Model save
        # ****************
        if (epoch+1) % 10 == 0:
            file_path = osp.join(cfg.model_dir, f'snapshot_{epoch}.pt')
            torch.save(model.state_dict(), file_path)

    # model save
    file_path = osp.join(cfg.model_dir, f'snapshot_{cfg.hyp.TRAINING.EPOCHS}.pt')
    torch.save(model.state_dict(), file_path)
    print('End training')
