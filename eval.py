import cv2
import numpy as np

from config import cfg
from collections import defaultdict
from dataset.dataset import CTDataset
from dataset.nia import NIADataset
from torch.utils.data import DataLoader
import torch
from models.unet import UNet
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import copy
from datetime import datetime
from sklearn.metrics import precision_recall_curve



def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint for evaluate')
    parser.add_argument('--is_vis',
                        help='True/False of visualization. If you store option, it means true',
                        action='store_true')
    args = parser.parse_args()
    return args


def oks(g, d, a):
    """ Calculate OKS score.
        gt_kps, pred_kps: numpy.ndarray([batch_size, num_joints, height, width)
    """
    sigmas = np.array([.25, .25, .25, .25])
    vars = (sigmas * 2) ** 2

    xg = g.transpose(0, 2, 1)[:,0::2]
    yg = g.transpose(0, 2, 1)[:,1::3]
    xd = d.transpose(0, 2, 1)[:,0::2]
    yd = d.transpose(0, 2, 1)[:,1::3]
    ious = np.zeros((d.shape[0]))
    for n_d in range(0, d.shape[0]):
        xdd = xd[n_d]
        ydd = yd[n_d]
        xgd = xg[n_d]
        ygd = yg[n_d]
        dx = xdd - xgd
        dy = ydd - ygd
        e = (dx ** 2 + dy ** 2) / vars / (a[n_d] + np.spacing(1)) / 2
        ious[n_d] = np.sum(np.exp(-e)) / e.shape[1] if e.shape[1] != 0 else 0.0
    return ious

def get_max_preds(batch_heatmaps):
    """ get predictions from score maps
        heatmaps: torch.tensor([batch_size, num_joints, height, width])
    """
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = torch.argmax(heatmaps_reshaped, dim=2)
    maxvals = torch.amax(heatmaps_reshaped, dim=2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = torch.tile(idx, (1, 1, 2))

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = torch.floor((preds[:, :, 1]) / width)

    pred_mask = torch.tile(torch.greater(maxvals, 0.0), (1, 1, 2))

    preds *= pred_mask
    return preds, maxvals


if __name__ == '__main__':
    args = make_args()
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print('Start evaluation. ', dt_string)

    # Select device (gpu | cpu)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')
    score = defaultdict(list)

    # ********************
    # 1. Load datasets
    # ********************
    dataset = NIADataset(data_split='test', tag='SAG')
    dataset_loader = CTDataset(dataset, transforms=cfg.data_transforms['test'])
    data_generator = DataLoader(dataset=dataset_loader, batch_size=cfg.batch_size, shuffle=False,
                                num_workers=cfg.num_thread, pin_memory=True)
    # ********************
    # 2. Load model
    # ********************
    # load model
    model = UNet(in_channels=3, n_classes=len(dataset.cat_name), n_channels=48).to(device)
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()

    # ****************
    # 3. Evaluate
    # ****************
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(dataset.cat_name) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    cnt = 0
    stp, sfp, recall, precision = 0, 0, 0, 0
    score_list, correct_list = [], []

    s = ('%4s' + '%37s' + '%5s' * 8 + '%42s' * 2) % ('No', 'DataID', 'TP', 'TN', 'FP', 'FN', 'sTP', 'sFP', 'P', 'R', 'GT', 'Pred')
    print()
    print(s)

    pbar = tqdm(enumerate(data_generator), total=len(data_generator))
    for step, (images, masks, images_bgr, areas, data_ids) in pbar:
        images = images.to(device, dtype=torch.float)
        masks = masks.to(device, dtype=torch.float)
        B, C, H, W = images.shape
        with torch.no_grad():
            y_pred_heatmap = model(images)
        preds, maxvals = get_max_preds(y_pred_heatmap)
        gt_kpts, _ = get_max_preds(masks)

        # calculate oks
        for oks_thrs in [0.50]:
            _score = oks(gt_kpts.cpu().numpy(), preds.cpu().numpy(), areas.cpu().numpy())
            mAP_score = np.where(_score > oks_thrs, 1, 0).mean()
            score[f'AP:{oks_thrs:.2f}'].append(float(mAP_score))

        iou_thrs = 0.50
        _score = oks(gt_kpts.cpu().numpy(), preds.cpu().numpy(), areas.cpu().numpy())
        for b in range(B):
            tp = int(_score[b] >= iou_thrs)
            tn = 0.0
            fp = 0.0
            fn = int(_score[b] < iou_thrs)

            sfp += fp
            stp += tp
            precision = round(precision + tp / (tp + fp + 0.001), 2)
            recall = round(recall + tp / (tp + fn + 0.001), 2)
            pbar.set_description(('%4s' + '%37s' + '%5s' * 8 + '%42s' * 2) %
                                 (cnt, data_ids[b], tp, tn, fp, fn, stp, sfp, precision, recall, gt_kpts[b].cpu().numpy().flatten().tolist(), preds[b].cpu().numpy().flatten().tolist()))
            score_list.append(int(_score[b]))
            correct_list.append(tp)
            cnt += 1

        # ****************
        # 4. Visualization (with first batch)
        # ****************
        if args.is_vis:
            for b in range(B):
                # GT
                gt_vis_img = copy.deepcopy(images_bgr[b]).cpu().numpy()
                vis_path = f'{cfg.vis_dir}/bi_{str(step*10 + b).zfill(3)}.jpg'
                gt_kpt = gt_kpts[b]
                for cat_id, gt_kpt in enumerate(gt_kpt):
                    gt_vis_img = cv2.circle(gt_vis_img, (int(gt_kpt[0]), int(gt_kpt[1])), 5, (255, 0, 0), -1)
                    gt_vis_img = cv2.putText(gt_vis_img, dataset.cat_name[cat_id], (int(gt_kpt[0])-2, int(gt_kpt[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1)

                # Pred
                pred_vis_img = copy.deepcopy(images_bgr[b]).cpu().numpy()
                pred = preds[b]
                for cat_id, pred_kpt in enumerate(pred):
                    pred_vis_img = cv2.circle(pred_vis_img, (int(pred_kpt[0]), int(pred_kpt[1])), 5, (0, 0, 255), -1)
                    pred_vis_img = cv2.putText(pred_vis_img, dataset.cat_name[cat_id], (int(pred_kpt[0])-2, int(pred_kpt[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1)

                merge_img = np.concatenate((gt_vis_img, pred_vis_img), axis=1)
                vis_path = f'{cfg.vis_dir}/{str(step*10 + b).zfill(3)}.jpg'
                cv2.imwrite(vis_path, merge_img)

    # ****************
    # 3. Print result
    # ****************
    print()
    for k, v in score.items():
        print(k, ': ', np.mean(v))

    # save graph
    fig, ax = plt.subplots()
    precision, recall, thresholds = precision_recall_curve(correct_list, score_list)
    # create precision recall curve
    ax.plot(recall, precision)

    # add axis labels to plot
    ax.axis([0,1,0,1])
    ax.set_title('Precision-Recall Curve')
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')

    # display plot
    plt.savefig(f'../data/PR_curve_kpt.png.png')

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print('End evaluation. ', dt_string)
