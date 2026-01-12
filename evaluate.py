from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import os
from dataloader.dsec_split import DSECsplit
from dataloader.dsec_full import DSECfull
from pathlib import Path

import flow_vis
from utils.visualization import gray_to_colormap

@torch.no_grad()
def get_vis_prog(model):
    visualizations = []
    loader = DSECfull('prog')

    for index, (voxel, depth_gt, valid, img) in enumerate(loader):
        voxel = voxel[None].cuda()
        _, pred = model(voxel.cuda())

        pred = pred[0].cpu().numpy()
        depth_vis = gray_to_colormap(pred)
        
        depth_vis_masked = depth_vis.copy()
        depth_vis_masked[~valid] = 0

        depth_gt_vis = (gray_to_colormap(depth_gt[0]))

        row1 = np.hstack([img, depth_vis])
        row2 = np.hstack([depth_gt_vis, depth_vis_masked])

        frame = np.vstack(row1, row2)

        visualizations.append((index, frame))

        break

    return visualizations

@torch.no_grad()                   
def validate_DSEC(model):
    model.eval()
    val_dataset = DSECsplit('val')
    
    epe_list = []
    seg_loss_list = []
    out_list = []

    seg_loss_fn = nn.CrossEntropyLoss(ignore_index=255, reduction = 'mean')

    # Visualization index
    vis_idx = np.random.randint(0, len(val_dataset))
    random_vis = None   

    bar = tqdm(enumerate(val_dataset),total=len(val_dataset), ncols=60)
    bar.set_description('Test')
    for index, (voxel1, voxel2, flow_map, valid2D, img, seg_gt) in bar:
        voxel1 = voxel1[None].cuda()
        voxel2 = voxel2[None].cuda() 
        flow_pred, seg_out = model(voxel1, voxel2)#[1,2,H,W]
        flow_pred = flow_pred[0].cpu()
        seg_out = seg_out[0].cpu()

        # Flow loss
        epe = torch.sum((flow_pred- flow_map)**2, dim=0).sqrt()#[H,W]
        mag = torch.sum(flow_map**2, dim=0).sqrt()#[H,W]

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid2D.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

        # Segmentation loss
        seg_loss = seg_loss_fn(seg_out[None], seg_gt[None].long())
        seg_loss_list.append(seg_loss)

        # # Generate visualization
        # if index == vis_idx:
        #     flow_pred = flow_pred.numpy()
        #     flow_map = flow_map.numpy()
        #     valid2D = valid2D.numpy()

        #     seg_pred = seg_out.detach().max(dim=0)[1].cpu().numpy()
        #     seg_gt = seg_gt.numpy()

        #     img = img.numpy()

        #     random_vis = get_vis_matrix(flow_pred, flow_map, valid2D, seg_pred, seg_gt, img)

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    seg_loss = np.mean(seg_loss_list)
    
    print("Validation DSEC-TEST: %f, %f" % (epe, f1))
    return {'dsec-epe': epe, 'dsec-f1': f1, 'seg_loss':seg_loss, 'visualization': random_vis}


if __name__ == "__main__":
   
    from argparse import ArgumentParser
    

    parser =ArgumentParser()
    parser.add_argument("-c", "--checkpoint", type=str, help="Path to a saved checkpoint file (.pth)")
    parser.add_argument("-b", "--input_bins", type=int, default=15, help="Number of input bins")

    args = parser.parse_args()
