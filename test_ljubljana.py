import matplotlib.pyplot as plt
import torch
import diffdrr.data
from diffdrr.drr import DRR
import numpy as np
import pandas as pd
import h5py
from diffdrr.pose import convert, axis_angle_to_matrix, matrix_to_axis_angle
import time
from torchvision.transforms import Compose, Lambda, Normalize, Resize
from torch.nn.parallel import DistributedDataParallel as DDP
from diffdrr.metrics import MultiscaleNormalizedCrossCorrelation2d

from models import regressor_attention, registration_attention
import torch.nn as nn
import os
from utils import warp_to_canonical_space, square
from diffdrr.pose import RigidTransform
from pydicom import dcmread
import tqdm
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
mean_absolute_error = nn.L1Loss(reduction='none')
ncc = MultiscaleNormalizedCrossCorrelation2d([None, 16], [0.5, 0.5]).to(device)

mtre_tot = []
mpd = []
alpha = []
beta = []
gamma = []
x = []
y = []
z = []
parameterization = "axis_angle"
convention = None
indices = [0, 25, 50, 75, 110, 150, 250, 375, 500, 750, 1000]
for id_number in range(1,11):
    dest_path = os.path.join("/lustre/fswork/projects/rech/gfu/uur34ii/data/ljubljana",
                             f"subject{id_number:02d}_results")
    os.makedirs(dest_path, exist_ok=True)
    model_dict = torch.load(
        f"/lustre/fswork/projects/rech/gfu/uur34ii/model/ljubljana_models/subject{id_number:02d}/model_1499.pth")
    datapath = f"/lustre/fswork/projects/rech/gfu/uur34ii/data/ljubljana/subject{id_number:02d}"
    xrays = Path(os.path.join(datapath, "xrays"))
    print(xrays)
    model1 = regressor_attention()
    model2 = registration_attention()

    model1.load_state_dict(model_dict["model_state_dict1"])
    model1.to("cuda")
    model1.eval()

    model2.load_state_dict(model_dict["model_state_dict2"])
    model2.to("cuda")
    model2.eval()

    for img, gt in tqdm.tqdm(list(zip(sorted([f for f in xrays.glob("*.dcm") if not f.name.endswith("_max.dcm")]),
                                      sorted(xrays.glob("*.pt"))))):
        pose = torch.load(gt)
        pose = RigidTransform(pose['pose'][0]).to(device)
        print(os.path.basename(os.path.splitext(img)[0]))
        img_path = img
        csv_file_path = f"/lustre/fswork/projects/rech/gfu/uur34ii/scripts/phd_federica/RigidPoseEstimation//fiducials/subject{id_number:02d}_fiducials.csv"
        df = pd.read_csv(csv_file_path)
        fiducials = np.array(df.values)
        fiducials = torch.tensor([fiducials]).to(torch.float)

        subject = diffdrr.data.read(
            f"/lustre/fswork/projects/rech/gfu/uur34ii/data/ljubljana/subject{id_number:02d}/volume.nii.gz",
            orientation="AP", bone_attenuation_multiplier=2, fiducials=fiducials)

        ds = dcmread(img)
        img = ds.pixel_array.astype(np.int32)
        img = torch.from_numpy(img).to(torch.float32)[None, None]

        # Get intrinsic parameters of the imaging system
        sdd = ds.DistanceSourceToDetector
        try:
            dely, delx = ds.PixelSpacing

        except AttributeError:
            try:
                dely, delx = ds.ImagerPixelSpacing
            except AttributeError:
                raise AttributeError("Cannot find pixel spacing in DICOM file")
        try:
            y0, x0 = ds.DetectorActiveOrigin

        except AttributeError:
            y0, x0 = 0.0, 0.0

        height = ds.Rows  # Rows
        width = ds.Columns  # Columns
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)

        if height != width:
            img = square(img, height,width)

        target_size = max(height, width)
        p = delx
        p_target = p * 2480 /256
        p = p * target_size/256
        height = int(256)
        width = int(256)

        transforms = Compose(
            [
                Resize((int(height), int(width)), antialias=True),
                Lambda(lambda x: (((x - x.min()) / (x.max() - x.min() + 1e-6)))),
            ]
        )

        img = transforms(img).to(device)
        img += 1
        img = img.max().log() - img.log()
        img = transforms(img).to(device)
        drr = DRR(
            subject,  # A torchio.Subject object storing the CT volume, origin, and voxel spacing
            sdd=sdd,  # Source-to-detector distance (i.e., the C-arm's focal length)
            height=height,  # Height of the DRR (if width is not seperately provided, the generated image is square)
            width=width,  # Height of the DRR (if width is not seperately provided, the generated image is square)
            delx=p,  # Pixel spacing (in mm)
            x0=-x0,
            y0=y0,
            renderer="trilinear",
            reverse_x_axis=False
        ).to(device)

        K_real = drr.detector.intrinsic

        ##### USE THIS img IF YOU WANT TO TEST ON DRR TEST SET #######
        #img = drr(pose)
        #img = transforms(img)

        drr = DRR(
            subject,  # A torchio.Subject object storing the CT volume, origin, and voxel spacing
            sdd=1175,  # Source-to-detector distance (i.e., the C-arm's focal length)
            height=height,  # Height of the DRR (if width is not seperately provided, the generated image is square)
            width=width,  # Height of the DRR (if width is not seperately provided, the generated image is square)
            delx=p_target,  # Pixel spacing (in mm)
            x0=-0,
            y0=0,
            renderer="trilinear",
            reverse_x_axis=False
        ).to(device)
        K_can = drr.detector.intrinsic
        img = warp_to_canonical_space(img, K_real.unsqueeze(0), K_can.unsqueeze(0))

        sid = 750.0
        time_i = time.time()

        r, t = model1(img)

        T = t + torch.tensor([np.float32(0), np.float32((0 + sid)), np.float32(0)], dtype=t.dtype, device=t.device)

        points2d = drr.perspective_projection(pose, subject.fiducials.cuda())

        pose_pred = convert(r, T, parameterization=parameterization, convention=convention)
        points2d_pred = drr.perspective_projection(pose_pred, subject.fiducials.cuda())

        mask1 = (points2d_pred[:, :, 0] >= 0) & (points2d_pred[:, :, 0] <= width) & (points2d_pred[:, :, 1] >= 0) & (
                points2d_pred[:, :, 1] <= height)
        mask2 = (points2d[:, :, 0] >= 0) & (points2d[:, :, 0] <= width) & (points2d[:, :, 1] >= 0) & (
                points2d[:, :, 1] <= height)

        mask = mask1 & mask2
        mask2d = mask.unsqueeze(-1).expand_as(points2d_pred)

        points2d_pred[~mask2d] = float('nan')
        points2d[~mask2d] = float('nan')

        mpd1 = ((points2d - points2d_pred).norm(dim=-1).nanmean()) * p

        init_pose = drr(r, T, parameterization=parameterization, convention=convention)
        init_pose = transforms(init_pose.sum(dim=1, keepdim=True))

        for iter in range(1):
            input2 = torch.cat((img, init_pose), dim=1)

            delta_r, delta_t = model2(input2)

            T = delta_t

            delta_pose = convert(delta_r, T, parameterization=parameterization, convention=convention)
            pose_pred = pose_pred.compose(delta_pose)

            init_pose = drr(pose_pred)
            init_pose = transforms(init_pose.sum(dim=1, keepdim=True))

        time_f = time.time()

        points2d = drr.perspective_projection(pose, subject.fiducials.cuda())
        points2d_pred = drr.perspective_projection(pose_pred, subject.fiducials.cuda())
        mask1 = (points2d_pred[:, :, 0] >= 0) & (points2d_pred[:, :, 0] <= width) & (points2d_pred[:, :, 1] >= 0) & (
                points2d_pred[:, :, 1] <= height)
        mask2 = (points2d[:, :, 0] >= 0) & (points2d[:, :, 0] <= width) & (points2d[:, :, 1] >= 0) & (
                points2d[:, :, 1] <= height)

        mask = mask1 & mask2
        mask2d = mask.unsqueeze(-1).expand_as(points2d_pred)

        points2d_pred[~mask2d] = float('nan')
        points2d[~mask2d] = float('nan')
        mpd2 = ((points2d - points2d_pred).norm(dim=-1).nanmean()) * p
        print(mpd2.cpu().detach().numpy().round(2))
        mpd.append(mpd2.nanmean().detach().cpu().numpy())
        points3d = pose(subject.fiducials.cuda())
        points3d_pred = pose_pred(subject.fiducials.cuda())
        mtre = (points3d - points3d_pred).norm(dim=-1).mean(dim=-1)
        print(mtre.squeeze().detach().cpu().numpy().round(2))
        mtre_tot.append(mtre.nanmean().detach().cpu().numpy())

        final_pose = drr(pose_pred)
        final_pose = transforms(final_pose.sum(dim=1, keepdim=True))

        drr_gt = drr(pose)
        drr_gt = transforms(drr_gt.sum(dim=1, keepdim=True))

        # Figure 1: DRR GT pose
        plt.figure(figsize=(6, 6))
        plt.imshow(img[0].squeeze(0).detach().cpu(), cmap='gray')
        '''plt.scatter(points2d.squeeze()[indices, 0].detach().cpu().numpy(),
                    points2d.squeeze()[indices, 1].detach().cpu().numpy(),
                    c='fuchsia', s=70, alpha=1, marker='o')'''
        plt.axis('off')
        plt.savefig(os.path.join(dest_path, os.path.basename(os.path.splitext(img_path)[0]) + "_gt_pose.png"),bbox_inches='tight', pad_inches=0)
        plt.close()

        # Figure 2: DRR final pose
        plt.figure(figsize=(6, 6))
        plt.imshow(final_pose[0].squeeze(0).squeeze(0).detach().cpu(), cmap='gray')
        plt.scatter(points2d.squeeze()[indices, 0].detach().cpu().numpy(),
                    points2d.squeeze()[indices, 1].detach().cpu().numpy(),
                    c='fuchsia', s=70, alpha=1, marker='o')
        plt.scatter(points2d_pred.squeeze()[indices, 0].detach().cpu().numpy(),
                    points2d_pred.squeeze()[indices, 1].detach().cpu().numpy(),
                    c='lime', s=70, alpha=1, marker='x')
        plt.axis('off')
        plt.savefig(os.path.join(dest_path, os.path.basename(os.path.splitext(img_path)[0]) + "_final_pose.png"),bbox_inches='tight', pad_inches=0)
        plt.close()

        # Figure 3: DiffMap pose
        diff_pose = (drr_gt[0].squeeze(0).squeeze(0).detach().cpu() - final_pose[0].squeeze(0).detach().cpu()).abs()
        plt.figure(figsize=(6, 6))
        plt.imshow(diff_pose, origin='upper', cmap='Reds', vmin=0, vmax=1)
        plt.axis('off')
        plt.savefig(os.path.join(dest_path, os.path.basename(os.path.splitext(img_path)[0]) + "_diff_pose.png"),bbox_inches='tight', pad_inches=0)
        plt.close()

        # Figure 4: DiffMap domain gap
        diff_domain = (img.squeeze(0).squeeze(0).detach().cpu() - drr_gt[0].squeeze(0).detach().cpu()).abs()
        plt.figure(figsize=(6, 6))
        plt.imshow(diff_domain, origin='upper', cmap='Reds', vmin=0, vmax=1)
        plt.axis('off')
        plt.savefig(os.path.join(dest_path, os.path.basename(os.path.splitext(img_path)[0]) + "_diff_domain.png"),bbox_inches='tight', pad_inches=0)
        plt.close()

        plt.figure(figsize=(6, 6))
        plt.imshow(drr_gt[0].squeeze(0).squeeze(0).detach().cpu(), cmap='gray')

        plt.axis('off')
        plt.savefig(os.path.join(dest_path, os.path.basename(os.path.splitext(img_path)[0]) + "_gt_pose_DRR.png"),bbox_inches='tight', pad_inches=0)
        plt.close()


        del final_pose, init_pose, img, input2
        #### Parameters error
        M = pose_pred.rotation
        angles_pred = diffdrr.pose.matrix_to_euler_angles(M, convention="ZXY") * (180.0 / torch.pi)
        angles_GT = diffdrr.pose.matrix_to_euler_angles(pose.rotation, convention="ZXY") * (180.0 / torch.pi)

        print(diffdrr.pose.matrix_to_axis_angle(pose_pred.rotation))
        print(diffdrr.pose.matrix_to_axis_angle(pose.rotation))

        R_T = pose.rotation.transpose(1, 2)  # Shape: (1, 3, 3)
        R_p = pose_pred.rotation.transpose(1, 2)  # Shape: (1, 3, 3)

        T = torch.einsum("bij, bj -> bi", R_p, pose_pred.translation)
        gt_t = torch.einsum("bij, bj -> bi", R_T, pose.translation)
        print(T)
        print(gt_t)


        alpha.append(mean_absolute_error((angles_pred).detach().cpu()[:, 0], angles_GT.detach().cpu()[:, 0]).tolist())
        beta.append(mean_absolute_error((angles_pred).detach().cpu()[:, 1], angles_GT.detach().cpu()[:, 1]).tolist())
        gamma.append(mean_absolute_error((angles_pred).detach().cpu()[:, 2], angles_GT.detach().cpu()[:, 2]).tolist())
        x.append(mean_absolute_error(T.detach().cpu()[:, 0], gt_t.detach().cpu()[:, 0]).tolist())
        y.append(mean_absolute_error(T.detach().cpu()[:, 1], gt_t.detach().cpu()[:, 1]).tolist())
        z.append(mean_absolute_error(T.detach().cpu()[:, 2], gt_t.detach().cpu()[:, 2]).tolist())
        #print(mean_absolute_error(T.detach().cpu()[:, 1], gt_t.detach().cpu()[:, 1]))



alpha = np.concatenate(alpha)
beta = np.concatenate(beta)
gamma = np.concatenate(gamma)
x = np.concatenate(x)
y = np.concatenate(y)
z = np.concatenate(z)

mean_alpha = np.mean(alpha)
std_alpha = np.std(alpha)

mean_beta = np.mean(beta)
std_beta = np.std(beta)

mean_gamma = np.mean(gamma)
std_gamma = np.std(gamma)

mean_x = np.mean(x)
std_x = np.std(x)

mean_y = np.mean(y)
std_y = np.std(y)

mean_z = np.mean(z)
std_z = np.std(z)

df = pd.DataFrame({
        'mpd': mpd,
        'mtre': mtre_tot,
        "rx": np.array(alpha),
        "ry": np.array(beta),
        "rz": np.array(gamma),
        "tx": np.array(x),
        "ty": np.array(y),
        "tz": np.array(z)
    })
df.to_csv(os.path.join(dest_path,f'baseline_real.csv'), index=False)
