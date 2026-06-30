import matplotlib.pyplot as plt
import torch
import diffdrr.data
from diffdrr.drr import DRR
import numpy as np
import pandas as pd
import h5py
from diffdrr.pose import convert, axis_angle_to_matrix,matrix_to_axis_angle
import time
from torchvision.transforms import Compose, Lambda, Normalize, Resize
from torchvision.transforms.functional import center_crop,gaussian_blur
import torchvision.transforms as transforms
from models import regressor_attention, registration_attention
import torch.nn as nn
import os
import cv2
from utils import sobel
from diffdrr.pose import RigidTransform
from diffdrr.metrics import MultiscaleNormalizedCrossCorrelation2d
mtre_tot =[]
mpd =[]
alpha = []
beta = []
gamma = []
x = []
y = []
z = []
parametrization = "axis_angle"
convention = None
for id_number in range(1,7):

    model_dict = torch.load(f"/lustre/fswork/projects/rech/gfu/uur34ii/model/deepfluoro_models/specimen{id_number}/model_1499.pth")
    model1 = regressor_attention()
    model2 = registration_attention()

    model1.load_state_dict(model_dict["model_state_dict1"])
    model1.to("cuda")
    model1.eval()

    model2.load_state_dict(model_dict["model_state_dict2"])
    model2.to("cuda")
    model2.eval()
    resize = transforms.Resize((256, 256))

    transform = Compose(
                [
                    Resize((256, 256), antialias=True),
                    Lambda(lambda x: (((x - x.min()) / (x.max() - x.min()+ 1e-6)))),
                ]
            )

    norm_drr = Compose(
                [
                    #Lambda(lambda x: (x.max() + x.min() - x)),
                    #Lambda(lambda x: (((x - x.min()) / (x.max() - x.min()))))
                    Lambda(lambda x: (((x - x.min()) / (x.max() - x.min()+ 1e-6)))),
                    #Normalize(mean=0.3080, std=0.1494),

                ]
            )

    csv_file_path = f"/lustre/fswork/projects/rech/gfu/uur34ii/data/6_DOF_estimation/fiducials_RAS/specimen_{id_number}.csv"
    df = pd.read_csv(csv_file_path)
    fiducials = np.array(df.values) # Extract as NumPy array
    fiducials = torch.tensor([fiducials]).to(torch.float)

    ##### CT masking ########
    subject = diffdrr.data.read(
        f"/lustre/fswork/projects/rech/gfu/uur34ii/data/6_DOF_estimation/volumes_RAS/specimen_{id}.nii.gz",
        orientation="PA",
        labelmap=f"/lustre/fswork/projects/rech/gfu/uur34ii/data/6_DOF_estimation/segm_RAS/specimen_{id}.nii.gz",
        labels=1, bone_attenuation_multiplier=1, fiducials=fiducials)

    # Initialize the DRR module for generating synthetic X-rays
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())

    drr = DRR(
        subject,  # A torchio.Subject object storing the CT volume, origin, and voxel spacing
        sdd=1020,  # Source-to-detector distance (i.e., the C-arm's focal length)
        height=256,  # Height of the DRR (if width is not seperately provided, the generated image is square)
        delx=0.1940000057220459*1436/256,  # Pixel spacing (in mm)
        renderer="trilinear",
        reverse_x_axis=True
    ).to(device)

    sid = -675.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    filename = "/lustre/fswork/projects/rech/gfu/uur34ii/data/6_DOF_estimation/ipcai_2020_full_res_data.h5"
    f = h5py.File(filename, "r")
    assert id_number in {1, 2, 3, 4, 5, 6}
    specimen_id = [
        "17-1882",
        "18-1109",
        "18-0725",
        "18-2799",
        "18-2800",
        "17-1905",
    ][id_number - 1]
    dest_path = os.path.join("/lustre/fswork/projects/rech/gfu/uur34ii/data/6_DOF_estimation/", f"specimen_{id_number}")
    dest_path2 = os.path.join("/lustre/fswork/projects/rech/gfu/uur34ii/data/6_DOF_estimation/")

    os.makedirs(dest_path, exist_ok=True)
    specimen = f[specimen_id]

    volume = specimen["vol/pixels"][:]
    volume = torch.from_numpy(np.swapaxes(volume, 0, 2)).unsqueeze(0)
    # volume = torch.from_numpy(volume).unsqueeze(0).flip(1).flip(2)

    affine = np.eye(4)
    affine[:3, :3] = specimen["vol/dir-mat"][:]
    affine[:3, 3:] = specimen["vol/origin"][:]
    affine = torch.from_numpy(affine).to(torch.float32)
    from torchio import ScalarImage

    volume = ScalarImage(tensor=volume, affine=affine)
    isocenter = volume.get_center()
    anatomical2world = RigidTransform(torch.tensor(
        [
            [1, 0, 0, -isocenter[0]],
            [0, 1, 0, -isocenter[1]],
            [0, 0, 1, -isocenter[2]],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
    ))

    flip_z = RigidTransform(torch.tensor(
        [
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ]
    ).to(torch.float32))

    rot_180 = RigidTransform(torch.tensor(
        [
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    ).to(torch.float32))
    #%%
    mpd1 = []

    time3 = []
    mean_absolute_error = nn.L1Loss(reduction = 'none')
    ncc = MultiscaleNormalizedCrossCorrelation2d([None, 16], [0.5, 0.5]).to(device)
    projections = specimen["projections"]
    for key in projections.keys():
        print(key)
        drr.density = diffdrr.data.transform_hu_to_density(drr.subject.volume.data,
                                                           bone_attenuation_multiplier=torch.tensor(2,
                                                                                                    dtype=torch.float32)).to(
            'cuda').squeeze(0)
        data = projections[key]

        pose = torch.from_numpy(data["gt-poses/cam-to-pelvis-vol"][:])
        extrinsic = torch.from_numpy(f['proj-params']["extrinsic"][:]).unsqueeze(0)
        world2camera = RigidTransform(extrinsic)
        pose = RigidTransform(pose)

        pose = (
            flip_z
            .compose(world2camera.inverse())
            .compose(pose)
            .compose(anatomical2world)
            .compose(rot_180)

        )

        reorient = RigidTransform(subject.reorient)

        if data["rot-180-for-up"][()] == 1:
            pose = rot_180.compose(pose)

        pose = reorient.inverse().compose(pose).to(device)

        data = projections[key]
        landmarks2d = data["gt-landmarks"]
        landmarks2d = torch.from_numpy((np.array(list(landmarks2d.values())))).squeeze(-1).unsqueeze(0).to(device)
        landmarks2d = (landmarks2d-50)*256 / 1436
        img = (data["image/pixels"][:])

        if data["rot-180-for-up"][()] == 1:
            landmarks2d = 256 - landmarks2d
            img = cv2.rotate(img, cv2.ROTATE_180)
        img = torch.from_numpy(img)
        img = img.unsqueeze(0).unsqueeze(0)
        img = img.max().log() - img.log()

        img = center_crop(img, (1436, 1436))


        #img = drr(pose) ### USE THIS img IF YOU WANT TO TEST ON DRRs
        img = transform(img)

        gradient = sobel(img.to('cpu'))
        gradient_img = (gradient - gradient.min()) / (gradient.max() - gradient.min())
        input = img.to("cuda", dtype=torch.float32)

        time_i = time.time()

        r, t = model1(input)

        T = t + torch.tensor([np.float32(0), np.float32((0 + sid)), np.float32(0)],
                             dtype=t.dtype, device=t.device)
        pose_pred = convert(r, T, parameterization=parametrization, convention =convention)

        points2d_pred = drr.perspective_projection(pose_pred, subject.fiducials.cuda())
        landmarks2d = drr.perspective_projection(pose, subject.fiducials.cuda())
        mask1 = (points2d_pred[:, :, 0] >= 0) & (points2d_pred[:, :, 0] <= 256) & (points2d_pred[:, :, 1] >= 0) & (
                points2d_pred[:, :, 1] <= 256)
        mask2 = (landmarks2d[:, :, 0] >= 0) & (landmarks2d[:, :, 0] <= 256) & (landmarks2d[:, :, 1] >= 0) & (
                landmarks2d[:, :, 1] <= 256)

        mask = mask1 & mask2
        mask2d = mask.unsqueeze(-1).expand_as(points2d_pred)

        points2d_pred[~mask2d] = float('nan')
        landmarks2d[~mask2d] = float('nan')
        mpd_batch = ((landmarks2d - points2d_pred).norm(dim=-1).nanmean()) *(0.1940000057220459*1436/256)
        print(mpd_batch)
        mpd1.append(mpd_batch.mean().detach().cpu().numpy())
        pose_pred = convert(r, T, parameterization=parametrization, convention =convention)

        init_pose = drr(r, T, parameterization=parametrization, convention =convention)
        init_pose = norm_drr(init_pose.sum(dim=1, keepdim=True))

        input2 = torch.cat((input, init_pose), dim=1)

        delta_r, delta_t = model2((input2))

        T = delta_t
        delta_pose = convert(delta_r, T, parameterization=parametrization, convention=convention)
        pose_pred = pose_pred.compose(delta_pose)
        drr.density = diffdrr.data.transform_hu_to_density(drr.subject.volume.data,
                                                           bone_attenuation_multiplier=torch.tensor(5,
                                                                                                    dtype=torch.float32)).to(
            'cuda').squeeze(0)
        final = drr(pose_pred)
        final_pose = norm_drr(final.sum(dim=1, keepdim=True))
        img_gt = drr(pose)
        img_gt = norm_drr(img_gt.sum(dim=1, keepdim=True))


        time_f = time.time()
        time3.append(time_f - time_i)
        points2d_pred = drr.perspective_projection(pose_pred, subject.fiducials.cuda())
        landmarks2d = drr.perspective_projection(pose, subject.fiducials.cuda())

        mask1 = (points2d_pred[:, :, 0] >= 0) & (points2d_pred[:, :, 0] <= 256) & (points2d_pred[:, :, 1] >= 0) & (
                points2d_pred[:, :, 1] <= 256)
        mask2 = (landmarks2d[:, :, 0] >= 0) & (landmarks2d[:, :, 0] <= 256) & (landmarks2d[:, :, 1] >= 0) & (
                landmarks2d[:, :, 1] <= 256)
        mask = mask1 & mask2
        mask2d = mask.unsqueeze(-1).expand_as(points2d_pred)

        points2d_pred[~mask2d] = float('nan')
        landmarks2d[~mask2d] = float('nan')
        mpd_batch = ((landmarks2d - points2d_pred).norm(dim=-1).nanmean()) *(0.1940000057220459*1436/256)
        print(mpd_batch)
        mpd.append(mpd_batch.mean().detach().cpu().numpy())

        points3d = pose(subject.fiducials.cuda())
        points3d_pred = pose_pred(subject.fiducials.cuda())
        mtre = (points3d - points3d_pred).norm(dim=-1).mean(dim=-1)
        print(mtre)
        mtre_tot.append(mtre.nanmean().detach().cpu().numpy())
        #### Parameters error
        M = pose_pred.rotation
        angles_pred = diffdrr.pose.matrix_to_euler_angles(M, convention="ZXY") * (180.0 / torch.pi)
        angles_GT = diffdrr.pose.matrix_to_euler_angles(pose.rotation, convention="ZXY") * (180.0 / torch.pi)
        R_T = pose.rotation.transpose(1, 2)  # Shape: (1, 3, 3)
        R_p = pose_pred.rotation.transpose(1, 2)  # Shape: (1, 3, 3)

        T= torch.einsum("bij, bj -> bi", R_p, pose_pred.translation)
        gt_t = torch.einsum("bij, bj -> bi", R_T, pose.translation)
        alpha.append(mean_absolute_error((angles_pred).detach().cpu()[:, 0], angles_GT.detach().cpu()[:, 0]).tolist())
        beta.append(mean_absolute_error((angles_pred).detach().cpu()[:, 1], angles_GT.detach().cpu()[:, 1]).tolist())
        gamma.append(mean_absolute_error((angles_pred).detach().cpu()[:, 2], angles_GT.detach().cpu()[:, 2]).tolist())
        x.append(mean_absolute_error(T.detach().cpu()[:, 0], gt_t.detach().cpu()[:, 0]).tolist())
        y.append(mean_absolute_error(T.detach().cpu()[:, 1], gt_t.detach().cpu()[:, 1]).tolist())
        z.append(mean_absolute_error(T.detach().cpu()[:, 2], gt_t.detach().cpu()[:, 2]).tolist())

        # Figure 1: DRR GT pose
        plt.figure(figsize=(6, 6))
        plt.imshow(input[0].squeeze(0).detach().cpu(), cmap='gray')
        plt.scatter(landmarks2d.squeeze()[:, 0].detach().cpu().numpy(),
                    landmarks2d.squeeze()[:, 1].detach().cpu().numpy(),
                    c='fuchsia', s=70, alpha=1, marker='o')
        plt.axis('off')
        plt.savefig(os.path.join(dest_path, os.path.basename(os.path.splitext(key)[0]) + "_gt_pose.png"),
                    bbox_inches='tight', pad_inches=0)
        plt.close()
        # Figure 2: DRR final pose
        plt.figure(figsize=(6, 6))
        plt.imshow(final_pose[0].squeeze(0).squeeze(0).detach().cpu(), cmap='gray')
        plt.scatter(landmarks2d.squeeze()[:, 0].detach().cpu().numpy(),
                    landmarks2d.squeeze()[:, 1].detach().cpu().numpy(),
                    c='fuchsia', s=70, alpha=1, marker='o')
        plt.scatter(points2d_pred.squeeze()[:, 0].detach().cpu().numpy(),
                    points2d_pred.squeeze()[:, 1].detach().cpu().numpy(),
                    c='lime', s=70, alpha=1, marker='x')
        plt.axis('off')
        plt.savefig(os.path.join(dest_path, os.path.basename(os.path.splitext(key)[0]) + "_final_pose.png"),
                    bbox_inches='tight', pad_inches=0)
        plt.close()

print(f"mPD: mean = {np.array(mpd).mean():.4f}, std = {np.array(mpd).std():.4f}")
print(f"mtre: mean = {np.nanmean(np.array(mtre_tot)):.4f}, std = {np.nanstd(np.array(mtre_tot)):.4f}")

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
