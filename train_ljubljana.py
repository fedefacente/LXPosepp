from matplotlib import pyplot as plt
import torch
from tqdm import tqdm
import numpy as np
import os
import pandas as pd
import wandb
from diffdrr.pose import convert, axis_angle_to_matrix, matrix_to_axis_angle
import diffdrr.data
from diffdrr.drr import DRR
from torchvision.transforms import Compose, Lambda, Normalize
from models import regressor_attention, registration_attention
from geodesic import GeodesicSE3, DoubleGeodesic
from diffdrr.metrics import MultiscaleNormalizedCrossCorrelation2d, DoubleGeodesicSE3, \
    GradientNormalizedCrossCorrelation2d
from utils import sobel
from pytorch_transformers.optimization import WarmupCosineSchedule
from timm.utils.agc import adaptive_clip_grad as adaptive_clip_grad_
import kornia
from kornia.augmentation import AugmentationSequential
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--id", type=int, default=1)
args = parser.parse_args()
id_number = args.id

performance = wandb.init(project="ljubljana", name=f'subject{id_number:02d}', mode='offline')


def train(
        model,
        reg_model1,
        optimizer,
        scheduler,
        drr,
        transforms,
        device,
        batch_size,
        n_epochs,
        n_batches_per_epoch,
        parameterization,
        convention,
        output_path,
        subject,
        aug,
        start_epoch,
        height,
        width,
        p):
    torch.cuda.empty_cache()
    n_batches_per_epoch_val = 2
    min_delta = 0.01
    patience = 8
    best_loss = np.inf  # init to infinity
    for epoch in range(start_epoch, n_epochs + 1):
        losses = []
        loss_cnn1 = []
        loss_cnn2 = []
        losses_val = []

        model.train()
        reg_model1.train()

        for _ in (itr := tqdm(range(n_batches_per_epoch), leave=False)):
            sid = 750.0
            torch.manual_seed(torch.seed())

            rx = torch.normal(mean=0.0, std=0.05, size=(batch_size, 1)).to(device)
            ry = torch.normal(mean=0.0, std=0.05, size=(batch_size, 1)).to(device)
            mask = torch.tensor([True]*(batch_size//2) + [False]*(batch_size//2))
            perm = torch.randperm(batch_size)
            mask = mask[perm].unsqueeze(1)
            samples1 = torch.normal(mean=-0.16, std=0.18, size=(batch_size, 1))
            samples2 = torch.normal(mean=1.56, std=0.03, size=(batch_size, 1))
            rz = torch.where(mask, samples1, samples2).to(device)
            rot = torch.cat([rx, ry, rz], dim=1).to(device)

            tx = torch.normal(mean=0.0, std=10, size=(batch_size, 1)).to(device)
            ty = torch.normal(mean=0, std=20, size=(batch_size, 1)).to(device)

            mask = torch.rand(batch_size, 1) < 0.5  # 50% probability for each mode
            samples1 = torch.normal(mean=-23, std=2.5, size=(batch_size, 1))
            samples2 = torch.normal(mean=0.4, std=1, size=(batch_size, 1))
            tz = torch.where(mask, samples1, samples2).to(device)


            xyz = torch.cat([tx, ty, tz], dim=1).to(device)
            xyz[:, 1] = xyz[:, 1] + np.float32((0 + sid))
            img = drr(rot, xyz, parameterization=parameterization, convention=convention)
            img = transforms(img)
            img = aug(img)

            r, t = model(img)
            T = t + torch.tensor([np.float32(0), np.float32((0 + sid)), np.float32(0)], dtype=tx.dtype,
                                 device=tx.device)
            pose = convert(rot, xyz, parameterization=parameterization, convention=convention)

            points2d = drr.perspective_projection(pose, subject.fiducials.cuda())

            pose_pred = convert(r, T, parameterization=parameterization, convention=convention)
            points2d_pred = drr.perspective_projection(pose_pred, subject.fiducials.cuda())

            mask1 = (points2d_pred[:, :, 0] >= 0) & (points2d_pred[:, :, 0] <= width) & (
                        points2d_pred[:, :, 1] >= 0) & (
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
            input2 = torch.cat((img, init_pose), dim=1)

            delta_r, delta_t = reg_model1(input2)

            T = delta_t

            pose = convert(rot, xyz, parameterization=parameterization, convention=convention)
            points2d = drr.perspective_projection(pose, subject.fiducials.cuda())

            delta_pose = convert(delta_r, T, parameterization=parameterization, convention=convention)
            pose_pred = pose_pred.compose(delta_pose)

            points2d_pred = drr.perspective_projection(pose_pred, subject.fiducials.cuda())
            mask1 = (points2d_pred[:, :, 0] >= 0) & (points2d_pred[:, :, 0] <= width) & (
                        points2d_pred[:, :, 1] >= 0) & (
                            points2d_pred[:, :, 1] <= height)
            mask2 = (points2d[:, :, 0] >= 0) & (points2d[:, :, 0] <= width) & (points2d[:, :, 1] >= 0) & (
                    points2d[:, :, 1] <= height)

            mask = mask1 & mask2
            mask2d = mask.unsqueeze(-1).expand_as(points2d_pred)

            points2d_pred[~mask2d] = float('nan')
            points2d[~mask2d] = float('nan')
            mpd2 = ((points2d - points2d_pred).norm(dim=-1).nanmean()) * p

            del img
            loss = ( (mpd2.nanmean())) + ( (mpd1.nanmean()))
            optimizer.zero_grad()
            loss.mean().backward()
            adaptive_clip_grad_(model.parameters())
            adaptive_clip_grad_(reg_model1.parameters())
            optimizer.step()
            scheduler.step()

            losses.append(loss.mean().item())
            loss_cnn1.append(mpd1.mean().item())
            loss_cnn2.append(mpd2.mean().item())

            itr.set_description(f"Epoch [{epoch}/{n_epochs}]")
            itr.set_postfix(
                loss=loss.mean().item(),
            )
        losses = torch.tensor(losses)
        loss_cnn1 = torch.tensor(loss_cnn1)
        loss_cnn2 = torch.tensor(loss_cnn2)

        tqdm.write(f"Epoch {epoch + 1:04d} | Loss {losses.nanmean().item():.4f}")
        performance.log({"Training loss": losses.nanmean().item()})
        performance.log({"mPD cnn1": loss_cnn1.nanmean()})
        performance.log({"mPD cnn2": loss_cnn2.nanmean()})

        if (epoch + 1) % 100 == 0:
            model.eval()
            reg_model1.eval()
            with (((torch.no_grad()))):
                for _ in (tqdm(range(n_batches_per_epoch_val), leave=False)):
                    sid = 750.0
                    torch.manual_seed(0)

                    rx = torch.normal(mean=0.0, std=0.05, size=(batch_size, 1)).to(device)
                    ry = torch.normal(mean=0.0, std=0.05, size=(batch_size, 1)).to(device)

                    mask = torch.tensor([True] * (batch_size // 2) + [False] * (batch_size // 2))
                    perm = torch.randperm(batch_size)
                    mask = mask[perm].unsqueeze(1)
                    samples1 = torch.normal(mean=-0.16, std=0.18, size=(batch_size, 1))
                    samples2 = torch.normal(mean=1.56, std=0.03, size=(batch_size, 1))
                    rz = torch.where(mask, samples1, samples2).to(device)

                    rot = torch.cat([rx, ry, rz], dim=1).to(device)

                    tx = torch.normal(mean=0.0, std=10, size=(batch_size, 1)).to(device)
                    ty = torch.normal(mean=0, std=20, size=(batch_size, 1)).to(device)

                    mask = torch.rand(batch_size, 1) < 0.5  # 50% probability for each mode
                    samples1 = torch.normal(mean=-23, std=2.5, size=(batch_size, 1))
                    samples2 = torch.normal(mean=0.4, std=1, size=(batch_size, 1))
                    tz = torch.where(mask, samples1, samples2).to(device)

                    #tz = torch.normal(mean=0.0, std=10, size=(batch_size, 1)).to(device)

                    xyz = torch.cat([tx, ty, tz], dim=1).to(device)
                    xyz[:, 1] = xyz[:, 1] + np.float32((0 + sid))

                    img = drr(rot, xyz, parameterization=parameterization, convention=convention)

                    img = transforms(img)
                    img = aug(img)

                    r, t = model(img)
                    T = t + torch.tensor([np.float32(0), np.float32((0 + sid)), np.float32(0)], dtype=tx.dtype,
                                         device=tx.device)

                    pose = convert(rot, xyz, parameterization=parameterization, convention=convention)
                    points2d = drr.perspective_projection(pose, subject.fiducials.cuda())

                    pose_pred = convert(r, T, parameterization=parameterization, convention=convention)
                    points2d_pred = drr.perspective_projection(pose_pred, subject.fiducials.cuda())

                    mask1 = (points2d_pred[:, :, 0] >= 0) & (points2d_pred[:, :, 0] <= width) & (
                                points2d_pred[:, :, 1] >= 0) & (
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
                    #ncc1 = 1 - ncc(init_pose, img)
                    input2 = torch.cat((img, init_pose), dim=1)

                    delta_r, delta_t = reg_model1(input2)

                    T = delta_t
                    pose = convert(rot, xyz, parameterization=parameterization, convention=convention)
                    points2d = drr.perspective_projection(pose, subject.fiducials.cuda())

                    delta_pose = convert(delta_r, T, parameterization=parameterization, convention=convention)
                    pose_pred = pose_pred.compose(delta_pose)

                    points2d_pred = drr.perspective_projection(pose_pred, subject.fiducials.cuda())
                    mask1 = (points2d_pred[:, :, 0] >= 0) & (points2d_pred[:, :, 0] <= width) & (
                                points2d_pred[:, :, 1] >= 0) & (
                                    points2d_pred[:, :, 1] <= height)
                    mask2 = (points2d[:, :, 0] >= 0) & (points2d[:, :, 0] <= width) & (points2d[:, :, 1] >= 0) & (
                            points2d[:, :, 1] <= height)

                    mask = mask1 & mask2
                    mask2d = mask.unsqueeze(-1).expand_as(points2d_pred)

                    points2d_pred[~mask2d] = float('nan')
                    points2d[~mask2d] = float('nan')

                    mpd2 = ((points2d - points2d_pred).norm(dim=-1).nanmean()) * p

                    #ncc2 = (1 - ncc(final_pose, img))

                    del img

                    loss = ((mpd2.nanmean())) + ((mpd1.nanmean()))
                    losses_val.append(loss.mean().item())

                loss = torch.tensor(losses_val).nanmean().item()
                tqdm.write(f"Epoch {epoch + 1:04d} | Validation loss {loss:.4f}")
                performance.log({"Validation loss": loss})
                torch.save(
                    {
                        "model_state_dict1": model.state_dict(),
                        "model_state_dict2": reg_model1.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "epoch": epoch,
                    },
                    os.path.join(output_path, f'model_{epoch}.pth'),
                )
                if loss < best_loss - min_delta:
                    counter = 0
                    best_loss = loss
                    torch.save(
                        {
                            "model_state_dict1": model.state_dict(),
                            "model_state_dict2": reg_model1.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "epoch": epoch,
                        },
                        os.path.join(output_path, 'best1.pth'),
                    )
                else:
                    counter += 1
                    if counter >= patience:
                        print("Early stopping.")
                        break


def main(
        parameterization="axis_angle",
        convention=None,
        lr=1e-3,
        batch_size=16,
        n_epochs=1500,
        n_batches_per_epoch=100,
        resume=False
):
    device = torch.device("cuda")
    csv_file_path = f"/lustre/fswork/projects/rech/gfu/uur34ii/data/ljubljana/fiducials/subject{id_number:02d}_50.csv"
    df = pd.read_csv(csv_file_path)
    fiducials = np.array(df.values)  # Extract as NumPy array
    fiducials = torch.tensor([fiducials]).to(torch.float)

    subject = diffdrr.data.read(
        f"/lustre/fswork/projects/rech/gfu/uur34ii/data/ljubljana/subject{id_number:02d}/volume.nii.gz",
        orientation="AP", bone_attenuation_multiplier=1, fiducials=fiducials)

    height = 1920  # Rows
    width = 2480  # Columns
    p = 0.15399999916553497
    p = p * width/256
    height = int(256)
    width = int(256)

    drr = DRR(
        subject,  # A torchio.Subject object storing the CT volume, origin, and voxel spacing
        sdd=1175,  # Source-to-detector distance (i.e., the C-arm's focal length)
        height=height,
        width=width,
        delx=p,
        renderer="trilinear",
        reverse_x_axis=False
    ).to(device)

    model = regressor_attention()
    reg_model1 = registration_attention()

    model = model.to(device)
    reg_model1 = reg_model1.to(device)

    optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'lr': lr},
        {'params': reg_model1.parameters(), 'lr': lr}])

    scheduler = WarmupCosineSchedule(
        optimizer,
        5 * n_batches_per_epoch,
        n_epochs * n_batches_per_epoch - 5 * n_batches_per_epoch,
    )

    output_path = f"/lustre/fswork/projects/rech/gfu/uur34ii/model/lju/subject{id_number:02d}"
    os.makedirs(output_path, exist_ok=True)

    transforms = Compose(
        [
            # Lambda(lambda x: (x.max() + x.min() - x)),
            Lambda(lambda x: (((x - x.min()) / (x.max() - x.min() + 1e-6)))),
            # Normalize(mean=0.3080, std=0.1494),
        ]
    )

    aug = AugmentationSequential(
        kornia.augmentation.RandomSaltAndPepperNoise(amount=(0.005, 0.01), salt_vs_pepper=(0.4, 0.6), p=0.5,same_on_batch=False, keepdim=True),
        kornia.augmentation.RandomSharpness(p=0.5, keepdim=True, same_on_batch=False),
        kornia.augmentation.RandomGamma(gamma=(0.6, 1.5), gain=(1.0, 1.0), same_on_batch=False, p=0.5, keepdim=True),
        kornia.augmentation.RandomGaussianNoise(mean=0.0, std=0.03, same_on_batch=False, p=0.5, keepdim=True),
        kornia.augmentation.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1), p=0.5, keepdim=True,
                                               same_on_batch=False),
        kornia.augmentation.RandomPlasmaContrast(roughness=(0.1, 0.5), p=1.0, keepdim=True, same_on_batch=False),
        kornia.augmentation.RandomPlasmaBrightness(roughness=(0.1, 0.5), intensity=(-1, 1), p=1.0, keepdim=True,same_on_batch=False),
        kornia.augmentation.RandomPlasmaShadow(roughness=(0.1, 0.7), shade_intensity=(-0.5, 0.5), shade_quantity=(0.0, 0.5), same_on_batch=False, p=1.0, keepdim=False),

        data_keys=["input"],
        same_on_batch=None,
    )

    if resume == False:
        start_epoch = 0
    else:
        model_dict = torch.load(
            f"/lustre/fswork/projects/rech/gfu/uur34ii/model/RigidReg/subject{id_number:02d}_50/model_599.pth")
        model.load_state_dict(model_dict["model_state_dict1"])
        reg_model1.load_state_dict(model_dict["model_state_dict2"])
        start_epoch = model_dict['epoch']
        optimizer.load_state_dict(model_dict["optimizer_state_dict"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        scheduler.load_state_dict(model_dict["scheduler_state_dict"])



    train(
        model,
        reg_model1,
        optimizer,
        scheduler,
        drr,
        transforms,
        device,
        batch_size,
        n_epochs,
        n_batches_per_epoch,
        parameterization,
        convention,
        output_path,
        subject,
        aug,
        start_epoch,
        height,
        width,
        p)


if __name__ == "__main__":
    main()