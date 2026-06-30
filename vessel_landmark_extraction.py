import nibabel as nib
import pandas as pd
import os
import numpy as np
import trimesh
from skimage import measure
import open3d as o3d
base_dir = "/srv/storage/epione@storage2.sophia.grid5000.fr/ffacente/data/ljubljana"
subjects = [d for d in os.listdir(base_dir) if d.startswith("subject")]
destination_path = "/srv/storage/epione@storage2.sophia.grid5000.fr/ffacente/data//ljubljana/fiducials_new"
os.makedirs(destination_path, exist_ok=True)
num_list = [14,50,100,500,1000,5000]
for subject in subjects:
    for num in num_list:
        input_path = os.path.join(base_dir,subject, 'volume.nii.gz')
        output_path = os.path.join(destination_path, subject + ".nii.gz")
        print(input_path)

        img = nib.load(input_path)
        binary_mask = (img.get_fdata() > 200).astype(np.uint8)
        affine = img.affine

        new_img = nib.Nifti1Image(binary_mask, affine)
        nib.save(new_img, output_path)

        verts, faces, _, _ = measure.marching_cubes(binary_mask, level=0.5)
        verts_h = np.c_[verts, np.ones(len(verts))]
        verts_world = verts_h @ affine.T
        verts_world = verts_world[:, :3]
        mesh = trimesh.Trimesh(vertices=verts_world, faces=faces)
        components = mesh.split(only_watertight=False)
        mesh = max(components, key=lambda m: m.area)

        mesh_o3d = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(mesh.vertices),
            o3d.utility.Vector3iVector(mesh.faces)
        )


        pcd_dense = mesh_o3d.sample_points_uniformly(number_of_points=5000)
        points = pcd_dense.farthest_point_down_sample(num_samples  =num)

        df = pd.DataFrame(points.points, columns=["r", "a", "s"])
        os.makedirs(destination_path, exist_ok=True)
        csv_file_path = os.path.join(destination_path, subject + f'_{num}.csv')
        print(csv_file_path)
        df.to_csv(csv_file_path, index=False)