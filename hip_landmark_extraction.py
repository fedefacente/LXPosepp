from totalsegmentator.python_api import totalsegmentator
import nibabel as nib
import pandas as pd
import os
import numpy as np
import trimesh
from skimage import measure
import open3d as o3d

main_folder = "/srv/storage/epione@storage2.sophia.grid5000.fr/ffacente/data//6_DOF_estimation/volumes_RAS"
file_names = os.listdir(main_folder)
destination_path = "/srv/storage/epione@storage2.sophia.grid5000.fr/ffacente/data//6_DOF_estimation/fiducials_new"

os.makedirs(destination_path, exist_ok=True)
num_list = [14, 50, 100, 500, 1000, 5000]
for name in file_names:
    for num in num_list:
        dest_name = os.path.splitext(os.path.splitext(name)[0])[0]

        input_path = os.path.join(main_folder, name)
        output_path = os.path.join(destination_path, name)
        if __name__ == "__main__":
            img = totalsegmentator(
                input_path,
                output_path,
                roi_subset=[
                    'hip_left', 'hip_right', 'sacrum', 'vertebrae_S1', 'vertebrae_L5', 'vertebrae_L4', 'vertebrae_L3', 'vertebrae_L2', 'vertebrae_L1'
                ],
                ml=True
            )
            img_data = img.get_fdata()
            binary_mask = (img_data > 0).astype(np.uint8)
            affine = img.affine

            new_img = nib.Nifti1Image(binary_mask, affine)
            # nib.save(new_img, output_path)

            verts, faces, _, _ = measure.marching_cubes(binary_mask, level=0.5)
            verts_h = np.c_[verts, np.ones(len(verts))]
            verts_world = verts_h @ affine.T
            verts_world = verts_world[:, :3]
            mesh = trimesh.Trimesh(vertices=verts_world, faces=faces)

            mesh_o3d = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(mesh.vertices),
                o3d.utility.Vector3iVector(mesh.faces)
            )

            points = mesh_o3d.sample_points_poisson_disk(number_of_points=14, init_factor=5)
            points, _ = trimesh.sample.sample_surface_even(mesh, count=50, radius=0.01)

            pcd_dense = mesh_o3d.sample_points_uniformly(number_of_points=5000)
            points = pcd_dense.farthest_point_down_sample(num_samples=num)

            df = pd.DataFrame(points.points, columns=["r", "a", "s"])
            os.makedirs(destination_path, exist_ok=True)
            csv_file_path = os.path.join(destination_path, dest_name + f'_{num}.csv')
            print(csv_file_path)
            df.to_csv(csv_file_path, index=False)