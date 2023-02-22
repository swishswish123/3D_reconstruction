import open3d as o3d
import numpy as np
from pathlib import Path


if __name__=='__main__':
    ########################## PARAMS ###################################

    method = 'online'
    type='random'
    folder = 'books'

    ########################## LOADING ALL ###################################

    project_path = Path(__file__).parent.resolve()
    # where 3D points and colors located
    reconstruction_output = f'{project_path}/reconstructions/{method}/{type}/{folder}'
    # 3D points
    xyz = np.load(f'{reconstruction_output}/points.npy')
    # colors
    c = np.load(f'{reconstruction_output}/colors.npy')
    
    ########################## PLOTTING ###################################

    # creating pointcloud with colors, saving it
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(c)
    o3d.io.write_point_cloud("./sync.ply", pcd)

    # visualizing point cloud ->>>>> doesn't work on mac m1
    pcd_load = o3d.io.read_point_cloud("./sync.ply")
        # convert Open3D.o3d.geometry.PointCloud to numpy array
    xyz_load = np.asarray(pcd_load.points)
    print('xyz_load')
    print(xyz_load)

    # save scatter as an image (change [0,1] range to [0,255] range with uint8 type)
    img = o3d.geometry.Image((xyz * 255).astype(np.uint8))
    o3d.io.write_image(".sync.png", img) 
    o3d.visualization.draw_geometries([img])
    o3d.visualization.draw_geometries([pcd])
    
    