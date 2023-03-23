import numpy as np
import pandas as pd 
import plotly.graph_objs as go
from pathlib import Path
import open3d as o3d

def visualise_reconstruction_plotly(reconstruction_output, min_num=-np.inf, max_num=np.inf):
    """
    function used to visualise 3D point clouds that were obtained using reconstruction.

    Args:
        reconstruction_output (str): path to folder where points.npy and colors.npy are located
        min_num (int): lower bound of what values not to visualise on plot
        max_num (int): upper bound of what values not to visualise on plot

    Returns:
        this will open a browser window with an interactive 3D point cloud created with plotly
    """
    ########################## LOADING ALL ###################################

    # 3D points
    x = np.load(f'{reconstruction_output}/points.npy')
    # colours
    c = np.load(f'{reconstruction_output}/colors.npy')

    # database of 3D points and colors
    df = pd.DataFrame({'X': x[:, 0], 'Y': x[:, 1], 'Z': x[:, 2], 'R': c[:, 0], 'G': c[:, 1], 'B': c[:, 2]})

    ########################## REMOVE OUTLIERS ###################################

    # find all rows with any col values larger than max_num
    df = df.drop(df[(df['X'] > max_num) | (df['Y'] > max_num) | (df['Z'] > max_num)].index, inplace=False)
    df = df.drop(df[(df['X'] < min_num) | (df['Y'] < min_num) | (df['Z'] < min_num)].index, inplace=False)

    ########################## PLOTTING ###################################

    # plot scatter graph with colors
    trace = go.Scatter3d(x=df.X,
                         y=df.Y,
                         z=df.Z,
                         mode='markers',
                         marker=dict(size=4,
                                     color=['rgb({},{},{})'.format(r, g, b) for r, g, b in
                                            zip(df.R.values, df.G.values, df.B.values)],
                                     opacity=0.9, )
                         )

    # removing automatic rescaling as we want aspect of all axes to be realistic, not scaled
    data = [trace]
    layout = go.Layout(margin=dict(l=0,
                                   r=0,
                                   b=0,
                                   t=0),
                       scene=dict(
                           aspectmode='data'
                       ))

    fig = go.Figure(data=data, layout=layout)
    fig.show()


def visualise_o3d(reconstruction_output):
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

def main():
    # ######################### PARAMS ###################################
    project_path = Path(__file__).parent.resolve()

    # method reconstruction was performed with:
    # sksurgery/ online/ prince /method_3
    method = 'opencv'
    # folder of type of video
    # random / phantom / EM_tracker_calib
    type = 'aruco'
    # folder where image folder located
    # RANDOM, UNDISTORTED: arrow / brain  / checkerboard_test_calibrated / gloves /
    # RANDOM, Distorted: books / points / spinal_section / spinal_section_pink
    # EM_TRACKING_CALIB testing_points /testing_lines
    # RANDOM, UNDISTORTED WITH MAC: mac_camera
    # PHANTOM: surface / right_in / phantom_surface_2 / both_mid / surface_undistorted
    folder = 'shelves_video'

    reconstruction_output = f'{project_path}/reconstructions/{method}/{type}/{folder}'

    # plot reconstruction
    visualise_reconstruction_plotly(reconstruction_output, min_num=-100, max_num=100)
    #visualise_o3d(reconstruction_output)



if __name__=='__main__':
    
    main()

    # MATPLOTLIB METHOD
    '''
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    #ax = fig.gca(projection='3d')
    ax.set_aspect("equal")
    ax.scatter(df.X,df.Y,df.Z, c=df[['R', 'G', 'B']]/255.0,s=10, marker='.')

    plt.savefig('3D_scatter_surface.png')
    print('done')
    '''