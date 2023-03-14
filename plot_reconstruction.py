import numpy as np
import pandas as pd 
import plotly.graph_objs as go
from pathlib import Path


if __name__=='__main__':
    
    ########################## PARAMS ###################################

    # method reconstruction was performed with:
    # sksurgery/ online/ prince /method_3 
    method = 'opencv'
    # folder of type of video
    # random / phantom / EM_tracker_calib
    type='aruco'
    # folder where image folder located
    # RANDOM, UNDISTORTED: arrow / brain  / checkerboard_test_calibrated / gloves / 
    # RANDOM, Distorted: books / points / spinal_section / spinal_section_pink
    # EM_TRACKING_CALIB testing_points /testing_lines
    # RANDOM, UNDISTORTED WITH MAC: mac_camera
    # PHANTOM: surface / right_in / phantom_surface_2 / both_mid / surface_undistorted
    folder = 'shelves_2'

    ########################## LOADING ALL ###################################

    project_path = Path(__file__).parent.resolve()
    reconstruction_output = f'{project_path}/reconstructions/{method}/{type}/{folder}'
    # 3D points
    x = np.load(f'{reconstruction_output}/points.npy')
    # colours
    c = np.load(f'{reconstruction_output}/colors.npy')
    
    # database of 3D points and colors
    df = pd.DataFrame({'X':x[:,0], 'Y':x[:,1], 'Z':x[:,2], 'R':c[:,0], 'G':c[:,1], 'B':c[:,2]})

    ########################## REMOVE OUTLIERS ###################################
    
    # find all rows with any col values larger than max_num
    max_num = 1000
    min_num = -1000
    df = df.drop(df[ (df['X'] > max_num) | (df['Y'] > max_num) | (df['Z'] > max_num) ].index, inplace=False)
    df = df.drop(df[ (df['X'] < min_num) | (df['Y'] < min_num) | (df['Z'] < min_num) ].index, inplace=False)

    ########################## PLOTTING ###################################

    # plot scatter graph with colors
    trace = go.Scatter3d(x=df.X,
                      y=df.Y,
                      z=df.Z,
                      mode='markers',
                      marker=dict(size=4,
                                  color=['rgb({},{},{})'.format(r,g,b) for r,g,b in zip(df.R.values, df.G.values, df.B.values)],
                                  opacity=0.9,)
                                  )

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
