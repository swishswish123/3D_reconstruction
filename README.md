# 3D reconstruction with feature matching
This code can perform a 3D monocular sparse reconstruction from video frames (or image frames with matching or overlapping areas) and their corresponding camera poses. 

It also provides code for calibrating your camera, and for recording images with their corresponding camera positions using aruCo markers.

## 1. Create conda environment
```
conda create --name reconstruction
conda activate reconstruction
```

## 2. Install the dependencies
install requirements.txt

`pip3 install -r requirements.txt`

alternatively install dependencies as follows:

`pip3 install opencv-python matplotlib torch scikit-surgerycore`

for visualisation:

`pip3 install pandas plotly`

* (open3d) -> only needed if you want to visualize with open3D
(open3d 0.15.1)

## 3. camera calibration
Before performing any 3D reconstruction or recording videos, you will need to calibrate the camera you will use for recording the data. 
The code for calibrating a camera is provided in `calibration/calibration.py`

For more information on camera calibration see [this scikit-surgery tutorial](https://mphy0026.readthedocs.io/en/latest/summerschool/camera_calibration_demo.html#summerschoolcameracalibration)
TO-DO - add calibration code, images of process and detailed instructions

## 4. Recording your data
When recording your data, you need to store your camera's position in space. For this, you can use aruCo markers. 

#### Generate your aruCo board
First you will need to generate and print an aruCo board which will help you know relative camera positions. 
You can do this with:

```
python aruco_board_creation.py
```

You can change the size and number of markers as you see fit.
Once you have your aruCo board, print it and ensure it is the correct size you specified. 

TODO- CAN ADD PIC HERE

#### Record your video with the aruCo markers

Now that you have your aruco board, you can record a video of your scene using `record_video_and_pose_aruco.py`. Don't forget to put the board somewhere visibly in the scene so that the poses can also be recorded!

**Note- if you have changed the aruco board parameters in the previous stage, you will have to change them in this file aswell.
```
python record_video_and_pose_aruco.py
```


## 5. 3D Reconstructing your data

1. clone git repo and install all above requirements
2. download the [example data](https://drive.google.com/file/d/1n9fJ-a9MQr3BucgcRqIGlF40omruhb4r/view?usp=share_link) and place it in a folder called assets/random
3. Open python file reconstruction.py 
4. choose the correct parameters needed for reconstruction (see next section)
5. run reconstruction.py

```
python reconstruction.py
```

After running reconstruction.py, your point cloud's coordinates and RGB colours will appear under the folder `"reconstructions/<chosen-triangulation-method>/<chosen-tracking-method>".`


## params for reconstruction
In order to run the reconstruction, there are the following variable parameters you will need to choose:

### `type` and `folder` data folder and subfolder
Choose correct data folder and subfolder name where the data you want to reconstruct is located.
This should be structured as follows:

```
3D_Reconstruction
│
├── assets
│   └── <type>
│       └── <folder>
│           ├── images
│           │   └── 00000000.png
│           │   └── XXXXXXXX.png
│           │   └── XXXXXXXX.png
│           │   └── ...
│           └── rvecs.npy
│           └── tvecs.npy
├── ...
├── reconstruction.py
└── README.md
```

`type` this is the folder right under assets
`folder` this is the folder name right under `type`

### `matching_method`: Keypoints and feature matching
Choose a method of feature matching ('superglue'/'sift'/'manual') 
The following are options for the `matching_method` argument:

#### 'superglue'
Matches points between images using Superpoint & superlue:
* Website: [psarlin.com/superglue](https://psarlin.com/superglue) 

Note that this will save the feature matches under a folder called `outputs` and therefore if you in future run feature matching for the same data with superglue it will load the matches instead of computing them again. 
If you would like to overrun the feature matches every time, you can change this under the superglue parameters defined in `match_pairs.py`. You can also change any other parameters there!

To get superglue feature matches without running reconstruction, you can run `match_pairs.py` to get the matches- this will appear under 'outputs' folder. Remember to edit the parameters.

#### 'sift'
This will match your features using sift. It won't save features in a folder.

#### 'manual'
This will let you manually label keypoint pairs. When you run `reconstruction.py`, a window will pop up with the image pairs as subplots. You can then label the images by simply clicking a point in the left image an then the corresponding point in the right image (or vice-versa). You could also alternatively click all the points in the left image, followed by all the points on the right image so long as the order in which you clicked the points is the same. 

TODO- CAN ADD PIC HERE

### `method`: 
method used for triangulation

### `tracking`: 
This is the tracking method used. Eg EM/aruCo

### `frame_rate`:
Rate at which frames are chosen from folder. Eg if 1, then all frames are used. If 2 then every other frame etc.

### `intrinsics` and `distortion`:
path to where intrinsix and distortion matrices are stored respectively


## Visualising 3D reconstruction

To visualise results:



open `plot_reconstruction.py` and change the `type`, `folder` and `method` to be the same as you used in `reconstruction.py` (or to whatever reconstruction you want to visualise)
Then run plot_reconstruction.py

```
python plot_reconstruction.py
```


## For running tests:
Run setup file for importing modules

`pip install -e .`

then run pytest inside the tests folder

```
cd tests
pytest
```
