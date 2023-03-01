### 3D reconstruction with feature matching

## Dependencies
see requirements.txt

From Superglue:
* Python 3 >= 3.5 (I am using 3.9)
* PyTorch >= 1.1
* OpenCV >= 3.4 (4.1.2.30 recommended for best GUI keyboard interaction, see this [note](#additional-notes))
* Matplotlib >= 3.1
* NumPy >= 1.18

`pip3 install numpy opencv-python torch matplotlib`

* scikit-surgerycore
* scipy
* pandas
* plotly
* (open3d) -> only needed if you want to visualize with open3D
(open3d 0.15.1)

## Keypoints and feature matching
If using superpoint and superglue for feature matching follow the following instructions:

### Superpoint & superlue
* Website: [psarlin.com/superglue](https://psarlin.com/superglue) for videos, slides, recent updates, and more visualizations.

Run `match_pairs.py` to get the matches- this will appear under 'outputs' folder

## Visualising 3D reconstruction

`plot_reconstruction.py`


# STEPS TO RUN RECONSTRUCTION

1. clone git repo and install all above requirements
2. download data and place it in a folder called assets/random https://drive.google.com/file/d/1n9fJ-a9MQr3BucgcRqIGlF40omruhb4r/view?usp=share_link
3. Open python file reconstruction.py 
4. choose correct data folder and subfolder name under "type"->folder under "assets" and "folder"->folder name after "type"
5. run reconstruction.py

To visualise results:
6. open plot_reconstruction and change the type and folder to be the same as in reconstruction.py
7. run plot_reconstruction.py