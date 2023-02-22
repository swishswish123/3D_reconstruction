### 3D reconstruction with feature matching

## Dependencies
see requirements.txt

* Python 3 >= 3.5
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


