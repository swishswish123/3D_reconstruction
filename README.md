<img src="assets/magicleap.png" width="240">

### 3D reconstruction with feature matching

## Dependencies
see requirements.txt

* Python 3 >= 3.5
* PyTorch >= 1.1
* OpenCV >= 3.4 (4.1.2.30 recommended for best GUI keyboard interaction, see this [note](#additional-notes))
* Matplotlib >= 3.1
* NumPy >= 1.18

`pip3 install numpy opencv-python torch matplotlib`

* scipy
(* open3d)

## Keypoints and feature matching
If using superpoint and superglue for feature matching follow the following instructions:

### Superpoint & superlue
* Website: [psarlin.com/superglue](https://psarlin.com/superglue) for videos, slides, recent updates, and more visualizations.

`match_pairs.py`: reads image pairs and creates matches stored in 'outputs' folder




