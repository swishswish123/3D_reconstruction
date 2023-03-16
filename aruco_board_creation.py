'''
this file will create an aruco board of specified dimensions and aruco dict type.
bug: can't figure out how to make image same dimension for printing
'''
import cv2
from pathlib import Path


if __name__=='__main__':
    # https://stackoverflow.com/questions/59789491/using-board-create-for-aruco-markers-leads-to-error-due-to-objpoints-not-in-the

    project_path = Path(__file__).parent.resolve()
    # folder will be structured as follows:
    # assets/type/folder/images

    # Settings for the marker
    markers_w = 5 # Number of markers in the X direction.
    markers_h = 7 # Number of markers in the y direction.
    marker_length = 0.04 # length of aruco marker (m)
    marker_separation = 0.01 # separation between markers (m)
    ar = cv2.aruco.DICT_4X4_50 # aruco dictionary we will use
    aruco_dict = cv2.aruco.Dictionary_get(ar) # dictionary of markers provided

    # creat an aruco Board (The ids in ascending order starting on 0)
    grid_board = cv2.aruco.GridBoard_create(markers_w,
                                            markers_h,
                                            marker_length,
                                            marker_separation,
                                            aruco_dict)

    # now we want to print this board so we create an image:
    # convert to image
    img_size = (2400 , 3450)
    img_margin = 100 # margin in pixels, so none of the markers are touching the image border. In this case the margin is 10.
    img = grid_board.draw(img_size, img_margin, 10)

    cv2.imwrite("aruco_board.png", img)

